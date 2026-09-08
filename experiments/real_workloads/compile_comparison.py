"""
TraceOpt vs torch.compile: Existing-System Comparison

The critical novelty test: does torch.compile / TorchInductor already
eliminate the synchronization that TraceOpt targets?

Modes tested:
  EAGER_GLOBAL:   Eager PyTorch + cudaDeviceSynchronize (current baseline)
  EAGER_EVENT:    Eager PyTorch + TraceOpt event rewrite
  COMPILE_GLOBAL: torch.compile + cudaDeviceSynchronize
  COMPILE_EVENT:  torch.compile + TraceOpt event rewrite
  COMPILE_ONLY:   torch.compile with no explicit sync (let compiler decide)
  MANUAL:         Expert double-buffered pipeline

The crucial comparisons:
  - EAGER_GLOBAL vs EAGER_EVENT: Does TraceOpt help eager PyTorch?
  - COMPILE_GLOBAL vs COMPILE_EVENT: Does TraceOpt help compiled PyTorch?
  - EAGER_EVENT vs COMPILE_ONLY: Is TraceOpt complementary to torch.compile?
  - COMPILE_ONLY vs MANUAL: Does the compiler match expert scheduling?

If COMPILE_ONLY already eliminates the sync → TraceOpt's target is solved.
If COMPILE_GLOBAL ≈ COMPILE_ONLY → compiler doesn't touch cross-stream sync.
If COMPILE_EVENT < COMPILE_GLOBAL → TraceOpt finds what the compiler misses.

Usage:
    python compile_comparison.py [--batch-size 16] [--stages 3]
"""

import sys
import os
import time
import argparse
import json
import random
import math

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def build_model(device):
    from torchvision.models import resnet18, ResNet18_Weights
    return resnet18(weights=ResNet18_Weights.DEFAULT).to(device).eval()


def build_compiled_model(device, mode='reduce-overhead'):
    """Build a torch.compile'd model."""
    model = build_model(device)
    try:
        compiled = torch.compile(model, mode=mode)
        return compiled
    except Exception as e:
        print(f"  torch.compile(mode='{mode}') failed: {e}")
        print(f"  Falling back to mode='default'")
        try:
            compiled = torch.compile(model, mode='default')
            return compiled
        except Exception as e2:
            print(f"  torch.compile(mode='default') also failed: {e2}")
            return None


# ============================================================
# Pipeline implementations
# ============================================================

def make_pipeline_eager_global(model, n_stages):
    """Eager model + cudaDeviceSynchronize between stages."""
    def run(preprocess_buf, bufs, streams, n_iters, device):
        if n_stages == 3:
            s_pre, s_inf, s_post = streams
            infer_buf, postprocess_buf = bufs
            for i in range(n_iters):
                with torch.cuda.stream(s_pre):
                    preprocessed = torch.nn.functional.interpolate(
                        preprocess_buf, size=(224, 224), mode='bilinear',
                        align_corners=False)
                    infer_buf.copy_(preprocessed)
                torch.cuda.synchronize()
                with torch.cuda.stream(s_inf):
                    output = model(infer_buf)
                    postprocess_buf.copy_(output)
                torch.cuda.synchronize()
                with torch.cuda.stream(s_post):
                    torch.topk(postprocess_buf, k=5, dim=1)
        elif n_stages == 4:
            s_pre, s_inf, s_post, s_red = streams
            infer_buf, postprocess_buf, reduce_buf = bufs
            for i in range(n_iters):
                with torch.cuda.stream(s_pre):
                    preprocessed = torch.nn.functional.interpolate(
                        preprocess_buf, size=(224, 224), mode='bilinear',
                        align_corners=False)
                    infer_buf.copy_(preprocessed)
                torch.cuda.synchronize()
                with torch.cuda.stream(s_inf):
                    output = model(infer_buf)
                    postprocess_buf.copy_(output)
                torch.cuda.synchronize()
                with torch.cuda.stream(s_post):
                    topk_vals, _ = torch.topk(postprocess_buf, k=5, dim=1)
                    reduce_buf.copy_(topk_vals)
                torch.cuda.synchronize()
                with torch.cuda.stream(s_red):
                    probs = torch.nn.functional.softmax(reduce_buf, dim=1)
                    probs.sum(dim=1, keepdim=True)
    return run


def make_pipeline_event(model, n_stages):
    """Model + TraceOpt event dependencies (no global sync)."""
    def run(preprocess_buf, bufs, streams, n_iters, device):
        if n_stages == 3:
            s_pre, s_inf, s_post = streams
            infer_buf, postprocess_buf = bufs
            evt_pre = torch.cuda.Event()
            evt_inf = torch.cuda.Event()
            for i in range(n_iters):
                with torch.cuda.stream(s_pre):
                    preprocessed = torch.nn.functional.interpolate(
                        preprocess_buf, size=(224, 224), mode='bilinear',
                        align_corners=False)
                    infer_buf.copy_(preprocessed)
                    evt_pre.record(s_pre)
                with torch.cuda.stream(s_inf):
                    s_inf.wait_event(evt_pre)
                    output = model(infer_buf)
                    postprocess_buf.copy_(output)
                    evt_inf.record(s_inf)
                with torch.cuda.stream(s_post):
                    s_post.wait_event(evt_inf)
                    torch.topk(postprocess_buf, k=5, dim=1)
        elif n_stages == 4:
            s_pre, s_inf, s_post, s_red = streams
            infer_buf, postprocess_buf, reduce_buf = bufs
            evt_pre = torch.cuda.Event()
            evt_inf = torch.cuda.Event()
            evt_post = torch.cuda.Event()
            for i in range(n_iters):
                with torch.cuda.stream(s_pre):
                    preprocessed = torch.nn.functional.interpolate(
                        preprocess_buf, size=(224, 224), mode='bilinear',
                        align_corners=False)
                    infer_buf.copy_(preprocessed)
                    evt_pre.record(s_pre)
                with torch.cuda.stream(s_inf):
                    s_inf.wait_event(evt_pre)
                    output = model(infer_buf)
                    postprocess_buf.copy_(output)
                    evt_inf.record(s_inf)
                with torch.cuda.stream(s_post):
                    s_post.wait_event(evt_inf)
                    topk_vals, _ = torch.topk(postprocess_buf, k=5, dim=1)
                    reduce_buf.copy_(topk_vals)
                    evt_post.record(s_post)
                with torch.cuda.stream(s_red):
                    s_red.wait_event(evt_post)
                    probs = torch.nn.functional.softmax(reduce_buf, dim=1)
                    probs.sum(dim=1, keepdim=True)
    return run


def make_pipeline_compile_only(model, n_stages):
    """
    torch.compile'd model, no explicit sync between stages.
    Let the compiler/runtime decide synchronization.
    """
    def run(preprocess_buf, bufs, streams, n_iters, device):
        if n_stages == 3:
            s_pre, s_inf, s_post = streams
            infer_buf, postprocess_buf = bufs
            for i in range(n_iters):
                with torch.cuda.stream(s_pre):
                    preprocessed = torch.nn.functional.interpolate(
                        preprocess_buf, size=(224, 224), mode='bilinear',
                        align_corners=False)
                    infer_buf.copy_(preprocessed)
                with torch.cuda.stream(s_inf):
                    output = model(infer_buf)
                    postprocess_buf.copy_(output)
                with torch.cuda.stream(s_post):
                    torch.topk(postprocess_buf, k=5, dim=1)
        elif n_stages == 4:
            s_pre, s_inf, s_post, s_red = streams
            infer_buf, postprocess_buf, reduce_buf = bufs
            for i in range(n_iters):
                with torch.cuda.stream(s_pre):
                    preprocessed = torch.nn.functional.interpolate(
                        preprocess_buf, size=(224, 224), mode='bilinear',
                        align_corners=False)
                    infer_buf.copy_(preprocessed)
                with torch.cuda.stream(s_inf):
                    output = model(infer_buf)
                    postprocess_buf.copy_(output)
                with torch.cuda.stream(s_post):
                    topk_vals, _ = torch.topk(postprocess_buf, k=5, dim=1)
                    reduce_buf.copy_(topk_vals)
                with torch.cuda.stream(s_red):
                    probs = torch.nn.functional.softmax(reduce_buf, dim=1)
                    probs.sum(dim=1, keepdim=True)
    return run


def make_pipeline_manual(model, n_stages):
    """Expert double-buffered pipeline with per-iteration events."""
    def run(preprocess_buf, bufs, streams, n_iters, device):
        if n_stages == 3:
            s_pre, s_inf, s_post = streams
            infer_bufs, post_bufs = bufs
            events_pre = [torch.cuda.Event() for _ in range(n_iters)]
            events_inf = [torch.cuda.Event() for _ in range(n_iters)]
            for i in range(n_iters):
                idx = i % 2
                with torch.cuda.stream(s_pre):
                    preprocessed = torch.nn.functional.interpolate(
                        preprocess_buf, size=(224, 224), mode='bilinear',
                        align_corners=False)
                    infer_bufs[idx].copy_(preprocessed)
                    events_pre[i].record(s_pre)
                with torch.cuda.stream(s_inf):
                    s_inf.wait_event(events_pre[i])
                    output = model(infer_bufs[idx])
                    post_bufs[idx].copy_(output)
                    events_inf[i].record(s_inf)
                with torch.cuda.stream(s_post):
                    s_post.wait_event(events_inf[i])
                    torch.topk(post_bufs[idx], k=5, dim=1)
        elif n_stages == 4:
            s_pre, s_inf, s_post, s_red = streams
            infer_bufs, post_bufs, reduce_bufs = bufs
            events_pre = [torch.cuda.Event() for _ in range(n_iters)]
            events_inf = [torch.cuda.Event() for _ in range(n_iters)]
            events_post = [torch.cuda.Event() for _ in range(n_iters)]
            for i in range(n_iters):
                idx = i % 2
                with torch.cuda.stream(s_pre):
                    preprocessed = torch.nn.functional.interpolate(
                        preprocess_buf, size=(224, 224), mode='bilinear',
                        align_corners=False)
                    infer_bufs[idx].copy_(preprocessed)
                    events_pre[i].record(s_pre)
                with torch.cuda.stream(s_inf):
                    s_inf.wait_event(events_pre[i])
                    output = model(infer_bufs[idx])
                    post_bufs[idx].copy_(output)
                    events_inf[i].record(s_inf)
                with torch.cuda.stream(s_post):
                    s_post.wait_event(events_inf[i])
                    topk_vals, _ = torch.topk(post_bufs[idx], k=5, dim=1)
                    reduce_bufs[idx].copy_(topk_vals)
                    events_post[i].record(s_post)
                with torch.cuda.stream(s_red):
                    s_red.wait_event(events_post[i])
                    probs = torch.nn.functional.softmax(reduce_bufs[idx],
                                                        dim=1)
                    probs.sum(dim=1, keepdim=True)
    return run


# ============================================================
# Measurement with bootstrap CI
# ============================================================

def bootstrap_ci(data, n_bootstrap=10000, ci=0.95):
    """Compute bootstrap confidence interval for the mean."""
    n = len(data)
    means = []
    for _ in range(n_bootstrap):
        sample = [data[random.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((1 - ci) / 2 * n_bootstrap)]
    hi = means[int((1 + ci) / 2 * n_bootstrap)]
    return lo, hi


def measure_pipeline(run_fn, preprocess_buf, bufs, streams, n_iters,
                     n_warmup, n_trials, device):
    """Measure a pipeline and return stats with CI.

    IMPORTANT: end event must be recorded AFTER all stream work completes.
    We drain all work streams into the default stream before recording end,
    so start.elapsed_time(end) captures actual GPU completion, not just
    CPU enqueue time.
    """
    # Warmup
    for _ in range(n_warmup):
        torch.cuda.synchronize()
        run_fn(preprocess_buf, bufs, streams, n_iters, device)
        torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(n_trials):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run_fn(preprocess_buf, bufs, streams, n_iters, device)
        # Drain all work streams into default stream before recording end
        for s in streams:
            drain_evt = torch.cuda.Event()
            drain_evt.record(s)
            torch.cuda.current_stream().wait_event(drain_evt)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    mean_ms = sum(times) / len(times)
    median_ms = times[len(times) // 2]
    ci_lo, ci_hi = bootstrap_ci(times)

    return {
        'mean_ms': mean_ms,
        'median_ms': median_ms,
        'min_ms': times[0],
        'max_ms': times[-1],
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
        'per_iter_ms': median_ms / n_iters,
        'times': times,
    }


# ============================================================
# Buffer allocation
# ============================================================

def alloc_single_bufs(batch_size, device, n_stages):
    if n_stages == 3:
        return (torch.randn(batch_size, 3, 224, 224, device=device),
                torch.randn(batch_size, 1000, device=device))
    elif n_stages == 4:
        return (torch.randn(batch_size, 3, 224, 224, device=device),
                torch.randn(batch_size, 1000, device=device),
                torch.randn(batch_size, 5, device=device))


def alloc_double_bufs(batch_size, device, n_stages):
    if n_stages == 3:
        return ([torch.randn(batch_size, 3, 224, 224, device=device)
                 for _ in range(2)],
                [torch.randn(batch_size, 1000, device=device)
                 for _ in range(2)])
    elif n_stages == 4:
        return ([torch.randn(batch_size, 3, 224, 224, device=device)
                 for _ in range(2)],
                [torch.randn(batch_size, 1000, device=device)
                 for _ in range(2)],
                [torch.randn(batch_size, 5, device=device)
                 for _ in range(2)])


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='TraceOpt vs torch.compile comparison')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--stages', type=int, default=3,
                        choices=[3, 4])
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--warmup', type=int, default=15)
    parser.add_argument('--output', default='compile_comparison.json')
    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 78)
    print("TraceOpt vs torch.compile Comparison")
    print("=" * 78)

    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        print(f"GPU: {prop.name} ({prop.total_memory / 1e9:.1f} GB)")
        print(f"SMs: {prop.multi_processor_count}")

    print(f"PyTorch: {torch.__version__}")
    print(f"Batch size: {args.batch_size}, Stages: {args.stages}")
    print(f"Iters/trial: {args.iters}, Trials: {args.trials}, "
          f"Warmup: {args.warmup}")
    print()

    # Build models
    print("Loading eager model...")
    eager_model = build_model(device)

    print("Compiling model with torch.compile...")
    compiled_model = build_compiled_model(device)
    has_compile = compiled_model is not None

    # Allocate buffers
    preprocess_buf = torch.randn(args.batch_size, 3, 224, 224, device=device)
    single_bufs = alloc_single_bufs(args.batch_size, device, args.stages)
    double_bufs = alloc_double_bufs(args.batch_size, device, args.stages)
    streams = [torch.cuda.Stream() for _ in range(args.stages)]

    # Build pipeline functions
    configs = []

    # Eager baselines
    configs.append(('EAGER_GLOBAL',
                    make_pipeline_eager_global(eager_model, args.stages),
                    single_bufs))
    configs.append(('EAGER_EVENT',
                    make_pipeline_event(eager_model, args.stages),
                    single_bufs))

    # Compiled versions
    if has_compile:
        configs.append(('COMPILE_GLOBAL',
                        make_pipeline_eager_global(compiled_model, args.stages),
                        single_bufs))
        configs.append(('COMPILE_EVENT',
                        make_pipeline_event(compiled_model, args.stages),
                        single_bufs))
        configs.append(('COMPILE_ONLY',
                        make_pipeline_compile_only(compiled_model, args.stages),
                        single_bufs))

    # Manual baseline
    configs.append(('MANUAL',
                    make_pipeline_manual(eager_model, args.stages),
                    double_bufs))

    # Run experiments
    results = {}

    with torch.no_grad():
        for name, run_fn, bufs in configs:
            print(f"\n--- {name} ---")
            try:
                r = measure_pipeline(run_fn, preprocess_buf, bufs, streams,
                                     args.iters, args.warmup, args.trials,
                                     device)
                results[name] = r
                print(f"  Median: {r['median_ms']:.2f}ms "
                      f"({r['per_iter_ms']:.3f}ms/iter)")
                print(f"  95% CI: [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]ms")
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()
                results[name] = {'error': str(e)}

    # ================================================================
    # Analysis
    # ================================================================
    print(f"\n\n{'='*78}")
    print(f"COMPARISON: {args.stages}-stage pipeline, bs={args.batch_size}")
    print(f"{'='*78}\n")

    # Table
    baseline = results.get('EAGER_GLOBAL', {})
    baseline_med = baseline.get('median_ms', 1)

    print(f"  {'Mode':<18s} {'Median':>9s} {'95% CI':>20s} "
          f"{'vs Eager':>9s} {'Per-iter':>10s}")
    print(f"  {'-'*68}")

    for name in ['EAGER_GLOBAL', 'EAGER_EVENT', 'COMPILE_GLOBAL',
                 'COMPILE_EVENT', 'COMPILE_ONLY', 'MANUAL']:
        r = results.get(name)
        if r is None or 'error' in r:
            continue
        spd = baseline_med / r['median_ms']
        pct = (spd - 1) * 100
        print(f"  {name:<18s} {r['median_ms']:>7.2f}ms "
              f"[{r['ci_lo']:>7.2f}, {r['ci_hi']:>7.2f}]ms "
              f"{spd:>7.3f}x  {r['per_iter_ms']:>8.3f}ms")

    # Key comparisons
    print(f"\n  KEY COMPARISONS:")
    print(f"  {'-'*60}")

    def compare(name_a, name_b, label):
        ra = results.get(name_a)
        rb = results.get(name_b)
        if ra is None or rb is None or 'error' in ra or 'error' in rb:
            return
        spd = ra['median_ms'] / rb['median_ms']
        delta = ra['median_ms'] - rb['median_ms']
        print(f"  {label}")
        print(f"    {name_a}: {ra['median_ms']:.2f}ms → "
              f"{name_b}: {rb['median_ms']:.2f}ms")
        print(f"    Δ = {delta:.2f}ms ({(spd-1)*100:+.1f}%)")
        return spd

    # Does TraceOpt help eager?
    s1 = compare('EAGER_GLOBAL', 'EAGER_EVENT',
                 '1. TraceOpt on eager PyTorch:')
    if s1 and s1 > 1.005:
        print(f"    → YES, TraceOpt helps eager by "
              f"{(s1-1)*100:.1f}%")
    elif s1:
        print(f"    → No significant difference")

    print()

    # Does torch.compile already solve it?
    s2 = compare('COMPILE_GLOBAL', 'COMPILE_ONLY',
                 '2. Does torch.compile remove sync?')
    if s2 and s2 > 1.005:
        print(f"    → YES, compiler removes sync ({(s2-1)*100:.1f}% gain)")
        print(f"    → TraceOpt's target is partially addressed")
    elif s2:
        print(f"    → NO, compiler preserves synchronization structure")
        print(f"    → TraceOpt addresses an unoptimized dimension")

    print()

    # Does TraceOpt help compiled code?
    s3 = compare('COMPILE_GLOBAL', 'COMPILE_EVENT',
                 '3. TraceOpt on compiled PyTorch:')
    if s3 and s3 > 1.005:
        print(f"    → YES, TraceOpt helps compiled code by "
              f"{(s3-1)*100:.1f}%")
        print(f"    → NOVELTY: torch.compile misses this optimization")
    elif s3:
        print(f"    → No significant difference")

    print()

    # Is TraceOpt complementary to torch.compile?
    eager_event = results.get('EAGER_EVENT')
    compile_only = results.get('COMPILE_ONLY')
    if (eager_event and compile_only and
            'error' not in eager_event and 'error' not in compile_only):
        print(f"  4. Complementarity:")
        print(f"    EAGER_EVENT:  {eager_event['median_ms']:.2f}ms "
              f"(sync weakening)")
        print(f"    COMPILE_ONLY: {compile_only['median_ms']:.2f}ms "
              f"(compilation)")
        if eager_event['median_ms'] < compile_only['median_ms']:
            print(f"    → TraceOpt (sync weakening) alone beats compilation")
        else:
            print(f"    → Compilation alone beats TraceOpt sync weakening")

    print()

    # How does TraceOpt compare to manual?
    compare('EAGER_EVENT', 'MANUAL',
            '5. TraceOpt vs expert manual:')

    # ================================================================
    # Conclusion
    # ================================================================
    print(f"\n{'='*78}")
    print("CONCLUSION")
    print(f"{'='*78}")

    compile_global = results.get('COMPILE_GLOBAL')
    compile_event = results.get('COMPILE_EVENT')
    eager_event_r = results.get('EAGER_EVENT')

    if (compile_global and compile_event and
            'error' not in compile_global and 'error' not in compile_event):
        compile_helps = (compile_global['median_ms'] /
                         compile_event['median_ms'])
        if compile_helps > 1.01:
            print(f"\n  STRONG NOVELTY: TraceOpt improves compiled code by "
                  f"{(compile_helps-1)*100:.1f}%.")
            print(f"  torch.compile does NOT eliminate the targeted "
                  f"synchronization.")
        elif compile_helps > 1.005:
            print(f"\n  WEAK NOVELTY: Small improvement on compiled code "
                  f"({(compile_helps-1)*100:.1f}%).")
        else:
            print(f"\n  NO NOVELTY on compiled code "
                  f"({(compile_helps-1)*100:.1f}%).")
            print(f"  Check if torch.compile already handles this case.")

    print(f"{'='*78}")

    # Save
    save_data = {
        'config': {
            'batch_size': args.batch_size,
            'stages': args.stages,
            'iters': args.iters,
            'trials': args.trials,
            'warmup': args.warmup,
            'pytorch_version': torch.__version__,
            'gpu': torch.cuda.get_device_properties(0).name
                   if torch.cuda.is_available() else 'unknown',
        },
        'results': {
            name: {k: v for k, v in r.items() if k != 'times'}
            for name, r in results.items()
        },
    }
    with open(args.output, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
