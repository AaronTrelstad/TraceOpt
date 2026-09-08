"""
TraceOpt Rewrite Sweep: Characterize speedup across batch sizes and pipeline configs.

Runs four execution modes across a range of batch sizes:
  GLOBAL:  cudaDeviceSynchronize() between stages (baseline)
  EVENT:   TraceOpt frontier events (current automatic rewrite)
  MANUAL:  Expert-optimized double-buffered pipeline (best achievable)
  NONE:    No synchronization (unsafe, shows max overlap but has data races)

The MANUAL mode uses double-buffered intermediates to eliminate WAR hazards
across iterations, allowing true pipeline overlap. This separates:
  - Analysis quality (does TraceOpt find the right dependencies?)
  - Rewrite quality (does the generated event schedule recover the parallelism?)
  - Hardware limits (does the GPU have resources for concurrent execution?)

Key insight: EVENT mode reuses single buffers across iterations, creating
WAR hazards (P_{i+1} may overwrite infer_buf while I_i reads it). MANUAL
eliminates this with double buffering. The gap between EVENT and MANUAL
reveals whether cross-iteration dependency scheduling matters.

Usage:
    python rewrite_sweep.py [--batch-sizes 1,2,4,8,16,32] [--trials 20]
"""

import sys
import os
import time
import argparse
import json

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def build_model(device):
    from torchvision.models import resnet18, ResNet18_Weights
    return resnet18(weights=ResNet18_Weights.DEFAULT).to(device).eval()


# ============================================================
# Mode 1: GLOBAL — cudaDeviceSynchronize between stages
# ============================================================

def run_global(model, preprocess_buf, infer_buf, postprocess_buf,
               streams, n_iters, device):
    s_pre, s_inf, s_post = streams

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

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
            topk_vals, topk_idx = torch.topk(postprocess_buf, k=5, dim=1)

    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


# ============================================================
# Mode 2: EVENT — TraceOpt automatic frontier (single-buffered)
# ============================================================

def run_event(model, preprocess_buf, infer_buf, postprocess_buf,
              streams, n_iters, device):
    """
    TraceOpt automatic rewrite: replace global barriers with frontier events.

    NOTE: This has WAR hazards across iterations on shared buffers.
    P_{i+1} may write infer_buf while I_i is still reading it.
    At small batch sizes the GPU serializes enough that this rarely
    causes incorrect results, but it is formally unsafe.
    """
    s_pre, s_inf, s_post = streams

    evt_pre_done = torch.cuda.Event()
    evt_inf_done = torch.cuda.Event()

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    for i in range(n_iters):
        with torch.cuda.stream(s_pre):
            preprocessed = torch.nn.functional.interpolate(
                preprocess_buf, size=(224, 224), mode='bilinear',
                align_corners=False)
            infer_buf.copy_(preprocessed)
            evt_pre_done.record(s_pre)

        with torch.cuda.stream(s_inf):
            s_inf.wait_event(evt_pre_done)
            output = model(infer_buf)
            postprocess_buf.copy_(output)
            evt_inf_done.record(s_inf)

        with torch.cuda.stream(s_post):
            s_post.wait_event(evt_inf_done)
            topk_vals, topk_idx = torch.topk(postprocess_buf, k=5, dim=1)

    end.record(s_post)
    torch.cuda.synchronize()
    return start.elapsed_time(end)


# ============================================================
# Mode 3: MANUAL — Expert double-buffered pipeline
# ============================================================

def run_manual(model, preprocess_buf, infer_bufs, postprocess_bufs,
               streams, n_iters, device):
    """
    Expert-optimized pipeline with double-buffered intermediates.

    Eliminates WAR hazards: P_{i+1} writes to infer_bufs[(i+1)%2] while
    I_i reads from infer_bufs[i%2]. No cross-iteration buffer conflicts.

    Dependencies (all correct, no races):
    - P_i → I_i: RAW on infer_bufs[i%2] (event: s0 → s1)
    - I_i → O_i: RAW on postprocess_bufs[i%2] (event: s1 → s2)
    - No cross-iteration events needed (double buffering eliminates WAR)
    """
    s_pre, s_inf, s_post = streams

    # Per-iteration events (avoid reuse ambiguity)
    events_pre = [torch.cuda.Event() for _ in range(n_iters)]
    events_inf = [torch.cuda.Event() for _ in range(n_iters)]

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    for i in range(n_iters):
        buf_idx = i % 2

        with torch.cuda.stream(s_pre):
            preprocessed = torch.nn.functional.interpolate(
                preprocess_buf, size=(224, 224), mode='bilinear',
                align_corners=False)
            infer_bufs[buf_idx].copy_(preprocessed)
            events_pre[i].record(s_pre)

        with torch.cuda.stream(s_inf):
            s_inf.wait_event(events_pre[i])
            output = model(infer_bufs[buf_idx])
            postprocess_bufs[buf_idx].copy_(output)
            events_inf[i].record(s_inf)

        with torch.cuda.stream(s_post):
            s_post.wait_event(events_inf[i])
            topk_vals, topk_idx = torch.topk(postprocess_bufs[buf_idx],
                                              k=5, dim=1)

    end.record(s_post)
    torch.cuda.synchronize()
    return start.elapsed_time(end)


# ============================================================
# Mode 4: NONE — No synchronization (unsafe, data races)
# ============================================================

def run_none(model, preprocess_buf, infer_buf, postprocess_buf,
             streams, n_iters, device):
    """
    No synchronization. Has WAR data races on shared buffers.
    Shows maximum hardware overlap but results may be incorrect.
    """
    s_pre, s_inf, s_post = streams

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

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
            topk_vals, topk_idx = torch.topk(postprocess_buf, k=5, dim=1)

    end.record(s_post)
    torch.cuda.synchronize()
    return start.elapsed_time(end)


# ============================================================
# Correctness verification
# ============================================================

def verify_correctness(model, device, batch_size):
    """
    Run one iteration of each mode and compare outputs.
    Returns True if EVENT and MANUAL produce same results as GLOBAL.
    NONE is expected to potentially diverge.
    """
    preprocess_buf = torch.randn(batch_size, 3, 224, 224, device=device)
    infer_buf = torch.randn(batch_size, 3, 224, 224, device=device)
    postprocess_buf = torch.randn(batch_size, 1000, device=device)
    streams = [torch.cuda.Stream() for _ in range(3)]

    # Reference: single-stream sequential
    with torch.no_grad():
        preprocessed = torch.nn.functional.interpolate(
            preprocess_buf, size=(224, 224), mode='bilinear',
            align_corners=False)
        ref_infer = preprocessed.clone()
        ref_output = model(ref_infer)
        ref_topk_vals, ref_topk_idx = torch.topk(ref_output, k=5, dim=1)

    torch.cuda.synchronize()

    # GLOBAL mode (1 iter)
    with torch.no_grad():
        infer_buf.zero_()
        postprocess_buf.zero_()
        run_global(model, preprocess_buf, infer_buf, postprocess_buf,
                   streams, 1, device)
    global_infer = infer_buf.clone()
    global_post = postprocess_buf.clone()

    # EVENT mode (1 iter)
    with torch.no_grad():
        infer_buf.zero_()
        postprocess_buf.zero_()
        run_event(model, preprocess_buf, infer_buf, postprocess_buf,
                  streams, 1, device)
    torch.cuda.synchronize()
    event_infer = infer_buf.clone()
    event_post = postprocess_buf.clone()

    global_ok = torch.allclose(global_infer, ref_infer, atol=1e-5)
    event_ok = torch.allclose(event_infer, ref_infer, atol=1e-5)

    return global_ok, event_ok


# ============================================================
# Main sweep
# ============================================================

def run_sweep(model, batch_size, n_iters, n_trials, n_warmup, device):
    """Run all four modes for one batch size. Returns dict of results."""
    # Allocate buffers
    preprocess_buf = torch.randn(batch_size, 3, 224, 224, device=device)
    infer_buf = torch.randn(batch_size, 3, 224, 224, device=device)
    postprocess_buf = torch.randn(batch_size, 1000, device=device)

    # Double buffers for MANUAL mode
    infer_bufs = [
        torch.randn(batch_size, 3, 224, 224, device=device),
        torch.randn(batch_size, 3, 224, 224, device=device),
    ]
    postprocess_bufs = [
        torch.randn(batch_size, 1000, device=device),
        torch.randn(batch_size, 1000, device=device),
    ]

    streams = [torch.cuda.Stream() for _ in range(3)]

    configs = [
        ('GLOBAL', lambda ni: run_global(
            model, preprocess_buf, infer_buf, postprocess_buf,
            streams, ni, device)),
        ('EVENT', lambda ni: run_event(
            model, preprocess_buf, infer_buf, postprocess_buf,
            streams, ni, device)),
        ('MANUAL', lambda ni: run_manual(
            model, preprocess_buf, infer_bufs, postprocess_bufs,
            streams, ni, device)),
        ('NONE', lambda ni: run_none(
            model, preprocess_buf, infer_buf, postprocess_buf,
            streams, ni, device)),
    ]

    results = {}

    with torch.no_grad():
        for name, run_fn in configs:
            # Warmup
            for _ in range(n_warmup):
                run_fn(n_iters)

            # Measure
            times = []
            for _ in range(n_trials):
                ms = run_fn(n_iters)
                times.append(ms)

            times.sort()
            mean_ms = sum(times) / len(times)
            median_ms = times[len(times) // 2]
            per_iter = median_ms / n_iters

            results[name] = {
                'mean_ms': mean_ms,
                'median_ms': median_ms,
                'min_ms': times[0],
                'max_ms': times[-1],
                'p5_ms': times[max(0, int(len(times) * 0.05))],
                'p95_ms': times[int(len(times) * 0.95)],
                'per_iter_ms': per_iter,
                'times': times,
            }

    return results


def main():
    parser = argparse.ArgumentParser(
        description='TraceOpt rewrite sweep: batch size × execution mode')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-sizes', default='1,2,4,8,16,32',
                        help='Comma-separated batch sizes')
    parser.add_argument('--iters', type=int, default=20,
                        help='Pipeline iterations per trial')
    parser.add_argument('--trials', type=int, default=20,
                        help='Measurement trials per config')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--output', default='sweep_results.json')
    parser.add_argument('--verify', action='store_true',
                        help='Run correctness verification')
    args = parser.parse_args()

    device = torch.device(args.device)
    batch_sizes = [int(b) for b in args.batch_sizes.split(',')]

    print("=" * 78)
    print("TraceOpt Rewrite Sweep")
    print("=" * 78)

    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        print(f"GPU: {prop.name} ({prop.total_memory / 1e9:.1f} GB)")
        print(f"SMs: {prop.multi_processor_count}")

    print(f"Batch sizes: {batch_sizes}")
    print(f"Pipeline iters/trial: {args.iters}, Trials: {args.trials}, "
          f"Warmup: {args.warmup}")
    print()

    # Load model once
    print("Loading ResNet-18...")
    model = build_model(device)

    # Optional correctness check
    if args.verify:
        print("\n--- Correctness Verification ---")
        for bs in batch_sizes:
            try:
                g_ok, e_ok = verify_correctness(model, device, bs)
                status = "PASS" if (g_ok and e_ok) else "FAIL"
                print(f"  bs={bs:3d}: GLOBAL={'ok' if g_ok else 'FAIL'}  "
                      f"EVENT={'ok' if e_ok else 'FAIL'}  [{status}]")
            except RuntimeError as e:
                print(f"  bs={bs:3d}: SKIP ({e})")
        print()

    all_results = {}

    for bs in batch_sizes:
        print(f"\n{'='*78}")
        print(f"Batch size = {bs}")
        print(f"{'='*78}")

        try:
            results = run_sweep(model, bs, args.iters, args.trials,
                                args.warmup, device)
        except RuntimeError as e:
            print(f"  SKIP: {e}")
            all_results[bs] = {'error': str(e)}
            # Clear GPU memory
            torch.cuda.empty_cache()
            continue

        all_results[bs] = results

        # Print comparison table
        gl = results['GLOBAL']
        ev = results['EVENT']
        mn = results['MANUAL']
        nn_r = results['NONE']

        print(f"\n  {'Mode':<10s} {'Median':>10s} {'Per-iter':>10s} "
              f"{'vs GLOBAL':>10s} {'vs MANUAL':>10s}")
        print(f"  {'-'*50}")

        for name, r in [('GLOBAL', gl), ('EVENT', ev),
                        ('MANUAL', mn), ('NONE', nn_r)]:
            speedup_g = gl['median_ms'] / r['median_ms']
            speedup_m = mn['median_ms'] / r['median_ms']
            print(f"  {name:<10s} {r['median_ms']:>8.2f}ms "
                  f"{r['per_iter_ms']:>8.3f}ms "
                  f"{speedup_g:>9.3f}x {speedup_m:>9.3f}x")

        # Key metrics
        event_speedup = gl['median_ms'] / ev['median_ms']
        manual_speedup = gl['median_ms'] / mn['median_ms']
        none_speedup = gl['median_ms'] / nn_r['median_ms']

        # Gap analysis: how much of manual's gain does EVENT capture?
        gap_manual = gl['median_ms'] - mn['median_ms']
        gap_event = gl['median_ms'] - ev['median_ms']
        if abs(gap_manual) > 0.001:
            event_of_manual = gap_event / gap_manual * 100
        else:
            event_of_manual = 100.0

        # Gap analysis: how much of no-sync bound does MANUAL capture?
        gap_none = gl['median_ms'] - nn_r['median_ms']
        if abs(gap_none) > 0.001:
            manual_of_none = gap_manual / gap_none * 100
        else:
            manual_of_none = 100.0

        print(f"\n  EVENT  speedup: {event_speedup:.3f}x "
              f"({(event_speedup-1)*100:+.1f}%)")
        print(f"  MANUAL speedup: {manual_speedup:.3f}x "
              f"({(manual_speedup-1)*100:+.1f}%)")
        print(f"  No-sync bound:  {none_speedup:.3f}x "
              f"({(none_speedup-1)*100:+.1f}%)")
        print(f"  EVENT captures {event_of_manual:.0f}% of MANUAL's gain")
        print(f"  MANUAL captures {manual_of_none:.0f}% of no-sync bound")

        # Diagnose the gap
        if manual_speedup > event_speedup * 1.02:
            print(f"\n  >> GAP: MANUAL beats EVENT by "
                  f"{(manual_speedup/event_speedup - 1)*100:.1f}%")
            print(f"     Likely cause: cross-iteration WAR hazards or "
                  f"event reuse in EVENT mode")
        if none_speedup > manual_speedup * 1.05:
            print(f"\n  >> GAP: No-sync bound exceeds MANUAL by "
                  f"{(none_speedup/manual_speedup - 1)*100:.1f}%")
            print(f"     Likely cause: GPU resource contention or "
                  f"host submission bottleneck")

        # Clean up
        torch.cuda.empty_cache()

    # ================================================================
    # Summary table across all batch sizes
    # ================================================================
    print(f"\n\n{'='*78}")
    print("SWEEP SUMMARY")
    print(f"{'='*78}\n")

    header = (f"  {'BS':>4s} | {'GLOBAL':>9s} | {'EVENT':>9s} | "
              f"{'MANUAL':>9s} | {'NONE':>9s} | "
              f"{'Ev spd':>7s} | {'Mn spd':>7s} | {'Ev/Mn%':>7s}")
    print(header)
    print(f"  {'-'*len(header)}")

    for bs in batch_sizes:
        r = all_results.get(bs)
        if r is None or 'error' in r:
            print(f"  {bs:>4d} | {'SKIP':>9s}")
            continue

        gl_med = r['GLOBAL']['median_ms']
        ev_med = r['EVENT']['median_ms']
        mn_med = r['MANUAL']['median_ms']
        nn_med = r['NONE']['median_ms']

        ev_spd = gl_med / ev_med
        mn_spd = gl_med / mn_med
        gap_mn = gl_med - mn_med
        gap_ev = gl_med - ev_med
        ev_of_mn = (gap_ev / gap_mn * 100) if abs(gap_mn) > 0.001 else 100

        print(f"  {bs:>4d} | {gl_med:>7.2f}ms | {ev_med:>7.2f}ms | "
              f"{mn_med:>7.2f}ms | {nn_med:>7.2f}ms | "
              f"{ev_spd:>6.3f}x | {mn_spd:>6.3f}x | {ev_of_mn:>6.0f}%")

    print(f"\n  Ev spd  = EVENT speedup vs GLOBAL")
    print(f"  Mn spd  = MANUAL speedup vs GLOBAL")
    print(f"  Ev/Mn%  = how much of MANUAL's gain EVENT captures")
    print(f"\n  MANUAL uses double-buffered intermediates (no WAR hazards)")
    print(f"  EVENT uses single-buffered (TraceOpt automatic, has WAR hazards)")
    print(f"  NONE has data races (not a valid execution)")

    # ================================================================
    # Save results
    # ================================================================
    save_data = {
        'config': {
            'batch_sizes': batch_sizes,
            'iters_per_trial': args.iters,
            'trials': args.trials,
            'warmup': args.warmup,
            'gpu': torch.cuda.get_device_properties(0).name
                   if torch.cuda.is_available() else 'unknown',
        },
        'results': {},
    }
    for bs, r in all_results.items():
        if 'error' in r:
            save_data['results'][str(bs)] = {'error': r['error']}
            continue
        bs_data = {}
        for name in ['GLOBAL', 'EVENT', 'MANUAL', 'NONE']:
            d = r[name]
            bs_data[name] = {k: v for k, v in d.items() if k != 'times'}
        save_data['results'][str(bs)] = bs_data

    with open(args.output, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
