"""
Timing Verification: Submission vs Completion for torch.compile + sync modes.

The compile_comparison.py had a timing bug: end.record() on the default stream
measures CPU enqueue time, not GPU completion time, for EVENT/NONE modes.

This script measures BOTH for every configuration and reports the difference.
It also runs multi-iteration correctness verification.

The critical test:
  - If completion_time ≈ submission_time → deep pipelining is real
  - If completion_time >> submission_time → the 5.5x was a measurement artifact

Usage:
    python timing_verification.py [--batch-size 16]
"""

import sys
import os
import time
import argparse
import json
import random

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def bootstrap_ci(data, n_bootstrap=10000, ci=0.95):
    n = len(data)
    means = []
    for _ in range(n_bootstrap):
        sample = [data[random.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((1 - ci) / 2 * n_bootstrap)]
    hi = means[int((1 + ci) / 2 * n_bootstrap)]
    return lo, hi


def build_model(device):
    from torchvision.models import resnet18, ResNet18_Weights
    return resnet18(weights=ResNet18_Weights.DEFAULT).to(device).eval()


def run_pipeline_3stage(model, preprocess_buf, infer_buf, post_buf,
                        streams, sync_mode, n_iters):
    """3-stage pipeline with configurable sync."""
    s_pre, s_inf, s_post = streams
    evt_pre = torch.cuda.Event()
    evt_inf = torch.cuda.Event()

    for i in range(n_iters):
        with torch.cuda.stream(s_pre):
            p = torch.nn.functional.interpolate(
                preprocess_buf, size=(224, 224), mode='bilinear',
                align_corners=False)
            infer_buf.copy_(p)

        if sync_mode == 'global':
            torch.cuda.synchronize()
        elif sync_mode == 'event':
            evt_pre.record(s_pre)

        with torch.cuda.stream(s_inf):
            if sync_mode == 'event':
                s_inf.wait_event(evt_pre)
            out = model(infer_buf)
            post_buf.copy_(out)

        if sync_mode == 'global':
            torch.cuda.synchronize()
        elif sync_mode == 'event':
            evt_inf.record(s_inf)

        with torch.cuda.stream(s_post):
            if sync_mode == 'event':
                s_post.wait_event(evt_inf)
            torch.topk(post_buf, k=5, dim=1)

        if sync_mode == 'global':
            torch.cuda.synchronize()


# ============================================================
# Test 1: Submission vs Completion timing
# ============================================================

def measure_submit_vs_complete(model, batch_size, device, sync_mode,
                               iter_counts, n_trials=10, n_warmup=5):
    """Measure both submission time and completion time."""
    preprocess_buf = torch.randn(batch_size, 3, 256, 256, device=device)
    infer_buf = torch.randn(batch_size, 3, 224, 224, device=device)
    post_buf = torch.randn(batch_size, 1000, device=device)
    streams = [torch.cuda.Stream() for _ in range(3)]

    results = {}

    with torch.no_grad():
        for n_iters in iter_counts:
            # Warmup
            for _ in range(n_warmup):
                torch.cuda.synchronize()
                run_pipeline_3stage(model, preprocess_buf, infer_buf, post_buf,
                                    streams, sync_mode, n_iters)
                torch.cuda.synchronize()

            submit_times = []
            complete_times = []

            for _ in range(n_trials):
                # === Submission time (CPU wall-clock, no final sync) ===
                torch.cuda.synchronize()  # clean state
                t0 = time.perf_counter_ns()
                run_pipeline_3stage(model, preprocess_buf, infer_buf, post_buf,
                                    streams, sync_mode, n_iters)
                t1 = time.perf_counter_ns()
                torch.cuda.synchronize()  # wait for GPU before next trial
                submit_ms = (t1 - t0) / 1e6
                submit_times.append(submit_ms)

                # === Completion time (CUDA events, draining all streams) ===
                torch.cuda.synchronize()  # clean state
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)

                start_evt.record()  # on default stream
                run_pipeline_3stage(model, preprocess_buf, infer_buf, post_buf,
                                    streams, sync_mode, n_iters)
                # Drain all work streams into default stream
                for s in streams:
                    drain = torch.cuda.Event()
                    drain.record(s)
                    torch.cuda.current_stream().wait_event(drain)
                end_evt.record()  # now on default stream, AFTER all work
                torch.cuda.synchronize()
                complete_ms = start_evt.elapsed_time(end_evt)
                complete_times.append(complete_ms)

            submit_times.sort()
            complete_times.sort()
            sub_med = submit_times[len(submit_times) // 2]
            comp_med = complete_times[len(complete_times) // 2]

            results[n_iters] = {
                'submit_median_ms': sub_med,
                'complete_median_ms': comp_med,
                'submit_per_iter': sub_med / n_iters,
                'complete_per_iter': comp_med / n_iters,
                'ratio': comp_med / max(sub_med, 0.001),
            }

    return results


# ============================================================
# Test 2: Multi-iteration correctness
# ============================================================

def verify_multi_iteration_correctness(model_eager, batch_size, device,
                                       n_iters=20):
    """
    Check if EVENT mode with shared buffers produces correct results
    across multiple iterations. Tests for cross-iteration WAR hazards.

    Strategy: run the pipeline with global sync (guaranteed correct) and
    with events, compare the FINAL output after all iterations.
    """
    preprocess_buf = torch.randn(batch_size, 3, 256, 256, device=device)
    streams = [torch.cuda.Stream() for _ in range(3)]

    # Also test compiled model if available
    models = {'eager': model_eager}
    try:
        compiled = torch.compile(model_eager, mode='reduce-overhead')
        dummy = torch.randn(batch_size, 3, 224, 224, device=device)
        with torch.no_grad():
            for _ in range(5):
                compiled(dummy)
                torch.cuda.synchronize()
        models['compile_RO'] = compiled
    except Exception:
        pass

    results = {}

    for model_label, model in models.items():
        for sync_mode in ['global', 'event', 'none']:
            # Use fresh buffers each time, seeded identically
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
            infer_buf = torch.zeros(batch_size, 3, 224, 224, device=device)
            post_buf = torch.zeros(batch_size, 1000, device=device)

            try:
                with torch.no_grad():
                    run_pipeline_3stage(model, preprocess_buf, infer_buf, post_buf,
                                        streams, sync_mode, n_iters)
                    torch.cuda.synchronize()

                # Capture final state
                final_post = post_buf.clone().cpu()
                topk_vals, topk_idx = torch.topk(post_buf, k=5, dim=1)

                key = f"{model_label}_{sync_mode}"
                results[key] = {
                    'post_buf_mean': final_post.mean().item(),
                    'post_buf_std': final_post.std().item(),
                    'topk_vals': topk_vals.cpu(),
                    'topk_idx': topk_idx.cpu(),
                }
            except RuntimeError as e:
                # 'none' mode can trigger CUDA assertions from data races
                # (corrupted tensors → topk out-of-bounds). Catch and record.
                key = f"{model_label}_{sync_mode}"
                results[key] = {
                    'post_buf_mean': float('nan'),
                    'post_buf_std': float('nan'),
                    'error': str(e),
                }
                print(f"    {key}: CUDA error (expected for none mode): {str(e)[:100]}")
                # Reset device to clear poisoned context
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    # Compare against global (guaranteed correct) baselines
    print(f"\n  Multi-iteration correctness ({n_iters} iterations):")

    verification = {}
    for model_label in models:
        ref_key = f"{model_label}_global"
        if ref_key not in results:
            continue
        ref = results[ref_key]
        for sync_mode in ['event', 'none']:
            test_key = f"{model_label}_{sync_mode}"
            if test_key not in results:
                continue
            test = results[test_key]

            val_diff = (ref['topk_vals'] - test['topk_vals']).abs().max().item()
            idx_match = torch.equal(ref['topk_idx'], test['topk_idx'])
            mean_diff = abs(ref['post_buf_mean'] - test['post_buf_mean'])

            if val_diff < 0.01 and idx_match:
                status = "PASS"
            elif val_diff < 0.1:
                status = f"MARGINAL (val_diff={val_diff:.6f})"
            else:
                status = f"FAIL (val_diff={val_diff:.4f}, mean_diff={mean_diff:.4f})"

            print(f"    {test_key:<30s} vs {ref_key}: {status}")
            verification[test_key] = {
                'status': status.split()[0],
                'val_diff': val_diff,
                'idx_match': idx_match,
                'mean_diff': mean_diff,
            }

    return verification


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Timing verification: submission vs completion')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--output', default='timing_verification.json')
    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 78)
    print("TIMING VERIFICATION: Submission vs Completion")
    print("=" * 78)

    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        print(f"GPU: {prop.name} ({prop.total_memory / 1e9:.1f} GB)")
    print(f"PyTorch: {torch.__version__}")
    print(f"Batch size: {args.batch_size}")

    all_results = {}

    # Build models
    eager_model = build_model(device)
    compiled_model = None
    try:
        compiled_model = torch.compile(build_model(device),
                                       mode='reduce-overhead')
        dummy = torch.randn(args.batch_size, 3, 224, 224, device=device)
        with torch.no_grad():
            for _ in range(5):
                compiled_model(dummy)
                torch.cuda.synchronize()
        print("torch.compile(reduce-overhead): OK")
    except Exception as e:
        print(f"torch.compile: FAILED ({e})")

    # =============================================
    # Test 1: Submission vs Completion
    # =============================================
    iter_counts = [1, 2, 4, 8, 16, 20, 32, 64]

    configs = [
        ('eager', eager_model),
    ]
    if compiled_model is not None:
        configs.append(('compile_RO', compiled_model))

    for model_label, model in configs:
        for sync_mode in ['global', 'event', 'none']:
            label = f"{model_label}_{sync_mode}"
            print(f"\n{'='*78}")
            print(f"SUBMIT vs COMPLETE: {label}")
            print(f"{'='*78}")

            r = measure_submit_vs_complete(
                model, args.batch_size, device, sync_mode,
                iter_counts, n_trials=10, n_warmup=5)

            print(f"\n  {'N':>4s} {'Submit(ms)':>11s} {'Complete(ms)':>13s} "
                  f"{'Sub/iter':>10s} {'Comp/iter':>10s} {'Ratio':>7s}")
            print(f"  {'-'*58}")
            for n in iter_counts:
                d = r[n]
                print(f"  {n:>4d} {d['submit_median_ms']:>9.2f}ms "
                      f"{d['complete_median_ms']:>11.2f}ms "
                      f"{d['submit_per_iter']:>8.3f}ms "
                      f"{d['complete_per_iter']:>8.3f}ms "
                      f"{d['ratio']:>6.2f}x")

            all_results[label] = {
                str(n): {k: v for k, v in d.items()}
                for n, d in r.items()
            }

    # =============================================
    # Summary comparison at N=20
    # =============================================
    print(f"\n\n{'='*78}")
    print(f"SUMMARY AT N=20 (the measurement used in compile_comparison.py)")
    print(f"{'='*78}\n")

    print(f"  {'Config':<25s} {'Submit':>10s} {'Complete':>10s} "
          f"{'Ratio':>7s} {'Complete/iter':>13s}")
    print(f"  {'-'*68}")

    for label in all_results:
        if '20' in all_results[label]:
            d = all_results[label]['20']
            print(f"  {label:<25s} {d['submit_median_ms']:>8.2f}ms "
                  f"{d['complete_median_ms']:>8.2f}ms "
                  f"{d['ratio']:>6.2f}x "
                  f"{d['complete_per_iter']:>11.3f}ms")

    # Check if compile results change
    eager_g = all_results.get('eager_global', {}).get('20', {})
    compile_g = all_results.get('compile_RO_global', {}).get('20', {})
    compile_e = all_results.get('compile_RO_event', {}).get('20', {})

    if eager_g and compile_g and compile_e:
        eg_comp = eager_g.get('complete_median_ms', 0)
        cg_comp = compile_g.get('complete_median_ms', 0)
        ce_comp = compile_e.get('complete_median_ms', 0)

        print(f"\n  CORRECTED compile comparison (completion time):")
        print(f"    EAGER_GLOBAL:       {eg_comp:.2f}ms")
        print(f"    COMPILE_GLOBAL:     {cg_comp:.2f}ms")
        print(f"    COMPILE_EVENT:      {ce_comp:.2f}ms")
        if cg_comp > 0:
            speedup = cg_comp / ce_comp
            print(f"    COMPILE_GLOBAL → COMPILE_EVENT: {speedup:.2f}x")
            if speedup > 1.05:
                print(f"    → REAL speedup of {(speedup-1)*100:.1f}% "
                      f"(completion-time verified)")
            elif speedup > 1.01:
                print(f"    → Small speedup of {(speedup-1)*100:.1f}%")
            else:
                print(f"    → No significant difference "
                      f"(the 5.5x was a measurement artifact)")

    # =============================================
    # Test 2: Multi-iteration correctness
    # =============================================
    print(f"\n\n{'='*78}")
    print("MULTI-ITERATION CORRECTNESS")
    print(f"{'='*78}")

    verification = verify_multi_iteration_correctness(
        eager_model, args.batch_size, device, n_iters=20)
    all_results['correctness'] = {
        k: {kk: vv for kk, vv in v.items() if kk != 'topk_vals'}
        for k, v in verification.items()
    } if verification else {}

    # =============================================
    # Stress test: more iterations
    # =============================================
    print(f"\n  Stress test (100 iterations):")
    verification_stress = verify_multi_iteration_correctness(
        eager_model, args.batch_size, device, n_iters=100)

    # Save
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
