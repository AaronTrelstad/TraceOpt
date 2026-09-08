"""
TraceOpt Rewrite Experiment: Baseline vs Optimized Execution

The decisive experiment: take a real multi-stream pipeline workload,
replace global cudaDeviceSynchronize() barriers with targeted
cudaEventRecord/cudaStreamWaitEvent from the dependency frontier,
and measure the actual latency difference.

This proves (or disproves) that weakening synchronization actually
makes the GPU faster, not just that the analysis identifies opportunities.

Usage:
    python rewrite_experiment.py [--device cuda] [--trials 50]
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


def run_baseline(model, preprocess_buf, infer_buf, postprocess_buf,
                 streams, n_iters, device):
    """
    Baseline: global cudaDeviceSynchronize() between pipeline stages.

    Stream 0: preprocess → [DEVICE_SYNC] → idle
    Stream 1: idle → [DEVICE_SYNC] → inference → [DEVICE_SYNC] → idle
    Stream 2: idle → [DEVICE_SYNC] → idle → [DEVICE_SYNC] → postprocess
    """
    s_pre, s_inf, s_post = streams

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for i in range(n_iters):
        # Stage 1: preprocess on stream 0
        with torch.cuda.stream(s_pre):
            preprocessed = torch.nn.functional.interpolate(
                preprocess_buf, size=(224, 224), mode='bilinear',
                align_corners=False)
            # Copy result to inference buffer
            infer_buf.copy_(preprocessed)

        # GLOBAL SYNC — this is the over-synchronization
        torch.cuda.synchronize()

        # Stage 2: inference on stream 1
        with torch.cuda.stream(s_inf):
            output = model(infer_buf)
            # Copy logits to postprocess buffer
            postprocess_buf.copy_(output)

        # GLOBAL SYNC — another over-synchronization
        torch.cuda.synchronize()

        # Stage 3: postprocess on stream 2
        with torch.cuda.stream(s_post):
            topk_vals, topk_idx = torch.topk(postprocess_buf, k=5, dim=1)

    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def run_traceopt(model, preprocess_buf, infer_buf, postprocess_buf,
                 streams, n_iters, device):
    """
    TraceOpt optimized: replace global barriers with targeted event deps.

    The dependency frontier says:
    - preprocess(s0) writes infer_buf → inference(s1) reads infer_buf
      → need event: record on s0, wait on s1
    - inference(s1) writes postprocess_buf → postprocess(s2) reads it
      → need event: record on s1, wait on s2

    No other cross-stream dependencies exist.
    Stream 0 work does NOT block stream 2.
    Stream 1 work does NOT block stream 0's next iteration.
    """
    s_pre, s_inf, s_post = streams

    # Events for the dependency frontier
    evt_pre_done = torch.cuda.Event()   # preprocess → inference
    evt_inf_done = torch.cuda.Event()   # inference → postprocess

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for i in range(n_iters):
        # Stage 1: preprocess on stream 0
        with torch.cuda.stream(s_pre):
            preprocessed = torch.nn.functional.interpolate(
                preprocess_buf, size=(224, 224), mode='bilinear',
                align_corners=False)
            infer_buf.copy_(preprocessed)
            # Record: preprocess done
            evt_pre_done.record(s_pre)

        # Stage 2: inference on stream 1 — only waits for preprocess
        with torch.cuda.stream(s_inf):
            s_inf.wait_event(evt_pre_done)  # targeted dependency
            output = model(infer_buf)
            postprocess_buf.copy_(output)
            # Record: inference done
            evt_inf_done.record(s_inf)

        # Stage 3: postprocess on stream 2 — only waits for inference
        with torch.cuda.stream(s_post):
            s_post.wait_event(evt_inf_done)  # targeted dependency
            topk_vals, topk_idx = torch.topk(postprocess_buf, k=5, dim=1)

    # Wait for all streams at the end (for timing)
    end.record(s_post)
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def run_no_sync(model, preprocess_buf, infer_buf, postprocess_buf,
                streams, n_iters, device):
    """
    No synchronization at all (unsafe, max overlap baseline).
    Shows the theoretical maximum throughput.
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


def main():
    parser = argparse.ArgumentParser(
        description='TraceOpt rewrite experiment: baseline vs optimized')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--iters', type=int, default=20,
                        help='Pipeline iterations per trial')
    parser.add_argument('--trials', type=int, default=30,
                        help='Number of measurement trials')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--output', default='rewrite_results.json')
    args = parser.parse_args()

    device = torch.device(args.device)
    batch_size = args.batch_size

    print("=" * 70)
    print("TraceOpt Rewrite Experiment")
    print("=" * 70)

    # GPU info
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        print(f"GPU: {prop.name} ({prop.total_memory / 1e9:.1f} GB)")
        print(f"SMs: {prop.multi_processor_count}")

    print(f"Batch size: {batch_size}")
    print(f"Pipeline iters per trial: {args.iters}")
    print(f"Trials: {args.trials} (warmup: {args.warmup})")
    print()

    # Load model
    print("Loading ResNet-18...")
    model = build_model(device)

    # Allocate separate buffers per stage
    preprocess_buf = torch.randn(batch_size, 3, 224, 224, device=device)
    infer_buf = torch.randn(batch_size, 3, 224, 224, device=device)
    postprocess_buf = torch.randn(batch_size, 1000, device=device)

    # Create streams
    streams = [torch.cuda.Stream() for _ in range(3)]

    configs = [
        ('GLOBAL (baseline)', run_baseline),
        ('EVENT (TraceOpt)', run_traceopt),
        ('NONE (unsafe max)', run_no_sync),
    ]

    all_results = {}

    for name, run_fn in configs:
        print(f"\n--- {name} ---")

        # Warmup
        for _ in range(args.warmup):
            run_fn(model, preprocess_buf, infer_buf, postprocess_buf,
                   streams, args.iters, device)

        # Measure
        times = []
        for t in range(args.trials):
            ms = run_fn(model, preprocess_buf, infer_buf, postprocess_buf,
                        streams, args.iters, device)
            times.append(ms)

        times.sort()
        mean_ms = sum(times) / len(times)
        median_ms = times[len(times) // 2]
        min_ms = times[0]
        max_ms = times[-1]
        p95 = times[int(len(times) * 0.95)]

        # Per-iteration
        mean_per_iter = mean_ms / args.iters
        median_per_iter = median_ms / args.iters

        print(f"  Total ({args.iters} iters): "
              f"mean={mean_ms:.2f}ms  median={median_ms:.2f}ms  "
              f"min={min_ms:.2f}ms  max={max_ms:.2f}ms  p95={p95:.2f}ms")
        print(f"  Per iteration: "
              f"mean={mean_per_iter:.3f}ms  median={median_per_iter:.3f}ms")

        all_results[name] = {
            'mean_ms': mean_ms,
            'median_ms': median_ms,
            'min_ms': min_ms,
            'max_ms': max_ms,
            'p95_ms': p95,
            'per_iter_mean_ms': mean_per_iter,
            'per_iter_median_ms': median_per_iter,
            'times': times,
        }

    # Compute speedups
    baseline = all_results['GLOBAL (baseline)']
    traceopt = all_results['EVENT (TraceOpt)']
    nosync = all_results['NONE (unsafe max)']

    speedup_median = baseline['median_ms'] / traceopt['median_ms']
    speedup_mean = baseline['mean_ms'] / traceopt['mean_ms']
    theoretical_max = baseline['median_ms'] / nosync['median_ms']

    # How much of the theoretical gap did TraceOpt close?
    gap_total = baseline['median_ms'] - nosync['median_ms']
    gap_closed = baseline['median_ms'] - traceopt['median_ms']
    gap_pct = gap_closed / max(gap_total, 0.001) * 100

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"  GLOBAL baseline:  {baseline['median_ms']:.2f}ms "
          f"({baseline['per_iter_median_ms']:.3f}ms/iter)")
    print(f"  TraceOpt EVENT:   {traceopt['median_ms']:.2f}ms "
          f"({traceopt['per_iter_median_ms']:.3f}ms/iter)")
    print(f"  No-sync (unsafe): {nosync['median_ms']:.2f}ms "
          f"({nosync['per_iter_median_ms']:.3f}ms/iter)")
    print()
    print(f"  Speedup (TraceOpt vs baseline): {speedup_median:.3f}x "
          f"(median), {speedup_mean:.3f}x (mean)")
    print(f"  Theoretical max speedup:        {theoretical_max:.3f}x")
    print(f"  Gap closed:                     {gap_pct:.1f}%")
    print()

    if speedup_median > 1.01:
        print(f"  CONCLUSION: Weakening synchronization provides "
              f"{(speedup_median - 1) * 100:.1f}% latency reduction.")
        print(f"  TraceOpt closes {gap_pct:.0f}% of the gap to "
              f"theoretical maximum overlap.")
    elif speedup_median > 0.99:
        print(f"  CONCLUSION: No significant difference. "
              f"The workload may not have enough concurrent work "
              f"to benefit from weakened synchronization.")
    else:
        print(f"  CONCLUSION: TraceOpt is slower. "
              f"Event overhead may exceed synchronization savings.")

    print()
    print(f"  Transformation applied:")
    print(f"    Before: 2 x cudaDeviceSynchronize() per iteration")
    print(f"    After:  2 x cudaEventRecord + 2 x cudaStreamWaitEvent")
    print(f"    (dependency frontier: preprocess→inference, "
          f"inference→postprocess)")
    print(f"{'='*70}")

    # Save results
    save_data = {
        'config': {
            'batch_size': batch_size,
            'iters_per_trial': args.iters,
            'trials': args.trials,
            'warmup': args.warmup,
            'gpu': torch.cuda.get_device_properties(0).name
                   if torch.cuda.is_available() else 'unknown',
        },
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'times'}
                    for k, v in all_results.items()},
        'speedup_median': speedup_median,
        'speedup_mean': speedup_mean,
        'theoretical_max_speedup': theoretical_max,
        'gap_closed_pct': gap_pct,
    }
    with open(args.output, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
