"""
Focused eager-mode timing experiment.

Measures EVENT vs GLOBAL speedup across multiple pipeline depths and batch
sizes with high trial count to get stable eager-mode numbers.

Uses CUDA event timing with stream draining (completion time, not submit time).

Usage:
    python eager_timing.py [--output eager_timing.json]
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


def bootstrap_ci(data, n_bootstrap=5000, ci=0.95):
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


def run_pipeline(model, preprocess_buf, infer_buf, post_buf,
                 streams, sync_mode, n_iters, n_stages):
    """Run N-stage pipeline with configurable sync."""
    if n_stages == 3:
        s_pre, s_inf, s_post = streams[:3]
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

    elif n_stages == 4:
        s_pre, s_inf, s_post, s_red = streams[:4]
        evt_pre = torch.cuda.Event()
        evt_inf = torch.cuda.Event()
        evt_post = torch.cuda.Event()
        reduce_buf = torch.zeros_like(post_buf)

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
                topk_vals, _ = torch.topk(post_buf, k=5, dim=1)
                reduce_buf[:, :5].copy_(topk_vals)

            if sync_mode == 'global':
                torch.cuda.synchronize()
            elif sync_mode == 'event':
                evt_post.record(s_post)

            with torch.cuda.stream(s_red):
                if sync_mode == 'event':
                    s_red.wait_event(evt_post)
                reduce_buf.mean(dim=1)

            if sync_mode == 'global':
                torch.cuda.synchronize()


def measure_config(model, batch_size, device, n_stages, n_iters=20,
                   n_warmup=15, n_trials=40):
    """Measure completion time for global and event sync modes."""

    preprocess_buf = torch.randn(batch_size, 3, 256, 256, device=device)
    infer_buf = torch.randn(batch_size, 3, 224, 224, device=device)
    post_buf = torch.randn(batch_size, 1000, device=device)
    streams = [torch.cuda.Stream() for _ in range(n_stages)]

    results = {}

    for sync_mode in ['global', 'event']:
        # Warmup
        with torch.no_grad():
            for _ in range(n_warmup):
                torch.cuda.synchronize()
                run_pipeline(model, preprocess_buf, infer_buf, post_buf,
                             streams, sync_mode, n_iters, n_stages)
                torch.cuda.synchronize()

        # Measure completion time with CUDA events + stream draining
        complete_times = []
        with torch.no_grad():
            for _ in range(n_trials):
                torch.cuda.synchronize()
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record()
                run_pipeline(model, preprocess_buf, infer_buf, post_buf,
                             streams, sync_mode, n_iters, n_stages)
                # Drain all work streams into default before end event
                for s in streams:
                    drain = torch.cuda.Event()
                    drain.record(s)
                    torch.cuda.current_stream().wait_event(drain)
                end_evt.record()
                torch.cuda.synchronize()
                complete_times.append(start_evt.elapsed_time(end_evt))

        complete_times.sort()
        med = complete_times[len(complete_times) // 2]
        lo, hi = bootstrap_ci(complete_times)

        results[sync_mode] = {
            'median_ms': med,
            'per_iter_ms': med / n_iters,
            'ci_lo': lo / n_iters,
            'ci_hi': hi / n_iters,
            'all_ms': complete_times,
        }

    speedup = results['global']['median_ms'] / results['event']['median_ms']
    speedup_pct = (speedup - 1) * 100

    return {
        'global': {k: v for k, v in results['global'].items() if k != 'all_ms'},
        'event': {k: v for k, v in results['event'].items() if k != 'all_ms'},
        'speedup': speedup,
        'speedup_pct': speedup_pct,
        'global_all': results['global']['all_ms'],
        'event_all': results['event']['all_ms'],
    }


def main():
    parser = argparse.ArgumentParser(
        description='Focused eager-mode timing experiment')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default='eager_timing.json')
    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 78)
    print("EAGER-MODE TIMING EXPERIMENT")
    print("=" * 78)

    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        print(f"GPU: {prop.name} ({prop.total_memory / 1e9:.1f} GB)")
    print(f"PyTorch: {torch.__version__}")

    model = build_model(device)

    # Test matrix: batch sizes × pipeline depths
    batch_sizes = [4, 8, 16, 32]
    n_stages_list = [3, 4]

    all_results = {}

    for bs in batch_sizes:
        for n_stages in n_stages_list:
            label = f"bs{bs}_{n_stages}stage"
            print(f"\n{'='*60}")
            print(f"CONFIG: {label} (eager, 40 trials, 20 iters)")
            print(f"{'='*60}")

            try:
                r = measure_config(model, bs, device, n_stages,
                                   n_iters=20, n_warmup=15, n_trials=40)

                print(f"  GLOBAL: {r['global']['per_iter_ms']:.3f} ms/iter "
                      f"[{r['global']['ci_lo']:.3f}, {r['global']['ci_hi']:.3f}]")
                print(f"  EVENT:  {r['event']['per_iter_ms']:.3f} ms/iter "
                      f"[{r['event']['ci_lo']:.3f}, {r['event']['ci_hi']:.3f}]")
                print(f"  Speedup: {r['speedup_pct']:+.1f}%")

                all_results[label] = {
                    'batch_size': bs,
                    'n_stages': n_stages,
                    'global': r['global'],
                    'event': r['event'],
                    'speedup': r['speedup'],
                    'speedup_pct': r['speedup_pct'],
                }
            except Exception as e:
                print(f"  ERROR: {e}")
                all_results[label] = {'error': str(e)}

    # Summary
    print(f"\n\n{'='*78}")
    print(f"SUMMARY")
    print(f"{'='*78}")
    print(f"{'Config':<25s} {'GLOBAL':>10s} {'EVENT':>10s} {'Speedup':>8s}")
    print(f"{'-'*55}")
    for label, r in all_results.items():
        if 'error' in r:
            continue
        print(f"{label:<25s} "
              f"{r['global']['per_iter_ms']:>8.3f}ms "
              f"{r['event']['per_iter_ms']:>8.3f}ms "
              f"{r['speedup_pct']:>+6.1f}%")

    # Save
    output = {
        'config': {
            'gpu': (torch.cuda.get_device_properties(0).name
                    if torch.cuda.is_available() else 'N/A'),
            'pytorch_version': torch.__version__,
            'n_iters': 20,
            'n_warmup': 15,
            'n_trials': 40,
            'mode': 'eager',
        },
        'results': all_results,
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
