"""
Dispatch/Compute Ratio Sweep: The predictive model experiment.

Tests whether TraceOpt's benefit is predictable from:
    R = T_GPU / T_CPU_dispatch

Creates configurations with different R values by varying:
  - Compilation mode: eager (R≈1), compile default (R≈1.5), compile RO (R≈5)
  - Model size: resnet18 vs resnet50 (changes T_GPU)
  - Batch size: varies both T_GPU and T_CPU
  - Artificial CPU load: adds CPU work between dispatches (reduces R)

For each config, measures:
  1. T_GPU: completion time per iteration with global sync
  2. T_CPU_dispatch: submission time per iteration with event sync
  3. R = T_GPU / T_CPU_dispatch
  4. TraceOpt speedup: T_global_complete / T_event_complete

If speedup is monotonically related to R, we have a predictive model.

Usage:
    python dispatch_ratio_sweep.py [--output dispatch_ratio_sweep.json]
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


def build_model(model_name, device):
    from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
    if model_name == 'resnet18':
        return resnet18(weights=ResNet18_Weights.DEFAULT).to(device).eval()
    elif model_name == 'resnet50':
        return resnet50(weights=ResNet50_Weights.DEFAULT).to(device).eval()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_pipeline_3stage(model, preprocess_buf, infer_buf, post_buf,
                        streams, sync_mode, n_iters, cpu_busy_us=0):
    """3-stage pipeline with configurable sync and optional CPU busy-wait."""
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

        # Optional CPU busy-wait to artificially increase T_CPU_dispatch
        if cpu_busy_us > 0:
            end_time = time.perf_counter_ns() + cpu_busy_us * 1000
            while time.perf_counter_ns() < end_time:
                pass

        with torch.cuda.stream(s_inf):
            if sync_mode == 'event':
                s_inf.wait_event(evt_pre)
            out = model(infer_buf)
            post_buf.copy_(out)

        if sync_mode == 'global':
            torch.cuda.synchronize()
        elif sync_mode == 'event':
            evt_inf.record(s_inf)

        if cpu_busy_us > 0:
            end_time = time.perf_counter_ns() + cpu_busy_us * 1000
            while time.perf_counter_ns() < end_time:
                pass

        with torch.cuda.stream(s_post):
            if sync_mode == 'event':
                s_post.wait_event(evt_inf)
            torch.topk(post_buf, k=5, dim=1)

        if sync_mode == 'global':
            torch.cuda.synchronize()


def measure_config(model, batch_size, device, compile_mode, n_iters=20,
                   n_warmup=10, n_trials=15, cpu_busy_us=0):
    """Measure both submission and completion time for global and event modes."""

    preprocess_buf = torch.randn(batch_size, 3, 256, 256, device=device)
    infer_buf = torch.randn(batch_size, 3, 224, 224, device=device)
    post_buf = torch.randn(batch_size, 1000, device=device)
    streams = [torch.cuda.Stream() for _ in range(3)]

    results = {}

    for sync_mode in ['global', 'event']:
        # Warmup
        with torch.no_grad():
            for _ in range(n_warmup):
                torch.cuda.synchronize()
                run_pipeline_3stage(model, preprocess_buf, infer_buf, post_buf,
                                    streams, sync_mode, n_iters, cpu_busy_us)
                torch.cuda.synchronize()

        submit_times = []
        complete_times = []

        with torch.no_grad():
            for _ in range(n_trials):
                # Submission time (CPU wall-clock)
                torch.cuda.synchronize()
                t0 = time.perf_counter_ns()
                run_pipeline_3stage(model, preprocess_buf, infer_buf, post_buf,
                                    streams, sync_mode, n_iters, cpu_busy_us)
                t1 = time.perf_counter_ns()
                torch.cuda.synchronize()
                submit_times.append((t1 - t0) / 1e6)

                # Completion time (CUDA events with stream draining)
                torch.cuda.synchronize()
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record()
                run_pipeline_3stage(model, preprocess_buf, infer_buf, post_buf,
                                    streams, sync_mode, n_iters, cpu_busy_us)
                for s in streams:
                    drain = torch.cuda.Event()
                    drain.record(s)
                    torch.cuda.current_stream().wait_event(drain)
                end_evt.record()
                torch.cuda.synchronize()
                complete_times.append(start_evt.elapsed_time(end_evt))

        submit_times.sort()
        complete_times.sort()
        sub_med = submit_times[len(submit_times) // 2]
        comp_med = complete_times[len(complete_times) // 2]

        results[sync_mode] = {
            'submit_median_ms': sub_med,
            'complete_median_ms': comp_med,
            'submit_per_iter': sub_med / n_iters,
            'complete_per_iter': comp_med / n_iters,
        }

    # Compute derived metrics
    t_gpu = results['global']['complete_per_iter']
    t_cpu_dispatch = results['event']['submit_per_iter']
    R = t_gpu / max(t_cpu_dispatch, 0.001)
    speedup = (results['global']['complete_median_ms'] /
               max(results['event']['complete_median_ms'], 0.001))

    return {
        'global': results['global'],
        'event': results['event'],
        'T_GPU_per_iter': t_gpu,
        'T_CPU_dispatch_per_iter': t_cpu_dispatch,
        'R': R,
        'speedup': speedup,
        'speedup_pct': (speedup - 1) * 100,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Dispatch/compute ratio sweep')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default='dispatch_ratio_sweep.json')
    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 78)
    print("DISPATCH/COMPUTE RATIO SWEEP")
    print("=" * 78)

    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        print(f"GPU: {prop.name} ({prop.total_memory / 1e9:.1f} GB)")
    print(f"PyTorch: {torch.__version__}")

    # Build all models
    print("\nBuilding models...")
    eager_r18 = build_model('resnet18', device)
    eager_r50 = build_model('resnet50', device)

    compiled_models = {}
    for mode_label, mode in [('default', 'default'), ('RO', 'reduce-overhead')]:
        for model_name, eager in [('r18', eager_r18), ('r50', eager_r50)]:
            key = f"compile_{mode_label}_{model_name}"
            try:
                m = torch.compile(build_model(
                    'resnet18' if model_name == 'r18' else 'resnet50', device),
                    mode=mode)
                # Warmup compile
                bs = 16 if model_name == 'r18' else 8
                dummy = torch.randn(bs, 3, 224, 224, device=device)
                with torch.no_grad():
                    for _ in range(5):
                        m(dummy)
                        torch.cuda.synchronize()
                compiled_models[key] = m
                print(f"  {key}: OK")
            except Exception as e:
                print(f"  {key}: FAILED ({e})")

    # Define configurations
    # Each config: (label, model, batch_size, compile_mode, cpu_busy_us)
    configs = []

    # Vary compilation mode (main R variation)
    for bs in [8, 16]:
        configs.append((f"eager_r18_bs{bs}", eager_r18, bs, 'eager', 0))
        if 'compile_default_r18' in compiled_models:
            configs.append((f"compile_default_r18_bs{bs}",
                            compiled_models['compile_default_r18'], bs,
                            'compile_default', 0))
        if 'compile_RO_r18' in compiled_models:
            configs.append((f"compile_RO_r18_bs{bs}",
                            compiled_models['compile_RO_r18'], bs,
                            'compile_RO', 0))

    # Vary model size (changes T_GPU)
    for bs in [8]:
        configs.append((f"eager_r50_bs{bs}", eager_r50, bs, 'eager', 0))
        if 'compile_RO_r50' in compiled_models:
            configs.append((f"compile_RO_r50_bs{bs}",
                            compiled_models['compile_RO_r50'], bs,
                            'compile_RO', 0))

    # Artificial CPU busy-wait (reduces R by increasing T_CPU_dispatch)
    if 'compile_RO_r18' in compiled_models:
        for cpu_us in [100, 250, 500, 1000, 2000]:
            configs.append((f"compile_RO_r18_bs16_cpu{cpu_us}us",
                            compiled_models['compile_RO_r18'], 16,
                            'compile_RO', cpu_us))

    # Run all configs
    all_results = {}

    for label, model, bs, compile_mode, cpu_busy_us in configs:
        print(f"\n{'='*60}")
        print(f"CONFIG: {label}")
        print(f"  batch_size={bs}, compile={compile_mode}, cpu_busy={cpu_busy_us}us")
        print(f"{'='*60}")

        try:
            r = measure_config(model, bs, device, compile_mode,
                               cpu_busy_us=cpu_busy_us)

            print(f"  T_GPU/iter:      {r['T_GPU_per_iter']:.3f} ms")
            print(f"  T_CPU_disp/iter: {r['T_CPU_dispatch_per_iter']:.3f} ms")
            print(f"  R = T_GPU/T_CPU: {r['R']:.2f}")
            print(f"  Speedup:         {r['speedup']:.4f}x "
                  f"({r['speedup_pct']:+.1f}%)")
            print(f"  Global complete: {r['global']['complete_median_ms']:.2f} ms")
            print(f"  Event complete:  {r['event']['complete_median_ms']:.2f} ms")

            all_results[label] = r
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[label] = {'error': str(e)}

    # Summary table sorted by R
    print(f"\n\n{'='*78}")
    print(f"SUMMARY: Speedup vs Dispatch/Compute Ratio")
    print(f"{'='*78}")
    print(f"{'Config':<40s} {'R':>6s} {'T_GPU':>8s} {'T_CPU':>8s} {'Speedup':>8s}")
    print(f"{'-'*72}")

    valid = [(k, v) for k, v in all_results.items() if 'R' in v]
    valid.sort(key=lambda x: x[1]['R'])

    for label, r in valid:
        print(f"{label:<40s} {r['R']:>6.2f} "
              f"{r['T_GPU_per_iter']:>6.3f}ms "
              f"{r['T_CPU_dispatch_per_iter']:>6.3f}ms "
              f"{r['speedup_pct']:>+6.1f}%")

    # Save
    output = {
        'config': {
            'gpu': (torch.cuda.get_device_properties(0).name
                    if torch.cuda.is_available() else 'N/A'),
            'pytorch_version': torch.__version__,
        },
        'results': all_results,
        'summary_by_R': [
            {'label': k, 'R': v['R'], 'speedup_pct': v['speedup_pct'],
             'T_GPU': v['T_GPU_per_iter'], 'T_CPU': v['T_CPU_dispatch_per_iter']}
            for k, v in valid
        ],
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
