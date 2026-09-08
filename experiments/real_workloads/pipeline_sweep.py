"""
TraceOpt Pipeline Depth Sweep: Vary number of stages and streams.

Tests the hypothesis: benefit ∝ available independent work, until
GPU resource saturation dominates.

Configurations:
  - 2-stage: preprocess → inference (2 streams)
  - 3-stage: preprocess → inference → postprocess (3 streams)
  - 4-stage: preprocess → inference → postprocess → reduce (4 streams)

Each configuration runs GLOBAL vs EVENT vs MANUAL.

Usage:
    python pipeline_sweep.py [--batch-size 16] [--trials 20]
"""

import sys
import os
import argparse
import json

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def build_model(device):
    from torchvision.models import resnet18, ResNet18_Weights
    return resnet18(weights=ResNet18_Weights.DEFAULT).to(device).eval()


# ============================================================
# 2-stage pipeline: preprocess → inference
# ============================================================

def run_2stage_global(model, bufs, streams, n_iters, device):
    s_pre, s_inf = streams
    preprocess_buf, infer_buf = bufs

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

    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def run_2stage_event(model, bufs, streams, n_iters, device):
    s_pre, s_inf = streams
    preprocess_buf, infer_buf = bufs
    evt = torch.cuda.Event()

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
            evt.record(s_pre)
        with torch.cuda.stream(s_inf):
            s_inf.wait_event(evt)
            output = model(infer_buf)

    end.record(s_inf)
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def run_2stage_manual(model, bufs, streams, n_iters, device):
    s_pre, s_inf = streams
    preprocess_buf = bufs[0]
    infer_bufs = bufs[1]  # list of 2
    events = [torch.cuda.Event() for _ in range(n_iters)]

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    for i in range(n_iters):
        idx = i % 2
        with torch.cuda.stream(s_pre):
            preprocessed = torch.nn.functional.interpolate(
                preprocess_buf, size=(224, 224), mode='bilinear',
                align_corners=False)
            infer_bufs[idx].copy_(preprocessed)
            events[i].record(s_pre)
        with torch.cuda.stream(s_inf):
            s_inf.wait_event(events[i])
            output = model(infer_bufs[idx])

    end.record(s_inf)
    torch.cuda.synchronize()
    return start.elapsed_time(end)


# ============================================================
# 3-stage pipeline: preprocess → inference → postprocess
# ============================================================

def run_3stage_global(model, bufs, streams, n_iters, device):
    s_pre, s_inf, s_post = streams
    preprocess_buf, infer_buf, postprocess_buf = bufs

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


def run_3stage_event(model, bufs, streams, n_iters, device):
    s_pre, s_inf, s_post = streams
    preprocess_buf, infer_buf, postprocess_buf = bufs
    evt_pre = torch.cuda.Event()
    evt_inf = torch.cuda.Event()

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
            evt_pre.record(s_pre)
        with torch.cuda.stream(s_inf):
            s_inf.wait_event(evt_pre)
            output = model(infer_buf)
            postprocess_buf.copy_(output)
            evt_inf.record(s_inf)
        with torch.cuda.stream(s_post):
            s_post.wait_event(evt_inf)
            topk_vals, topk_idx = torch.topk(postprocess_buf, k=5, dim=1)

    end.record(s_post)
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def run_3stage_manual(model, bufs, streams, n_iters, device):
    s_pre, s_inf, s_post = streams
    preprocess_buf = bufs[0]
    infer_bufs = bufs[1]  # list of 2
    post_bufs = bufs[2]   # list of 2
    events_pre = [torch.cuda.Event() for _ in range(n_iters)]
    events_inf = [torch.cuda.Event() for _ in range(n_iters)]

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

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
            topk_vals, topk_idx = torch.topk(post_bufs[idx], k=5, dim=1)

    end.record(s_post)
    torch.cuda.synchronize()
    return start.elapsed_time(end)


# ============================================================
# 4-stage: preprocess → inference → postprocess → reduce
# ============================================================

def run_4stage_global(model, bufs, streams, n_iters, device):
    s_pre, s_inf, s_post, s_red = streams
    preprocess_buf, infer_buf, postprocess_buf, reduce_buf = bufs

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
            reduce_buf.copy_(topk_vals)
        torch.cuda.synchronize()
        with torch.cuda.stream(s_red):
            # Lightweight reduction: softmax + sum
            probs = torch.nn.functional.softmax(reduce_buf, dim=1)
            confidence = probs.sum(dim=1, keepdim=True)

    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def run_4stage_event(model, bufs, streams, n_iters, device):
    s_pre, s_inf, s_post, s_red = streams
    preprocess_buf, infer_buf, postprocess_buf, reduce_buf = bufs
    evt_pre = torch.cuda.Event()
    evt_inf = torch.cuda.Event()
    evt_post = torch.cuda.Event()

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
            evt_pre.record(s_pre)
        with torch.cuda.stream(s_inf):
            s_inf.wait_event(evt_pre)
            output = model(infer_buf)
            postprocess_buf.copy_(output)
            evt_inf.record(s_inf)
        with torch.cuda.stream(s_post):
            s_post.wait_event(evt_inf)
            topk_vals, topk_idx = torch.topk(postprocess_buf, k=5, dim=1)
            reduce_buf.copy_(topk_vals)
            evt_post.record(s_post)
        with torch.cuda.stream(s_red):
            s_red.wait_event(evt_post)
            probs = torch.nn.functional.softmax(reduce_buf, dim=1)
            confidence = probs.sum(dim=1, keepdim=True)

    end.record(s_red)
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def run_4stage_manual(model, bufs, streams, n_iters, device):
    s_pre, s_inf, s_post, s_red = streams
    preprocess_buf = bufs[0]
    infer_bufs = bufs[1]    # list of 2
    post_bufs = bufs[2]     # list of 2
    reduce_bufs = bufs[3]   # list of 2
    events_pre = [torch.cuda.Event() for _ in range(n_iters)]
    events_inf = [torch.cuda.Event() for _ in range(n_iters)]
    events_post = [torch.cuda.Event() for _ in range(n_iters)]

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

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
            topk_vals, topk_idx = torch.topk(post_bufs[idx], k=5, dim=1)
            reduce_bufs[idx].copy_(topk_vals)
            events_post[i].record(s_post)
        with torch.cuda.stream(s_red):
            s_red.wait_event(events_post[i])
            probs = torch.nn.functional.softmax(reduce_bufs[idx], dim=1)
            confidence = probs.sum(dim=1, keepdim=True)

    end.record(s_red)
    torch.cuda.synchronize()
    return start.elapsed_time(end)


# ============================================================
# Experiment runner
# ============================================================

def alloc_bufs(batch_size, device, n_stages):
    """Allocate buffers for each pipeline depth."""
    preprocess_buf = torch.randn(batch_size, 3, 224, 224, device=device)

    if n_stages == 2:
        single = (preprocess_buf,
                  torch.randn(batch_size, 3, 224, 224, device=device))
        manual = (preprocess_buf,
                  [torch.randn(batch_size, 3, 224, 224, device=device)
                   for _ in range(2)])
        return single, manual

    elif n_stages == 3:
        single = (preprocess_buf,
                  torch.randn(batch_size, 3, 224, 224, device=device),
                  torch.randn(batch_size, 1000, device=device))
        manual = (preprocess_buf,
                  [torch.randn(batch_size, 3, 224, 224, device=device)
                   for _ in range(2)],
                  [torch.randn(batch_size, 1000, device=device)
                   for _ in range(2)])
        return single, manual

    elif n_stages == 4:
        single = (preprocess_buf,
                  torch.randn(batch_size, 3, 224, 224, device=device),
                  torch.randn(batch_size, 1000, device=device),
                  torch.randn(batch_size, 5, device=device))
        manual = (preprocess_buf,
                  [torch.randn(batch_size, 3, 224, 224, device=device)
                   for _ in range(2)],
                  [torch.randn(batch_size, 1000, device=device)
                   for _ in range(2)],
                  [torch.randn(batch_size, 5, device=device)
                   for _ in range(2)])
        return single, manual


STAGE_CONFIGS = {
    2: {
        'GLOBAL': run_2stage_global,
        'EVENT': run_2stage_event,
        'MANUAL': run_2stage_manual,
    },
    3: {
        'GLOBAL': run_3stage_global,
        'EVENT': run_3stage_event,
        'MANUAL': run_3stage_manual,
    },
    4: {
        'GLOBAL': run_4stage_global,
        'EVENT': run_4stage_event,
        'MANUAL': run_4stage_manual,
    },
}


def measure(run_fn, bufs, model, streams, n_iters, n_warmup, n_trials, device):
    with torch.no_grad():
        for _ in range(n_warmup):
            run_fn(model, bufs, streams, n_iters, device)

        times = []
        for _ in range(n_trials):
            ms = run_fn(model, bufs, streams, n_iters, device)
            times.append(ms)

    times.sort()
    return {
        'mean_ms': sum(times) / len(times),
        'median_ms': times[len(times) // 2],
        'min_ms': times[0],
        'max_ms': times[-1],
        'per_iter_ms': times[len(times) // 2] / n_iters,
    }


def main():
    parser = argparse.ArgumentParser(
        description='TraceOpt pipeline depth sweep')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--stages', default='2,3,4',
                        help='Comma-separated stage counts')
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--output', default='pipeline_sweep_results.json')
    args = parser.parse_args()

    device = torch.device(args.device)
    stage_counts = [int(s) for s in args.stages.split(',')]

    print("=" * 78)
    print("TraceOpt Pipeline Depth Sweep")
    print("=" * 78)

    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        print(f"GPU: {prop.name} ({prop.total_memory / 1e9:.1f} GB)")

    print(f"Batch size: {args.batch_size}")
    print(f"Stages: {stage_counts}")
    print(f"Iters/trial: {args.iters}, Trials: {args.trials}")

    model = build_model(device)
    all_results = {}

    for n_stages in stage_counts:
        print(f"\n{'='*78}")
        print(f"{n_stages}-STAGE PIPELINE ({n_stages} streams, "
              f"{n_stages - 1} sync points)")
        print(f"{'='*78}")

        streams = [torch.cuda.Stream() for _ in range(n_stages)]
        single_bufs, manual_bufs = alloc_bufs(args.batch_size, device,
                                               n_stages)

        stage_results = {}
        fns = STAGE_CONFIGS[n_stages]

        for mode_name in ['GLOBAL', 'EVENT', 'MANUAL']:
            fn = fns[mode_name]
            bufs = manual_bufs if mode_name == 'MANUAL' else single_bufs
            r = measure(fn, bufs, model, streams, args.iters,
                        args.warmup, args.trials, device)
            stage_results[mode_name] = r

        gl = stage_results['GLOBAL']
        ev = stage_results['EVENT']
        mn = stage_results['MANUAL']

        print(f"\n  {'Mode':<10s} {'Median':>10s} {'Per-iter':>10s} "
              f"{'vs GLOBAL':>10s}")
        print(f"  {'-'*40}")
        for name, r in [('GLOBAL', gl), ('EVENT', ev), ('MANUAL', mn)]:
            spd = gl['median_ms'] / r['median_ms']
            print(f"  {name:<10s} {r['median_ms']:>8.2f}ms "
                  f"{r['per_iter_ms']:>8.3f}ms {spd:>9.3f}x")

        ev_spd = gl['median_ms'] / ev['median_ms']
        mn_spd = gl['median_ms'] / mn['median_ms']
        print(f"\n  EVENT speedup:  {ev_spd:.3f}x ({(ev_spd-1)*100:+.1f}%)")
        print(f"  MANUAL speedup: {mn_spd:.3f}x ({(mn_spd-1)*100:+.1f}%)")

        if mn_spd > ev_spd * 1.02:
            print(f"  >> MANUAL beats EVENT: double-buffering matters "
                  f"for {n_stages}-stage pipeline")

        all_results[n_stages] = {
            name: {k: v for k, v in r.items()}
            for name, r in stage_results.items()
        }
        torch.cuda.empty_cache()

    # Summary
    print(f"\n\n{'='*78}")
    print("PIPELINE DEPTH SUMMARY")
    print(f"{'='*78}\n")

    header = (f"  {'Stages':>6s} | {'Syncs':>5s} | {'GLOBAL':>9s} | "
              f"{'EVENT':>9s} | {'MANUAL':>9s} | "
              f"{'Ev spd':>7s} | {'Mn spd':>7s}")
    print(header)
    print(f"  {'-'*len(header)}")

    for ns in stage_counts:
        r = all_results[ns]
        gl_med = r['GLOBAL']['median_ms']
        ev_med = r['EVENT']['median_ms']
        mn_med = r['MANUAL']['median_ms']
        ev_spd = gl_med / ev_med
        mn_spd = gl_med / mn_med
        print(f"  {ns:>6d} | {ns-1:>5d} | {gl_med:>7.2f}ms | "
              f"{ev_med:>7.2f}ms | {mn_med:>7.2f}ms | "
              f"{ev_spd:>6.3f}x | {mn_spd:>6.3f}x")

    print(f"\n  Hypothesis: more stages → more sync points → more benefit")
    print(f"  from weakening (if GPU has resources for concurrent execution)")

    # Save
    save_data = {
        'config': {
            'batch_size': args.batch_size,
            'stage_counts': stage_counts,
            'iters': args.iters,
            'trials': args.trials,
            'gpu': torch.cuda.get_device_properties(0).name
                   if torch.cuda.is_available() else 'unknown',
        },
        'results': {str(k): v for k, v in all_results.items()},
    }
    with open(args.output, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
