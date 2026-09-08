"""
Nsight Systems capture: Run 4 key configurations with NVTX annotations.

Generates profiles that show:
  - CPU submission timeline
  - CUDA streams and kernel concurrency
  - Synchronization intervals and GPU idle periods
  - Critical-path duration

Usage (wrap with nsys):
    nsys profile -o eager_global python nsight_capture.py --mode eager_global
    nsys profile -o eager_event  python nsight_capture.py --mode eager_event
    nsys profile -o compile_global python nsight_capture.py --mode compile_global
    nsys profile -o compile_only python nsight_capture.py --mode compile_only

Or use the helper script:
    bash nsight_capture_all.sh
"""

import sys
import os
import argparse

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# NVTX support
try:
    import torch.cuda.nvtx as nvtx
    HAS_NVTX = True
except ImportError:
    HAS_NVTX = False


def nvtx_range(name):
    if HAS_NVTX:
        return nvtx.range(name)
    from contextlib import nullcontext
    return nullcontext()


def build_model(device):
    from torchvision.models import resnet18, ResNet18_Weights
    return resnet18(weights=ResNet18_Weights.DEFAULT).to(device).eval()


def run_pipeline(model, preprocess_buf, infer_buf, post_buf,
                 streams, sync_mode, n_iters):
    """Run 3-stage pipeline with NVTX annotations."""
    s_pre, s_inf, s_post = streams
    evt_pre = torch.cuda.Event()
    evt_inf = torch.cuda.Event()

    for i in range(n_iters):
        with nvtx_range(f"iter_{i}"):
            with nvtx_range("preprocess"):
                with torch.cuda.stream(s_pre):
                    p = torch.nn.functional.interpolate(
                        preprocess_buf, size=(224, 224), mode='bilinear',
                        align_corners=False)
                    infer_buf.copy_(p)

            if sync_mode == 'global':
                with nvtx_range("sync_pre"):
                    torch.cuda.synchronize()
            elif sync_mode == 'event':
                evt_pre.record(s_pre)

            with nvtx_range("inference"):
                with torch.cuda.stream(s_inf):
                    if sync_mode == 'event':
                        s_inf.wait_event(evt_pre)
                    out = model(infer_buf)
                    post_buf.copy_(out)

            if sync_mode == 'global':
                with nvtx_range("sync_inf"):
                    torch.cuda.synchronize()
            elif sync_mode == 'event':
                evt_inf.record(s_inf)

            with nvtx_range("postprocess"):
                with torch.cuda.stream(s_post):
                    if sync_mode == 'event':
                        s_post.wait_event(evt_inf)
                    torch.topk(post_buf, k=5, dim=1)

            if sync_mode == 'global':
                with nvtx_range("sync_post"):
                    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(
        description='Nsight Systems capture for TraceOpt')
    parser.add_argument('--mode', required=True,
                        choices=['eager_global', 'eager_event',
                                 'compile_global', 'compile_event',
                                 'compile_only'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--iters', type=int, default=20,
                        help='Iterations to profile (keep small for timeline)')
    parser.add_argument('--warmup', type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    use_compile = args.mode.startswith('compile')

    if args.mode.endswith('_global'):
        sync_mode = 'global'
    elif args.mode.endswith('_event'):
        sync_mode = 'event'
    elif args.mode.endswith('_only'):
        sync_mode = 'none'

    print(f"Mode: {args.mode}")
    print(f"Compiled: {use_compile}, Sync: {sync_mode}")
    print(f"NVTX: {HAS_NVTX}")

    # Build model
    if use_compile:
        eager = build_model(device)
        model = torch.compile(eager, mode='reduce-overhead')
    else:
        model = build_model(device)

    # Allocate
    preprocess_buf = torch.randn(args.batch_size, 3, 256, 256, device=device)
    infer_buf = torch.randn(args.batch_size, 3, 224, 224, device=device)
    post_buf = torch.randn(args.batch_size, 1000, device=device)
    streams = [torch.cuda.Stream() for _ in range(3)]

    # Warmup (outside profiling region)
    print("Warmup...")
    with torch.no_grad():
        for _ in range(args.warmup):
            run_pipeline(model, preprocess_buf, infer_buf, post_buf,
                         streams, sync_mode, args.iters)
            torch.cuda.synchronize()

    # Profiled region
    print(f"Profiling {args.iters} iterations...")
    torch.cuda.synchronize()

    if HAS_NVTX:
        nvtx.range_push(f"profiled_{args.mode}")

    with torch.no_grad():
        run_pipeline(model, preprocess_buf, infer_buf, post_buf,
                     streams, sync_mode, args.iters)

    torch.cuda.synchronize()
    if HAS_NVTX:
        nvtx.range_pop()

    print("Done.")


if __name__ == '__main__':
    main()
