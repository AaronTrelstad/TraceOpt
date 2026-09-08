"""
Workload 3: Multi-Stream Pipeline Serving

Pattern: Serving pipeline where preprocessing, inference, and postprocessing
run on separate CUDA streams for throughput. Framework inserts
cudaDeviceSynchronize() between stages "for safety" even when only
specific cross-stream dependencies exist.

    Stream 0: preprocess(batch_N)   →  [sync]  →  infer(batch_N-1)
    Stream 1: infer(batch_N-1)      →  [sync]  →  postprocess(batch_N-2)
    Stream 2: postprocess(batch_N-2)

The global sync between stages blocks ALL streams when only the specific
producer→consumer dependency needs to be enforced.

Expected TraceOpt finding: WEAKENABLE (cross-stream data deps required,
but many induced orderings are removable).
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import torch
import torch.nn as nn
from profiler import SyncProfiler


def build_model(model_name: str, device: torch.device):
    if model_name == 'resnet18':
        from torchvision.models import resnet18, ResNet18_Weights
        return resnet18(weights=ResNet18_Weights.DEFAULT).to(device).eval()
    elif model_name == 'mobilenet_v2':
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        return mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).to(device).eval()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_workload(model_name: str = 'resnet18', batch_size: int = 16,
                 n_batches: int = 8, device_str: str = 'cuda',
                 verbose: bool = False):
    """
    Three-stage pipeline with over-synchronized stage boundaries.

    Stage 1 (stream 0): Preprocessing (resize, normalize, augment)
    Stage 2 (stream 1): Model inference
    Stage 3 (stream 2): Postprocessing (top-k, format results)

    Over-synchronization: cudaDeviceSynchronize() between stages
    instead of targeted event dependencies.
    """
    device = torch.device(device_str)
    print(f"Loading {model_name} for pipeline serving...")
    model = build_model(model_name, device)

    # Three pipeline streams
    stream_preprocess = torch.cuda.Stream()
    stream_infer = torch.cuda.Stream()
    stream_postprocess = torch.cuda.Stream()

    # Allocate buffers for each stage (different memory regions)
    preprocess_buf = torch.randn(batch_size, 3, 224, 224, device=device)
    infer_buf = torch.randn(batch_size, 3, 224, 224, device=device)
    postprocess_buf = torch.randn(batch_size, 1000, device=device)

    print(f"Running {n_batches} batches in 3-stage pipeline...")
    profiler = SyncProfiler(verbose=verbose)

    with torch.no_grad():
        with profiler.capture():
            for i in range(n_batches):
                # --- Stage 1: Preprocess on stream 0 ---
                with torch.cuda.stream(stream_preprocess):
                    profiler._record('kernel', f'preprocess_{i}',
                                     stream_id=0,
                                     input_ptrs=[(preprocess_buf.data_ptr(),
                                                  preprocess_buf.nelement() * 4)],
                                     output_ptrs=[(infer_buf.data_ptr(),
                                                   infer_buf.nelement() * 4)])
                    # Simulate preprocessing
                    preprocessed = preprocess_buf.clone()
                    preprocessed = torch.nn.functional.interpolate(
                        preprocessed, size=(224, 224), mode='bilinear',
                        align_corners=False)

                # --- OVER-SYNCHRONIZATION: global sync between stages ---
                # This is what frameworks often insert "for safety"
                torch.cuda.synchronize()

                # --- Stage 2: Inference on stream 1 ---
                with torch.cuda.stream(stream_infer):
                    profiler._record('kernel', f'inference_{i}',
                                     stream_id=1,
                                     input_ptrs=[(infer_buf.data_ptr(),
                                                  infer_buf.nelement() * 4)],
                                     output_ptrs=[(postprocess_buf.data_ptr(),
                                                   postprocess_buf.nelement() * 4)])
                    output = model(infer_buf)

                # --- OVER-SYNCHRONIZATION: another global sync ---
                torch.cuda.synchronize()

                # --- Stage 3: Postprocess on stream 2 ---
                with torch.cuda.stream(stream_postprocess):
                    profiler._record('kernel', f'postprocess_{i}',
                                     stream_id=2,
                                     input_ptrs=[(postprocess_buf.data_ptr(),
                                                  postprocess_buf.nelement() * 4)],
                                     output_ptrs=[(postprocess_buf.data_ptr() + 0x100000,
                                                   batch_size * 5 * 4)])
                    # top-k
                    topk_vals, topk_idx = torch.topk(postprocess_buf, k=5, dim=1)

                if verbose or i == 0:
                    torch.cuda.synchronize()
                    print(f"  Batch {i}: top prediction idx={topk_idx[0, 0].item()}")

    return profiler


def main():
    parser = argparse.ArgumentParser(description='Multi-stream pipeline workload')
    parser.add_argument('--model', default='resnet18')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--batches', type=int, default=6)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--save-trace', type=str, default=None)
    args = parser.parse_args()

    profiler = run_workload(
        model_name=args.model,
        batch_size=args.batch_size,
        n_batches=args.batches,
        device_str=args.device,
        verbose=args.verbose,
    )

    result = profiler.analyze()
    print(profiler.summary(result))

    if args.save_trace:
        profiler.save_trace(args.save_trace)
        print(f"Trace saved to {args.save_trace}")


if __name__ == '__main__':
    main()
