"""
Workload 1: Inference with Loss/Metric Logging

Pattern: Model inference on stream 0, but periodic .item() calls or
torch.cuda.synchronize() for logging metrics force global barriers
that block independent work.

This is the most common over-synchronization in PyTorch:
    loss = criterion(output, target)
    loss_val = loss.item()          # <-- implicit cudaDeviceSynchronize!
    logger.log("loss", loss_val)

The .item() synchronizes the ENTIRE device, but only needs the loss
tensor's value. Any concurrent work on other streams is needlessly blocked.

Expected TraceOpt finding: WEAKENABLE barriers (some orderings removable).
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import torch
import torch.nn as nn
from profiler import SyncProfiler


def build_model(model_name: str, device: torch.device):
    """Load a real pretrained model."""
    if model_name == 'resnet50':
        from torchvision.models import resnet50, ResNet50_Weights
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
    elif model_name == 'resnet18':
        from torchvision.models import resnet18, ResNet18_Weights
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    elif model_name == 'mobilenet_v2':
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device).eval()
    return model


def run_workload(model_name: str = 'resnet18', batch_size: int = 32,
                 n_batches: int = 10, device_str: str = 'cuda',
                 verbose: bool = False):
    """
    Simulate inference with periodic logging syncs.

    Pattern:
        for batch in data:
            output = model(batch)          # kernel on stream 0
            loss = criterion(output, target)
            loss_val = loss.item()         # GLOBAL SYNC (over-constrained!)
            # ... could have preprocessing on another stream ...
    """
    device = torch.device(device_str)

    print(f"Loading {model_name}...")
    model = build_model(model_name, device)

    # Create synthetic input (representing real image batches)
    dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
    dummy_target = torch.randint(0, 1000, (batch_size,), device=device)
    criterion = nn.CrossEntropyLoss()

    # Create a second stream for "preprocessing" (simulating data pipeline)
    preprocess_stream = torch.cuda.Stream()

    print(f"Running {n_batches} batches (batch_size={batch_size})...")
    profiler = SyncProfiler(verbose=verbose)

    with torch.no_grad():
        with profiler.capture():
            for i in range(n_batches):
                # --- Stream 0: inference ---
                # Record kernel launch for the forward pass
                profiler._record('kernel', f'forward_batch_{i}',
                                 stream_id=0,
                                 input_ptrs=[(dummy_input.data_ptr(),
                                              dummy_input.nelement() * 4)],
                                 output_ptrs=[])

                output = model(dummy_input)

                # Record the output tensor
                profiler._record('kernel', f'loss_compute_{i}',
                                 stream_id=0,
                                 input_ptrs=[(output.data_ptr(),
                                              output.nelement() * 4)],
                                 output_ptrs=[(output.data_ptr(),
                                               output.nelement() * 4)])

                loss = criterion(output, dummy_target)

                # --- Concurrent: preprocess next batch on stream 1 ---
                with torch.cuda.stream(preprocess_stream):
                    profiler._record('kernel', f'preprocess_batch_{i+1}',
                                     stream_id=1,
                                     input_ptrs=[(dummy_input.data_ptr() + 0x100000,
                                                  dummy_input.nelement() * 4)],
                                     output_ptrs=[(dummy_input.data_ptr() + 0x200000,
                                                   dummy_input.nelement() * 4)])
                    # Simulate preprocessing work
                    next_batch = torch.randn(batch_size, 3, 224, 224,
                                             device=device)

                # --- OVER-SYNCHRONIZATION: .item() blocks everything ---
                # This is the natural pattern we're targeting.
                # loss.item() triggers cudaDeviceSynchronize internally.
                loss_val = loss.item()
                # profiler hooks capture the .item() and sync automatically

                # After sync, record CPU logging
                profiler._record('cpu_op', f'log_loss_{i}', stream_id=-1)

                if verbose or i == 0:
                    print(f"  Batch {i}: loss={loss_val:.4f}")

    return profiler


def main():
    parser = argparse.ArgumentParser(description='Inference + logging workload')
    parser.add_argument('--model', default='resnet18',
                        choices=['resnet18', 'resnet50', 'mobilenet_v2'])
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--batches', type=int, default=5)
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

    # Analyze
    result = profiler.analyze()
    print(profiler.summary(result))

    if args.save_trace:
        profiler.save_trace(args.save_trace)
        print(f"Trace saved to {args.save_trace}")


if __name__ == '__main__':
    main()
