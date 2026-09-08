"""
Workload 2: Training with Gradient Accumulation + AMP Scaler Check

Pattern: Mixed-precision training uses GradScaler which calls:
    scaler.step(optimizer)
    scaler.update()

Internally, GradScaler checks for inf/nan in gradients via:
    found_inf = torch.isinf(grad).any().item()

This .item() call triggers cudaDeviceSynchronize, blocking ALL streams
including any concurrent data loading, preprocessing, or communication.

The sync is needed for the host to read the inf check result, but any
independent GPU work on other streams should NOT be blocked.

Expected TraceOpt finding: WEAKENABLE (host needs sync for inf check,
but independent streams are over-constrained).
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
        return resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
    elif model_name == 'resnet50':
        from torchvision.models import resnet50, ResNet50_Weights
        return resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_workload(model_name: str = 'resnet18', batch_size: int = 16,
                 n_steps: int = 10, accum_steps: int = 4,
                 device_str: str = 'cuda', verbose: bool = False):
    """
    Mixed-precision training with gradient accumulation.

    Over-synchronization sources:
    1. GradScaler.step() calls .item() to check for inf grads
    2. Periodic loss logging via .item()
    3. torch.cuda.synchronize() before timing measurements

    Independent work that gets blocked:
    - Data prefetching on a separate stream
    - Gradient all-reduce (simulated)
    """
    device = torch.device(device_str)
    print(f"Loading {model_name} for training...")
    model = build_model(model_name, device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scaler = torch.amp.GradScaler()
    criterion = nn.CrossEntropyLoss()

    dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
    dummy_target = torch.randint(0, 1000, (batch_size,), device=device)

    # Second stream for simulated data prefetch
    prefetch_stream = torch.cuda.Stream()

    print(f"Training {n_steps} steps (accum={accum_steps})...")
    profiler = SyncProfiler(verbose=verbose)

    with profiler.capture():
        for step in range(n_steps):
            # --- Stream 1: prefetch next batch (independent work) ---
            with torch.cuda.stream(prefetch_stream):
                profiler._record('kernel', f'prefetch_step_{step}',
                                 stream_id=1,
                                 input_ptrs=[(0xA0000000, batch_size * 3 * 224 * 224 * 4)],
                                 output_ptrs=[(0xB0000000, batch_size * 3 * 224 * 224 * 4)])
                next_batch = torch.randn(batch_size, 3, 224, 224, device=device)

            # --- Stream 0: forward + backward ---
            profiler._record('kernel', f'forward_step_{step}',
                             stream_id=0,
                             input_ptrs=[(dummy_input.data_ptr(),
                                          dummy_input.nelement() * 4)],
                             output_ptrs=[])

            with torch.amp.autocast(device_type='cuda'):
                output = model(dummy_input)
                loss = criterion(output, dummy_target)
                loss = loss / accum_steps

            profiler._record('kernel', f'backward_step_{step}',
                             stream_id=0,
                             input_ptrs=[(loss.data_ptr(), loss.nelement() * 4)],
                             output_ptrs=[])

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                # --- OVER-SYNCHRONIZATION 1: GradScaler inf check ---
                # scaler.step() internally does .item() on the inf check
                profiler._record('kernel', f'grad_unscale_{step}',
                                 stream_id=0,
                                 output_ptrs=[(0xC0000000, 4)])

                # Simulate the inf check that GradScaler does
                grad_check = torch.tensor([0.0], device=device)
                profiler._record('kernel', f'inf_check_{step}',
                                 stream_id=0,
                                 input_ptrs=[(0xC0000000, 4)],
                                 output_ptrs=[(grad_check.data_ptr(),
                                               grad_check.nelement() * 4)])

                # This .item() is the culprit — global sync for a single scalar
                found_inf = grad_check.item()
                # (profiler hooks capture this automatically)

                profiler._record('cpu_op', f'scaler_decision_{step}',
                                 stream_id=-1)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                profiler._record('kernel', f'optimizer_step_{step}',
                                 stream_id=0,
                                 output_ptrs=[(dummy_input.data_ptr(),
                                               dummy_input.nelement() * 4)])

            # --- OVER-SYNCHRONIZATION 2: loss logging ---
            if step % 2 == 0:
                loss_val = (loss * accum_steps).item()
                profiler._record('cpu_op', f'log_loss_{step}', stream_id=-1)
                if verbose or step == 0:
                    print(f"  Step {step}: loss={loss_val:.4f}")

    return profiler


def main():
    parser = argparse.ArgumentParser(description='Training + grad accum workload')
    parser.add_argument('--model', default='resnet18')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--accum', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--save-trace', type=str, default=None)
    args = parser.parse_args()

    profiler = run_workload(
        model_name=args.model,
        batch_size=args.batch_size,
        n_steps=args.steps,
        accum_steps=args.accum,
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
