"""
torch.compile Diagnostic: What actually changes between COMPILE_GLOBAL and COMPILE_EVENT?

The compile_comparison.py shows:
  COMPILE_GLOBAL: 35-41ms
  COMPILE_EVENT:   6-7ms
  COMPILE_ONLY:    6-7ms

That 5-6x reduction CANNOT be explained by removing 2-3 sync points.
Something else is happening. This script investigates:

1. Is torch.compile using CUDA graphs (reduce-overhead mode)?
2. Does the compiled model actually run on the assigned stream?
3. How much time does the CPU spend between iterations?
4. What is the per-stage GPU time?
5. Does COMPILE_ONLY produce correct results?
6. What happens with mode='default' (no CUDA graphs)?

Key hypothesis: In COMPILE_GLOBAL, torch.cuda.synchronize() blocks the CPU,
preventing it from submitting the next iteration. With compiled kernels being
fast to dispatch, this creates GPU idle bubbles. With events or no sync, the
CPU races ahead and the GPU stays fully utilized across iterations.

Usage:
    python compile_diagnostic.py [--batch-size 16] [--stages 3]
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
# Diagnostic 1: Per-stage GPU timing
# ============================================================

def measure_per_stage(model, batch_size, device, n_iters=10, compiled=False):
    """Time each pipeline stage individually on the GPU."""
    preprocess_buf = torch.randn(batch_size, 3, 256, 256, device=device)
    infer_buf = torch.randn(batch_size, 3, 224, 224, device=device)
    post_buf = torch.randn(batch_size, 1000, device=device)

    s_pre = torch.cuda.Stream()
    s_inf = torch.cuda.Stream()
    s_post = torch.cuda.Stream()

    label = "compiled" if compiled else "eager"

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            with torch.cuda.stream(s_pre):
                p = torch.nn.functional.interpolate(
                    preprocess_buf, size=(224, 224), mode='bilinear',
                    align_corners=False)
                infer_buf.copy_(p)
            torch.cuda.synchronize()
            with torch.cuda.stream(s_inf):
                out = model(infer_buf)
                post_buf.copy_(out)
            torch.cuda.synchronize()
            with torch.cuda.stream(s_post):
                torch.topk(post_buf, k=5, dim=1)
            torch.cuda.synchronize()

    # Measure each stage
    stages = {}
    with torch.no_grad():
        # Preprocess
        times = []
        for _ in range(n_iters):
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(s_pre)
            with torch.cuda.stream(s_pre):
                p = torch.nn.functional.interpolate(
                    preprocess_buf, size=(224, 224), mode='bilinear',
                    align_corners=False)
                infer_buf.copy_(p)
            e.record(s_pre)
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        stages['preprocess'] = sum(times) / len(times)

        # Inference
        times = []
        for _ in range(n_iters):
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(s_inf)
            with torch.cuda.stream(s_inf):
                out = model(infer_buf)
                post_buf.copy_(out)
            e.record(s_inf)
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        stages['inference'] = sum(times) / len(times)

        # Postprocess
        times = []
        for _ in range(n_iters):
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(s_post)
            with torch.cuda.stream(s_post):
                torch.topk(post_buf, k=5, dim=1)
            e.record(s_post)
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        stages['postprocess'] = sum(times) / len(times)

    stages['total_serial'] = sum(stages.values())
    print(f"\n  Per-stage GPU time ({label}, bs={batch_size}):")
    for name, t in stages.items():
        print(f"    {name:<15s}: {t:.3f}ms")
    return stages


# ============================================================
# Diagnostic 2: CPU submission timing
# ============================================================

def measure_cpu_submission(model, batch_size, device, n_iters=20,
                           compiled=False, sync_mode='global'):
    """Measure CPU wall-clock time between iteration submissions."""
    preprocess_buf = torch.randn(batch_size, 3, 256, 256, device=device)
    infer_buf = torch.randn(batch_size, 3, 224, 224, device=device)
    post_buf = torch.randn(batch_size, 1000, device=device)

    s_pre = torch.cuda.Stream()
    s_inf = torch.cuda.Stream()
    s_post = torch.cuda.Stream()
    evt_pre = torch.cuda.Event()
    evt_inf = torch.cuda.Event()

    label = f"{'compiled' if compiled else 'eager'}_{sync_mode}"

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            with torch.cuda.stream(s_pre):
                p = torch.nn.functional.interpolate(
                    preprocess_buf, size=(224, 224), mode='bilinear',
                    align_corners=False)
                infer_buf.copy_(p)
            torch.cuda.synchronize()
            with torch.cuda.stream(s_inf):
                out = model(infer_buf)
                post_buf.copy_(out)
            torch.cuda.synchronize()

    # Measure CPU time per iteration
    cpu_times = []
    with torch.no_grad():
        torch.cuda.synchronize()
        for i in range(n_iters):
            t0 = time.perf_counter_ns()

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
            # no sync for 'none' mode

            t1 = time.perf_counter_ns()
            cpu_times.append((t1 - t0) / 1e6)  # ms

        # Final sync to ensure everything completed
        torch.cuda.synchronize()

    mean_cpu = sum(cpu_times) / len(cpu_times)
    min_cpu = min(cpu_times)
    max_cpu = max(cpu_times)

    print(f"\n  CPU submission time per iteration ({label}):")
    print(f"    Mean: {mean_cpu:.3f}ms, Min: {min_cpu:.3f}ms, "
          f"Max: {max_cpu:.3f}ms")

    return {
        'mean_ms': mean_cpu,
        'min_ms': min_cpu,
        'max_ms': max_cpu,
        'times': cpu_times,
    }


# ============================================================
# Diagnostic 3: Compile mode comparison
# ============================================================

def compare_compile_modes(model_eager, batch_size, device, n_iters=20,
                          n_trials=10, n_warmup=5):
    """Compare reduce-overhead (CUDA graphs) vs default (no graphs)."""
    preprocess_buf = torch.randn(batch_size, 3, 256, 256, device=device)
    infer_buf = torch.randn(batch_size, 3, 224, 224, device=device)
    post_buf = torch.randn(batch_size, 1000, device=device)

    s_pre = torch.cuda.Stream()
    s_inf = torch.cuda.Stream()
    s_post = torch.cuda.Stream()
    evt_pre = torch.cuda.Event()
    evt_inf = torch.cuda.Event()

    modes_to_test = [
        ('eager', model_eager),
    ]

    # Try compiling with different modes
    for compile_mode in ['default', 'reduce-overhead']:
        try:
            compiled = torch.compile(model_eager, mode=compile_mode)
            # Warmup to trigger compilation
            with torch.no_grad():
                for _ in range(3):
                    compiled(infer_buf)
                    torch.cuda.synchronize()
            modes_to_test.append((f'compile_{compile_mode}', compiled))
            print(f"  torch.compile(mode='{compile_mode}'): OK")
        except Exception as e:
            print(f"  torch.compile(mode='{compile_mode}'): FAILED ({e})")

    results = {}

    for model_label, model in modes_to_test:
        model_results = {}

        for sync_mode in ['global', 'event', 'none']:
            def run_pipeline():
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

            # Warmup
            with torch.no_grad():
                for _ in range(n_warmup):
                    torch.cuda.synchronize()
                    run_pipeline()
                    torch.cuda.synchronize()

            # Measure
            times = []
            with torch.no_grad():
                for _ in range(n_trials):
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    run_pipeline()
                    end.record()
                    torch.cuda.synchronize()
                    times.append(start.elapsed_time(end))

            times.sort()
            median = times[len(times) // 2]
            model_results[sync_mode] = {
                'median_ms': median,
                'per_iter_ms': median / n_iters,
                'mean_ms': sum(times) / len(times),
            }

        results[model_label] = model_results

    return results


# ============================================================
# Diagnostic 4: Correctness verification
# ============================================================

def verify_correctness(model_eager, batch_size, device, n_iters=5):
    """Check if all modes produce identical results."""
    torch.manual_seed(42)
    input_data = torch.randn(batch_size, 3, 256, 256, device=device)
    infer_buf = torch.randn(batch_size, 3, 224, 224, device=device)
    post_buf = torch.randn(batch_size, 1000, device=device)

    s_pre = torch.cuda.Stream()
    s_inf = torch.cuda.Stream()
    s_post = torch.cuda.Stream()
    evt_pre = torch.cuda.Event()
    evt_inf = torch.cuda.Event()

    modes = {
        'eager': model_eager,
    }

    try:
        compiled = torch.compile(model_eager, mode='reduce-overhead')
        with torch.no_grad():
            for _ in range(3):
                compiled(infer_buf)
                torch.cuda.synchronize()
        modes['compiled'] = compiled
    except Exception:
        pass

    results = {}

    for model_label, model in modes.items():
        for sync_mode in ['global', 'event', 'none']:
            # Reset buffers to known state
            torch.manual_seed(42)
            test_input = input_data.clone()
            test_infer = torch.zeros_like(infer_buf)
            test_post = torch.zeros_like(post_buf)

            with torch.no_grad():
                # Run single iteration with full sync to get reference
                with torch.cuda.stream(s_pre):
                    p = torch.nn.functional.interpolate(
                        test_input, size=(224, 224), mode='bilinear',
                        align_corners=False)
                    test_infer.copy_(p)
                if sync_mode == 'global':
                    torch.cuda.synchronize()
                elif sync_mode == 'event':
                    evt_pre.record(s_pre)
                with torch.cuda.stream(s_inf):
                    if sync_mode == 'event':
                        s_inf.wait_event(evt_pre)
                    out = model(test_infer)
                    test_post.copy_(out)
                if sync_mode == 'global':
                    torch.cuda.synchronize()
                elif sync_mode == 'event':
                    evt_inf.record(s_inf)
                with torch.cuda.stream(s_post):
                    if sync_mode == 'event':
                        s_post.wait_event(evt_inf)
                    topk_vals, topk_idx = torch.topk(test_post, k=5, dim=1)

                # Always sync at end to get result
                torch.cuda.synchronize()

            key = f"{model_label}_{sync_mode}"
            results[key] = {
                'topk_vals': topk_vals.cpu(),
                'topk_idx': topk_idx.cpu(),
            }

    # Compare all against eager_global reference
    ref_key = 'eager_global'
    if ref_key not in results:
        print("  Cannot verify: eager_global not available")
        return {}

    ref_vals = results[ref_key]['topk_vals']
    ref_idx = results[ref_key]['topk_idx']

    print(f"\n  Correctness verification (vs eager_global):")
    verification = {}
    for key, r in results.items():
        if key == ref_key:
            continue
        vals_match = torch.allclose(ref_vals, r['topk_vals'], atol=1e-4)
        idx_match = torch.equal(ref_idx, r['topk_idx'])
        status = "PASS" if (vals_match and idx_match) else "FAIL"
        if not vals_match:
            max_diff = (ref_vals - r['topk_vals']).abs().max().item()
            status += f" (max_val_diff={max_diff:.6f})"
        print(f"    {key:<25s}: {status}")
        verification[key] = status
    return verification


# ============================================================
# Diagnostic 5: Stream activity check
# ============================================================

def check_stream_activity(model, batch_size, device, compiled=False):
    """Check if compiled model actually dispatches work to assigned stream."""
    infer_buf = torch.randn(batch_size, 3, 224, 224, device=device)

    s0 = torch.cuda.Stream()  # non-default stream
    label = "compiled" if compiled else "eager"

    # Time on default stream
    with torch.no_grad():
        for _ in range(5):
            model(infer_buf)
            torch.cuda.synchronize()

        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(10):
            model(infer_buf)
        e.record()
        torch.cuda.synchronize()
        default_ms = s.elapsed_time(e) / 10

    # Time on non-default stream
    with torch.no_grad():
        for _ in range(5):
            with torch.cuda.stream(s0):
                model(infer_buf)
            torch.cuda.synchronize()

        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(s0)
        with torch.cuda.stream(s0):
            for _ in range(10):
                model(infer_buf)
        e.record(s0)
        torch.cuda.synchronize()
        stream_ms = s.elapsed_time(e) / 10

    print(f"\n  Stream activity check ({label}):")
    print(f"    Default stream: {default_ms:.3f}ms/call")
    print(f"    Non-default stream: {stream_ms:.3f}ms/call")
    ratio = stream_ms / max(default_ms, 0.001)
    if abs(ratio - 1.0) < 0.1:
        print(f"    → Similar timing, model runs on assigned stream")
    else:
        print(f"    → Ratio: {ratio:.2f}x — possible stream behavior difference")
    return {'default_ms': default_ms, 'stream_ms': stream_ms}


# ============================================================
# Diagnostic 6: Iteration depth test
# ============================================================

def measure_iteration_scaling(model, batch_size, device, compiled=False,
                              sync_mode='global', max_iters=40):
    """
    Measure total time vs iteration count.
    If time scales linearly: sequential execution.
    If time is sublinear: pipelining/overlap is happening.
    """
    preprocess_buf = torch.randn(batch_size, 3, 256, 256, device=device)
    infer_buf = torch.randn(batch_size, 3, 224, 224, device=device)
    post_buf = torch.randn(batch_size, 1000, device=device)

    s_pre = torch.cuda.Stream()
    s_inf = torch.cuda.Stream()
    s_post = torch.cuda.Stream()
    evt_pre = torch.cuda.Event()
    evt_inf = torch.cuda.Event()

    label = f"{'compiled' if compiled else 'eager'}_{sync_mode}"

    def run(n):
        for i in range(n):
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

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            torch.cuda.synchronize()
            run(5)
            torch.cuda.synchronize()

    # Measure at different iteration counts
    iter_counts = [1, 2, 5, 10, 20, max_iters]
    scaling = {}
    with torch.no_grad():
        for n in iter_counts:
            times = []
            for _ in range(5):
                torch.cuda.synchronize()
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                run(n)
                e.record()
                torch.cuda.synchronize()
                times.append(s.elapsed_time(e))
            times.sort()
            median = times[len(times) // 2]
            scaling[n] = median

    print(f"\n  Iteration scaling ({label}):")
    print(f"    {'Iters':>6s} {'Total(ms)':>10s} {'Per-iter(ms)':>12s} "
          f"{'Ratio vs 1':>12s}")
    base_per_iter = scaling[1]
    for n in iter_counts:
        per_iter = scaling[n] / n
        ratio = per_iter / base_per_iter
        print(f"    {n:>6d} {scaling[n]:>10.2f} {per_iter:>12.3f} "
              f"{ratio:>12.3f}")

    return scaling


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='torch.compile diagnostic investigation')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--output', default='compile_diagnostic.json')
    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 78)
    print("torch.compile DIAGNOSTIC")
    print("Investigating the 41ms → 7ms anomaly")
    print("=" * 78)

    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        print(f"GPU: {prop.name} ({prop.total_memory / 1e9:.1f} GB)")
        print(f"SMs: {prop.multi_processor_count}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Batch size: {args.batch_size}")

    # Check torch.compile backend info
    print(f"\ntorch.compile backend: ", end="")
    try:
        dynamo = torch._dynamo
        print(f"dynamo available, backends: {dynamo.list_backends()}")
    except Exception as e:
        print(f"dynamo info unavailable: {e}")

    print(f"\nCUDA graphs support: ", end="")
    try:
        g = torch.cuda.CUDAGraph()
        print("yes")
        del g
    except Exception as e:
        print(f"no ({e})")

    all_results = {}

    # Build models
    print("\n--- Building models ---")
    eager_model = build_model(device)

    compiled_models = {}
    for mode in ['default', 'reduce-overhead']:
        try:
            m = torch.compile(build_model(device), mode=mode)
            # Trigger compilation
            dummy = torch.randn(args.batch_size, 3, 224, 224, device=device)
            with torch.no_grad():
                for _ in range(5):
                    m(dummy)
                    torch.cuda.synchronize()
            compiled_models[mode] = m
            print(f"  torch.compile(mode='{mode}'): OK")
        except Exception as e:
            print(f"  torch.compile(mode='{mode}'): FAILED ({e})")

    # =============================================
    # Test 1: Per-stage GPU timing
    # =============================================
    print(f"\n{'='*78}")
    print("TEST 1: Per-stage GPU time (isolating each stage)")
    print(f"{'='*78}")

    stage_eager = measure_per_stage(eager_model, args.batch_size, device,
                                    compiled=False)
    all_results['per_stage_eager'] = stage_eager

    for mode, m in compiled_models.items():
        stage_comp = measure_per_stage(m, args.batch_size, device,
                                       compiled=True)
        all_results[f'per_stage_compile_{mode}'] = stage_comp

    # =============================================
    # Test 2: CPU submission timing
    # =============================================
    print(f"\n{'='*78}")
    print("TEST 2: CPU wall-clock time per iteration")
    print("(How long does the CPU spend between iterations?)")
    print(f"{'='*78}")

    for sync_mode in ['global', 'event', 'none']:
        r = measure_cpu_submission(eager_model, args.batch_size, device,
                                   compiled=False, sync_mode=sync_mode)
        all_results[f'cpu_eager_{sync_mode}'] = r

    for mode, m in compiled_models.items():
        for sync_mode in ['global', 'event', 'none']:
            r = measure_cpu_submission(m, args.batch_size, device,
                                       compiled=True, sync_mode=sync_mode)
            all_results[f'cpu_compile_{mode}_{sync_mode}'] = r

    # =============================================
    # Test 3: Compile mode comparison (default vs reduce-overhead)
    # =============================================
    print(f"\n{'='*78}")
    print("TEST 3: Compile mode × sync mode factorial")
    print("(default=no CUDA graphs, reduce-overhead=CUDA graphs)")
    print(f"{'='*78}")

    mode_results = compare_compile_modes(eager_model, args.batch_size, device)
    all_results['factorial'] = mode_results

    print(f"\n  {'Model':<25s} {'global':>10s} {'event':>10s} "
          f"{'none':>10s}")
    print(f"  {'-'*55}")
    for model_label, sync_results in mode_results.items():
        gl = sync_results['global']['median_ms']
        ev = sync_results['event']['median_ms']
        nn = sync_results['none']['median_ms']
        print(f"  {model_label:<25s} {gl:>8.2f}ms {ev:>8.2f}ms "
              f"{nn:>8.2f}ms")

    print(f"\n  Key ratios:")
    if 'eager' in mode_results and 'compile_reduce-overhead' in mode_results:
        eg = mode_results['eager']['global']['median_ms']
        ee = mode_results['eager']['event']['median_ms']
        crg = mode_results['compile_reduce-overhead']['global']['median_ms']
        cre = mode_results['compile_reduce-overhead']['event']['median_ms']
        crn = mode_results['compile_reduce-overhead']['none']['median_ms']
        print(f"    Eager: global→event = {eg:.2f}→{ee:.2f}ms "
              f"({eg/ee:.2f}x)")
        print(f"    Compile-RO: global→event = {crg:.2f}→{cre:.2f}ms "
              f"({crg/cre:.2f}x)")
        print(f"    Compile-RO: global→none  = {crg:.2f}→{crn:.2f}ms "
              f"({crg/crn:.2f}x)")
        print(f"    Compile-RO: event→none   = {cre:.2f}→{crn:.2f}ms "
              f"({cre/crn:.2f}x)")

    if 'compile_default' in mode_results:
        cdg = mode_results['compile_default']['global']['median_ms']
        cde = mode_results['compile_default']['event']['median_ms']
        cdn = mode_results['compile_default']['none']['median_ms']
        print(f"    Compile-default: global→event = {cdg:.2f}→{cde:.2f}ms "
              f"({cdg/cde:.2f}x)")
        print(f"    Compile-default: global→none  = {cdg:.2f}→{cdn:.2f}ms "
              f"({cdg/cdn:.2f}x)")

    # =============================================
    # Test 4: Correctness
    # =============================================
    print(f"\n{'='*78}")
    print("TEST 4: Correctness verification")
    print(f"{'='*78}")

    verification = verify_correctness(eager_model, args.batch_size, device)
    all_results['correctness'] = verification

    # =============================================
    # Test 5: Stream activity
    # =============================================
    print(f"\n{'='*78}")
    print("TEST 5: Stream activity (does compiled model use assigned stream?)")
    print(f"{'='*78}")

    stream_eager = check_stream_activity(eager_model, args.batch_size, device,
                                          compiled=False)
    all_results['stream_eager'] = stream_eager

    for mode, m in compiled_models.items():
        stream_comp = check_stream_activity(m, args.batch_size, device,
                                             compiled=True)
        all_results[f'stream_compile_{mode}'] = stream_comp

    # =============================================
    # Test 6: Iteration scaling
    # =============================================
    print(f"\n{'='*78}")
    print("TEST 6: Iteration scaling")
    print("(Linear = serialized, Sublinear = pipelined)")
    print(f"{'='*78}")

    for sync_mode in ['global', 'event']:
        scaling = measure_iteration_scaling(
            eager_model, args.batch_size, device,
            compiled=False, sync_mode=sync_mode)
        all_results[f'scaling_eager_{sync_mode}'] = scaling

    if 'reduce-overhead' in compiled_models:
        for sync_mode in ['global', 'event', 'none']:
            scaling = measure_iteration_scaling(
                compiled_models['reduce-overhead'], args.batch_size, device,
                compiled=True, sync_mode=sync_mode)
            all_results[f'scaling_compile_ro_{sync_mode}'] = scaling

    # =============================================
    # Summary
    # =============================================
    print(f"\n\n{'='*78}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*78}")

    print(f"""
The 41ms → 7ms question:

If CPU submission time with global sync >> CPU submission time with events,
then the bottleneck is CPU blocking, not GPU synchronization overhead.

torch.compile makes kernel dispatch nearly instant, so:
  - With global sync: CPU blocks after each stage, creating GPU idle bubbles
  - With events: CPU races ahead, GPU stays busy across iterations
  - The 5x speedup is from eliminating CPU-side blocking, not GPU-side sync

This means the speedup is real but the causal attribution is:
  "Removing CPU-blocking synchronization enables deep pipelining of
   compiled iterations" — NOT "TraceOpt optimizes compiled code by 5x."

Check the CPU submission times above to confirm this hypothesis.
""")

    # Save
    save_data = {k: v for k, v in all_results.items()
                 if not isinstance(v, dict) or 'times' not in v}
    # Strip raw times from nested dicts
    clean = {}
    for k, v in all_results.items():
        if isinstance(v, dict):
            clean[k] = {kk: vv for kk, vv in v.items()
                        if kk != 'times' and not isinstance(vv, list)}
            if not clean[k]:
                clean[k] = v
        else:
            clean[k] = v

    with open(args.output, 'w') as f:
        json.dump(clean, f, indent=2, default=str)
    print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
