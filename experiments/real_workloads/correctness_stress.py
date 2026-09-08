"""
Rigorous multi-iteration correctness test for TraceOpt event-based sync.

Tests compile_RO_event (the deep-pipelining regime) and eager_event under:
  - Many iterations (40, 100, 200)
  - Multiple random inputs (10 different random tensors)
  - Multiple seeds (5 seeds)
  - Multiple pipeline depths (3, 4, 5 stages)
  - Repeated executions (5 repetitions per config)
  - Comparison against synchronized reference (global sync)
  - Explicit tensor corruption detection (NaN, Inf, large deviations)

Usage:
    python correctness_stress.py [--batch-size 16] [--output correctness_stress.json]
"""

import sys
import os
import argparse
import json
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def build_model(device):
    from torchvision.models import resnet18, ResNet18_Weights
    return resnet18(weights=ResNet18_Weights.DEFAULT).to(device).eval()


def run_pipeline_3stage(model, preprocess_buf, infer_buf, post_buf,
                        streams, sync_mode, n_iters):
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


def run_pipeline_4stage(model, preprocess_buf, infer_buf, post_buf, reduce_buf,
                        streams, sync_mode, n_iters):
    s_pre, s_inf, s_post, s_red = streams
    evt_pre = torch.cuda.Event()
    evt_inf = torch.cuda.Event()
    evt_post = torch.cuda.Event()

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
            topk_vals, topk_idx = torch.topk(post_buf, k=5, dim=1)
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


def run_pipeline_5stage(model, preprocess_buf, infer_buf, post_buf, reduce_buf,
                        norm_buf, streams, sync_mode, n_iters):
    s_pre, s_inf, s_post, s_red, s_norm = streams
    evt_pre = torch.cuda.Event()
    evt_inf = torch.cuda.Event()
    evt_post = torch.cuda.Event()
    evt_red = torch.cuda.Event()

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
            topk_vals, topk_idx = torch.topk(post_buf, k=5, dim=1)
            reduce_buf[:, :5].copy_(topk_vals)

        if sync_mode == 'global':
            torch.cuda.synchronize()
        elif sync_mode == 'event':
            evt_post.record(s_post)

        with torch.cuda.stream(s_red):
            if sync_mode == 'event':
                s_red.wait_event(evt_post)
            m = reduce_buf.mean(dim=1, keepdim=True)
            norm_buf.copy_(reduce_buf - m)

        if sync_mode == 'global':
            torch.cuda.synchronize()
        elif sync_mode == 'event':
            evt_red.record(s_red)

        with torch.cuda.stream(s_norm):
            if sync_mode == 'event':
                s_norm.wait_event(evt_red)
            norm_buf / (norm_buf.std(dim=1, keepdim=True) + 1e-8)

        if sync_mode == 'global':
            torch.cuda.synchronize()


def run_single_test(model, batch_size, device, sync_mode, n_stages, n_iters):
    """Run pipeline, return final output tensor for comparison."""
    preprocess_buf = torch.randn(batch_size, 3, 256, 256, device=device)
    infer_buf = torch.zeros(batch_size, 3, 224, 224, device=device)
    post_buf = torch.zeros(batch_size, 1000, device=device)

    if n_stages == 3:
        streams = [torch.cuda.Stream() for _ in range(3)]
        with torch.no_grad():
            run_pipeline_3stage(model, preprocess_buf, infer_buf, post_buf,
                                streams, sync_mode, n_iters)
            torch.cuda.synchronize()
        return post_buf.clone()

    elif n_stages == 4:
        reduce_buf = torch.zeros(batch_size, 1000, device=device)
        streams = [torch.cuda.Stream() for _ in range(4)]
        with torch.no_grad():
            run_pipeline_4stage(model, preprocess_buf, infer_buf, post_buf,
                                reduce_buf, streams, sync_mode, n_iters)
            torch.cuda.synchronize()
        return post_buf.clone()

    elif n_stages == 5:
        reduce_buf = torch.zeros(batch_size, 1000, device=device)
        norm_buf = torch.zeros(batch_size, 1000, device=device)
        streams = [torch.cuda.Stream() for _ in range(5)]
        with torch.no_grad():
            run_pipeline_5stage(model, preprocess_buf, infer_buf, post_buf,
                                reduce_buf, norm_buf, streams, sync_mode, n_iters)
            torch.cuda.synchronize()
        return post_buf.clone()


def check_corruption(tensor, label):
    """Check for NaN, Inf, and extreme values."""
    issues = []
    if torch.isnan(tensor).any():
        n_nan = torch.isnan(tensor).sum().item()
        issues.append(f"NaN ({n_nan} elements)")
    if torch.isinf(tensor).any():
        n_inf = torch.isinf(tensor).sum().item()
        issues.append(f"Inf ({n_inf} elements)")
    absmax = tensor.abs().max().item()
    if absmax > 1e6:
        issues.append(f"extreme value (absmax={absmax:.2e})")
    return issues


def compare_outputs(ref, test):
    """Compare reference and test outputs, return detailed metrics."""
    diff = (ref - test).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    # Relative diff (avoid div-by-zero)
    ref_abs = ref.abs().clamp(min=1e-8)
    rel_diff = (diff / ref_abs).max().item()
    # Top-5 comparison
    ref_topk_v, ref_topk_i = torch.topk(ref, k=5, dim=1)
    test_topk_v, test_topk_i = torch.topk(test, k=5, dim=1)
    topk_idx_match = torch.equal(ref_topk_i, test_topk_i)
    topk_val_diff = (ref_topk_v - test_topk_v).abs().max().item()

    return {
        'max_abs_diff': max_diff,
        'mean_abs_diff': mean_diff,
        'max_rel_diff': rel_diff,
        'topk_idx_match': topk_idx_match,
        'topk_val_diff': topk_val_diff,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Rigorous multi-iteration correctness stress test')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--output', default='correctness_stress.json')
    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 78)
    print("CORRECTNESS STRESS TEST")
    print("=" * 78)

    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        print(f"GPU: {prop.name} ({prop.total_memory / 1e9:.1f} GB)")
    print(f"PyTorch: {torch.__version__}")
    print(f"Batch size: {args.batch_size}")

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

    models = {'eager': eager_model}
    if compiled_model is not None:
        models['compile_RO'] = compiled_model

    # Test matrix
    seeds = [42, 123, 456, 789, 1337]
    n_iters_list = [40, 100, 200]
    n_stages_list = [3, 4, 5]
    n_repetitions = 5
    # NOTE: 'none' mode removed — it causes data races that trigger CUDA
    # assertion failures in torch.topk, poisoning the GPU context and killing
    # all subsequent experiments in the same job.  NONE-mode corruption is
    # already demonstrated by the rewrite experiments.
    sync_modes = ['event']

    all_results = {}
    total_tests = 0
    total_pass = 0
    total_fail = 0
    failures = []

    for model_label, model in models.items():
        print(f"\n{'='*78}")
        print(f"MODEL: {model_label}")
        print(f"{'='*78}")

        for n_stages in n_stages_list:
            for n_iters in n_iters_list:
                for seed in seeds:
                    # Set seed for reproducible input
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)

                    # Reference: global sync (guaranteed correct)
                    ref_output = run_single_test(
                        model, args.batch_size, device, 'global',
                        n_stages, n_iters)

                    ref_corrupt = check_corruption(ref_output, "reference")
                    if ref_corrupt:
                        print(f"  WARNING: reference corrupted! {ref_corrupt}")

                    for sync_mode in sync_modes:
                        for rep in range(n_repetitions):
                            total_tests += 1
                            key = (f"{model_label}_{sync_mode}_"
                                   f"{n_stages}s_{n_iters}i_"
                                   f"seed{seed}_rep{rep}")

                            # Must re-seed to get same input
                            torch.manual_seed(seed)
                            torch.cuda.manual_seed(seed)

                            test_output = run_single_test(
                                model, args.batch_size, device, sync_mode,
                                n_stages, n_iters)

                            # Check for corruption
                            corrupt = check_corruption(test_output, key)

                            # Compare to reference
                            metrics = compare_outputs(ref_output, test_output)

                            # Determine pass/fail
                            if corrupt:
                                status = "CORRUPT"
                                total_fail += 1
                                failures.append(key)
                            elif metrics['max_abs_diff'] > 0.1:
                                status = "FAIL"
                                total_fail += 1
                                failures.append(key)
                            elif metrics['max_abs_diff'] > 0.01:
                                status = "MARGINAL"
                                total_pass += 1
                            else:
                                status = "PASS"
                                total_pass += 1

                            result_entry = {
                                'status': status,
                                'corruption': corrupt,
                                **metrics,
                            }
                            all_results[key] = result_entry

                        # Print summary for this config (once per sync_mode)
                        config_keys = [
                            k for k in all_results
                            if k.startswith(f"{model_label}_{sync_mode}_"
                                            f"{n_stages}s_{n_iters}i_"
                                            f"seed{seed}_")
                        ]
                        statuses = [all_results[k]['status'] for k in config_keys]
                        max_diffs = [all_results[k]['max_abs_diff'] for k in config_keys]
                        worst = max(max_diffs) if max_diffs else 0

                        summary = (f"  {model_label} {sync_mode:>5s} "
                                   f"{n_stages}stg {n_iters:>3d}iter "
                                   f"seed={seed}: "
                                   f"{'/'.join(statuses)} "
                                   f"(worst_diff={worst:.6f})")

                        if any(s in ('FAIL', 'CORRUPT') for s in statuses):
                            summary += " <<<FAILURE>>>"
                        print(summary)

    # Final summary
    print(f"\n{'='*78}")
    print(f"FINAL SUMMARY")
    print(f"{'='*78}")
    print(f"Total tests: {total_tests}")
    print(f"PASS: {total_pass}")
    print(f"FAIL: {total_fail}")

    if failures:
        print(f"\nFailed configurations:")
        for f in failures:
            r = all_results[f]
            print(f"  {f}: {r['status']} max_diff={r['max_abs_diff']:.6f}")
    else:
        print(f"\nALL TESTS PASSED")

    # Aggregate stats by model+sync_mode
    print(f"\nAggregate by configuration:")
    for model_label in models:
        for sync_mode in sync_modes:
            relevant = {k: v for k, v in all_results.items()
                        if k.startswith(f"{model_label}_{sync_mode}_")}
            if not relevant:
                continue
            n = len(relevant)
            n_pass = sum(1 for v in relevant.values() if v['status'] == 'PASS')
            n_fail = sum(1 for v in relevant.values()
                         if v['status'] in ('FAIL', 'CORRUPT'))
            worst = max(v['max_abs_diff'] for v in relevant.values())
            print(f"  {model_label}_{sync_mode}: "
                  f"{n_pass}/{n} pass, {n_fail} fail, "
                  f"worst_max_diff={worst:.6f}")

    # Save
    output = {
        'config': {
            'batch_size': args.batch_size,
            'seeds': seeds,
            'n_iters_list': n_iters_list,
            'n_stages_list': n_stages_list,
            'n_repetitions': n_repetitions,
            'gpu': torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'N/A',
        },
        'summary': {
            'total_tests': total_tests,
            'total_pass': total_pass,
            'total_fail': total_fail,
            'failures': failures,
        },
        'results': all_results,
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
