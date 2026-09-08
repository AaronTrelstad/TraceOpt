"""
TraceOpt Milestone 4: Run all real workload experiments.

Runs three workloads, analyzes each with TraceOpt, and produces
a combined report showing naturally occurring over-synchronization.

Usage:
    python run_all_workloads.py [--device cuda] [--verbose]
"""

import sys
import os
import time
import argparse
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from spec import SyncClassification


def run_workload_safe(name, run_fn, **kwargs):
    """Run a workload with error handling."""
    print(f"\n{'='*70}")
    print(f"WORKLOAD: {name}")
    print(f"{'='*70}")

    start = time.time()
    try:
        profiler = run_fn(**kwargs)
        elapsed = time.time() - start
        print(f"\nExecution time: {elapsed:.1f}s")

        result = profiler.analyze()
        print(profiler.summary(result))

        return {
            'name': name,
            'elapsed_s': elapsed,
            'n_operations': result['n_operations'],
            'n_sync_barriers': result['n_sync_constraints'],
            'n_semantic_deps': result['n_semantic_deps'],
            'n_orderings': result['n_induced_orderings'],
            'n_required': result['n_required'],
            'n_removable': result['n_removable'],
            'n_covered': result['n_covered'],
            'overconstraint_ratio': result['overconstraint_ratio'],
            'n_frontier_events': result.get('n_frontier_events', 0),
            'barriers': [{
                'op_id': sr.sync.sync_op_id,
                'classification': sr.classification.name,
                'host_required': sr.host_obs.required,
                'n_required': sr.n_required,
                'n_removable': sr.n_removable,
                'n_covered': sr.n_covered,
                'n_frontier_events': sr.frontier.n_events
                    if hasattr(sr, 'frontier') else 0,
                'frontier_events': [
                    {'producer': e.producer, 'consumer': e.consumer,
                     'producer_stream': e.producer_stream,
                     'consumer_stream': e.consumer_stream}
                    for e in sr.frontier.events
                ] if hasattr(sr, 'frontier') else [],
            } for sr in result['sync_results']],
            'error': None,
        }
    except Exception as e:
        elapsed = time.time() - start
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            'name': name,
            'elapsed_s': elapsed,
            'error': str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description='TraceOpt Milestone 4: Real workload analysis')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--output', default='workload_results.json')
    parser.add_argument('--workload', default='all',
                        choices=['all', 'inference', 'training', 'pipeline'])
    args = parser.parse_args()

    results = []

    # Workload 1: Inference + logging
    if args.workload in ('all', 'inference'):
        from workload_inference_logging import run_workload as run_inference
        r = run_workload_safe(
            'Inference + Loss Logging',
            run_inference,
            model_name='resnet18',
            batch_size=8,
            n_batches=5,
            device_str=args.device,
            verbose=args.verbose,
        )
        results.append(r)

    # Workload 2: Training with grad accumulation + AMP
    if args.workload in ('all', 'training'):
        from workload_training_grad_accum import run_workload as run_training
        r = run_workload_safe(
            'Training + GradScaler Inf Check',
            run_training,
            model_name='resnet18',
            batch_size=8,
            n_steps=8,
            accum_steps=4,
            device_str=args.device,
            verbose=args.verbose,
        )
        results.append(r)

    # Workload 3: Multi-stream pipeline
    if args.workload in ('all', 'pipeline'):
        from workload_multi_stream_pipeline import run_workload as run_pipeline
        r = run_workload_safe(
            'Multi-Stream Pipeline Serving',
            run_pipeline,
            model_name='resnet18',
            batch_size=8,
            n_batches=6,
            device_str=args.device,
            verbose=args.verbose,
        )
        results.append(r)

    # Combined report
    print(f"\n\n{'='*70}")
    print("MILESTONE 4: COMBINED RESULTS")
    print(f"{'='*70}\n")

    total_barriers = 0
    total_weakenable = 0
    total_redundant = 0
    total_required = 0
    total_orderings = 0
    total_removable_orderings = 0
    total_frontier = 0
    total_required_edges = 0

    header = (f"{'Workload':<35s} {'Barr':>5s} {'WEAK':>5s} "
              f"{'REQ':>4s} {'P×Q':>6s} {'Req':>5s} "
              f"{'Front':>5s} {'Overcon%':>8s}")
    print(header)
    print("-" * len(header))

    for r in results:
        if r.get('error'):
            print(f"{r['name']:<35s} ERROR: {r['error']}")
            continue

        n_weak = sum(1 for b in r['barriers']
                     if b['classification'] == 'WEAKENABLE')
        n_redun = sum(1 for b in r['barriers']
                      if b['classification'] == 'PROVABLY_REDUNDANT')
        n_req = sum(1 for b in r['barriers']
                    if b['classification'] == 'REQUIRED')
        nb = r['n_sync_barriers']
        n_front = r.get('n_frontier_events', 0)

        total_barriers += nb
        total_weakenable += n_weak
        total_redundant += n_redun
        total_required += n_req
        total_orderings += r['n_orderings']
        total_removable_orderings += r['n_removable']
        total_frontier += n_front
        total_required_edges += r['n_required']

        print(f"{r['name']:<35s} {nb:>5d} {n_weak:>5d} "
              f"{n_req:>4d} {r['n_orderings']:>6d} {r['n_required']:>5d} "
              f"{n_front:>5d} {r['overconstraint_ratio']:>7.1%}")

    print("-" * len(header))
    overall_ratio = (total_removable_orderings / max(total_orderings, 1))
    print(f"{'TOTAL':<35s} {total_barriers:>5d} {total_weakenable:>5d} "
          f"{total_required:>4d} {total_orderings:>6d} {total_required_edges:>5d} "
          f"{total_frontier:>5d} {overall_ratio:>7.1%}")

    print(f"\nKey findings:")
    print(f"  - {total_weakenable}/{total_barriers} barriers are WEAKENABLE "
          f"(global sync → targeted events)")
    print(f"  - {total_required_edges} required Cartesian-product edges "
          f"reduce to {total_frontier} minimal frontier events")
    print(f"  - Compression: {total_required_edges} edges → "
          f"{total_frontier} events "
          f"({total_frontier/max(total_required_edges,1):.0%} of naive)")
    print(f"  - {total_removable_orderings}/{total_orderings} "
          f"induced orderings are removable ({overall_ratio:.0%})")

    print(f"\n  The actual optimization replaces {total_weakenable} global "
          f"barriers with {total_frontier} targeted event dependencies.")

    if total_weakenable > 0 or total_redundant > 0:
        print(f"\n  CONCLUSION: Over-synchronization exists naturally "
              f"in common ML pipelines.")
        print(f"  TraceOpt identifies {total_weakenable + total_redundant} "
              f"optimization opportunities across {len(results)} workloads.")
    else:
        print(f"\n  NOTE: No over-synchronization detected. "
              f"Review trace capture completeness.")

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
