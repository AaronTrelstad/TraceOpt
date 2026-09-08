"""
TraceOpt Real Workload Profiler

Captures CUDA synchronization events from PyTorch workloads and converts
them to TraceOpt's Operation format for analysis.

Two capture modes:
1. Hook-based: monkey-patches torch.cuda.synchronize and torch.Tensor.item()
2. Profiler-based: uses torch.profiler to capture kernel launches + syncs

Usage:
    from profiler import SyncProfiler
    profiler = SyncProfiler()
    with profiler.capture():
        # ... your PyTorch code ...
    ops = profiler.to_operations()
    result = profiler.analyze()
"""

import sys
import os
import time
import threading
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
from contextlib import contextmanager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import torch
import torch.cuda

from spec import (
    Operation, OpType, TensorID, SyncConstraint,
    SyncClassification, OrderingStatus,
)
from analyzer import TraceOptAnalyzer


@dataclass
class CapturedEvent:
    """A single captured CUDA event during profiling."""
    event_id: int
    event_type: str           # 'kernel', 'sync', 'memcpy', 'item', 'cpu_op'
    name: str
    stream_id: int = 0
    timestamp_ns: int = 0
    # Tensor metadata (for kernels)
    input_ptrs: List[Tuple[int, int]] = field(default_factory=list)   # [(ptr, size)]
    output_ptrs: List[Tuple[int, int]] = field(default_factory=list)  # [(ptr, size)]
    # Sync metadata
    sync_type: str = ''       # 'device', 'stream', 'event'
    target_stream: Optional[int] = None
    # Origin tracking: where did this sync come from?
    origin: str = 'unknown'   # 'framework', 'library_api', 'user_explicit', 'instrumentation'


class SyncProfiler:
    """
    Captures CUDA synchronization patterns from live PyTorch execution.

    Instruments:
    - torch.cuda.synchronize() → DEVICE_SYNC
    - torch.cuda.Stream.synchronize() → STREAM_SYNC
    - torch.Tensor.item() → HOST_SCALAR_READ + implicit sync
    - Kernel launches via torch.profiler
    - D2H memcpy (tensor.cpu(), tensor.item())
    """

    def __init__(self, verbose: bool = False):
        self.events: List[CapturedEvent] = []
        self._event_counter = 0
        self._lock = threading.Lock()
        self._original_sync = None
        self._original_item = None
        self._original_cpu = None
        self._capturing = False
        self.verbose = verbose
        # Track tensor data pointers seen during kernels
        self._tensor_registry: Dict[int, int] = {}  # data_ptr -> size

    def _next_id(self) -> int:
        with self._lock:
            eid = self._event_counter
            self._event_counter += 1
            return eid

    def _record(self, event_type: str, name: str, stream_id: int = 0,
                input_ptrs=None, output_ptrs=None,
                sync_type: str = '', target_stream=None,
                origin: str = 'unknown'):
        ev = CapturedEvent(
            event_id=self._next_id(),
            event_type=event_type,
            name=name,
            stream_id=stream_id,
            timestamp_ns=time.time_ns(),
            input_ptrs=input_ptrs or [],
            output_ptrs=output_ptrs or [],
            sync_type=sync_type,
            target_stream=target_stream,
            origin=origin,
        )
        self.events.append(ev)
        if self.verbose:
            print(f"  [TRACE] {ev.event_type:>8s} | {ev.name} "
                  f"| stream={ev.stream_id} | origin={ev.origin}")
        return ev

    def _get_tensor_io(self, args, kwargs=None):
        """Extract tensor data pointers from function arguments."""
        inputs = []
        outputs = []
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.is_cuda:
                ptr = arg.data_ptr()
                size = arg.nelement() * arg.element_size()
                inputs.append((ptr, size))
                self._tensor_registry[ptr] = size
            elif isinstance(arg, (list, tuple)):
                for a in arg:
                    if isinstance(a, torch.Tensor) and a.is_cuda:
                        ptr = a.data_ptr()
                        size = a.nelement() * a.element_size()
                        inputs.append((ptr, size))
                        self._tensor_registry[ptr] = size
        return inputs, outputs

    @contextmanager
    def capture(self):
        """Context manager that instruments PyTorch CUDA operations."""
        self._install_hooks()
        self._capturing = True
        try:
            yield self
        finally:
            self._capturing = False
            self._remove_hooks()

    def _install_hooks(self):
        # Patch torch.cuda.synchronize
        self._original_sync = torch.cuda.synchronize
        profiler = self

        def patched_sync(device=None):
            profiler._original_sync(device)
            if profiler._capturing:
                profiler._record('sync', 'cudaDeviceSynchronize',
                                 stream_id=-1, sync_type='device',
                                 origin='user_explicit')

        torch.cuda.synchronize = patched_sync

        # Patch Tensor.item() — triggers implicit device sync
        self._original_item = torch.Tensor.item
        def patched_item(tensor_self):
            result = profiler._original_item(tensor_self)
            if profiler._capturing and tensor_self.is_cuda:
                ptr = tensor_self.data_ptr()
                size = tensor_self.nelement() * tensor_self.element_size()
                profiler._record('item', f'Tensor.item() ptr={ptr:#x}',
                                 stream_id=-1,
                                 input_ptrs=[(ptr, size)],
                                 sync_type='device',
                                 origin='library_api')
            return result

        torch.Tensor.item = patched_item

        # Patch Tensor.cpu() — D2H memcpy
        self._original_cpu = torch.Tensor.cpu
        def patched_cpu(tensor_self, *args, **kwargs):
            result = profiler._original_cpu(tensor_self, *args, **kwargs)
            if profiler._capturing and tensor_self.is_cuda:
                ptr = tensor_self.data_ptr()
                size = tensor_self.nelement() * tensor_self.element_size()
                profiler._record('memcpy', f'D2H ptr={ptr:#x}',
                                 stream_id=0,
                                 input_ptrs=[(ptr, size)],
                                 sync_type='',
                                 origin='library_api')
            return result

        torch.Tensor.cpu = patched_cpu

    def _remove_hooks(self):
        if self._original_sync:
            torch.cuda.synchronize = self._original_sync
        if self._original_item:
            torch.Tensor.item = self._original_item
        if self._original_cpu:
            torch.Tensor.cpu = self._original_cpu

    def capture_profiler_trace(self, activities=None):
        """
        Use torch.profiler to capture kernel launches alongside our hooks.
        Returns the profiler context manager.
        """
        if activities is None:
            activities = [
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        return torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            with_stack=False,
        )

    def ingest_profiler_events(self, prof):
        """
        Convert torch.profiler events into CapturedEvents.
        Extracts kernel launches with their stream IDs.
        """
        for ev in prof.key_averages():
            if ev.device_type is not None and 'cuda' in str(ev.device_type).lower():
                self._record('kernel', ev.key, stream_id=0)

    def to_operations(self) -> List[Operation]:
        """
        Convert captured events to TraceOpt Operation objects.

        Mapping:
        - 'kernel' → KERNEL_LAUNCH
        - 'sync' (device) → DEVICE_SYNC
        - 'sync' (stream) → STREAM_SYNC
        - 'item' → HOST_SCALAR_READ + DEVICE_SYNC
        - 'memcpy' (D2H) → MEMCPY_D2H
        - 'cpu_op' → CPU_OP
        """
        ops = []
        op_id = 0

        for ev in self.events:
            if ev.event_type == 'kernel':
                ops.append(Operation(
                    op_id=op_id,
                    op_type=OpType.KERNEL_LAUNCH,
                    name=ev.name,
                    stream_id=ev.stream_id,
                    inputs=frozenset(
                        TensorID(ptr, size) for ptr, size in ev.input_ptrs
                    ),
                    outputs=frozenset(
                        TensorID(ptr, size) for ptr, size in ev.output_ptrs
                    ),
                ))
                op_id += 1

            elif ev.event_type == 'sync':
                if ev.sync_type == 'device':
                    ops.append(Operation(
                        op_id=op_id,
                        op_type=OpType.DEVICE_SYNC,
                        name=ev.name,
                        stream_id=-1,
                    ))
                    op_id += 1
                elif ev.sync_type == 'stream':
                    ops.append(Operation(
                        op_id=op_id,
                        op_type=OpType.STREAM_SYNC,
                        name=ev.name,
                        stream_id=ev.stream_id,
                        target_stream=ev.target_stream,
                    ))
                    op_id += 1

            elif ev.event_type == 'item':
                # .item() = HOST_SCALAR_READ (implicit sync)
                ops.append(Operation(
                    op_id=op_id,
                    op_type=OpType.HOST_SCALAR_READ,
                    name=ev.name,
                    stream_id=-1,
                    inputs=frozenset(
                        TensorID(ptr, size) for ptr, size in ev.input_ptrs
                    ),
                ))
                op_id += 1
                # The .item() also triggers a device sync
                ops.append(Operation(
                    op_id=op_id,
                    op_type=OpType.DEVICE_SYNC,
                    name='implicit_sync_from_item',
                    stream_id=-1,
                ))
                op_id += 1

            elif ev.event_type == 'memcpy':
                ops.append(Operation(
                    op_id=op_id,
                    op_type=OpType.MEMCPY_D2H,
                    name=ev.name,
                    stream_id=ev.stream_id,
                    inputs=frozenset(
                        TensorID(ptr, size) for ptr, size in ev.input_ptrs
                    ),
                ))
                op_id += 1

            elif ev.event_type == 'cpu_op':
                ops.append(Operation(
                    op_id=op_id,
                    op_type=OpType.CPU_OP,
                    name=ev.name,
                    stream_id=-1,
                ))
                op_id += 1

        return ops

    def analyze(self) -> Dict:
        """Convert captured events to Operations and run TraceOpt analyzer."""
        ops = self.to_operations()
        analyzer = TraceOptAnalyzer(ops)
        return analyzer.analyze()

    def summary(self, result: Optional[Dict] = None) -> str:
        """Generate a human-readable summary of the analysis."""
        if result is None:
            result = self.analyze()

        lines = []
        lines.append("=" * 60)
        lines.append("TraceOpt Real Workload Analysis")
        lines.append("=" * 60)
        lines.append(f"Operations captured: {result['n_operations']}")
        lines.append(f"Sync barriers found: {result['n_sync_constraints']}")
        lines.append(f"Semantic dependencies: {result['n_semantic_deps']}")
        lines.append(f"Induced orderings: {result['n_induced_orderings']}")
        lines.append(f"  Required: {result['n_required']}")
        lines.append(f"  Removable: {result['n_removable']}")
        lines.append(f"  Covered: {result['n_covered']}")
        lines.append(f"Overconstraint ratio: {result['overconstraint_ratio']:.1%}")
        lines.append(f"Frontier events: {result.get('n_frontier_events', '?')}")
        lines.append(f"  (vs {result['n_required']} required Cartesian-product edges)")
        lines.append("")

        for sr in result['sync_results']:
            host_tag = " [HOST_REQ]" if sr.host_obs.required else ""
            frontier_info = ""
            if hasattr(sr, 'frontier') and sr.frontier.n_events > 0:
                frontier_info = f" → {sr.frontier.n_events} event(s)"
            lines.append(
                f"  Barrier op={sr.sync.sync_op_id}: "
                f"{sr.classification.name}{host_tag} "
                f"({sr.n_required} req, {sr.n_removable} rem, "
                f"{sr.n_covered} cov){frontier_info}"
            )
            if hasattr(sr, 'frontier') and sr.frontier.n_events > 0:
                for ev in sr.frontier.events:
                    lines.append(
                        f"    event: record s{ev.producer_stream} "
                        f"op {ev.producer} → wait s{ev.consumer_stream} "
                        f"op {ev.consumer}"
                    )

        lines.append("=" * 60)
        return "\n".join(lines)

    def save_trace(self, path: str):
        """Save captured events to JSON for offline analysis."""
        data = []
        for ev in self.events:
            data.append({
                'event_id': ev.event_id,
                'event_type': ev.event_type,
                'name': ev.name,
                'stream_id': ev.stream_id,
                'timestamp_ns': ev.timestamp_ns,
                'input_ptrs': ev.input_ptrs,
                'output_ptrs': ev.output_ptrs,
                'sync_type': ev.sync_type,
                'target_stream': ev.target_stream,
                'origin': ev.origin,
            })
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_trace(cls, path: str) -> 'SyncProfiler':
        """Load a saved trace for offline analysis."""
        profiler = cls()
        with open(path) as f:
            data = json.load(f)
        for d in data:
            profiler.events.append(CapturedEvent(
                event_id=d['event_id'],
                event_type=d['event_type'],
                name=d['name'],
                stream_id=d['stream_id'],
                timestamp_ns=d['timestamp_ns'],
                input_ptrs=[tuple(x) for x in d['input_ptrs']],
                output_ptrs=[tuple(x) for x in d['output_ptrs']],
                sync_type=d['sync_type'],
                target_stream=d.get('target_stream'),
                origin=d.get('origin', 'unknown'),
            ))
        profiler._event_counter = len(profiler.events)
        return profiler
