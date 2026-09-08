"""
TraceOpt Analyzer v2: Constraint-Based Synchronization Analysis

Key change: we analyze SYNCHRONIZATION CONSTRAINTS (barriers), not edges.

For each barrier, we ask:
  "Which of the induced A→B orderings are semantically required?"

A barrier with predecessors={A,C} and successors={B,D} induces 4 orderings.
If only A→B is required, the barrier can be weakened to a single event.

Safety guarantee: ZERO false removals within the supported programming model.
"""

from spec import (
    Operation, OpType, TensorID, GPU_OP_TYPES,
    SyncConstraint, SemanticDep, DepType,
    InducedOrderingResult, OrderingStatus, SyncAnalysisResult,
    SyncClassification, HostObservability, DependencyFrontier,
    ranges_overlap, has_write_read_overlap, has_write_write_overlap,
)
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict


class TraceOptAnalyzer:
    """
    Analyzes an execution trace to:
    1. Extract synchronization constraints (barriers)
    2. Infer semantic dependencies (data, alias, lifetime, host)
    3. Classify each induced ordering as REQUIRED / REMOVABLE / COVERED
    """

    def __init__(self, operations: List[Operation]):
        self.ops = {op.op_id: op for op in operations}
        self.op_list = sorted(operations, key=lambda o: o.op_id)

        # Index ops by stream
        self._by_stream: Dict[int, List[Operation]] = defaultdict(list)
        for op in self.op_list:
            if op.stream_id >= 0:
                self._by_stream[op.stream_id].append(op)

    # ================================================================
    # Step 1: Extract synchronization constraints
    # ================================================================

    def extract_sync_constraints(self) -> List[SyncConstraint]:
        """
        Find all synchronization barriers and their induced orderings.
        """
        constraints = []
        sync_id = 0

        for op in self.op_list:
            if op.op_type == OpType.DEVICE_SYNC:
                # Device sync: ALL preceding GPU ops → ALL subsequent GPU ops
                preds = {o.op_id for o in self.op_list
                         if o.op_id < op.op_id
                         and o.op_type in GPU_OP_TYPES}
                succs = {o.op_id for o in self.op_list
                         if o.op_id > op.op_id
                         and o.op_type in GPU_OP_TYPES}

                if preds and succs:
                    constraints.append(SyncConstraint(
                        sync_id=sync_id,
                        sync_op_id=op.op_id,
                        sync_type=OpType.DEVICE_SYNC,
                        predecessors=preds,
                        successors=succs,
                    ))
                    sync_id += 1

            elif op.op_type == OpType.STREAM_SYNC:
                target = (op.target_stream
                          if op.target_stream is not None
                          else op.stream_id)
                preds = {o.op_id for o in self._by_stream.get(target, [])
                         if o.op_id < op.op_id}
                succs = {o.op_id for o in self.op_list
                         if o.op_id > op.op_id
                         and o.op_type in GPU_OP_TYPES}

                if preds and succs:
                    constraints.append(SyncConstraint(
                        sync_id=sync_id,
                        sync_op_id=op.op_id,
                        sync_type=OpType.STREAM_SYNC,
                        predecessors=preds,
                        successors=succs,
                    ))
                    sync_id += 1

        return constraints

    # ================================================================
    # Step 2: Infer semantic dependencies
    # ================================================================

    def infer_semantic_deps(self) -> List[SemanticDep]:
        """
        Infer all semantic dependencies from the trace.

        Rules:
        1. DATA: B reads tensor written by A (same data_ptr)
        2. ALIAS: Overlapping memory on different streams (write-read or write-write)
        3. MEMORY_LIFETIME: Buffer free/realloc chains
        4. HOST_CONTROL: D2H → sync → CPU decision → kernel
        """
        deps = []

        # Track writers and users per data_ptr
        last_writer: Dict[int, int] = {}   # ptr -> op_id of most recent writer
        all_writers: Dict[int, List[int]] = defaultdict(list)  # ptr -> [op_ids]
        last_free: Dict[int, int] = {}     # ptr -> op_id of free
        last_alloc: Dict[int, int] = {}    # ptr -> op_id of alloc

        for op in self.op_list:
            # Rule 1: DATA dependencies (explicit tensor I/O match)
            for inp in op.inputs:
                if inp.data_ptr in last_writer:
                    writer_id = last_writer[inp.data_ptr]
                    # Only create cross-stream deps (same-stream is inherent)
                    writer_op = self.ops[writer_id]
                    if writer_op.stream_id != op.stream_id or op.stream_id < 0:
                        deps.append(SemanticDep(
                            writer_id, op.op_id, DepType.DATA,
                            f"op {op.op_id} reads ptr {inp.data_ptr} "
                            f"written by op {writer_id}"
                        ))
                    # Same-stream data dep is covered by stream order
                    elif writer_op.stream_id == op.stream_id:
                        deps.append(SemanticDep(
                            writer_id, op.op_id, DepType.STREAM_ORDER,
                            f"same-stream data dep: op {writer_id} → "
                            f"op {op.op_id} via ptr {inp.data_ptr}"
                        ))

            # Rule 2: ALIAS dependencies (overlapping memory, different streams)
            for prev_op in self.op_list:
                if prev_op.op_id >= op.op_id:
                    break
                if prev_op.stream_id == op.stream_id and prev_op.stream_id >= 0:
                    continue  # Same stream — hardware enforces

                # Write-read overlap (including exact ptr match)
                if has_write_read_overlap(prev_op, op):
                    deps.append(SemanticDep(
                        prev_op.op_id, op.op_id, DepType.ALIAS,
                        f"write-read memory overlap between "
                        f"op {prev_op.op_id} and op {op.op_id}"
                    ))

                # Write-write overlap (race hazard)
                if has_write_write_overlap(prev_op, op):
                    deps.append(SemanticDep(
                        prev_op.op_id, op.op_id, DepType.ALIAS,
                        f"write-write memory overlap between "
                        f"op {prev_op.op_id} and op {op.op_id}"
                    ))

            # Rule 3: MEMORY_LIFETIME
            if op.op_type == OpType.FREE and op.alloc_ptr is not None:
                ptr = op.alloc_ptr
                # Must wait for last writer to finish
                if ptr in last_writer:
                    deps.append(SemanticDep(
                        last_writer[ptr], op.op_id, DepType.MEMORY_LIFETIME,
                        f"free of ptr {ptr} must wait for writer op {last_writer[ptr]}"
                    ))
                last_free[ptr] = op.op_id

            if op.op_type == OpType.MALLOC and op.alloc_ptr is not None:
                ptr = op.alloc_ptr
                if ptr in last_free:
                    deps.append(SemanticDep(
                        last_free[ptr], op.op_id, DepType.MEMORY_LIFETIME,
                        f"alloc at ptr {ptr} must wait for free op {last_free[ptr]}"
                    ))
                last_alloc[ptr] = op.op_id

            # Kernel using recently reallocated buffer
            if op.op_type in GPU_OP_TYPES:
                for inp in op.inputs:
                    ptr = inp.data_ptr
                    if ptr in last_alloc and ptr in last_free:
                        if last_alloc[ptr] > last_free[ptr]:
                            deps.append(SemanticDep(
                                last_alloc[ptr], op.op_id,
                                DepType.MEMORY_LIFETIME,
                                f"op {op.op_id} uses reallocated ptr {ptr}, "
                                f"must wait for alloc op {last_alloc[ptr]}"
                            ))

            # Rule 4: HOST_CONTROL
            if op.op_type == OpType.HOST_SCALAR_READ:
                for inp in op.inputs:
                    if inp.data_ptr in last_writer:
                        deps.append(SemanticDep(
                            last_writer[inp.data_ptr], op.op_id,
                            DepType.HOST_CONTROL,
                            f".item()/.cpu() on ptr {inp.data_ptr} "
                            f"requires op {last_writer[inp.data_ptr]} to complete"
                        ))

            if op.op_type == OpType.MEMCPY_D2H:
                for inp in op.inputs:
                    if inp.data_ptr in last_writer:
                        deps.append(SemanticDep(
                            last_writer[inp.data_ptr], op.op_id,
                            DepType.DATA,
                            f"D2H copy reads ptr {inp.data_ptr}"
                        ))

            # CPU_OP after sync that follows D2H → host control dependency
            if op.op_type == OpType.CPU_OP:
                # Find the preceding sync
                for prev in reversed(self.op_list):
                    if prev.op_id >= op.op_id:
                        continue
                    if prev.op_type in (OpType.DEVICE_SYNC, OpType.STREAM_SYNC):
                        # Check if there's a D2H before the sync
                        has_d2h = any(
                            p.op_type == OpType.MEMCPY_D2H
                            for p in self.op_list if p.op_id < prev.op_id
                        )
                        if has_d2h:
                            deps.append(SemanticDep(
                                prev.op_id, op.op_id, DepType.HOST_CONTROL,
                                f"CPU op {op.op_id} depends on sync "
                                f"op {prev.op_id} for host-visible state"
                            ))
                        break

                # CPU op → next kernel (host determines kernel params)
                for next_op in self.op_list:
                    if next_op.op_id <= op.op_id:
                        continue
                    if next_op.op_type == OpType.KERNEL_LAUNCH:
                        deps.append(SemanticDep(
                            op.op_id, next_op.op_id, DepType.HOST_CONTROL,
                            f"CPU op {op.op_id} may determine params "
                            f"for kernel {next_op.op_id}"
                        ))
                        break

            # D2H → sync is required when host needs the result
            if op.op_type in (OpType.DEVICE_SYNC, OpType.STREAM_SYNC):
                # Check if there's a CPU op after that needs host data
                has_cpu_after = any(
                    n.op_type == OpType.CPU_OP
                    for n in self.op_list if n.op_id > op.op_id
                )
                if has_cpu_after:
                    for prev in self.op_list:
                        if prev.op_id >= op.op_id:
                            break
                        if prev.op_type == OpType.MEMCPY_D2H:
                            deps.append(SemanticDep(
                                prev.op_id, op.op_id, DepType.HOST_CONTROL,
                                f"D2H op {prev.op_id} must complete before "
                                f"sync op {op.op_id} for host visibility"
                            ))

            # Update writers
            for out in op.outputs:
                last_writer[out.data_ptr] = op.op_id
                all_writers[out.data_ptr].append(op.op_id)

        return deps

    # ================================================================
    # Step 3: Check if two ops are on the same stream
    # ================================================================

    def _same_stream(self, op_a_id: int, op_b_id: int) -> bool:
        """Check if two ops are on the same CUDA stream."""
        a = self.ops[op_a_id]
        b = self.ops[op_b_id]
        return (a.stream_id == b.stream_id
                and a.stream_id >= 0
                and b.stream_id >= 0)

    # ================================================================
    # Step 3b: Detect host-observability requirements
    # ================================================================

    def _detect_host_observability(self,
                                    sync: SyncConstraint) -> HostObservability:
        """
        Detect whether a barrier serves host-visible semantics.

        A barrier has host-observable requirements when:
        1. A D2H transfer or HOST_SCALAR_READ precedes the barrier
        2. A CPU operation after the barrier consumes the result
        3. The barrier ensures GPU→host transfer completes before CPU reads

        This is SEPARATE from GPU→GPU ordering analysis.
        """
        sync_op = self.ops[sync.sync_op_id]
        d2h_ops = []
        host_scalar_ops = []
        cpu_consumers = []

        # Find D2H and HOST_SCALAR_READ ops before the barrier
        for op in self.op_list:
            if op.op_id >= sync.sync_op_id:
                break
            if op.op_type == OpType.MEMCPY_D2H:
                d2h_ops.append(op.op_id)
            if op.op_type == OpType.HOST_SCALAR_READ:
                host_scalar_ops.append(op.op_id)

        # Find CPU ops after the barrier
        for op in self.op_list:
            if op.op_id <= sync.sync_op_id:
                continue
            if op.op_type == OpType.CPU_OP:
                cpu_consumers.append(op.op_id)

        # Host observability is required when D2H (or scalar read)
        # precedes the barrier AND CPU work follows it
        if d2h_ops and cpu_consumers:
            return HostObservability(
                required=True,
                reason=(f"D2H ops {d2h_ops} must complete before "
                        f"CPU ops {cpu_consumers} can read results"),
                d2h_ops=d2h_ops,
                cpu_consumers=cpu_consumers,
            )

        if host_scalar_ops and cpu_consumers:
            return HostObservability(
                required=True,
                reason=(f"Host scalar reads {host_scalar_ops} require "
                        f"GPU completion before CPU ops {cpu_consumers}"),
                d2h_ops=host_scalar_ops,
                cpu_consumers=cpu_consumers,
            )

        return HostObservability()

    # ================================================================
    # Step 4: Classify induced orderings
    # ================================================================

    def classify_sync(self, sync: SyncConstraint,
                      deps: List[SemanticDep]) -> SyncAnalysisResult:
        """
        For each (predecessor, successor) ordering induced by a sync barrier,
        classify it as REQUIRED, REMOVABLE, or COVERED.
        """
        # Build dependency lookup: (src, dst) -> list of deps
        dep_lookup: Dict[Tuple[int, int], List[SemanticDep]] = defaultdict(list)
        for d in deps:
            dep_lookup[(d.src, d.dst)].append(d)

        # Build reachability via CROSS-STREAM semantic deps only.
        # STREAM_ORDER deps are enforced by hardware and should NOT
        # make cross-stream orderings appear transitively required.
        adj = defaultdict(set)
        for d in deps:
            if d.dep_type != DepType.STREAM_ORDER:
                adj[d.src].add(d.dst)

        results = []
        for pred_id in sync.predecessors:
            for succ_id in sync.successors:
                # Check direct semantic dependency
                direct_deps = dep_lookup.get((pred_id, succ_id), [])

                # Check transitive reachability
                reachable = self._is_reachable(adj, pred_id, succ_id)

                # Check same-stream (stream order already enforces)
                same_stream = self._same_stream(pred_id, succ_id)

                # Check conservative: could there be a hidden dep?
                conservative_dep = self._conservative_dep_check(
                    pred_id, succ_id
                )

                if direct_deps:
                    if same_stream:
                        status = OrderingStatus.COVERED
                        reason = "required but already covered by stream order"
                    else:
                        status = OrderingStatus.REQUIRED
                        reason = "direct semantic dependency"
                    results.append(InducedOrderingResult(
                        sync_id=sync.sync_id,
                        predecessor=pred_id,
                        successor=succ_id,
                        status=status,
                        backing_deps=direct_deps,
                        replacement=(
                            f"event: record after op {pred_id}, "
                            f"wait before op {succ_id}"
                            if status == OrderingStatus.REQUIRED else
                            f"stream order on stream "
                            f"{self.ops[pred_id].stream_id}"
                        ),
                    ))
                elif reachable:
                    if same_stream:
                        status = OrderingStatus.COVERED
                    else:
                        status = OrderingStatus.REQUIRED
                    results.append(InducedOrderingResult(
                        sync_id=sync.sync_id,
                        predecessor=pred_id,
                        successor=succ_id,
                        status=status,
                        backing_deps=[SemanticDep(
                            pred_id, succ_id, DepType.DATA,
                            "transitive dependency"
                        )],
                        replacement=f"transitive via intermediate ops",
                    ))
                elif conservative_dep:
                    # Conservative: mark as required when we detect
                    # potential memory aliasing or lifetime issues
                    status = OrderingStatus.REQUIRED
                    results.append(InducedOrderingResult(
                        sync_id=sync.sync_id,
                        predecessor=pred_id,
                        successor=succ_id,
                        status=status,
                        backing_deps=[conservative_dep],
                        replacement=f"event (conservative: {conservative_dep.reason})",
                    ))
                else:
                    status = OrderingStatus.REMOVABLE
                    results.append(InducedOrderingResult(
                        sync_id=sync.sync_id,
                        predecessor=pred_id,
                        successor=succ_id,
                        status=status,
                        backing_deps=[],
                        replacement="REMOVE: no semantic dependency",
                    ))

        host_obs = self._detect_host_observability(sync)
        result = SyncAnalysisResult(
            sync=sync, orderings=results, host_obs=host_obs)
        result.classification = result.classify_barrier()

        # Compute minimal dependency frontier
        op_streams = {op_id: op.stream_id for op_id, op in self.ops.items()}
        result.frontier = result.compute_dependency_frontier(op_streams)

        return result

    def _conservative_dep_check(self, pred_id: int,
                                 succ_id: int) -> Optional[SemanticDep]:
        """
        Conservative check for hidden dependencies that might not
        appear in explicit tensor I/O tracking.

        Returns a SemanticDep if a potential hidden dependency is found.
        """
        pred = self.ops[pred_id]
        succ = self.ops[succ_id]

        # Check write-read memory overlap
        if has_write_read_overlap(pred, succ):
            return SemanticDep(
                pred_id, succ_id, DepType.ALIAS,
                f"write-read memory overlap"
            )

        # Check write-write memory overlap
        if has_write_write_overlap(pred, succ):
            return SemanticDep(
                pred_id, succ_id, DepType.ALIAS,
                f"write-write memory overlap"
            )

        # Check if there's a free/malloc cycle between them
        pred_ptrs = ({t.data_ptr for t in pred.outputs} |
                     {t.data_ptr for t in pred.inputs})
        succ_ptrs = ({t.data_ptr for t in succ.outputs} |
                     {t.data_ptr for t in succ.inputs})

        for op in self.op_list:
            if op.op_id <= pred_id or op.op_id >= succ_id:
                continue
            if op.op_type == OpType.FREE and op.alloc_ptr is not None:
                if op.alloc_ptr in pred_ptrs or op.alloc_ptr in succ_ptrs:
                    return SemanticDep(
                        pred_id, succ_id, DepType.MEMORY_LIFETIME,
                        f"buffer lifecycle: free at ptr {op.alloc_ptr}"
                    )
            if op.op_type == OpType.MALLOC and op.alloc_ptr is not None:
                if op.alloc_ptr in pred_ptrs or op.alloc_ptr in succ_ptrs:
                    return SemanticDep(
                        pred_id, succ_id, DepType.MEMORY_LIFETIME,
                        f"buffer lifecycle: alloc at ptr {op.alloc_ptr}"
                    )

        return None

    def _is_reachable(self, adj: Dict[int, Set[int]],
                       src: int, dst: int) -> bool:
        """BFS reachability."""
        visited = set()
        queue = [src]
        while queue:
            node = queue.pop(0)
            if node == dst:
                return True
            if node in visited:
                continue
            visited.add(node)
            queue.extend(adj.get(node, set()))
        return False

    # ================================================================
    # Step 5: Full analysis
    # ================================================================

    def analyze(self) -> Dict:
        """
        Complete analysis:
        1. Extract sync constraints
        2. Infer semantic deps
        3. Classify each sync's induced orderings
        """
        constraints = self.extract_sync_constraints()
        deps = self.infer_semantic_deps()

        sync_results = []
        for sync in constraints:
            result = self.classify_sync(sync, deps)
            sync_results.append(result)

        # Aggregate metrics
        total_orderings = sum(len(r.orderings) for r in sync_results)
        total_required = sum(r.n_required for r in sync_results)
        total_removable = sum(r.n_removable for r in sync_results)
        total_covered = sum(r.n_covered for r in sync_results)
        total_frontier_events = sum(
            r.frontier.n_events for r in sync_results
            if hasattr(r, 'frontier'))

        return {
            'n_operations': len(self.ops),
            'n_sync_constraints': len(constraints),
            'n_semantic_deps': len(deps),
            'n_induced_orderings': total_orderings,
            'n_required': total_required,
            'n_removable': total_removable,
            'n_covered': total_covered,
            'n_frontier_events': total_frontier_events,
            'overconstraint_ratio': (
                total_removable / max(total_orderings, 1)
            ),
            'sync_results': sync_results,
            'semantic_deps': deps,
        }
