"""
TraceOpt Formal Specification (v2)

Key change from v1: Synchronization is modeled as a CONSTRAINT (barrier)
that induces a set of orderings, not as a vertex with edges.

We classify the INDUCED orderings (A → B via sync), not the
sync-to-vertex edges (A → sync, sync → B).

The question for each induced ordering is:
  "Is A → B semantically required? If so, can a weaker constraint
   (event dependency) replace the global barrier?"
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Set, Dict, List, Optional, Tuple, FrozenSet
from collections import defaultdict


# ============================================================
# 1. Operation Types
# ============================================================

class OpType(Enum):
    KERNEL_LAUNCH = auto()
    MEMCPY_H2D = auto()
    MEMCPY_D2H = auto()
    MEMCPY_D2D = auto()
    MEMSET = auto()
    DEVICE_SYNC = auto()
    STREAM_SYNC = auto()
    EVENT_RECORD = auto()
    EVENT_WAIT = auto()
    EVENT_SYNC = auto()
    MALLOC = auto()
    FREE = auto()
    CPU_OP = auto()
    HOST_SCALAR_READ = auto()


GPU_OP_TYPES = {
    OpType.KERNEL_LAUNCH, OpType.MEMCPY_H2D, OpType.MEMCPY_D2H,
    OpType.MEMCPY_D2D, OpType.MEMSET,
}


# ============================================================
# 2. Core Data Structures
# ============================================================

@dataclass(frozen=True)
class TensorID:
    """Identifies a tensor by its data pointer and size."""
    data_ptr: int
    size_bytes: int
    device: int = 0


@dataclass
class Operation:
    """A single operation in the execution trace."""
    op_id: int
    op_type: OpType
    name: str
    stream_id: int = 0
    device_id: int = 0
    # Tensor I/O
    inputs: FrozenSet[TensorID] = field(default_factory=frozenset)
    outputs: FrozenSet[TensorID] = field(default_factory=frozenset)
    # For sync ops
    target_stream: Optional[int] = None
    target_event: Optional[int] = None
    # For memory ops
    alloc_ptr: Optional[int] = None
    alloc_size: Optional[int] = None


# ============================================================
# 3. Synchronization Constraint (the key abstraction)
# ============================================================

@dataclass
class SyncConstraint:
    """
    A synchronization barrier that induces orderings.

    A device-wide sync with predecessors={A,C} and successors={B,D}
    induces: A→B, A→D, C→B, C→D.

    We classify EACH induced ordering as required or removable.
    """
    sync_id: int
    sync_op_id: int            # The operation that is the sync
    sync_type: OpType          # DEVICE_SYNC, STREAM_SYNC, etc.
    predecessors: Set[int]     # op_ids that must complete before sync
    successors: Set[int]       # op_ids that wait on sync

    def induced_orderings(self) -> Set[Tuple[int, int]]:
        """All (predecessor, successor) orderings induced by this barrier."""
        return {(p, s) for p in self.predecessors for s in self.successors}


# ============================================================
# 4. Dependency Types
# ============================================================

class DepType(Enum):
    """Types of semantic dependencies."""
    DATA = auto()           # B reads tensor written by A
    ALIAS = auto()          # Overlapping memory regions (may-alias)
    MEMORY_LIFETIME = auto()  # Buffer reuse / allocator constraint
    HOST_CONTROL = auto()   # Host reads GPU result → decides next op
    STREAM_ORDER = auto()   # Same-stream sequential ordering


@dataclass(frozen=True)
class SemanticDep:
    """A required semantic dependency between two operations."""
    src: int        # op_id
    dst: int        # op_id
    dep_type: DepType
    reason: str     # Human-readable explanation


# ============================================================
# 5. Induced Ordering Classification (per ordering pair)
# ============================================================

class OrderingStatus(Enum):
    """Per-ordering classification (internal detail)."""
    REQUIRED = auto()       # Semantic dependency exists → keep
    REMOVABLE = auto()      # No semantic dependency → can weaken
    COVERED = auto()        # Required, but already covered by stream order


# ============================================================
# 6. Barrier-Level Classification (the primary output)
# ============================================================

class SyncClassification(Enum):
    """
    Per-barrier classification — every sync point gets exactly one label.

    REQUIRED:           All induced orderings are semantically required.
                        The sync enforces necessary dependencies.
    PROVABLY_REDUNDANT: No induced ordering is semantically required.
                        The entire sync can be removed.
    WEAKENABLE:         Some orderings are required, but the global barrier
                        is stronger than necessary. Replace with targeted
                        event dependencies.
    UNKNOWN:            The analyzer cannot prove safety. Preserve the
                        original synchronization.
    """
    REQUIRED = auto()
    PROVABLY_REDUNDANT = auto()
    WEAKENABLE = auto()
    UNKNOWN = auto()


@dataclass
class InducedOrderingResult:
    """Classification of one (predecessor, successor) pair from a sync."""
    sync_id: int
    predecessor: int
    successor: int
    status: OrderingStatus
    backing_deps: List[SemanticDep]   # Why it's required (empty if removable)
    replacement: Optional[str] = None  # Suggested weaker constraint


@dataclass
class HostObservability:
    """
    Analysis of whether a barrier serves host-visible semantics.

    A barrier has host-observable requirements when:
    - A D2H transfer or HOST_SCALAR_READ precedes the barrier
    - A CPU operation after the barrier reads the transferred value
    - The barrier ensures GPU→host transfer completes before CPU reads

    This is SEPARATE from GPU→GPU ordering analysis.
    """
    required: bool = False
    reason: str = ""
    d2h_ops: List[int] = field(default_factory=list)
    cpu_consumers: List[int] = field(default_factory=list)


@dataclass
class SyncAnalysisResult:
    """
    Full analysis of one synchronization barrier.

    Contains two separate analyses:
    1. GPU ordering impact: per-ordering REQUIRED/REMOVABLE/COVERED
    2. Host observability: does the barrier serve host-visible semantics?

    The barrier-level classification considers BOTH dimensions.
    """
    sync: SyncConstraint
    orderings: List[InducedOrderingResult]
    host_obs: HostObservability = field(default_factory=HostObservability)
    classification: SyncClassification = SyncClassification.UNKNOWN

    @property
    def n_required(self) -> int:
        return sum(1 for o in self.orderings
                   if o.status == OrderingStatus.REQUIRED)

    @property
    def n_removable(self) -> int:
        return sum(1 for o in self.orderings
                   if o.status == OrderingStatus.REMOVABLE)

    @property
    def n_covered(self) -> int:
        return sum(1 for o in self.orderings
                   if o.status == OrderingStatus.COVERED)

    def classify_barrier(self) -> SyncClassification:
        """
        Derive barrier-level classification from BOTH GPU ordering
        and host-observability analysis.

        PROVABLY_REDUNDANT: no GPU orderings required AND no host semantics
        WEAKENABLE: some GPU orderings removable while others (or host) required
        REQUIRED: all GPU orderings required AND/OR only host semantics
                  with nothing to weaken
        UNKNOWN: insufficient information
        """
        n_req = self.n_required
        n_rem = self.n_removable
        n_cov = self.n_covered
        host_req = self.host_obs.required

        if not host_req and n_req == 0:
            # No GPU ordering required, no host semantics
            return SyncClassification.PROVABLY_REDUNDANT
        elif host_req and n_rem > 0:
            # Host needs the sync, but some GPU orderings are removable
            # → can separate host sync from unnecessary GPU blocking
            return SyncClassification.WEAKENABLE
        elif not host_req and n_rem > 0:
            # No host requirement, some GPU orderings removable
            return SyncClassification.WEAKENABLE
        elif host_req and n_rem == 0 and n_req == 0:
            # Host needs sync, no GPU orderings to weaken
            return SyncClassification.REQUIRED
        elif n_rem == 0 and n_cov == 0:
            # All GPU orderings required, no removable
            return SyncClassification.REQUIRED
        else:
            # Has required GPU orderings + covered, no removable
            return SyncClassification.REQUIRED

    @property
    def replacement_suggestion(self) -> str:
        """Generate the replacement suggestion for this sync."""
        if self.classification == SyncClassification.PROVABLY_REDUNDANT:
            return "REMOVE: no semantic dependencies across this barrier"

        parts = []

        if self.host_obs.required:
            parts.append(f"PRESERVE host sync: {self.host_obs.reason}")

        required = [o for o in self.orderings
                    if o.status == OrderingStatus.REQUIRED]
        if required:
            event_deps = [
                f"event: record after op {o.predecessor}, "
                f"wait before op {o.successor}"
                for o in required
            ]
            parts.append("GPU events:\n    " + "\n    ".join(event_deps))

        if self.classification == SyncClassification.WEAKENABLE:
            removable = [o for o in self.orderings
                         if o.status == OrderingStatus.REMOVABLE]
            return (f"WEAKEN: replace global barrier. "
                    f"Remove {len(removable)} unnecessary GPU ordering(s).\n  "
                    + "\n  ".join(parts))
        else:
            return ("REQUIRED:\n  " + "\n  ".join(parts))


# ============================================================
# 6. Helper Functions
# ============================================================

def ranges_overlap(t1: TensorID, t2: TensorID) -> bool:
    """Check if two tensor memory ranges overlap (including exact match)."""
    if t1.device != t2.device:
        return False
    start1, end1 = t1.data_ptr, t1.data_ptr + t1.size_bytes
    start2, end2 = t2.data_ptr, t2.data_ptr + t2.size_bytes
    return start1 < end2 and start2 < end1


def has_write_read_overlap(writer: Operation, reader: Operation) -> bool:
    """Check if reader reads any memory that writer writes."""
    for out in writer.outputs:
        for inp in reader.inputs:
            if ranges_overlap(out, inp):
                return True
    return False


def has_write_write_overlap(op1: Operation, op2: Operation) -> bool:
    """Check if two ops write overlapping memory (race hazard)."""
    for out1 in op1.outputs:
        for out2 in op2.outputs:
            if ranges_overlap(out1, out2):
                return True
    return False
