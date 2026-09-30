"""NEXUS local-first runtime primitives."""

from .conflict_resolver import (
    ConflictResolver,
    EntitySnapshot,
    MergeAction,
    MergeDecision,
    MergeEvidence,
    materialize_orset,
    materialize_pn_counter,
)
from .event_store import ClockRelation, EventStore, SyncEvent, compare_vector_clocks
from .merge_policy import (
    MergePolicyKind,
    MergePolicyRegistry,
    MergePolicySpec,
    UnknownMergePolicyError,
)

__all__ = [
    "ClockRelation",
    "ConflictResolver",
    "EntitySnapshot",
    "EventStore",
    "MergeAction",
    "MergeDecision",
    "MergeEvidence",
    "MergePolicyKind",
    "MergePolicyRegistry",
    "MergePolicySpec",
    "SyncEvent",
    "UnknownMergePolicyError",
    "compare_vector_clocks",
    "materialize_orset",
    "materialize_pn_counter",
]
