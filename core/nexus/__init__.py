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
from .merge_applier import ApplyResult, ApplyStatus, MergeApplier
from .sync_daemon import (
    PROTOCOL_VERSION,
    SyncAck,
    SyncBatch,
    SyncDaemon,
    SyncReport,
    SyncTransport,
)
from .merge_policy import (
    MergePolicyKind,
    MergePolicyRegistry,
    MergePolicySpec,
    UnknownMergePolicyError,
)

__all__ = [
    "ApplyResult",
    "ApplyStatus",
    "ClockRelation",
    "ConflictResolver",
    "EntitySnapshot",
    "EventStore",
    "MergeAction",
    "MergeApplier",
    "MergeDecision",
    "MergeEvidence",
    "MergePolicyKind",
    "MergePolicyRegistry",
    "MergePolicySpec",
    "SyncAck",
    "SyncBatch",
    "SyncDaemon",
    "SyncEvent",
    "SyncReport",
    "SyncTransport",
    "PROTOCOL_VERSION",
    "UnknownMergePolicyError",
    "compare_vector_clocks",
    "materialize_orset",
    "materialize_pn_counter",
    "KeyringPeerKeyStore",
    "NonceReplayCache",
    "PeerEndpoint",
    "PeerRegistry",
    "SignedSyncHttpEndpoint",
    "TailscaleCLI",
    "TailscaleHttpTransport",
    "generate_pairwise_secret",
]

from .tailscale_transport import (
    KeyringPeerKeyStore,
    NonceReplayCache,
    PeerEndpoint,
    PeerRegistry,
    SignedSyncHttpEndpoint,
    TailscaleCLI,
    TailscaleHttpTransport,
    generate_pairwise_secret,
)
