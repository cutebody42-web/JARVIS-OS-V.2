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
from .peer_auth import (
    AUTH_VERSION,
    DeviceSigner,
    PeerAuthenticator,
    PeerRegistry,
    TrustedPeer,
    load_or_create_device_signer,
)
from .signed_transport import (
    MEDIA_TYPE,
    SYNC_PATH,
    SignedHTTPSyncTransport,
    SignedSyncEndpoint,
    SyncTransportError,
)
from .sync_node import NexusSyncNode, SyncIdentity
from .sync_scheduler import PeerSchedule, SyncScheduler
from .merge_policy import (
    MergePolicyKind,
    MergePolicyRegistry,
    MergePolicySpec,
    UnknownMergePolicyError,
)

__all__ = [
    "AUTH_VERSION",
    "ApplyResult",
    "ApplyStatus",
    "ClockRelation",
    "ConflictResolver",
    "EntitySnapshot",
    "DeviceSigner",
    "EventStore",
    "MergeAction",
    "MergeApplier",
    "MergeDecision",
    "MergeEvidence",
    "MergePolicyKind",
    "MergePolicyRegistry",
    "MergePolicySpec",
    "MEDIA_TYPE",
    "NexusSyncNode",
    "PeerAuthenticator",
    "PeerRegistry",
    "PeerSchedule",
    "SYNC_PATH",
    "SignedHTTPSyncTransport",
    "SignedSyncEndpoint",
    "SyncAck",
    "SyncBatch",
    "SyncDaemon",
    "SyncEvent",
    "SyncReport",
    "SyncTransport",
    "SyncTransportError",
    "SyncIdentity",
    "SyncScheduler",
    "TrustedPeer",
    "PROTOCOL_VERSION",
    "UnknownMergePolicyError",
    "compare_vector_clocks",
    "materialize_orset",
    "load_or_create_device_signer",
    "materialize_pn_counter",
]
