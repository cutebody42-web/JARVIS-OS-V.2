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
from .integration_hub import IntegrationState, NEXUSIntegrationHub
from .merge_applier import ApplyResult, ApplyStatus, MergeApplier
from .sync_daemon import (
    PROTOCOL_VERSION,
    SyncAck,
    SyncBatch,
    SyncDaemon,
    SyncReport,
    SyncTransport,
)
from .pairing import (
    PAIRING_VERSION,
    PairingManager,
    PairingOffer,
    PairingRequest,
    PendingPairing,
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
from .worker_pool import (
    BoundedWorkerPool,
    WorkerLease,
    WorkerPoolBusy,
    WorkerPoolClosed,
    WorkerSnapshot,
)
from .merge_policy import (
    MergePolicyKind,
    MergePolicyRegistry,
    MergePolicySpec,
    UnknownMergePolicyError,
)

__all__ = [
    "AUTH_VERSION",
    "ApplyResult",
    "BoundedWorkerPool",
    "ApplyStatus",
    "ClockRelation",
    "ConflictResolver",
    "EntitySnapshot",
    "DeviceSigner",
    "EventStore",
    "IntegrationState",
    "MergeAction",
    "MergeApplier",
    "MergeDecision",
    "MergeEvidence",
    "MergePolicyKind",
    "MergePolicyRegistry",
    "MergePolicySpec",
    "MEDIA_TYPE",
    "NEXUSIntegrationHub",
    "NexusSyncNode",
    "PAIRING_VERSION",
    "PairingManager",
    "PairingOffer",
    "PairingRequest",
    "PendingPairing",
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
    "WorkerLease",
    "WorkerPoolBusy",
    "WorkerPoolClosed",
    "WorkerSnapshot",
    "PROTOCOL_VERSION",
    "UnknownMergePolicyError",
    "compare_vector_clocks",
    "materialize_orset",
    "load_or_create_device_signer",
    "materialize_pn_counter",
]
