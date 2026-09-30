"""Deployable composition for one authenticated NEXUS sync node."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.nexus.event_store import EventStore
from core.nexus.merge_applier import MergeApplier
from core.nexus.pairing import PairingManager
from core.nexus.peer_auth import (
    DeviceSigner,
    PeerAuthenticator,
    PeerRegistry,
    TrustedPeer,
    load_or_create_device_signer,
)
from core.nexus.signed_transport import SignedHTTPSyncTransport, SignedSyncEndpoint
from core.nexus.sync_daemon import SyncDaemon
from core.nexus.sync_scheduler import SyncScheduler
from core.secret_store import SecretStore


@dataclass(frozen=True)
class SyncIdentity:
    device_id: str
    public_key: str


class NexusSyncNode:
    """Own all durable/authenticated sync components for one device."""

    def __init__(
        self,
        state_dir: str | Path,
        device_id: str,
        *,
        secret_store: SecretStore | None = None,
        signer: DeviceSigner | None = None,
        http_session=None,
    ):
        self.store = EventStore(state_dir, device_id)
        self.applier = MergeApplier(self.store)
        self.daemon = SyncDaemon(self.store, self.applier)
        self.registry = PeerRegistry(self.store)
        self.signer = signer or load_or_create_device_signer(
            device_id,
            secret_store=secret_store,
        )
        self.authenticator = PeerAuthenticator(
            device_id,
            self.signer,
            self.registry,
        )
        self.pairing = PairingManager(
            self.registry,
            device_id,
            self.signer,
        )
        self.endpoint = SignedSyncEndpoint(self.daemon, self.authenticator)
        self.transport = SignedHTTPSyncTransport(
            device_id,
            self.registry,
            self.authenticator,
            session=http_session,
        )
        self.scheduler = SyncScheduler(
            self.daemon,
            self.registry,
            self.transport,
        )

    @property
    def identity(self) -> SyncIdentity:
        return SyncIdentity(self.store.device_id, self.signer.public_b64)

    def trust_peer(self, peer_id: str, public_key: str, endpoint: str) -> TrustedPeer:
        return self.registry.trust_peer(peer_id, public_key, endpoint)

    def revoke_peer(self, peer_id: str) -> bool:
        return self.registry.revoke_peer(peer_id)

    def recover(self) -> dict[str, int]:
        """Recover durable local/inbound work before normal peer scheduling."""
        local = self.applier.apply_pending_local()
        inbound = self.applier.apply_pending()
        exported = self.store.flush_pending()
        return {
            "local_applied": len(local),
            "inbound_applied": len(inbound),
            "outbox_exported": exported,
        }
