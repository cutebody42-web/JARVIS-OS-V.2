"""Signed NEXUS sync protocol and HTTP transport adapter.

The SyncDaemon remains transport-neutral. This module serializes its batches and
ACKs into Ed25519-authenticated envelopes and moves them over an explicitly
trusted HTTP(S) peer origin (typically a Tailscale/MagicDNS address).

HTTP redirects are rejected. The peer registry, not response headers or DNS
discovery, decides where a device may send semantic state.
"""

from __future__ import annotations

from typing import Any, Mapping

from core.nexus.event_store import SyncEvent
from core.nexus.peer_auth import (
    MAX_ENVELOPE_BYTES,
    PeerAuthenticator,
    PeerRegistry,
)
from core.nexus.sync_daemon import (
    PROTOCOL_VERSION,
    SyncAck,
    SyncBatch,
    SyncDaemon,
    SyncTransport,
)


MEDIA_TYPE = "application/vnd.nexus-sync+json"
SYNC_PATH = "/nexus/sync/v1/batch"
MAX_HTTP_RESPONSE_BYTES = MAX_ENVELOPE_BYTES


class SyncTransportError(RuntimeError):
    pass


def batch_to_payload(batch: SyncBatch) -> dict[str, Any]:
    if not isinstance(batch, SyncBatch):
        raise TypeError("batch must be SyncBatch")
    return {
        "protocol_version": batch.protocol_version,
        "sender_device": batch.sender_device,
        "batch_id": batch.batch_id,
        "events": [event.to_dict() for event in batch.events],
    }


def batch_from_payload(value: Mapping[str, Any]) -> SyncBatch:
    if not isinstance(value, Mapping):
        raise TypeError("batch payload must be an object")
    allowed = {"protocol_version", "sender_device", "batch_id", "events"}
    if set(value) != allowed:
        raise ValueError("batch payload schema mismatch")
    events = value["events"]
    if not isinstance(events, list) or not events:
        raise ValueError("batch events must be a non-empty list")
    if len(events) > 50:
        raise ValueError("batch exceeds protocol event-count limit")
    return SyncBatch(
        sender_device=value["sender_device"],
        batch_id=value["batch_id"],
        events=tuple(SyncEvent.from_dict(item) for item in events),
        protocol_version=value["protocol_version"],
    )


def ack_to_payload(ack: SyncAck) -> dict[str, Any]:
    if not isinstance(ack, SyncAck):
        raise TypeError("ack must be SyncAck")
    return {
        "protocol_version": ack.protocol_version,
        "receiver_device": ack.receiver_device,
        "batch_id": ack.batch_id,
        "durable_event_ids": list(ack.durable_event_ids),
        "applied_event_ids": list(ack.applied_event_ids),
        "failed_event_ids": list(ack.failed_event_ids),
        "pending_event_ids": list(ack.pending_event_ids),
    }


def ack_from_payload(value: Mapping[str, Any]) -> SyncAck:
    if not isinstance(value, Mapping):
        raise TypeError("ack payload must be an object")
    allowed = {
        "protocol_version",
        "receiver_device",
        "batch_id",
        "durable_event_ids",
        "applied_event_ids",
        "failed_event_ids",
        "pending_event_ids",
    }
    if set(value) != allowed:
        raise ValueError("ack payload schema mismatch")
    for field in (
        "durable_event_ids",
        "applied_event_ids",
        "failed_event_ids",
        "pending_event_ids",
    ):
        if not isinstance(value[field], list):
            raise ValueError(f"{field} must be a list")
    return SyncAck(
        receiver_device=value["receiver_device"],
        batch_id=value["batch_id"],
        durable_event_ids=tuple(value["durable_event_ids"]),
        applied_event_ids=tuple(value["applied_event_ids"]),
        failed_event_ids=tuple(value["failed_event_ids"]),
        pending_event_ids=tuple(value["pending_event_ids"]),
        protocol_version=value["protocol_version"],
    )


class SignedSyncEndpoint:
    """Pure request handler usable by FastAPI, tests or another HTTP server."""

    def __init__(self, daemon: SyncDaemon, authenticator: PeerAuthenticator):
        if not isinstance(daemon, SyncDaemon):
            raise TypeError("daemon must be SyncDaemon")
        if not isinstance(authenticator, PeerAuthenticator):
            raise TypeError("authenticator must be PeerAuthenticator")
        if daemon.device_id != authenticator.local_device:
            raise ValueError("daemon and authenticator device ids must match")
        self.daemon = daemon
        self.authenticator = authenticator

    def handle_batch(self, encoded: bytes) -> bytes:
        peer, payload, message_id = self.authenticator.verify(
            encoded,
            expected_kind="sync.batch",
        )
        batch = batch_from_payload(payload)
        if batch.sender_device != peer.peer_id:
            raise PermissionError("signed sender does not match batch origin")
        if message_id != batch.batch_id:
            raise PermissionError("signed message id does not match batch id")

        ack = self.daemon.receive_batch(batch)
        return self.authenticator.sign(
            "sync.ack",
            peer.peer_id,
            ack_to_payload(ack),
            message_id="ack:" + batch.batch_id,
        )


class SignedHTTPSyncTransport(SyncTransport):
    """HTTP client for a peer whose origin/public key are explicitly trusted."""

    def __init__(
        self,
        local_device: str,
        registry: PeerRegistry,
        authenticator: PeerAuthenticator,
        *,
        session=None,
        timeout_seconds: float = 15.0,
    ):
        if authenticator.local_device != local_device:
            raise ValueError("local device and authenticator must match")
        if isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.local_device = local_device
        self.registry = registry
        self.authenticator = authenticator
        self._session = session
        self._timeout = timeout_seconds

    def _client(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session

    @staticmethod
    def _read_bounded(response) -> bytes:
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=65536):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_HTTP_RESPONSE_BYTES:
                raise SyncTransportError("Sync response exceeds the byte budget.")
            chunks.append(chunk)
        return b"".join(chunks)

    def send_batch(self, peer_id: str, batch: SyncBatch) -> SyncAck:
        if not isinstance(batch, SyncBatch):
            raise TypeError("batch must be SyncBatch")
        if batch.sender_device != self.local_device:
            raise ValueError("transport can only send local-origin batches")
        peer = self.registry.get_peer(peer_id)
        if peer is None:
            raise PermissionError("peer is not actively trusted")

        encoded = self.authenticator.sign(
            "sync.batch",
            peer_id,
            batch_to_payload(batch),
            message_id=batch.batch_id,
        )

        try:
            response = self._client().post(
                peer.endpoint + SYNC_PATH,
                data=encoded,
                headers={"Content-Type": MEDIA_TYPE, "Accept": MEDIA_TYPE},
                timeout=self._timeout,
                allow_redirects=False,
                stream=True,
            )
            if response.status_code != 200:
                raise SyncTransportError("Peer rejected the sync request.")
            content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if content_type != MEDIA_TYPE:
                raise SyncTransportError("Peer returned an unexpected content type.")
            body = self._read_bounded(response)
        except SyncTransportError:
            raise
        except Exception as exc:
            raise SyncTransportError(
                f"Peer transport failed ({type(exc).__name__})."
            ) from None
        finally:
            try:
                response.close()
            except (UnboundLocalError, Exception):
                pass

        sender, payload, message_id = self.authenticator.verify(
            body,
            expected_kind="sync.ack",
            expected_sender=peer_id,
        )
        if sender.peer_id != peer_id:
            raise PermissionError("ACK signer does not match peer")
        if message_id != "ack:" + batch.batch_id:
            raise PermissionError("ACK message id does not match batch")
        ack = ack_from_payload(payload)
        if ack.receiver_device != peer_id or ack.batch_id != batch.batch_id:
            raise PermissionError("ACK payload does not match request")
        return ack
