"""Signed HTTP transport intended for Tailscale-connected NEXUS peers.

Tailscale provides the encrypted network path and automatically chooses direct,
peer-relay, or DERP connectivity. This module adds application-level peer
identity, message integrity, replay protection, and deterministic batch/ACK
serialization on top of that path.

No raw secrets are logged or persisted here. Production keys are resolved
through PeerKeyStore (KeyringPeerKeyStore by default); tests can inject a
memory-only store.
"""

from __future__ import annotations

import base64
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import ipaddress
import json
import secrets
import subprocess
import threading
from typing import Mapping, Protocol, runtime_checkable

from core.nexus.event_store import SyncEvent
from core.nexus.sync_daemon import (
    PROTOCOL_VERSION,
    SyncAck,
    SyncBatch,
    SyncDaemon,
    SyncTransport,
    _validate_peer_id,
)


SYNC_PATH = "/nexus/sync/v1/batch"
SIGNATURE_HEADER = "X-Nexus-Signature"
DEVICE_HEADER = "X-Nexus-Device"
TIMESTAMP_HEADER = "X-Nexus-Timestamp"
NONCE_HEADER = "X-Nexus-Nonce"
PROTOCOL_HEADER = "X-Nexus-Protocol"


class SyncAuthenticationError(RuntimeError):
    pass


class TailscaleDiscoveryError(RuntimeError):
    pass


@dataclass(frozen=True)
class PeerEndpoint:
    device_id: str
    host: str
    port: int = 8443
    scheme: str = "http"
    enabled: bool = True

    def __post_init__(self) -> None:
        _validate_peer_id(self.device_id)
        if not isinstance(self.host, str) or not self.host.strip():
            raise ValueError("peer host must be non-empty")
        if any(ch in self.host for ch in "/?#@"):
            raise ValueError("peer host must be a hostname or IP, not a URL")
        if isinstance(self.port, bool) or not isinstance(self.port, int) or not 1 <= self.port <= 65535:
            raise ValueError("peer port must be between 1 and 65535")
        if self.scheme not in {"http", "https"}:
            raise ValueError("peer scheme must be http or https")

    @property
    def base_url(self) -> str:
        host = self.host
        try:
            parsed = ipaddress.ip_address(host)
        except ValueError:
            parsed = None
        if isinstance(parsed, ipaddress.IPv6Address):
            host = f"[{host}]"
        return f"{self.scheme}://{host}:{self.port}"


class PeerRegistry:
    def __init__(self, peers: Mapping[str, PeerEndpoint]):
        normalized = {}
        for peer_id, endpoint in peers.items():
            _validate_peer_id(peer_id)
            if not isinstance(endpoint, PeerEndpoint):
                raise TypeError("peer registry values must be PeerEndpoint")
            if endpoint.device_id != peer_id:
                raise ValueError("peer registry key must match endpoint device_id")
            normalized[peer_id] = endpoint
        self._peers = dict(normalized)

    def get(self, peer_id: str) -> PeerEndpoint:
        _validate_peer_id(peer_id)
        try:
            endpoint = self._peers[peer_id]
        except KeyError:
            raise KeyError(f"Unknown sync peer: {peer_id}") from None
        if not endpoint.enabled:
            raise PermissionError(f"Sync peer is disabled: {peer_id}")
        return endpoint

    def known(self, peer_id: str) -> bool:
        endpoint = self._peers.get(peer_id)
        return bool(endpoint and endpoint.enabled)


@runtime_checkable
class PeerKeyStore(Protocol):
    def get_secret(self, peer_id: str) -> bytes:
        ...


class KeyringPeerKeyStore:
    """Store pairwise sync keys in the OS credential store."""

    def __init__(self, service_name: str = "nexus-sync"):
        if not service_name.strip():
            raise ValueError("service_name must be non-empty")
        self.service_name = service_name

    def get_secret(self, peer_id: str) -> bytes:
        _validate_peer_id(peer_id)
        import keyring

        encoded = keyring.get_password(self.service_name, peer_id)
        if not encoded:
            raise KeyError(f"No sync key provisioned for peer {peer_id}")
        try:
            secret = base64.urlsafe_b64decode(encoded.encode("ascii"))
        except Exception:
            raise ValueError("Stored sync key is invalid") from None
        if len(secret) < 32:
            raise ValueError("Stored sync key is too short")
        return secret

    def set_secret(self, peer_id: str, secret: bytes) -> None:
        _validate_peer_id(peer_id)
        if not isinstance(secret, bytes) or len(secret) < 32:
            raise ValueError("sync secret must be at least 32 bytes")
        import keyring

        keyring.set_password(
            self.service_name,
            peer_id,
            base64.urlsafe_b64encode(secret).decode("ascii"),
        )


def generate_pairwise_secret() -> bytes:
    return secrets.token_bytes(32)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def batch_to_dict(batch: SyncBatch) -> dict:
    return {
        "protocol_version": batch.protocol_version,
        "sender_device": batch.sender_device,
        "batch_id": batch.batch_id,
        "events": [event.to_dict() for event in batch.events],
    }


def batch_from_dict(value: object) -> SyncBatch:
    if not isinstance(value, dict):
        raise ValueError("sync batch body must be an object")
    allowed = {"protocol_version", "sender_device", "batch_id", "events"}
    if set(value) != allowed:
        raise ValueError("sync batch body has missing or unknown fields")
    events = value["events"]
    if not isinstance(events, list) or not events:
        raise ValueError("sync batch events must be a non-empty list")
    return SyncBatch(
        protocol_version=value["protocol_version"],
        sender_device=value["sender_device"],
        batch_id=value["batch_id"],
        events=tuple(SyncEvent.from_dict(item) for item in events),
    )


def ack_to_dict(ack: SyncAck) -> dict:
    return {
        "protocol_version": ack.protocol_version,
        "receiver_device": ack.receiver_device,
        "batch_id": ack.batch_id,
        "durable_event_ids": list(ack.durable_event_ids),
        "applied_event_ids": list(ack.applied_event_ids),
        "failed_event_ids": list(ack.failed_event_ids),
        "pending_event_ids": list(ack.pending_event_ids),
    }


def ack_from_dict(value: object) -> SyncAck:
    if not isinstance(value, dict):
        raise ValueError("sync ACK body must be an object")
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
        raise ValueError("sync ACK body has missing or unknown fields")
    for key in (
        "durable_event_ids",
        "applied_event_ids",
        "failed_event_ids",
        "pending_event_ids",
    ):
        if not isinstance(value[key], list):
            raise ValueError(f"{key} must be a list")
    return SyncAck(
        protocol_version=value["protocol_version"],
        receiver_device=value["receiver_device"],
        batch_id=value["batch_id"],
        durable_event_ids=tuple(value["durable_event_ids"]),
        applied_event_ids=tuple(value["applied_event_ids"]),
        failed_event_ids=tuple(value["failed_event_ids"]),
        pending_event_ids=tuple(value["pending_event_ids"]),
    )


def _signature_payload(
    *,
    role: str,
    device_id: str,
    timestamp: str,
    nonce: str,
    body: bytes,
) -> bytes:
    digest = hashlib.sha256(body).hexdigest()
    return "\n".join(
        [role, device_id, timestamp, nonce, digest]
    ).encode("utf-8")


def sign_payload(
    secret: bytes,
    *,
    role: str,
    device_id: str,
    timestamp: str,
    nonce: str,
    body: bytes,
) -> str:
    if not isinstance(secret, bytes) or len(secret) < 32:
        raise ValueError("sync secret must be at least 32 bytes")
    message = _signature_payload(
        role=role,
        device_id=device_id,
        timestamp=timestamp,
        nonce=nonce,
        body=body,
    )
    return hmac.new(secret, message, hashlib.sha256).hexdigest()


def verify_signature(
    secret: bytes,
    signature: str,
    *,
    role: str,
    device_id: str,
    timestamp: str,
    nonce: str,
    body: bytes,
) -> bool:
    expected = sign_payload(
        secret,
        role=role,
        device_id=device_id,
        timestamp=timestamp,
        nonce=nonce,
        body=body,
    )
    return hmac.compare_digest(expected, str(signature or ""))


def _header(headers: Mapping[str, str], name: str) -> str:
    wanted = name.casefold()
    for key, value in headers.items():
        if str(key).casefold() == wanted:
            return str(value)
    raise SyncAuthenticationError(f"Missing required sync header: {name}")


def _parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise SyncAuthenticationError("Invalid sync timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SyncAuthenticationError("Sync timestamp must include timezone")
    return parsed.astimezone(timezone.utc)


class NonceReplayCache:
    def __init__(self, max_entries: int = 4096):
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        self._max_entries = max_entries
        self._queue = deque()
        self._seen = set()
        self._lock = threading.Lock()

    def check_and_mark(self, peer_id: str, nonce: str) -> None:
        key = (peer_id, nonce)
        with self._lock:
            if key in self._seen:
                raise SyncAuthenticationError("Replay nonce already used")
            self._seen.add(key)
            self._queue.append(key)
            while len(self._queue) > self._max_entries:
                expired = self._queue.popleft()
                self._seen.discard(expired)


class TailscaleCLI:
    """Minimal safe discovery wrapper around the Tailscale CLI."""

    def __init__(self, executable: str = "tailscale", timeout_seconds: float = 3.0):
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.executable = executable
        self.timeout_seconds = timeout_seconds

    def _run(self, args: list[str]) -> str:
        try:
            result = subprocess.run(
                [self.executable, *args],
                capture_output=True,
                text=True,
                check=True,
                timeout=self.timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError):
            raise TailscaleDiscoveryError("Tailscale CLI discovery failed") from None
        return result.stdout.strip()

    @staticmethod
    def _validate_tail_ipv4(value: str) -> str:
        try:
            address = ipaddress.ip_address(value.strip())
        except ValueError:
            raise TailscaleDiscoveryError("Tailscale returned an invalid IPv4 address") from None
        if not isinstance(address, ipaddress.IPv4Address):
            raise TailscaleDiscoveryError("Expected a Tailscale IPv4 address")
        if address not in ipaddress.ip_network("100.64.0.0/10"):
            raise TailscaleDiscoveryError("IPv4 address is outside Tailscale CGNAT range")
        return str(address)

    def local_ipv4(self) -> str:
        output = self._run(["ip", "-4"])
        first = output.splitlines()[0] if output else ""
        return self._validate_tail_ipv4(first)

    def peer_ipv4(self, hostname: str) -> str:
        if not isinstance(hostname, str) or not hostname.strip():
            raise ValueError("hostname must be non-empty")
        output = self._run(["ip", "-4", hostname])
        first = output.splitlines()[0] if output else ""
        return self._validate_tail_ipv4(first)


class TailscaleHttpTransport(SyncTransport):
    def __init__(
        self,
        *,
        local_device_id: str,
        registry: PeerRegistry,
        key_store: PeerKeyStore,
        session=None,
        timeout_seconds: float = 10.0,
        clock=None,
    ):
        _validate_peer_id(local_device_id)
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not isinstance(registry, PeerRegistry):
            raise TypeError("registry must be PeerRegistry")
        if not isinstance(key_store, PeerKeyStore):
            raise TypeError("key_store must implement PeerKeyStore")
        self.local_device_id = local_device_id
        self.registry = registry
        self.key_store = key_store
        self.session = session
        self.timeout_seconds = timeout_seconds
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def _http(self):
        if self.session is None:
            import requests
            self.session = requests.Session()
        return self.session

    def send_batch(self, peer_id: str, batch: SyncBatch) -> SyncAck:
        endpoint = self.registry.get(peer_id)
        if batch.sender_device != self.local_device_id:
            raise ValueError("batch sender_device must match local transport identity")

        body = _canonical_json(batch_to_dict(batch))
        secret = self.key_store.get_secret(peer_id)
        now = self.clock().astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        nonce = secrets.token_urlsafe(18)
        signature = sign_payload(
            secret,
            role="request",
            device_id=self.local_device_id,
            timestamp=now,
            nonce=nonce,
            body=body,
        )
        headers = {
            "Content-Type": "application/json",
            DEVICE_HEADER: self.local_device_id,
            TIMESTAMP_HEADER: now,
            NONCE_HEADER: nonce,
            SIGNATURE_HEADER: signature,
            PROTOCOL_HEADER: str(PROTOCOL_VERSION),
        }

        try:
            response = self._http().post(
                endpoint.base_url + SYNC_PATH,
                data=body,
                headers=headers,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            response_body = bytes(response.content)
            response_headers = response.headers
        except Exception as exc:
            raise ConnectionError(
                f"NEXUS sync transport failed ({type(exc).__name__})"
            ) from None

        remote_device = _header(response_headers, DEVICE_HEADER)
        if remote_device != peer_id:
            raise SyncAuthenticationError("Sync response identity does not match peer")
        response_timestamp = _header(response_headers, TIMESTAMP_HEADER)
        response_nonce = _header(response_headers, NONCE_HEADER)
        response_signature = _header(response_headers, SIGNATURE_HEADER)
        if not verify_signature(
            secret,
            response_signature,
            role="response",
            device_id=peer_id,
            timestamp=response_timestamp,
            nonce=response_nonce,
            body=response_body,
        ):
            raise SyncAuthenticationError("Invalid sync response signature")

        try:
            payload = json.loads(response_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("Sync response is not valid JSON") from None
        return ack_from_dict(payload)


class SignedSyncHttpEndpoint:
    """Framework-neutral server-side handler for one signed sync POST."""

    def __init__(
        self,
        *,
        daemon: SyncDaemon,
        registry: PeerRegistry,
        key_store: PeerKeyStore,
        replay_cache: NonceReplayCache | None = None,
        max_clock_skew_seconds: float = 90.0,
        clock=None,
    ):
        if not isinstance(daemon, SyncDaemon):
            raise TypeError("daemon must be SyncDaemon")
        if not isinstance(registry, PeerRegistry):
            raise TypeError("registry must be PeerRegistry")
        if not isinstance(key_store, PeerKeyStore):
            raise TypeError("key_store must implement PeerKeyStore")
        if max_clock_skew_seconds <= 0:
            raise ValueError("max_clock_skew_seconds must be positive")
        self.daemon = daemon
        self.registry = registry
        self.key_store = key_store
        self.replay_cache = replay_cache or NonceReplayCache()
        self.max_clock_skew_seconds = max_clock_skew_seconds
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def _authenticate(self, headers: Mapping[str, str], body: bytes) -> str:
        peer_id = _header(headers, DEVICE_HEADER)
        self.registry.get(peer_id)
        protocol = _header(headers, PROTOCOL_HEADER)
        if protocol != str(PROTOCOL_VERSION):
            raise SyncAuthenticationError("Unsupported sync protocol header")
        timestamp = _header(headers, TIMESTAMP_HEADER)
        nonce = _header(headers, NONCE_HEADER)
        signature = _header(headers, SIGNATURE_HEADER)

        observed = _parse_timestamp(timestamp)
        now = self.clock().astimezone(timezone.utc)
        if abs((now - observed).total_seconds()) > self.max_clock_skew_seconds:
            raise SyncAuthenticationError("Sync request timestamp is outside allowed skew")

        secret = self.key_store.get_secret(peer_id)
        if not verify_signature(
            secret,
            signature,
            role="request",
            device_id=peer_id,
            timestamp=timestamp,
            nonce=nonce,
            body=body,
        ):
            raise SyncAuthenticationError("Invalid sync request signature")

        self.replay_cache.check_and_mark(peer_id, nonce)
        return peer_id

    def handle(
        self,
        *,
        headers: Mapping[str, str],
        body: bytes,
    ) -> tuple[int, dict[str, str], bytes]:
        try:
            peer_id = self._authenticate(headers, body)
        except (SyncAuthenticationError, KeyError, PermissionError):
            return 401, {"Content-Type": "application/json"}, b'{"error":"unauthorized"}'

        try:
            payload = json.loads(body.decode("utf-8"))
            batch = batch_from_dict(payload)
            if batch.sender_device != peer_id:
                raise SyncAuthenticationError("Batch sender does not match signed peer")
            ack = self.daemon.receive_batch(batch)
            response_body = _canonical_json(ack_to_dict(ack))
        except Exception:
            response_body = b'{"error":"invalid_sync_batch"}'
            status = 400
        else:
            status = 200

        secret = self.key_store.get_secret(peer_id)
        now = self.clock().astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        nonce = secrets.token_urlsafe(18)
        response_headers = {
            "Content-Type": "application/json",
            DEVICE_HEADER: self.daemon.device_id,
            TIMESTAMP_HEADER: now,
            NONCE_HEADER: nonce,
            PROTOCOL_HEADER: str(PROTOCOL_VERSION),
            SIGNATURE_HEADER: sign_payload(
                secret,
                role="response",
                device_id=self.daemon.device_id,
                timestamp=now,
                nonce=nonce,
                body=response_body,
            ),
        }
        return status, response_headers, response_body
