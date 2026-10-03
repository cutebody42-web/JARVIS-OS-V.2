"""Application-level device identity and peer trust for NEXUS sync.

Tailscale protects the network path. This module independently authenticates
NEXUS devices so tailnet membership alone never grants semantic-sync authority.

Private Ed25519 keys live in SecretStore; only public keys are stored in the
sync database. Trust and revocation are explicit local-owner operations.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import hashlib
import re
from typing import Any, Mapping
from urllib.parse import urlparse, urlunparse
from uuid import uuid4

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

from core.nexus.event_store import EventStore
from core.secret_store import SecretStore, get_secret_store


AUTH_VERSION = 1
MAX_ENVELOPE_BYTES = 1024 * 1024
MAX_CLOCK_SKEW_SECONDS = 15 * 60
_DEVICE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(value: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError("encoded key/signature must be non-empty text")
    try:
        return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except Exception as exc:
        raise ValueError("invalid base64url value") from exc


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("issued_at must be non-empty")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("issued_at must include timezone")
    return parsed.astimezone(timezone.utc)


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _device(value: str) -> str:
    if not isinstance(value, str) or not _DEVICE_RE.fullmatch(value):
        raise ValueError("device id must be a safe 1-64 character identifier")
    return value


def normalize_peer_endpoint(value: str) -> str:
    """Validate an explicitly owner-trusted HTTP(S) peer origin.

    Network reachability/least privilege is delegated to the tailnet policy;
    this function prevents URL confusion, credentials, redirects-by-config,
    paths and fragments.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("peer endpoint must be non-empty")
    raw = value.strip()
    if "://" not in raw:
        raw = "http://" + raw
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("peer endpoint must be an HTTP(S) origin")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("peer endpoint cannot contain credentials")
    if parsed.path not in {"", "/"} or parsed.params or parsed.query or parsed.fragment:
        raise ValueError("peer endpoint must not include path/query/fragment")
    if parsed.port is not None and not 1 <= parsed.port <= 65535:
        raise ValueError("invalid peer endpoint port")
    return urlunparse(parsed).rstrip("/")


class PeerRole(str, Enum):
    NODE = "node"
    COMPANION = "companion"


@dataclass(frozen=True)
class TrustedPeer:
    peer_id: str
    public_key: str
    endpoint: str | None
    revoked: bool = False
    role: PeerRole = PeerRole.NODE


class DeviceSigner:
    """In-memory Ed25519 signer. No private key is exposed by envelopes."""

    def __init__(self, private_key: Ed25519PrivateKey):
        if not isinstance(private_key, Ed25519PrivateKey):
            raise TypeError("private_key must be Ed25519PrivateKey")
        self._private_key = private_key

    @classmethod
    def generate(cls) -> "DeviceSigner":
        return cls(Ed25519PrivateKey.generate())

    @classmethod
    def from_private_b64(cls, value: str) -> "DeviceSigner":
        raw = _unb64(value)
        if len(raw) != 32:
            raise ValueError("Ed25519 private key must be 32 bytes")
        return cls(Ed25519PrivateKey.from_private_bytes(raw))

    @property
    def private_b64(self) -> str:
        raw = self._private_key.private_bytes(
            Encoding.Raw,
            PrivateFormat.Raw,
            NoEncryption(),
        )
        return _b64(raw)

    @property
    def public_b64(self) -> str:
        raw = self._private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        return _b64(raw)

    @property
    def fingerprint(self) -> str:
        raw = self._private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        digest = hashlib.sha256(raw).hexdigest()[:24]
        return "-".join(digest[i:i + 4] for i in range(0, len(digest), 4))

    def sign(self, data: bytes) -> str:
        if not isinstance(data, bytes):
            raise TypeError("signed data must be bytes")
        return _b64(self._private_key.sign(data))

    @staticmethod
    def verify(public_key_b64: str, signature_b64: str, data: bytes) -> bool:
        try:
            key = Ed25519PublicKey.from_public_bytes(_unb64(public_key_b64))
            key.verify(_unb64(signature_b64), data)
            return True
        except (ValueError, InvalidSignature):
            return False


def load_or_create_device_signer(
    device_id: str,
    *,
    secret_store: SecretStore | None = None,
) -> DeviceSigner:
    """Load a device identity from OS-backed SecretStore or generate it once."""
    device_id = _device(device_id)
    store = secret_store or get_secret_store()
    key_name = f"nexus.sync.ed25519.{device_id}"
    existing = store.get_persistent(key_name)
    if existing:
        return DeviceSigner.from_private_b64(existing)
    signer = DeviceSigner.generate()
    store.set(key_name, signer.private_b64)
    return signer


class PeerRegistry:
    """Explicit trusted-peer public keys and configured tailnet origins."""

    def __init__(self, event_store: EventStore):
        if not isinstance(event_store, EventStore):
            raise TypeError("event_store must be EventStore")
        self._store = event_store
        with self._store._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS nexus_trusted_peers (
                    peer_id TEXT PRIMARY KEY,
                    public_key TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    revoked INTEGER NOT NULL DEFAULT 0 CHECK(revoked IN (0,1)),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            columns = {
                row["name"]
                for row in db.execute("PRAGMA table_info(nexus_trusted_peers)")
            }
            if "role" not in columns:
                db.execute(
                    "ALTER TABLE nexus_trusted_peers "
                    "ADD COLUMN role TEXT NOT NULL DEFAULT 'node'"
                )

    def _upsert_peer_in_db(
        self,
        db,
        peer_id: str,
        public_key: str,
        *,
        role: PeerRole,
        endpoint: str | None,
    ) -> TrustedPeer:
        peer_id = _device(peer_id)
        if peer_id == self._store.device_id:
            raise ValueError("cannot trust the local device as a remote peer")
        raw_key = _unb64(public_key)
        if len(raw_key) != 32:
            raise ValueError("peer Ed25519 public key must be 32 bytes")
        if not isinstance(role, PeerRole):
            raise TypeError("role must be PeerRole")

        if role is PeerRole.NODE:
            stored_endpoint = normalize_peer_endpoint(endpoint)
            exposed_endpoint: str | None = stored_endpoint
        else:
            if endpoint not in {None, ""}:
                raise ValueError("companion peers do not expose sync endpoints")
            stored_endpoint = ""
            exposed_endpoint = None

        now = _utc_now().isoformat().replace("+00:00", "Z")
        db.execute(
            """
            INSERT INTO nexus_trusted_peers(
                peer_id, public_key, endpoint, revoked, created_at, updated_at, role
            ) VALUES (?, ?, ?, 0, ?, ?, ?)
            ON CONFLICT(peer_id) DO UPDATE SET
                public_key=excluded.public_key,
                endpoint=excluded.endpoint,
                revoked=0,
                updated_at=excluded.updated_at,
                role=excluded.role
            """,
            (peer_id, public_key, stored_endpoint, now, now, role.value),
        )
        return TrustedPeer(
            peer_id,
            public_key,
            exposed_endpoint,
            False,
            role,
        )

    def _trust_peer_in_db(
        self,
        db,
        peer_id: str,
        public_key: str,
        endpoint: str,
    ) -> TrustedPeer:
        return self._upsert_peer_in_db(
            db,
            peer_id,
            public_key,
            role=PeerRole.NODE,
            endpoint=endpoint,
        )

    def _trust_companion_in_db(
        self,
        db,
        peer_id: str,
        public_key: str,
    ) -> TrustedPeer:
        return self._upsert_peer_in_db(
            db,
            peer_id,
            public_key,
            role=PeerRole.COMPANION,
            endpoint=None,
        )

    def trust_peer(self, peer_id: str, public_key: str, endpoint: str) -> TrustedPeer:
        with self._store._connect() as db:
            return self._trust_peer_in_db(db, peer_id, public_key, endpoint)

    def trust_companion(self, peer_id: str, public_key: str) -> TrustedPeer:
        with self._store._connect() as db:
            return self._trust_companion_in_db(db, peer_id, public_key)

    def revoke_peer(self, peer_id: str) -> bool:
        peer_id = _device(peer_id)
        now = _utc_now().isoformat().replace("+00:00", "Z")
        with self._store._connect() as db:
            cursor = db.execute(
                """
                UPDATE nexus_trusted_peers
                SET revoked=1, updated_at=?
                WHERE peer_id=? AND revoked=0
                """,
                (now, peer_id),
            )
        return cursor.rowcount == 1

    def get_peer(self, peer_id: str, *, include_revoked: bool = False) -> TrustedPeer | None:
        peer_id = _device(peer_id)
        with self._store._connect() as db:
            row = db.execute(
                """
                SELECT peer_id, public_key, endpoint, revoked, role
                FROM nexus_trusted_peers WHERE peer_id=?
                """,
                (peer_id,),
            ).fetchone()
        if row is None or (bool(row["revoked"]) and not include_revoked):
            return None
        role = PeerRole(row["role"])
        endpoint = row["endpoint"] if role is PeerRole.NODE else None
        return TrustedPeer(
            row["peer_id"],
            row["public_key"],
            endpoint,
            bool(row["revoked"]),
            role,
        )

    def active_peers(self) -> tuple[TrustedPeer, ...]:
        with self._store._connect() as db:
            rows = db.execute(
                """
                SELECT peer_id, public_key, endpoint, revoked, role
                FROM nexus_trusted_peers
                WHERE revoked=0
                ORDER BY peer_id
                """
            ).fetchall()
        return tuple(
            TrustedPeer(
                row["peer_id"],
                row["public_key"],
                row["endpoint"] if PeerRole(row["role"]) is PeerRole.NODE else None,
                False,
                PeerRole(row["role"]),
            )
            for row in rows
        )

    def active_sync_peers(self) -> tuple[TrustedPeer, ...]:
        return tuple(
            peer
            for peer in self.active_peers()
            if peer.role is PeerRole.NODE and peer.endpoint is not None
        )


class PeerAuthenticator:
    """Sign and verify canonical NEXUS envelopes against explicit peer trust."""

    def __init__(
        self,
        local_device: str,
        signer: DeviceSigner,
        registry: PeerRegistry,
        *,
        clock=_utc_now,
        max_clock_skew_seconds: int = MAX_CLOCK_SKEW_SECONDS,
    ):
        self.local_device = _device(local_device)
        if not isinstance(signer, DeviceSigner):
            raise TypeError("signer must be DeviceSigner")
        if not isinstance(registry, PeerRegistry):
            raise TypeError("registry must be PeerRegistry")
        if isinstance(max_clock_skew_seconds, bool) or max_clock_skew_seconds <= 0:
            raise ValueError("max_clock_skew_seconds must be positive")
        self.signer = signer
        self.registry = registry
        self._clock = clock
        self._max_skew = max_clock_skew_seconds

    def sign(
        self,
        kind: str,
        receiver_device: str,
        payload: Mapping[str, Any],
        *,
        message_id: str | None = None,
    ) -> bytes:
        receiver_device = _device(receiver_device)
        if receiver_device == self.local_device:
            raise ValueError("cannot sign peer envelope to self")
        if self.registry.get_peer(receiver_device) is None:
            raise PermissionError("receiver is not an active trusted peer")
        if not isinstance(kind, str) or not re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}", kind):
            raise ValueError("invalid envelope kind")
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("auth clock must be timezone-aware")
        unsigned = {
            "version": AUTH_VERSION,
            "kind": kind,
            "sender_device": self.local_device,
            "receiver_device": receiver_device,
            "message_id": message_id or uuid4().hex,
            "issued_at": now.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "payload": dict(payload),
        }
        signature = self.signer.sign(_canonical(unsigned))
        encoded = _canonical({**unsigned, "signature": signature})
        if len(encoded) > MAX_ENVELOPE_BYTES:
            raise ValueError("signed envelope exceeds byte budget")
        return encoded

    def verify(
        self,
        encoded: bytes,
        *,
        expected_kind: str,
        expected_sender: str | None = None,
    ) -> tuple[TrustedPeer, dict[str, Any], str]:
        if not isinstance(encoded, bytes) or not encoded or len(encoded) > MAX_ENVELOPE_BYTES:
            raise ValueError("invalid signed envelope size")
        try:
            value = json.loads(encoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid signed envelope JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("signed envelope must be an object")
        allowed = {
            "version", "kind", "sender_device", "receiver_device",
            "message_id", "issued_at", "payload", "signature",
        }
        if set(value) != allowed:
            raise ValueError("signed envelope schema mismatch")
        if value["version"] != AUTH_VERSION or value["kind"] != expected_kind:
            raise ValueError("signed envelope protocol mismatch")
        sender = _device(value["sender_device"])
        receiver = _device(value["receiver_device"])
        if receiver != self.local_device:
            raise PermissionError("signed envelope targets another device")
        if expected_sender is not None and sender != _device(expected_sender):
            raise PermissionError("signed envelope sender mismatch")
        peer = self.registry.get_peer(sender)
        if peer is None:
            raise PermissionError("sender is not an active trusted peer")
        if not isinstance(value["message_id"], str) or not value["message_id"]:
            raise ValueError("invalid signed envelope message id")
        if not isinstance(value["payload"], dict):
            raise ValueError("signed envelope payload must be an object")

        issued = _timestamp(value["issued_at"])
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("auth clock must be timezone-aware")
        skew = abs((now.astimezone(timezone.utc) - issued).total_seconds())
        if skew > self._max_skew:
            raise PermissionError("signed envelope is outside the accepted clock window")

        signature = value["signature"]
        unsigned = {key: value[key] for key in allowed if key != "signature"}
        if not DeviceSigner.verify(peer.public_key, signature, _canonical(unsigned)):
            raise PermissionError("signed envelope signature is invalid")

        return peer, dict(value["payload"]), value["message_id"]
