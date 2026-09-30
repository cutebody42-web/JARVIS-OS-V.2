"""Owner-controlled NEXUS device pairing and revocation registry.

The registry stores device metadata, pairing-ticket hashes and key fingerprints
only. Pairwise secrets live in an injected credential store (OS keyring in
production). Pairing tickets are one-time, short-lived and never persisted in
plaintext.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
import hashlib
import hmac
from pathlib import Path
import secrets
import sqlite3
from typing import Protocol, runtime_checkable
from uuid import uuid4

from core.nexus.sync_daemon import _validate_peer_id
from core.nexus.tailscale_transport import PeerEndpoint, PeerRegistry


class DeviceKind(str, Enum):
    DESKTOP = "desktop"
    PHONE = "phone"
    TABLET = "tablet"


class DeviceState(str, Enum):
    ACTIVE = "active"
    REVOKED = "revoked"


@dataclass(frozen=True)
class PairingTicket:
    ticket_id: str
    token: str
    expires_at: datetime


@dataclass(frozen=True)
class PairedDevice:
    device_id: str
    kind: DeviceKind
    host: str
    state: DeviceState
    key_fingerprint: str
    paired_at: datetime
    revoked_at: datetime | None = None


@dataclass(frozen=True)
class PairingResult:
    device: PairedDevice
    pairwise_secret: bytes


@runtime_checkable
class PairingSecretStore(Protocol):
    def set_secret(self, peer_id: str, secret: bytes) -> None:
        ...

    def delete_secret(self, peer_id: str) -> None:
        ...


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("stored timestamp is not timezone-aware")
    return parsed.astimezone(timezone.utc)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _fingerprint(secret: bytes) -> str:
    return hashlib.sha256(secret).hexdigest()[:24]


class DeviceRegistry:
    def __init__(
        self,
        directory: str | Path,
        *,
        clock=None,
        default_port: int = 8443,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.db_path = self.directory / "devices.sqlite"
        self._clock = clock or _utcnow
        self._default_port = default_port
        if not 1 <= default_port <= 65535:
            raise ValueError("default_port must be between 1 and 65535")
        self._init_db()

    def _connect(self):
        db = sqlite3.connect(self.db_path, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        return db

    def _init_db(self):
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS pairing_tickets (
                    ticket_id TEXT PRIMARY KEY,
                    token_hash TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    consumed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS trusted_devices (
                    device_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    host TEXT NOT NULL,
                    port INTEGER NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('active','revoked')),
                    key_fingerprint TEXT NOT NULL,
                    paired_at TEXT NOT NULL,
                    revoked_at TEXT
                );
                """
            )

    def begin_pairing(
        self,
        kind: DeviceKind,
        *,
        ttl_seconds: int = 300,
    ) -> PairingTicket:
        if not isinstance(kind, DeviceKind):
            raise TypeError("kind must be DeviceKind")
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int) or not 30 <= ttl_seconds <= 900:
            raise ValueError("pairing TTL must be between 30 and 900 seconds")

        now = self._clock().astimezone(timezone.utc)
        expires = now + timedelta(seconds=ttl_seconds)
        ticket_id = uuid4().hex
        token = secrets.token_urlsafe(32)
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO pairing_tickets(ticket_id, token_hash, kind, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (ticket_id, _token_hash(token), kind.value, _iso(expires)),
            )
        return PairingTicket(ticket_id, token, expires)

    def complete_pairing(
        self,
        ticket_id: str,
        token: str,
        *,
        device_id: str,
        host: str,
        secret_store: PairingSecretStore,
        port: int | None = None,
    ) -> PairingResult:
        _validate_peer_id(device_id)
        if not isinstance(host, str) or not host.strip() or any(c in host for c in "/?#@"):
            raise ValueError("host must be a hostname or IP")
        if not isinstance(secret_store, PairingSecretStore):
            raise TypeError("secret_store must support set_secret/delete_secret")
        use_port = self._default_port if port is None else port
        if isinstance(use_port, bool) or not isinstance(use_port, int) or not 1 <= use_port <= 65535:
            raise ValueError("port must be between 1 and 65535")

        now = self._clock().astimezone(timezone.utc)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM pairing_tickets WHERE ticket_id = ?",
                (ticket_id,),
            ).fetchone()
            if row is None:
                raise PermissionError("Unknown pairing ticket")
            if row["consumed_at"] is not None:
                raise PermissionError("Pairing ticket already consumed")
            if now > _parse(row["expires_at"]):
                raise PermissionError("Pairing ticket expired")
            if not isinstance(token, str) or not hmac.compare_digest(
                row["token_hash"], _token_hash(token)
            ):
                raise PermissionError("Pairing token is invalid")

            existing = db.execute(
                "SELECT state FROM trusted_devices WHERE device_id = ?",
                (device_id,),
            ).fetchone()
            if existing is not None and existing["state"] == DeviceState.ACTIVE.value:
                raise PermissionError("Device is already paired")

            secret = secrets.token_bytes(32)
            fingerprint = _fingerprint(secret)
            # Credential-store write happens before DB commit. On DB failure we
            # remove the just-created secret, preserving fail-closed semantics.
            secret_store.set_secret(device_id, secret)
            try:
                db.execute(
                    """
                    INSERT INTO trusted_devices(
                        device_id, kind, host, port, state, key_fingerprint,
                        paired_at, revoked_at
                    ) VALUES (?, ?, ?, ?, 'active', ?, ?, NULL)
                    ON CONFLICT(device_id) DO UPDATE SET
                        kind=excluded.kind,
                        host=excluded.host,
                        port=excluded.port,
                        state='active',
                        key_fingerprint=excluded.key_fingerprint,
                        paired_at=excluded.paired_at,
                        revoked_at=NULL
                    """,
                    (
                        device_id,
                        row["kind"],
                        host.strip(),
                        use_port,
                        fingerprint,
                        _iso(now),
                    ),
                )
                db.execute(
                    "UPDATE pairing_tickets SET consumed_at=? WHERE ticket_id=?",
                    (_iso(now), ticket_id),
                )
                db.commit()
            except Exception:
                db.rollback()
                secret_store.delete_secret(device_id)
                raise

        device = self.get(device_id)
        return PairingResult(device, secret)

    def get(self, device_id: str) -> PairedDevice:
        _validate_peer_id(device_id)
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM trusted_devices WHERE device_id = ?",
                (device_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown device {device_id!r}")
        return PairedDevice(
            device_id=row["device_id"],
            kind=DeviceKind(row["kind"]),
            host=row["host"],
            state=DeviceState(row["state"]),
            key_fingerprint=row["key_fingerprint"],
            paired_at=_parse(row["paired_at"]),
            revoked_at=_parse(row["revoked_at"]),
        )

    def list_devices(self, *, include_revoked: bool = False) -> tuple[PairedDevice, ...]:
        query = "SELECT device_id FROM trusted_devices"
        if not include_revoked:
            query += " WHERE state='active'"
        query += " ORDER BY device_id"
        with self._connect() as db:
            ids = [row["device_id"] for row in db.execute(query)]
        return tuple(self.get(device_id) for device_id in ids)

    def revoke(self, device_id: str, secret_store: PairingSecretStore) -> PairedDevice:
        _validate_peer_id(device_id)
        if not isinstance(secret_store, PairingSecretStore):
            raise TypeError("secret_store must support set_secret/delete_secret")
        now = self._clock().astimezone(timezone.utc)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT state FROM trusted_devices WHERE device_id=?",
                (device_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown device {device_id!r}")
            if row["state"] != DeviceState.REVOKED.value:
                db.execute(
                    """
                    UPDATE trusted_devices
                    SET state='revoked', revoked_at=?
                    WHERE device_id=?
                    """,
                    (_iso(now), device_id),
                )
            db.commit()
        secret_store.delete_secret(device_id)
        return self.get(device_id)

    def peer_registry(self) -> PeerRegistry:
        peers = {}
        with self._connect() as db:
            for row in db.execute(
                "SELECT * FROM trusted_devices WHERE state='active' ORDER BY device_id"
            ):
                peers[row["device_id"]] = PeerEndpoint(
                    row["device_id"],
                    row["host"],
                    port=row["port"],
                )
        return PeerRegistry(peers)
