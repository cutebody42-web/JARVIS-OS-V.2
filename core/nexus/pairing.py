"""Explicit one-time pairing for NEXUS nodes and client-only companions.

Pairing is an invitation flow, not ambient trust:
1. desktop creates a short-lived offer with a one-time secret;
2. candidate proves knowledge of that secret and presents its Ed25519 public key;
3. owner explicitly approves the pending candidate;
4. only then is the device written to PeerRegistry.

NODE peers expose an inbound sync endpoint.
COMPANION peers (phones/tablets) are client-only and expose no inbound endpoint.

Only SHA-256(secret) is stored in local.db; the plaintext pairing secret is
present only in the short-lived QR/deep-link offer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
import secrets
from typing import Any, Mapping
from uuid import uuid4

from core.nexus.peer_auth import (
    DeviceSigner,
    PeerRegistry,
    PeerRole,
    TrustedPeer,
    normalize_peer_endpoint,
)


PAIRING_VERSION = 2
MAX_PAIRING_SECONDS = 10 * 60


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _ts(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("pairing timestamp must include timezone")
    return parsed.astimezone(timezone.utc)


def _secret_hash(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def _normalize_candidate_endpoint(
    role: PeerRole,
    endpoint: str | None,
) -> str | None:
    if role is PeerRole.NODE:
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("node pairing requires a sync endpoint")
        return normalize_peer_endpoint(endpoint)
    if role is PeerRole.COMPANION:
        if endpoint not in {None, ""}:
            raise ValueError("companion pairing does not accept an inbound endpoint")
        return None
    raise ValueError("unsupported pairing role")


@dataclass(frozen=True)
class PairingOffer:
    version: int
    pairing_id: str
    inviter_device: str
    inviter_public_key: str
    inviter_endpoint: str
    secret: str
    expires_at: str

    def public_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "pairing_id": self.pairing_id,
            "inviter_device": self.inviter_device,
            "inviter_public_key": self.inviter_public_key,
            "inviter_endpoint": self.inviter_endpoint,
            "secret": self.secret,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PairingOffer":
        if not isinstance(value, Mapping):
            raise TypeError("pairing offer must be an object")
        required = {
            "version",
            "pairing_id",
            "inviter_device",
            "inviter_public_key",
            "inviter_endpoint",
            "secret",
            "expires_at",
        }
        if set(value) != required:
            raise ValueError("pairing offer schema mismatch")
        if value["version"] != PAIRING_VERSION:
            raise ValueError("unsupported pairing offer version")
        return cls(
            version=value["version"],
            pairing_id=value["pairing_id"],
            inviter_device=value["inviter_device"],
            inviter_public_key=value["inviter_public_key"],
            inviter_endpoint=normalize_peer_endpoint(value["inviter_endpoint"]),
            secret=value["secret"],
            expires_at=value["expires_at"],
        )


@dataclass(frozen=True)
class PairingRequest:
    version: int
    pairing_id: str
    candidate_device: str
    candidate_public_key: str
    candidate_role: PeerRole
    candidate_endpoint: str | None
    proof: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "pairing_id": self.pairing_id,
            "candidate_device": self.candidate_device,
            "candidate_public_key": self.candidate_public_key,
            "candidate_role": self.candidate_role.value,
            "candidate_endpoint": self.candidate_endpoint,
            "proof": self.proof,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PairingRequest":
        if not isinstance(value, Mapping):
            raise TypeError("pairing request must be an object")
        required = {
            "version",
            "pairing_id",
            "candidate_device",
            "candidate_public_key",
            "candidate_role",
            "candidate_endpoint",
            "proof",
        }
        if set(value) != required:
            raise ValueError("pairing request schema mismatch")
        if value["version"] != PAIRING_VERSION:
            raise ValueError("unsupported pairing request version")
        try:
            role = PeerRole(value["candidate_role"])
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid pairing candidate role") from exc
        endpoint = _normalize_candidate_endpoint(role, value["candidate_endpoint"])
        return cls(
            version=value["version"],
            pairing_id=value["pairing_id"],
            candidate_device=value["candidate_device"],
            candidate_public_key=value["candidate_public_key"],
            candidate_role=role,
            candidate_endpoint=endpoint,
            proof=value["proof"],
        )


@dataclass(frozen=True)
class PendingPairing:
    pairing_id: str
    candidate_device: str
    candidate_public_key: str
    candidate_role: PeerRole
    candidate_endpoint: str | None
    created_at: str


class PairingManager:
    def __init__(
        self,
        registry: PeerRegistry,
        local_device: str,
        signer: DeviceSigner,
        *,
        clock=_utc_now,
    ):
        if not isinstance(registry, PeerRegistry):
            raise TypeError("registry must be PeerRegistry")
        if not isinstance(signer, DeviceSigner):
            raise TypeError("signer must be DeviceSigner")
        self.registry = registry
        self.local_device = local_device
        self.signer = signer
        self._clock = clock

        with self.registry._store._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS nexus_pairing_sessions (
                    pairing_id TEXT PRIMARY KEY,
                    secret_hash TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    state TEXT NOT NULL
                        CHECK(state IN ('offered','pending','approved','cancelled','expired')),
                    candidate_device TEXT,
                    candidate_public_key TEXT,
                    candidate_endpoint TEXT,
                    candidate_role TEXT NOT NULL DEFAULT 'node',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            columns = {
                row["name"]
                for row in db.execute("PRAGMA table_info(nexus_pairing_sessions)")
            }
            if "candidate_role" not in columns:
                db.execute(
                    "ALTER TABLE nexus_pairing_sessions "
                    "ADD COLUMN candidate_role TEXT NOT NULL DEFAULT 'node'"
                )

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("pairing clock must be timezone-aware")
        return now.astimezone(timezone.utc)

    def create_offer(self, endpoint: str, *, ttl_seconds: int = 300) -> PairingOffer:
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int):
            raise TypeError("ttl_seconds must be an integer")
        if not 0 < ttl_seconds <= MAX_PAIRING_SECONDS:
            raise ValueError("pairing TTL exceeds the allowed bound")

        normalized_endpoint = normalize_peer_endpoint(endpoint)
        now = self._now()
        expires = datetime.fromtimestamp(now.timestamp() + ttl_seconds, timezone.utc)
        pairing_id = uuid4().hex
        secret = secrets.token_urlsafe(32)
        created = now.isoformat().replace("+00:00", "Z")
        expires_at = expires.isoformat().replace("+00:00", "Z")

        with self.registry._store._connect() as db:
            db.execute(
                """
                INSERT INTO nexus_pairing_sessions(
                    pairing_id, secret_hash, expires_at, state, created_at, updated_at
                ) VALUES (?, ?, ?, 'offered', ?, ?)
                """,
                (pairing_id, _secret_hash(secret), expires_at, created, created),
            )

        return PairingOffer(
            PAIRING_VERSION,
            pairing_id,
            self.local_device,
            self.signer.public_b64,
            normalized_endpoint,
            secret,
            expires_at,
        )

    @staticmethod
    def build_request(
        offer: PairingOffer,
        candidate_device: str,
        candidate_signer: DeviceSigner,
        candidate_endpoint: str | None = None,
        *,
        candidate_role: PeerRole = PeerRole.NODE,
    ) -> PairingRequest:
        if not isinstance(offer, PairingOffer):
            raise TypeError("offer must be PairingOffer")
        if offer.version != PAIRING_VERSION:
            raise ValueError("unsupported pairing offer version")
        if not isinstance(candidate_signer, DeviceSigner):
            raise TypeError("candidate_signer must be DeviceSigner")
        if not isinstance(candidate_role, PeerRole):
            raise TypeError("candidate_role must be PeerRole")

        endpoint = _normalize_candidate_endpoint(candidate_role, candidate_endpoint)
        unsigned = {
            "version": PAIRING_VERSION,
            "pairing_id": offer.pairing_id,
            "candidate_device": candidate_device,
            "candidate_public_key": candidate_signer.public_b64,
            "candidate_role": candidate_role.value,
            "candidate_endpoint": endpoint,
        }
        proof = hmac.new(
            _secret_hash(offer.secret).encode("ascii"),
            _canonical(unsigned),
            hashlib.sha256,
        ).hexdigest()

        return PairingRequest(
            version=PAIRING_VERSION,
            pairing_id=offer.pairing_id,
            candidate_device=candidate_device,
            candidate_public_key=candidate_signer.public_b64,
            candidate_role=candidate_role,
            candidate_endpoint=endpoint,
            proof=proof,
        )

    def receive_request(self, request: PairingRequest) -> PendingPairing:
        if not isinstance(request, PairingRequest):
            raise TypeError("request must be PairingRequest")
        if request.version != PAIRING_VERSION:
            raise ValueError("unsupported pairing request version")

        endpoint = _normalize_candidate_endpoint(
            request.candidate_role,
            request.candidate_endpoint,
        )
        now = self._now()

        with self.registry._store._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM nexus_pairing_sessions WHERE pairing_id=?",
                (request.pairing_id,),
            ).fetchone()
            if row is None or row["state"] != "offered":
                db.rollback()
                raise PermissionError("pairing offer is absent or already used")

            if now > _ts(row["expires_at"]):
                db.execute(
                    """
                    UPDATE nexus_pairing_sessions
                    SET state='expired', updated_at=?
                    WHERE pairing_id=?
                    """,
                    (
                        now.isoformat().replace("+00:00", "Z"),
                        request.pairing_id,
                    ),
                )
                db.commit()
                raise PermissionError("pairing offer expired")

            unsigned = {
                "version": request.version,
                "pairing_id": request.pairing_id,
                "candidate_device": request.candidate_device,
                "candidate_public_key": request.candidate_public_key,
                "candidate_role": request.candidate_role.value,
                "candidate_endpoint": endpoint,
            }
            expected_proof = hmac.new(
                row["secret_hash"].encode("ascii"),
                _canonical(unsigned),
                hashlib.sha256,
            ).hexdigest()
            if not hmac.compare_digest(expected_proof, request.proof):
                db.rollback()
                raise PermissionError("pairing proof is invalid")

            updated = now.isoformat().replace("+00:00", "Z")
            db.execute(
                """
                UPDATE nexus_pairing_sessions
                SET state='pending',
                    candidate_device=?,
                    candidate_public_key=?,
                    candidate_endpoint=?,
                    candidate_role=?,
                    updated_at=?
                WHERE pairing_id=?
                """,
                (
                    request.candidate_device,
                    request.candidate_public_key,
                    endpoint,
                    request.candidate_role.value,
                    updated,
                    request.pairing_id,
                ),
            )
            db.commit()

        return PendingPairing(
            pairing_id=request.pairing_id,
            candidate_device=request.candidate_device,
            candidate_public_key=request.candidate_public_key,
            candidate_role=request.candidate_role,
            candidate_endpoint=endpoint,
            created_at=updated,
        )

    def pending(self) -> tuple[PendingPairing, ...]:
        with self.registry._store._connect() as db:
            rows = db.execute(
                """
                SELECT pairing_id, candidate_device, candidate_public_key,
                       candidate_role, candidate_endpoint, updated_at
                FROM nexus_pairing_sessions
                WHERE state='pending'
                ORDER BY updated_at
                """
            ).fetchall()

        pending: list[PendingPairing] = []
        for row in rows:
            role = PeerRole(row["candidate_role"])
            pending.append(
                PendingPairing(
                    pairing_id=row["pairing_id"],
                    candidate_device=row["candidate_device"],
                    candidate_public_key=row["candidate_public_key"],
                    candidate_role=role,
                    candidate_endpoint=(
                        row["candidate_endpoint"] if role is PeerRole.NODE else None
                    ),
                    created_at=row["updated_at"],
                )
            )
        return tuple(pending)

    def approve(self, pairing_id: str) -> TrustedPeer:
        now = self._now()
        with self.registry._store._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM nexus_pairing_sessions WHERE pairing_id=?",
                (pairing_id,),
            ).fetchone()
            if row is None or row["state"] != "pending":
                db.rollback()
                raise PermissionError("pairing request is not awaiting owner approval")

            role = PeerRole(row["candidate_role"])
            if role is PeerRole.COMPANION:
                peer = self.registry._trust_companion_in_db(
                    db,
                    row["candidate_device"],
                    row["candidate_public_key"],
                )
            else:
                peer = self.registry._trust_peer_in_db(
                    db,
                    row["candidate_device"],
                    row["candidate_public_key"],
                    row["candidate_endpoint"],
                )

            db.execute(
                """
                UPDATE nexus_pairing_sessions
                SET state='approved', updated_at=?
                WHERE pairing_id=?
                """,
                (
                    now.isoformat().replace("+00:00", "Z"),
                    pairing_id,
                ),
            )
            db.commit()
        return peer

    def cancel(self, pairing_id: str) -> bool:
        now = self._now().isoformat().replace("+00:00", "Z")
        with self.registry._store._connect() as db:
            cursor = db.execute(
                """
                UPDATE nexus_pairing_sessions
                SET state='cancelled', updated_at=?
                WHERE pairing_id=? AND state IN ('offered','pending')
                """,
                (now, pairing_id),
            )
        return cursor.rowcount == 1
