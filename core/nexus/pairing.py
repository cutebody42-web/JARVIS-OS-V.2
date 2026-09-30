"""Explicit one-time pairing for adding a phone or another NEXUS device.

Pairing is an invitation flow, not ambient trust:
1. desktop creates a short-lived offer with a one-time secret;
2. candidate device proves knowledge of that secret and presents its public key;
3. owner explicitly approves the pending candidate;
4. only then is the peer written to PeerRegistry.

The one-time secret is stored only as a SHA-256 hash in local.db.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
import secrets
import sqlite3
from typing import Any, Mapping
from uuid import uuid4

from core.nexus.peer_auth import (
    DeviceSigner,
    PeerRegistry,
    TrustedPeer,
    normalize_peer_endpoint,
)


PAIRING_VERSION = 1
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


@dataclass(frozen=True)
class PairingRequest:
    version: int
    pairing_id: str
    candidate_device: str
    candidate_public_key: str
    candidate_endpoint: str
    proof: str


@dataclass(frozen=True)
class PendingPairing:
    pairing_id: str
    candidate_device: str
    candidate_public_key: str
    candidate_endpoint: str
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
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
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

        endpoint = normalize_peer_endpoint(endpoint)
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
            endpoint,
            secret,
            expires_at,
        )

    @staticmethod
    def build_request(
        offer: PairingOffer,
        candidate_device: str,
        candidate_signer: DeviceSigner,
        candidate_endpoint: str,
    ) -> PairingRequest:
        if not isinstance(offer, PairingOffer):
            raise TypeError("offer must be PairingOffer")
        if offer.version != PAIRING_VERSION:
            raise ValueError("unsupported pairing offer version")
        if not isinstance(candidate_signer, DeviceSigner):
            raise TypeError("candidate_signer must be DeviceSigner")

        candidate_endpoint = normalize_peer_endpoint(candidate_endpoint)
        unsigned = {
            "version": PAIRING_VERSION,
            "pairing_id": offer.pairing_id,
            "candidate_device": candidate_device,
            "candidate_public_key": candidate_signer.public_b64,
            "candidate_endpoint": candidate_endpoint,
        }
        proof = hmac.new(
            offer.secret.encode("utf-8"),
            _canonical(unsigned),
            hashlib.sha256,
        ).hexdigest()
        return PairingRequest(proof=proof, **unsigned)

    def receive_request(self, request: PairingRequest) -> PendingPairing:
        if not isinstance(request, PairingRequest):
            raise TypeError("request must be PairingRequest")
        if request.version != PAIRING_VERSION:
            raise ValueError("unsupported pairing request version")
        candidate_endpoint = normalize_peer_endpoint(request.candidate_endpoint)

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
                    (now.isoformat().replace("+00:00", "Z"), request.pairing_id),
                )
                db.commit()
                raise PermissionError("pairing offer expired")

            # We cannot reverse a stored hash to recover the secret, so the
            # proof is validated against a derived server-side verifier token.
            # Store HMAC(secret_hash, unsigned) semantics instead of plaintext.
            unsigned = {
                "version": request.version,
                "pairing_id": request.pairing_id,
                "candidate_device": request.candidate_device,
                "candidate_public_key": request.candidate_public_key,
                "candidate_endpoint": candidate_endpoint,
            }
            # Candidate proof is transformed once more using the stored secret
            # hash. This prevents accepting arbitrary proof strings from DB data.
            expected_binding = hmac.new(
                row["secret_hash"].encode("ascii"),
                request.proof.encode("ascii") + _canonical(unsigned),
                hashlib.sha256,
            ).hexdigest()

            # The corresponding binding is saved transiently by the offer
            # issuer through request proof verification below.
            # Because only the secret hash is durable, validate request proof by
            # comparing a server-generated challenge digest stored at offer time.
            # Legacy rows without that column are not admitted.
            columns = {
                item["name"]
                for item in db.execute("PRAGMA table_info(nexus_pairing_sessions)")
            }
            if "proof_binding" not in columns:
                db.execute(
                    "ALTER TABLE nexus_pairing_sessions ADD COLUMN proof_binding TEXT"
                )
                columns.add("proof_binding")
            binding = row["proof_binding"] if "proof_binding" in row.keys() else None
            if binding is None or not hmac.compare_digest(binding, expected_binding):
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
                    updated_at=?
                WHERE pairing_id=?
                """,
                (
                    request.candidate_device,
                    request.candidate_public_key,
                    candidate_endpoint,
                    updated,
                    request.pairing_id,
                ),
            )
            db.commit()

        return PendingPairing(
            request.pairing_id,
            request.candidate_device,
            request.candidate_public_key,
            candidate_endpoint,
            updated,
        )

    def register_request_proof(self, offer: PairingOffer, request: PairingRequest) -> None:
        """Bind a candidate proof to a locally issued offer before network intake.

        This helper is primarily for transports that receive the candidate
        request as a complete payload. The binding contains no plaintext secret.
        """
        unsigned = {
            "version": request.version,
            "pairing_id": request.pairing_id,
            "candidate_device": request.candidate_device,
            "candidate_public_key": request.candidate_public_key,
            "candidate_endpoint": normalize_peer_endpoint(request.candidate_endpoint),
        }
        if request.pairing_id != offer.pairing_id:
            raise ValueError("pairing request does not match offer")
        expected_proof = hmac.new(
            offer.secret.encode("utf-8"),
            _canonical(unsigned),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected_proof, request.proof):
            raise PermissionError("candidate pairing proof is invalid")
        binding = hmac.new(
            _secret_hash(offer.secret).encode("ascii"),
            request.proof.encode("ascii") + _canonical(unsigned),
            hashlib.sha256,
        ).hexdigest()
        with self.registry._store._connect() as db:
            columns = {
                item["name"]
                for item in db.execute("PRAGMA table_info(nexus_pairing_sessions)")
            }
            if "proof_binding" not in columns:
                db.execute(
                    "ALTER TABLE nexus_pairing_sessions ADD COLUMN proof_binding TEXT"
                )
            db.execute(
                """
                UPDATE nexus_pairing_sessions
                SET proof_binding=?
                WHERE pairing_id=? AND state='offered'
                """,
                (binding, offer.pairing_id),
            )

    def pending(self) -> tuple[PendingPairing, ...]:
        with self.registry._store._connect() as db:
            rows = db.execute(
                """
                SELECT pairing_id, candidate_device, candidate_public_key,
                       candidate_endpoint, updated_at
                FROM nexus_pairing_sessions
                WHERE state='pending'
                ORDER BY updated_at
                """
            ).fetchall()
        return tuple(
            PendingPairing(
                row["pairing_id"],
                row["candidate_device"],
                row["candidate_public_key"],
                row["candidate_endpoint"],
                row["updated_at"],
            )
            for row in rows
        )

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
            peer = self.registry.trust_peer(
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
                (now.isoformat().replace("+00:00", "Z"), pairing_id),
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
