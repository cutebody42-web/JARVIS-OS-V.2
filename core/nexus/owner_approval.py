"""Biometric companion approval queue for sensitive JARVIS operations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hmac
import re
from typing import Callable
from uuid import uuid4

from core.nexus.event_store import EventStore


_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class OwnerApproval:
    approval_id: str
    summary: str
    action_digest: str
    created_at: str
    expires_at: str
    state: str
    decided_by: str | None = None
    consumed: bool = False


class OwnerApprovalManager:
    def __init__(self, store: EventStore, *, clock: Callable[[], datetime] = _utc_now):
        if not isinstance(store, EventStore):
            raise TypeError("store must be EventStore")
        self.store = store
        self._clock = clock
        with self.store._connect() as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS nexus_owner_approvals (
                    approval_id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL,
                    action_digest TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('pending','approved','rejected','expired')),
                    decided_by TEXT,
                    decided_at TEXT,
                    consumed INTEGER NOT NULL DEFAULT 0 CHECK(consumed IN (0,1))
                )
                """
            )

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("approval clock must be timezone-aware")
        return now.astimezone(timezone.utc)

    @staticmethod
    def _parse(value: str) -> datetime:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("approval timestamp must include timezone")
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _row(row) -> OwnerApproval:
        return OwnerApproval(
            row["approval_id"],
            row["summary"],
            row["action_digest"],
            row["created_at"],
            row["expires_at"],
            row["state"],
            row["decided_by"],
            bool(row["consumed"]),
        )

    def _expire_locked(self, db) -> None:
        now = self._now()
        # Compare actual instants: ISO strings with optional fractional seconds
        # do not sort reliably within the same second. An unused approved grant
        # has the same deadline as the pending request and must expire too.
        rows = db.execute(
            "SELECT approval_id, expires_at FROM nexus_owner_approvals "
            "WHERE state IN ('pending','approved') AND consumed=0"
        ).fetchall()
        expired = [
            (row["approval_id"],)
            for row in rows
            if self._parse(row["expires_at"]) <= now
        ]
        db.executemany(
            "UPDATE nexus_owner_approvals SET state='expired' WHERE approval_id=?",
            expired,
        )

    @staticmethod
    def _digest(value: str) -> str:
        if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value.casefold()):
            raise ValueError("action_digest must be a SHA-256 hex digest")
        return value.casefold()

    def create(
        self,
        summary: str,
        action_digest: str,
        *,
        ttl_seconds: int = 180,
    ) -> OwnerApproval:
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("approval summary must be non-empty")
        if len(summary.strip()) > 500:
            raise ValueError("approval summary is too long")
        digest = self._digest(action_digest)
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int) or not 15 <= ttl_seconds <= 600:
            raise ValueError("approval TTL must be 15..600 seconds")

        now = self._now()
        expires = datetime.fromtimestamp(now.timestamp() + ttl_seconds, timezone.utc)
        value = OwnerApproval(
            uuid4().hex,
            summary.strip(),
            digest,
            now.isoformat().replace("+00:00", "Z"),
            expires.isoformat().replace("+00:00", "Z"),
            "pending",
        )
        with self.store._connect() as db:
            db.execute(
                """
                INSERT INTO nexus_owner_approvals(
                    approval_id, summary, action_digest, created_at, expires_at,
                    state, decided_by, decided_at, consumed
                ) VALUES (?, ?, ?, ?, ?, 'pending', NULL, NULL, 0)
                """,
                (
                    value.approval_id,
                    value.summary,
                    value.action_digest,
                    value.created_at,
                    value.expires_at,
                ),
            )
        return value

    def pending(self) -> tuple[OwnerApproval, ...]:
        with self.store._connect() as db:
            self._expire_locked(db)
            rows = db.execute(
                """
                SELECT approval_id, summary, action_digest, created_at, expires_at,
                       state, decided_by, consumed
                FROM nexus_owner_approvals
                WHERE state='pending'
                ORDER BY created_at
                """
            ).fetchall()
        return tuple(self._row(row) for row in rows)

    def decide(
        self,
        approval_id: str,
        *,
        peer_id: str,
        approved: bool,
        user_verified: bool,
        action_digest: str | None = None,
    ) -> OwnerApproval:
        if user_verified is not True:
            raise PermissionError("Companion approval requires local biometric verification.")
        if not isinstance(approved, bool):
            raise ValueError("approved must be boolean")
        if not isinstance(approval_id, str) or not approval_id:
            raise ValueError("approval_id is required")
        if not isinstance(peer_id, str) or not peer_id:
            raise ValueError("peer_id is required")
        digest = self._digest(action_digest) if action_digest is not None else None
        now = self._now()
        with self.store._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._expire_locked(db)
            row = db.execute(
                """
                SELECT approval_id, summary, action_digest, created_at, expires_at,
                       state, decided_by, consumed
                FROM nexus_owner_approvals
                WHERE approval_id=?
                """,
                (approval_id,),
            ).fetchone()
            if row is None:
                db.rollback()
                raise KeyError("approval request not found")
            current = self._row(row)
            if current.state != "pending":
                db.rollback()
                raise PermissionError(f"approval request is already {current.state}")
            if digest is not None and not hmac.compare_digest(current.action_digest, digest):
                db.rollback()
                raise PermissionError("approval decision is bound to a different action")
            state = "approved" if approved else "rejected"
            db.execute(
                """
                UPDATE nexus_owner_approvals
                SET state=?, decided_by=?, decided_at=?
                WHERE approval_id=? AND state='pending'
                """,
                (
                    state,
                    peer_id,
                    now.isoformat().replace("+00:00", "Z"),
                    approval_id,
                ),
            )
            db.commit()
        return self.get(approval_id)

    def get(self, approval_id: str) -> OwnerApproval:
        with self.store._connect() as db:
            self._expire_locked(db)
            row = db.execute(
                """
                SELECT approval_id, summary, action_digest, created_at, expires_at,
                       state, decided_by, consumed
                FROM nexus_owner_approvals WHERE approval_id=?
                """,
                (approval_id,),
            ).fetchone()
        if row is None:
            raise KeyError("approval request not found")
        return self._row(row)

    def consume(self, approval_id: str, *, action_digest: str) -> OwnerApproval:
        digest = self._digest(action_digest)
        with self.store._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._expire_locked(db)
            row = db.execute(
                """
                SELECT approval_id, summary, action_digest, created_at, expires_at,
                       state, decided_by, consumed
                FROM nexus_owner_approvals WHERE approval_id=?
                """,
                (approval_id,),
            ).fetchone()
            if row is None:
                db.rollback()
                raise KeyError("approval request not found")
            value = self._row(row)
            if value.state != "approved" or value.consumed:
                db.rollback()
                raise PermissionError("approval is not active")
            if value.action_digest != digest:
                db.rollback()
                raise PermissionError("approval is bound to a different action")
            db.execute(
                "UPDATE nexus_owner_approvals SET consumed=1 WHERE approval_id=?",
                (approval_id,),
            )
            db.commit()
        return self.get(approval_id)
