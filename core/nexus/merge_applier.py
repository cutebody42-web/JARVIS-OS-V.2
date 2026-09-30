"""Transactional application of NEXUS merge decisions.

This layer is the ACID boundary between:
- durable inbound sync events in nexus_sync_events,
- pure conflict resolution, and
- the materialized semantic entity state in nexus_entities.

For entities stored in this same SQLite database, memory mutation, evidence,
and apply_state are committed atomically. A crash before COMMIT rolls all of
them back and leaves the inbound event pending.

This is exactly-once *materialization inside NEXUS local.db*. It does not claim
exactly-once semantics for external side effects such as filesystem, network,
or Windows actions; those remain governed by ActionReceipt/idempotency.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import sqlite3
from typing import Any, Callable

from core.nexus.conflict_resolver import (
    ConflictResolver,
    EntitySnapshot,
    MergeAction,
    MergeDecision,
)
from core.nexus.event_store import EventStore, SyncEvent


class ApplyStatus(str, Enum):
    APPLIED = "applied"
    MERGED = "merged"
    SKIPPED = "skipped"
    ALREADY_APPLIED = "already_applied"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ApplyResult:
    event_id: str
    status: ApplyStatus
    decision: MergeDecision | None = None


class MergeApplier:
    """Apply accepted inbound events transactionally into local materialized state."""

    def __init__(
        self,
        event_store: EventStore,
        resolver: ConflictResolver | None = None,
        *,
        before_commit_hook: Callable[[sqlite3.Connection, SyncEvent, MergeDecision], None]
        | None = None,
    ):
        if not isinstance(event_store, EventStore):
            raise TypeError("event_store must be EventStore")
        self._store = event_store
        self._resolver = resolver or ConflictResolver()
        self._before_commit_hook = before_commit_hook
        self._init_tables()

    def _init_tables(self) -> None:
        with self._store._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS nexus_entities (
                    entity_id TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    value_json TEXT,
                    vclock_json TEXT NOT NULL,
                    last_event_id TEXT NOT NULL,
                    last_device TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    tombstone INTEGER NOT NULL CHECK(tombstone IN (0, 1))
                );

                CREATE TABLE IF NOT EXISTS nexus_merge_evidence (
                    event_id TEXT PRIMARY KEY,
                    action TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    policy TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    winner_event_id TEXT,
                    loser_event_id TEXT,
                    FOREIGN KEY(event_id) REFERENCES nexus_sync_events(event_id)
                );

                CREATE INDEX IF NOT EXISTS idx_nexus_entities_type
                    ON nexus_entities(entity_type);
                """
            )

    @staticmethod
    def _encode_json(value: Any) -> str:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    @staticmethod
    def _decode_json(value: str | None) -> Any:
        return None if value is None else json.loads(value)

    @staticmethod
    def _row_to_snapshot(row: sqlite3.Row | None) -> EntitySnapshot | None:
        if row is None:
            return None
        return EntitySnapshot(
            entity_id=row["entity_id"],
            entity_type=row["entity_type"],
            value=MergeApplier._decode_json(row["value_json"]),
            vclock=json.loads(row["vclock_json"]),
            last_event_id=row["last_event_id"],
            last_device=row["last_device"],
            timestamp=row["timestamp"],
            tombstone=bool(row["tombstone"]),
        )

    def _load_snapshot(
        self, db: sqlite3.Connection, entity_id: str
    ) -> EntitySnapshot | None:
        row = db.execute(
            "SELECT * FROM nexus_entities WHERE entity_id = ?",
            (entity_id,),
        ).fetchone()
        return self._row_to_snapshot(row)

    def get_snapshot(self, entity_id: str) -> EntitySnapshot | None:
        with self._store._connect() as db:
            return self._load_snapshot(db, entity_id)

    def get_apply_state(self, event_id: str) -> str | None:
        with self._store._connect() as db:
            row = db.execute(
                "SELECT apply_state FROM nexus_sync_events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        return row["apply_state"] if row else None

    def get_evidence(self, event_id: str) -> dict[str, Any] | None:
        with self._store._connect() as db:
            row = db.execute(
                "SELECT * FROM nexus_merge_evidence WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        return dict(row) if row else None

    def _store_snapshot(
        self, db: sqlite3.Connection, snapshot: EntitySnapshot
    ) -> None:
        db.execute(
            """
            INSERT INTO nexus_entities(
                entity_id, entity_type, value_json, vclock_json,
                last_event_id, last_device, timestamp, tombstone
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(entity_id) DO UPDATE SET
                entity_type = excluded.entity_type,
                value_json = excluded.value_json,
                vclock_json = excluded.vclock_json,
                last_event_id = excluded.last_event_id,
                last_device = excluded.last_device,
                timestamp = excluded.timestamp,
                tombstone = excluded.tombstone
            """,
            (
                snapshot.entity_id,
                snapshot.entity_type,
                None if snapshot.value is None else self._encode_json(snapshot.value),
                self._encode_json(dict(snapshot.vclock)),
                snapshot.last_event_id,
                snapshot.last_device,
                snapshot.timestamp,
                int(snapshot.tombstone),
            ),
        )

    def _store_evidence(
        self, db: sqlite3.Connection, event: SyncEvent, decision: MergeDecision
    ) -> None:
        evidence = decision.evidence
        db.execute(
            """
            INSERT INTO nexus_merge_evidence(
                event_id, action, relation, policy, reason,
                winner_event_id, loser_event_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(event_id) DO UPDATE SET
                action = excluded.action,
                relation = excluded.relation,
                policy = excluded.policy,
                reason = excluded.reason,
                winner_event_id = excluded.winner_event_id,
                loser_event_id = excluded.loser_event_id
            """,
            (
                event.id,
                decision.action.value,
                evidence.relation.value,
                evidence.policy.value,
                evidence.reason,
                evidence.winner_event_id,
                evidence.loser_event_id,
            ),
        )

    @staticmethod
    def _status_for(decision: MergeDecision) -> ApplyStatus:
        if decision.action is MergeAction.APPLY:
            return ApplyStatus.APPLIED
        if decision.action is MergeAction.MERGE:
            return ApplyStatus.MERGED
        if decision.action is MergeAction.SKIP:
            return ApplyStatus.SKIPPED
        if decision.action is MergeAction.REJECT:
            return ApplyStatus.REJECTED
        raise ValueError(f"Unsupported merge action: {decision.action}")

    def _apply_direction(self, event: SyncEvent, expected_direction: str) -> ApplyResult:
        if not isinstance(event, SyncEvent):
            raise TypeError("event must be SyncEvent")
        if expected_direction not in {"inbound", "outbound"}:
            raise ValueError("expected_direction must be inbound or outbound")

        db = self._store._connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            ledger = db.execute(
                """
                SELECT direction, apply_state
                FROM nexus_sync_events
                WHERE event_id = ?
                """,
                (event.id,),
            ).fetchone()
            if ledger is None:
                raise ValueError("event must exist in EventStore before apply")
            if ledger["direction"] != expected_direction:
                raise ValueError(
                    f"merge applier expected {expected_direction} event, "
                    f"got {ledger['direction']}"
                )
            if ledger["apply_state"] == "applied":
                db.rollback()
                return ApplyResult(event.id, ApplyStatus.ALREADY_APPLIED)
            if ledger["apply_state"] == "failed":
                db.rollback()
                return ApplyResult(event.id, ApplyStatus.REJECTED)

            local = self._load_snapshot(db, event.entity_id)
            decision = self._resolver.resolve(local, event)

            if decision.action is not MergeAction.REJECT:
                self._store_snapshot(db, decision.snapshot)

            self._store_evidence(db, event, decision)
            apply_state = (
                "failed" if decision.action is MergeAction.REJECT else "applied"
            )
            db.execute(
                """
                UPDATE nexus_sync_events
                SET apply_state = ?
                WHERE event_id = ? AND direction = ?
                """,
                (apply_state, event.id, expected_direction),
            )

            if self._before_commit_hook is not None:
                self._before_commit_hook(db, event, decision)

            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

        if expected_direction == "outbound" and apply_state == "applied":
            # File export is intentionally outside the SQLite transaction. If the
            # process dies here, EventStore.flush_pending() recovers it later.
            self._store.flush_pending()

        return ApplyResult(event.id, self._status_for(decision), decision)

    def apply_event(self, event: SyncEvent) -> ApplyResult:
        """Resolve and materialize one accepted inbound event transactionally."""
        return self._apply_direction(event, "inbound")

    def apply_local_event(self, event: SyncEvent) -> ApplyResult:
        """Materialize one locally authored outbound event before it can sync."""
        return self._apply_direction(event, "outbound")

    def apply_pending(self, limit: int | None = None) -> tuple[ApplyResult, ...]:
        """Replay durable pending inbound events after startup or reconnect."""
        return tuple(
            self.apply_event(event)
            for event in self._store.pending_inbound(limit=limit)
        )

    def apply_pending_local(self, limit: int | None = None) -> tuple[ApplyResult, ...]:
        """Replay local events that were durably authored but not materialized."""
        return tuple(
            self.apply_local_event(event)
            for event in self._store.pending_local(limit=limit)
        )
