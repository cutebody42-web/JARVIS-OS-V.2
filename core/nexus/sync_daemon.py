"""Transport-neutral at-least-once peer synchronization for NEXUS.

This module deliberately does not open sockets or know about Tailscale.
It defines the durable delivery/ACK contract that any authenticated transport
(HTTP/WebSocket over Tailscale, test loopback, future LAN transport) must satisfy.

Delivery guarantee:
- outbound events are eligible only after local materialization is applied;
- delivery attempts are recorded before network send;
- missing/lost ACKs cause safe retransmission;
- receiver ACKs only after the event is durably present in EventStore;
- duplicate delivery is expected and safe because EventStore + MergeApplier are idempotent.

Backpressure:
- every sync_peer() sends at most one bounded batch;
- callers schedule repeated sync_peer() invocations;
- max events and max serialized bytes are enforced per batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
import sqlite3
from typing import Protocol, runtime_checkable
from uuid import uuid4

from core.nexus.event_store import EventStore, SyncEvent
from core.nexus.merge_applier import ApplyStatus, MergeApplier


PROTOCOL_VERSION = 1
_PEER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_peer_id(peer_id: str) -> str:
    if not isinstance(peer_id, str) or not _PEER_RE.fullmatch(peer_id):
        raise ValueError("peer_id must be a safe 1-64 character identifier")
    return peer_id


@dataclass(frozen=True)
class SyncBatch:
    sender_device: str
    batch_id: str
    events: tuple[SyncEvent, ...]
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _validate_peer_id(self.sender_device)
        if not isinstance(self.batch_id, str) or not self.batch_id.strip():
            raise ValueError("batch_id must be non-empty")
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("unsupported sync protocol version")
        if not self.events:
            raise ValueError("sync batch must contain at least one event")
        for event in self.events:
            if not isinstance(event, SyncEvent):
                raise TypeError("sync batch events must be SyncEvent instances")
            if event.device != self.sender_device:
                raise ValueError(
                    "v1 sync batches only carry direct-origin events from sender_device"
                )


@dataclass(frozen=True)
class SyncAck:
    receiver_device: str
    batch_id: str
    durable_event_ids: tuple[str, ...]
    applied_event_ids: tuple[str, ...] = ()
    failed_event_ids: tuple[str, ...] = ()
    pending_event_ids: tuple[str, ...] = ()
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _validate_peer_id(self.receiver_device)
        if not isinstance(self.batch_id, str) or not self.batch_id.strip():
            raise ValueError("batch_id must be non-empty")
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("unsupported sync protocol version")
        for group in (
            self.durable_event_ids,
            self.applied_event_ids,
            self.failed_event_ids,
            self.pending_event_ids,
        ):
            if any(not isinstance(item, str) or not item for item in group):
                raise ValueError("ack event ids must be non-empty strings")


@dataclass(frozen=True)
class SyncReport:
    peer_id: str
    batch_id: str | None
    selected: int
    acked: int
    remaining: int
    attempts_recorded: int
    error: str | None = None


@runtime_checkable
class SyncTransport(Protocol):
    def send_batch(self, peer_id: str, batch: SyncBatch) -> SyncAck:
        """Send one batch to an authenticated peer and return its durable ACK."""
        ...


class SyncDaemon:
    """Durable sender/receiver state machine, independent of network transport."""

    def __init__(
        self,
        event_store: EventStore,
        merge_applier: MergeApplier,
        *,
        max_batch_events: int = 50,
        max_batch_bytes: int = 512 * 1024,
    ):
        if not isinstance(event_store, EventStore):
            raise TypeError("event_store must be EventStore")
        if not isinstance(merge_applier, MergeApplier):
            raise TypeError("merge_applier must be MergeApplier")
        if isinstance(max_batch_events, bool) or not isinstance(max_batch_events, int) or max_batch_events <= 0:
            raise ValueError("max_batch_events must be a positive integer")
        if isinstance(max_batch_bytes, bool) or not isinstance(max_batch_bytes, int) or max_batch_bytes <= 0:
            raise ValueError("max_batch_bytes must be a positive integer")

        self._store = event_store
        self._applier = merge_applier
        self._max_batch_events = max_batch_events
        self._max_batch_bytes = max_batch_bytes
        self._init_delivery_table()

    @property
    def device_id(self) -> str:
        return self._store.device_id

    def _init_delivery_table(self) -> None:
        with self._store._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS nexus_sync_delivery (
                    peer_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'pending'
                        CHECK(state IN ('pending', 'acked')),
                    attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts >= 0),
                    last_error TEXT,
                    last_attempt_at TEXT,
                    acked_at TEXT,
                    PRIMARY KEY(peer_id, event_id),
                    FOREIGN KEY(event_id) REFERENCES nexus_sync_events(event_id)
                );

                CREATE INDEX IF NOT EXISTS idx_nexus_sync_delivery_pending
                    ON nexus_sync_delivery(peer_id, state, attempts);
                """
            )

    def _pending_for_peer(self, peer_id: str) -> tuple[SyncEvent, ...]:
        _validate_peer_id(peer_id)
        if peer_id == self.device_id:
            raise ValueError("cannot synchronize a device to itself")

        query = """
            SELECT e.*
            FROM nexus_sync_events AS e
            LEFT JOIN nexus_sync_delivery AS d
                ON d.peer_id = ? AND d.event_id = e.event_id
            WHERE e.direction = 'outbound'
              AND e.apply_state = 'applied'
              AND (d.state IS NULL OR d.state != 'acked')
            ORDER BY e.rowid
        """

        selected: list[SyncEvent] = []
        bytes_used = 0
        with self._store._connect() as db:
            for row in db.execute(query, (peer_id,)):
                event = self._store._row_to_event(row)
                encoded_size = len(event.to_json_line().encode("utf-8"))
                if selected and (
                    len(selected) >= self._max_batch_events
                    or bytes_used + encoded_size > self._max_batch_bytes
                ):
                    break
                if not selected and encoded_size > self._max_batch_bytes:
                    # A single valid event must be deliverable; selecting it makes
                    # the oversized condition observable instead of starvation.
                    selected.append(event)
                    break
                selected.append(event)
                bytes_used += encoded_size
                if len(selected) >= self._max_batch_events:
                    break
        return tuple(selected)

    def pending_for_peer_count(self, peer_id: str) -> int:
        _validate_peer_id(peer_id)
        with self._store._connect() as db:
            row = db.execute(
                """
                SELECT COUNT(*) AS count
                FROM nexus_sync_events AS e
                LEFT JOIN nexus_sync_delivery AS d
                    ON d.peer_id = ? AND d.event_id = e.event_id
                WHERE e.direction = 'outbound'
                  AND e.apply_state = 'applied'
                  AND (d.state IS NULL OR d.state != 'acked')
                """,
                (peer_id,),
            ).fetchone()
        return int(row["count"])

    def delivery_state(self, peer_id: str, event_id: str) -> dict | None:
        _validate_peer_id(peer_id)
        with self._store._connect() as db:
            row = db.execute(
                """
                SELECT peer_id, event_id, state, attempts, last_error,
                       last_attempt_at, acked_at
                FROM nexus_sync_delivery
                WHERE peer_id = ? AND event_id = ?
                """,
                (peer_id, event_id),
            ).fetchone()
        return dict(row) if row else None

    def _record_attempts(
        self, peer_id: str, events: tuple[SyncEvent, ...]
    ) -> None:
        now = _utc_now()
        with self._store._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for event in events:
                db.execute(
                    """
                    INSERT INTO nexus_sync_delivery(
                        peer_id, event_id, state, attempts, last_attempt_at
                    ) VALUES (?, ?, 'pending', 1, ?)
                    ON CONFLICT(peer_id, event_id) DO UPDATE SET
                        attempts = nexus_sync_delivery.attempts + 1,
                        last_attempt_at = excluded.last_attempt_at,
                        last_error = NULL
                    """,
                    (peer_id, event.id, now),
                )

    def _record_transport_error(
        self, peer_id: str, events: tuple[SyncEvent, ...], exc: Exception
    ) -> None:
        safe = f"{type(exc).__name__}"
        with self._store._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for event in events:
                db.execute(
                    """
                    UPDATE nexus_sync_delivery
                    SET last_error = ?
                    WHERE peer_id = ? AND event_id = ?
                    """,
                    (safe, peer_id, event.id),
                )

    def _accept_ack(
        self,
        peer_id: str,
        batch: SyncBatch,
        ack: SyncAck,
    ) -> int:
        if not isinstance(ack, SyncAck):
            raise TypeError("transport must return SyncAck")
        if ack.batch_id != batch.batch_id:
            raise ValueError("sync ACK batch_id does not match request")
        if ack.receiver_device != peer_id:
            raise ValueError("sync ACK receiver does not match peer_id")

        sent_ids = {event.id for event in batch.events}
        durable = set(ack.durable_event_ids)
        status_ids = (
            set(ack.applied_event_ids)
            | set(ack.failed_event_ids)
            | set(ack.pending_event_ids)
        )
        if not durable.issubset(sent_ids):
            raise ValueError("sync ACK contains event ids that were not sent")
        if not status_ids.issubset(durable):
            raise ValueError("sync ACK status ids must be durable first")

        now = _utc_now()
        with self._store._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for event_id in durable:
                db.execute(
                    """
                    UPDATE nexus_sync_delivery
                    SET state = 'acked',
                        acked_at = ?,
                        last_error = NULL
                    WHERE peer_id = ? AND event_id = ?
                    """,
                    (now, peer_id, event_id),
                )
        return len(durable)

    def sync_peer(self, peer_id: str, transport: SyncTransport) -> SyncReport:
        """Send at most one bounded batch. Call again for remaining events."""
        _validate_peer_id(peer_id)
        events = self._pending_for_peer(peer_id)
        if not events:
            return SyncReport(
                peer_id=peer_id,
                batch_id=None,
                selected=0,
                acked=0,
                remaining=0,
                attempts_recorded=0,
            )

        batch = SyncBatch(
            sender_device=self.device_id,
            batch_id=uuid4().hex,
            events=events,
        )

        # Record the attempt before network I/O. If the process dies after the
        # remote receives but before local ACK persistence, retry is safe.
        self._record_attempts(peer_id, events)

        try:
            ack = transport.send_batch(peer_id, batch)
            acked = self._accept_ack(peer_id, batch, ack)
        except Exception as exc:
            self._record_transport_error(peer_id, events, exc)
            return SyncReport(
                peer_id=peer_id,
                batch_id=batch.batch_id,
                selected=len(events),
                acked=0,
                remaining=self.pending_for_peer_count(peer_id),
                attempts_recorded=len(events),
                error=type(exc).__name__,
            )

        return SyncReport(
            peer_id=peer_id,
            batch_id=batch.batch_id,
            selected=len(events),
            acked=acked,
            remaining=self.pending_for_peer_count(peer_id),
            attempts_recorded=len(events),
        )

    def receive_batch(self, batch: SyncBatch) -> SyncAck:
        """Durably receive one direct-origin batch and attempt local materialization."""
        if not isinstance(batch, SyncBatch):
            raise TypeError("batch must be SyncBatch")
        if batch.sender_device == self.device_id:
            raise ValueError("loopback sync batches are not allowed")

        # Validate the whole batch before any durable mutation.
        for event in batch.events:
            if event.device != batch.sender_device:
                raise ValueError("event origin does not match authenticated batch sender")
            if event.device == self.device_id:
                raise ValueError("remote batch cannot contain local-origin events")

        durable: list[str] = []
        applied: list[str] = []
        failed: list[str] = []
        pending: list[str] = []

        for event in batch.events:
            self._store.accept_remote(event)
            if not self._store.seen(event.id):
                # Should be unreachable unless EventStore semantics change.
                raise RuntimeError("event was not durably accepted")

            durable.append(event.id)
            state = self._applier.get_apply_state(event.id)

            if state == "pending":
                try:
                    result = self._applier.apply_event(event)
                    if result.status in {
                        ApplyStatus.APPLIED,
                        ApplyStatus.MERGED,
                        ApplyStatus.SKIPPED,
                        ApplyStatus.ALREADY_APPLIED,
                    }:
                        state = "applied"
                    elif result.status is ApplyStatus.REJECTED:
                        state = "failed"
                except Exception:
                    # Durable ACK is still valid. Unknown/new policy or an
                    # application bug stays pending for local recovery; sender
                    # must not redeliver forever once durability is confirmed.
                    state = "pending"

            if state == "applied":
                applied.append(event.id)
            elif state == "failed":
                failed.append(event.id)
            else:
                pending.append(event.id)

        return SyncAck(
            receiver_device=self.device_id,
            batch_id=batch.batch_id,
            durable_event_ids=tuple(dict.fromkeys(durable)),
            applied_event_ids=tuple(dict.fromkeys(applied)),
            failed_event_ids=tuple(dict.fromkeys(failed)),
            pending_event_ids=tuple(dict.fromkeys(pending)),
        )
