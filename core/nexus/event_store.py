"""Durable semantic event journal for NEXUS device sync.

This module deliberately synchronizes semantic events, never SQLite WAL pages or
raw database files. SQLite WAL may still be used *locally* for crash-safe storage.

The local SQLite database is the durable ledger. JSONL files under outbox/ are
transport spools derived from that ledger. If export is interrupted, flush_pending()
recreates any missing spool files without losing the event.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import os
from pathlib import Path
import re
import sqlite3
from types import MappingProxyType
from typing import Any, Iterable, Mapping
from uuid import uuid4


SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = {1, 2}
_DEVICE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class ClockRelation(str, Enum):
    BEFORE = "before"
    AFTER = "after"
    EQUAL = "equal"
    CONCURRENT = "concurrent"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("timestamp must be a non-empty string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return parsed


def _validate_device_id(device_id: str) -> str:
    if not isinstance(device_id, str) or not _DEVICE_RE.fullmatch(device_id):
        raise ValueError("device_id must be 1-64 safe identifier characters")
    return device_id


def _normalize_clock(clock: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(clock, Mapping):
        raise TypeError("vclock must be a mapping")
    normalized: dict[str, int] = {}
    for device, counter in clock.items():
        _validate_device_id(device)
        if isinstance(counter, bool) or not isinstance(counter, int) or counter < 0:
            raise ValueError("vector clock counters must be non-negative integers")
        normalized[device] = counter
    return normalized


def compare_vector_clocks(
    left: Mapping[str, int], right: Mapping[str, int]
) -> ClockRelation:
    """Compare two vector clocks, treating missing components as zero."""
    lclock = _normalize_clock(left)
    rclock = _normalize_clock(right)
    keys = set(lclock) | set(rclock)
    left_lt = any(lclock.get(k, 0) < rclock.get(k, 0) for k in keys)
    left_gt = any(lclock.get(k, 0) > rclock.get(k, 0) for k in keys)
    if left_lt and left_gt:
        return ClockRelation.CONCURRENT
    if left_lt:
        return ClockRelation.BEFORE
    if left_gt:
        return ClockRelation.AFTER
    return ClockRelation.EQUAL


def _freeze_json(value: Any) -> Any:
    """Validate JSON data and detach it from mutable caller-owned objects."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("payload cannot contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("payload object keys must be strings")
            frozen[key] = _freeze_json(item)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    raise TypeError(f"payload contains unsupported JSON type: {type(value).__name__}")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class SyncEvent:
    id: str
    device: str
    entity_id: str
    entity_type: str
    type: str
    vclock: Mapping[str, int]
    payload: Mapping[str, Any]
    timestamp: str
    tombstone: bool = False
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"unsupported sync event schema: {self.schema_version}")
        if not isinstance(self.id, str) or not self.id.strip() or len(self.id) > 128:
            raise ValueError("event id must be a non-empty string up to 128 characters")
        _validate_device_id(self.device)
        if not isinstance(self.entity_id, str) or not self.entity_id.strip():
            raise ValueError("entity_id must be a non-empty string")
        if not isinstance(self.entity_type, str) or not self.entity_type.strip():
            raise ValueError("entity_type must be a non-empty string")
        if self.schema_version >= 2 and self.entity_type == "legacy.unknown":
            raise ValueError("schema v2 events require an explicit entity_type")
        if not isinstance(self.type, str) or not self.type.strip():
            raise ValueError("event type must be a non-empty string")
        _parse_timestamp(self.timestamp)
        if not isinstance(self.tombstone, bool):
            raise TypeError("tombstone must be bool")

        clock = _normalize_clock(self.vclock)
        if clock.get(self.device, 0) <= 0:
            raise ValueError("event vector clock must include its originating device")
        if not isinstance(self.payload, Mapping):
            raise TypeError("payload must be a JSON object")

        frozen_payload = _freeze_json(self.payload)
        # Verify canonical JSON serialization now, not during network transport.
        json.dumps(_thaw_json(frozen_payload), sort_keys=True, separators=(",", ":"), allow_nan=False)

        object.__setattr__(self, "vclock", MappingProxyType(clock))
        object.__setattr__(self, "payload", frozen_payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "device": self.device,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "vclock": dict(self.vclock),
            "type": self.type,
            "payload": _thaw_json(self.payload),
            "timestamp": self.timestamp,
            "tombstone": self.tombstone,
        }

    def to_json_line(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SyncEvent":
        if not isinstance(value, Mapping):
            raise TypeError("event must be an object")
        allowed = {
            "schema_version",
            "id",
            "device",
            "entity_id",
            "entity_type",
            "vclock",
            "type",
            "payload",
            "timestamp",
            "tombstone",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown sync event fields: {sorted(unknown)}")
        schema_version = value.get("schema_version", SCHEMA_VERSION)
        required = {"id", "device", "entity_id", "vclock", "type", "payload", "timestamp"}
        if schema_version >= 2:
            required.add("entity_type")
        missing = required - set(value)
        if missing:
            raise ValueError(f"missing sync event fields: {sorted(missing)}")
        return cls(
            schema_version=schema_version,
            id=value["id"],
            device=value["device"],
            entity_id=value["entity_id"],
            entity_type=value.get("entity_type", "legacy.unknown"),
            vclock=value["vclock"],
            type=value["type"],
            payload=value["payload"],
            timestamp=value["timestamp"],
            tombstone=value.get("tombstone", False),
        )

    @classmethod
    def from_json_line(cls, line: str) -> "SyncEvent":
        if not isinstance(line, str) or not line.strip():
            raise ValueError("event JSONL line cannot be empty")
        return cls.from_dict(json.loads(line))


class EventStore:
    """Crash-safe local ledger plus append-only JSONL transport spool."""

    def __init__(self, root: str | Path, device_id: str):
        self.root = Path(root)
        self.device_id = _validate_device_id(device_id)
        self.db_path = self.root / "local.db"
        self.outbox_dir = self.root / "outbox"
        self.root.mkdir(parents=True, exist_ok=True)
        self.outbox_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _init_db(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS nexus_sync_clock (
                    device_id TEXT PRIMARY KEY,
                    counter INTEGER NOT NULL CHECK(counter >= 0)
                );

                CREATE TABLE IF NOT EXISTS nexus_sync_events (
                    event_id TEXT PRIMARY KEY,
                    device_id TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    entity_type TEXT NOT NULL DEFAULT 'legacy.unknown',
                    event_type TEXT NOT NULL,
                    vclock_json TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    tombstone INTEGER NOT NULL CHECK(tombstone IN (0, 1)),
                    direction TEXT NOT NULL CHECK(direction IN ('outbound', 'inbound')),
                    exported INTEGER NOT NULL CHECK(exported IN (0, 1)),
                    apply_state TEXT NOT NULL DEFAULT 'applied'
                        CHECK(apply_state IN ('pending', 'applied', 'failed')),
                    schema_version INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_nexus_sync_pending
                    ON nexus_sync_events(direction, exported, timestamp);
                """
            )
            columns = {
                row["name"] for row in db.execute("PRAGMA table_info(nexus_sync_events)")
            }
            if "entity_type" not in columns:
                db.execute(
                    "ALTER TABLE nexus_sync_events ADD COLUMN entity_type "
                    "TEXT NOT NULL DEFAULT 'legacy.unknown'"
                )
            if "apply_state" not in columns:
                db.execute(
                    "ALTER TABLE nexus_sync_events ADD COLUMN apply_state "
                    "TEXT NOT NULL DEFAULT 'applied'"
                )
                db.execute(
                    "UPDATE nexus_sync_events SET apply_state = 'pending' "
                    "WHERE direction = 'inbound'"
                )
            db.execute(
                "INSERT OR IGNORE INTO nexus_sync_clock(device_id, counter) VALUES (?, 0)",
                (self.device_id,),
            )

    def _snapshot_clock(self, db: sqlite3.Connection) -> dict[str, int]:
        return {
            row["device_id"]: row["counter"]
            for row in db.execute(
                "SELECT device_id, counter FROM nexus_sync_clock ORDER BY device_id"
            )
        }

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> SyncEvent:
        return SyncEvent(
            id=row["event_id"],
            device=row["device_id"],
            entity_id=row["entity_id"],
            entity_type=row["entity_type"],
            type=row["event_type"],
            vclock=json.loads(row["vclock_json"]),
            payload=json.loads(row["payload_json"]),
            timestamp=row["timestamp"],
            tombstone=bool(row["tombstone"]),
            schema_version=row["schema_version"],
        )

    def create_event(
        self,
        event_type: str,
        entity_id: str,
        payload: Mapping[str, Any],
        *,
        entity_type: str,
        tombstone: bool = False,
        event_id: str | None = None,
        timestamp: str | None = None,
        export: bool = True,
    ) -> SyncEvent:
        """Persist a local event, then best-effort materialize its JSONL spool.

        The SQLite insert commits before spool export. Therefore an interrupted
        export cannot lose the event; flush_pending() recovers it later.
        """
        event_id = event_id or uuid4().hex
        timestamp = timestamp or _utc_now()

        # Build a temporary event to validate payload/type before taking the write lock.
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute(
                "SELECT counter FROM nexus_sync_clock WHERE device_id = ?",
                (self.device_id,),
            ).fetchone()
            next_counter = (current["counter"] if current else 0) + 1
            db.execute(
                """
                INSERT INTO nexus_sync_clock(device_id, counter) VALUES (?, ?)
                ON CONFLICT(device_id) DO UPDATE SET counter = excluded.counter
                """,
                (self.device_id, next_counter),
            )
            clock = self._snapshot_clock(db)
            event = SyncEvent(
                id=event_id,
                device=self.device_id,
                entity_id=entity_id,
                entity_type=entity_type,
                type=event_type,
                vclock=clock,
                payload=payload,
                timestamp=timestamp,
                tombstone=tombstone,
            )
            db.execute(
                """
                INSERT INTO nexus_sync_events(
                    event_id, device_id, entity_id, entity_type, event_type, vclock_json,
                    payload_json, timestamp, tombstone, direction, exported,
                    apply_state, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'outbound', 0, 'applied', ?)
                """,
                (
                    event.id,
                    event.device,
                    event.entity_id,
                    event.entity_type,
                    event.type,
                    json.dumps(dict(event.vclock), sort_keys=True, separators=(",", ":")),
                    json.dumps(
                        _thaw_json(event.payload),
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    ),
                    event.timestamp,
                    int(event.tombstone),
                    event.schema_version,
                ),
            )

        if export:
            self.flush_pending()
        return event

    def _spool_path(self, event: SyncEvent) -> Path:
        stamp = _parse_timestamp(event.timestamp).astimezone(timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        return self.outbox_dir / f"{stamp}_{event.device}_{event.id[:12]}.jsonl"

    def _write_event_file(self, event: SyncEvent) -> Path:
        destination = self._spool_path(event)
        canonical = event.to_json_line().encode("utf-8")
        if destination.exists():
            if destination.read_bytes() != canonical:
                raise RuntimeError(f"outbox collision for event {event.id}")
            return destination

        temporary = destination.with_suffix(destination.suffix + f".{uuid4().hex}.tmp")
        try:
            with open(temporary, "xb") as handle:
                handle.write(canonical)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            return destination
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def flush_pending(self, limit: int | None = None) -> int:
        if limit is not None and (isinstance(limit, bool) or limit <= 0):
            raise ValueError("limit must be a positive integer")
        query = """
            SELECT * FROM nexus_sync_events
            WHERE direction = 'outbound' AND exported = 0
            ORDER BY rowid
        """
        params: tuple[Any, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (limit,)

        with self._connect() as db:
            rows = list(db.execute(query, params))

        exported = 0
        for row in rows:
            event = self._row_to_event(row)
            self._write_event_file(event)
            with self._connect() as db:
                db.execute(
                    """
                    UPDATE nexus_sync_events
                    SET exported = 1
                    WHERE event_id = ? AND direction = 'outbound'
                    """,
                    (event.id,),
                )
            exported += 1
        return exported

    def accept_remote(self, event: SyncEvent) -> bool:
        """Durably record a remote event once. Payload application happens elsewhere."""
        if not isinstance(event, SyncEvent):
            raise TypeError("event must be SyncEvent")
        if event.device == self.device_id:
            # Loopback delivery is a dedupe check, not a new inbound event.
            return False

        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            exists = db.execute(
                "SELECT 1 FROM nexus_sync_events WHERE event_id = ?", (event.id,)
            ).fetchone()
            if exists:
                return False

            db.execute(
                """
                INSERT INTO nexus_sync_events(
                    event_id, device_id, entity_id, entity_type, event_type, vclock_json,
                    payload_json, timestamp, tombstone, direction, exported,
                    apply_state, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'inbound', 1, 'pending', ?)
                """,
                (
                    event.id,
                    event.device,
                    event.entity_id,
                    event.entity_type,
                    event.type,
                    json.dumps(dict(event.vclock), sort_keys=True, separators=(",", ":")),
                    json.dumps(
                        _thaw_json(event.payload),
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    ),
                    event.timestamp,
                    int(event.tombstone),
                    event.schema_version,
                ),
            )
            for device, counter in event.vclock.items():
                db.execute(
                    """
                    INSERT INTO nexus_sync_clock(device_id, counter) VALUES (?, ?)
                    ON CONFLICT(device_id) DO UPDATE SET
                        counter = MAX(nexus_sync_clock.counter, excluded.counter)
                    """,
                    (device, counter),
                )
        return True

    def ingest_jsonl(self, lines: Iterable[str]) -> tuple[int, int]:
        """Validate and dedupe JSONL events. Returns (accepted, duplicates)."""
        accepted = 0
        duplicates = 0
        for line in lines:
            event = SyncEvent.from_json_line(line)
            if self.accept_remote(event):
                accepted += 1
            else:
                duplicates += 1
        return accepted, duplicates

    def seen(self, event_id: str) -> bool:
        with self._connect() as db:
            return (
                db.execute(
                    "SELECT 1 FROM nexus_sync_events WHERE event_id = ?", (event_id,)
                ).fetchone()
                is not None
            )

    def get_event(self, event_id: str) -> SyncEvent | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM nexus_sync_events WHERE event_id = ?", (event_id,)
            ).fetchone()
        return self._row_to_event(row) if row else None

    def pending_count(self) -> int:
        with self._connect() as db:
            row = db.execute(
                """
                SELECT COUNT(*) AS count FROM nexus_sync_events
                WHERE direction = 'outbound' AND exported = 0
                """
            ).fetchone()
        return int(row["count"])

    def pending_inbound(self, limit: int | None = None) -> tuple[SyncEvent, ...]:
        """Return durable remote events that still need merge/application.

        This method does not mark events applied. The future merge-applier must
        commit application state together with memory mutation, or otherwise use
        event-id idempotency, before changing apply_state.
        """
        if limit is not None and (isinstance(limit, bool) or limit <= 0):
            raise ValueError("limit must be a positive integer")
        query = """
            SELECT * FROM nexus_sync_events
            WHERE direction = 'inbound' AND apply_state = 'pending'
            ORDER BY rowid
        """
        params: tuple[Any, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (limit,)
        with self._connect() as db:
            return tuple(self._row_to_event(row) for row in db.execute(query, params))

    def pending_apply_count(self) -> int:
        with self._connect() as db:
            row = db.execute(
                """
                SELECT COUNT(*) AS count FROM nexus_sync_events
                WHERE direction = 'inbound' AND apply_state = 'pending'
                """
            ).fetchone()
        return int(row["count"])

    def current_clock(self) -> dict[str, int]:
        with self._connect() as db:
            return self._snapshot_clock(db)
