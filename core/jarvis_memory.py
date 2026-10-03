"""Cross-device semantic continuity memory for the single JARVIS identity.

Memory is stored as semantic events inside the existing NEXUS EventStore. The
same events are materialized locally and synchronized to paired devices; raw
SQLite/WAL files are never copied between machines.

Design:
- immutable turn entities preserve conversational history;
- project checkpoints use LWW-map semantics per field;
- one current handoff register gives a compact "continue from here" snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping
from uuid import uuid4

from core.nexus.event_store import EventStore
from core.nexus.merge_applier import MergeApplier


JARVIS_MEMORY_NAMESPACE = "jarvis"


@dataclass(frozen=True)
class MemoryTurn:
    entity_id: str
    session_id: str
    role: str
    content: str
    project_id: str | None
    timestamp: str
    device_id: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ProjectState:
    project_id: str
    fields: Mapping[str, Any]


class JarvisMemory:
    def __init__(self, store: EventStore, applier: MergeApplier | None = None):
        if not isinstance(store, EventStore):
            raise TypeError("store must be EventStore")
        self.store = store
        self.applier = applier or MergeApplier(store)

    def recover(self) -> dict[str, int]:
        local = self.applier.apply_pending_local()
        inbound = self.applier.apply_pending()
        exported = self.store.flush_pending()
        return {
            "local_applied": len(local),
            "inbound_applied": len(inbound),
            "exported": exported,
        }

    @staticmethod
    def _clean_text(value: str, *, field: str, limit: int = 32000) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be non-empty text")
        text = value.strip()
        if len(text) > limit:
            raise ValueError(f"{field} exceeds the configured memory budget")
        return text

    @staticmethod
    def _project_entity(project_id: str) -> str:
        project = JarvisMemory._clean_text(project_id, field="project_id", limit=160)
        return "jarvis.project:" + project

    @staticmethod
    def _materialize_map(snapshot) -> dict[str, Any]:
        if snapshot is None or snapshot.tombstone or not snapshot.value:
            return {}
        state = snapshot.value
        entries = state.get("entries", {}) if isinstance(state, dict) else {}
        result: dict[str, Any] = {}
        for key, item in entries.items():
            if not isinstance(item, dict) or item.get("tombstone"):
                continue
            result[key] = item.get("value")
        return result

    def append_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        project_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MemoryTurn:
        session = self._clean_text(session_id, field="session_id", limit=160)
        role_value = self._clean_text(role, field="role", limit=32).casefold()
        if role_value not in {"user", "assistant", "system"}:
            raise ValueError("role must be user, assistant or system")
        text = self._clean_text(content, field="content")
        project = None if project_id is None else self._clean_text(
            project_id, field="project_id", limit=160
        )
        meta = dict(metadata or {})
        # Round-trip JSON now so callers cannot inject mutable/non-JSON objects.
        meta = json.loads(json.dumps(meta, sort_keys=True, allow_nan=False))

        entity_id = "jarvis.turn:" + uuid4().hex
        value = {
            "session_id": session,
            "role": role_value,
            "content": text,
            "project_id": project,
            "metadata": meta,
        }
        event = self.store.create_event(
            "memory.turn.upsert",
            entity_id,
            {"value": value},
            entity_type="memory.fact",
        )
        self.applier.apply_local_event(event)
        snapshot = self.applier.get_snapshot(entity_id)
        return MemoryTurn(
            entity_id=entity_id,
            session_id=session,
            role=role_value,
            content=text,
            project_id=project,
            timestamp=snapshot.timestamp,
            device_id=snapshot.last_device,
            metadata=meta,
        )

    def remember_fact(
        self,
        key: str,
        value: Any,
        *,
        source: str = "owner",
        confidence: float = 1.0,
    ) -> str:
        fact_key = self._clean_text(key, field="key", limit=200)
        source_value = self._clean_text(source, field="source", limit=80)
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise TypeError("confidence must be numeric")
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

        clean_value = json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
        entity_id = "jarvis.fact:" + fact_key
        event = self.store.create_event(
            "memory.fact.upsert",
            entity_id,
            {
                "value": {
                    "key": fact_key,
                    "value": clean_value,
                    "source": source_value,
                    "confidence": confidence,
                }
            },
            entity_type="memory.fact",
        )
        self.applier.apply_local_event(event)
        return event.id

    def get_fact(self, key: str) -> Any | None:
        fact_key = self._clean_text(key, field="key", limit=200)
        snapshot = self.applier.get_snapshot("jarvis.fact:" + fact_key)
        if snapshot is None or snapshot.tombstone or not isinstance(snapshot.value, dict):
            return None
        return snapshot.value.get("value")

    def set_project_field(self, project_id: str, key: str, value: Any) -> str:
        field = self._clean_text(key, field="project field", limit=120)
        clean_value = json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
        event = self.store.create_event(
            "memory.project.set",
            self._project_entity(project_id),
            {"key": field, "value": clean_value},
            entity_type="memory.project",
        )
        self.applier.apply_local_event(event)
        return event.id

    def delete_project_field(self, project_id: str, key: str) -> str:
        field = self._clean_text(key, field="project field", limit=120)
        event = self.store.create_event(
            "memory.project.delete",
            self._project_entity(project_id),
            {"key": field},
            entity_type="memory.project",
        )
        self.applier.apply_local_event(event)
        return event.id

    def checkpoint_project(
        self,
        project_id: str,
        *,
        summary: str | None = None,
        phase: str | None = None,
        open_tasks: list[str] | None = None,
        decisions: list[str] | None = None,
        artifacts: list[str] | None = None,
    ) -> ProjectState:
        fields = {
            "summary": summary,
            "phase": phase,
            "open_tasks": open_tasks,
            "decisions": decisions,
            "artifacts": artifacts,
        }
        for key, value in fields.items():
            if value is not None:
                self.set_project_field(project_id, key, value)
        return self.project_state(project_id)

    def project_state(self, project_id: str) -> ProjectState:
        entity_id = self._project_entity(project_id)
        snapshot = self.applier.get_snapshot(entity_id)
        return ProjectState(
            project_id=project_id,
            fields=self._materialize_map(snapshot),
        )

    def write_handoff(
        self,
        *,
        project_id: str | None,
        summary: str,
        next_actions: list[str] | tuple[str, ...] = (),
        session_id: str | None = None,
    ) -> str:
        summary_text = self._clean_text(summary, field="summary", limit=16000)
        value = {
            "project_id": project_id,
            "summary": summary_text,
            "next_actions": list(next_actions),
            "session_id": session_id,
        }
        event = self.store.create_event(
            "memory.handoff.upsert",
            "jarvis.handoff:current",
            {"value": value},
            entity_type="memory.handoff",
        )
        self.applier.apply_local_event(event)
        return event.id

    def current_handoff(self) -> dict[str, Any] | None:
        snapshot = self.applier.get_snapshot("jarvis.handoff:current")
        if snapshot is None or snapshot.tombstone:
            return None
        return dict(snapshot.value or {})

    def recent_turns(
        self,
        *,
        limit: int = 20,
        project_id: str | None = None,
    ) -> tuple[MemoryTurn, ...]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 200:
            raise ValueError("limit must be between 1 and 200")

        query = """
            SELECT entity_id, value_json, timestamp, last_device
            FROM nexus_entities
            WHERE entity_type='memory.fact'
              AND tombstone=0
              AND entity_id LIKE 'jarvis.turn:%'
            ORDER BY timestamp DESC, entity_id DESC
            LIMIT ?
        """
        rows = []
        with self.store._connect() as db:
            for row in db.execute(query, (limit * 4,)):
                value = json.loads(row["value_json"]) if row["value_json"] else {}
                if not isinstance(value, dict):
                    continue
                if project_id is not None and value.get("project_id") != project_id:
                    continue
                rows.append(
                    MemoryTurn(
                        entity_id=row["entity_id"],
                        session_id=str(value.get("session_id") or ""),
                        role=str(value.get("role") or ""),
                        content=str(value.get("content") or ""),
                        project_id=value.get("project_id"),
                        timestamp=row["timestamp"],
                        device_id=row["last_device"],
                        metadata=dict(value.get("metadata") or {}),
                    )
                )
                if len(rows) >= limit:
                    break
        rows.reverse()
        return tuple(rows)

    def continuity_context(self, *, turn_limit: int = 8) -> str:
        """Compact prompt context shared across desktop/mobile JARVIS surfaces."""
        parts: list[str] = []
        handoff = self.current_handoff()
        if handoff:
            parts.append(
                "Current handoff: "
                + json.dumps(handoff, ensure_ascii=False, sort_keys=True)
            )
            project_id = handoff.get("project_id")
            if isinstance(project_id, str) and project_id:
                project = self.project_state(project_id)
                if project.fields:
                    parts.append(
                        "Active project state: "
                        + json.dumps(project.fields, ensure_ascii=False, sort_keys=True)
                    )

        turns = self.recent_turns(limit=turn_limit)
        if turns:
            transcript = "\n".join(
                ("Owner" if item.role == "user" else "JARVIS")
                + ": "
                + item.content
                for item in turns
                if item.role in {"user", "assistant"}
            )
            if transcript:
                parts.append("Recent synchronized conversation:\n" + transcript)

        return "\n\n".join(parts)
