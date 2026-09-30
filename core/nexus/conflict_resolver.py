"""Pure deterministic conflict resolution for NEXUS semantic events.

Durability and dedupe remain in event_store.py. This module never writes SQLite,
files, network sockets or application memory. It returns a decision that a later
transactional merge-applier can commit together with the inbound event's apply
state.

Vector clocks decide causality. They do *not* totally order concurrent writes.
For LWW-style policies, concurrent writes use an explicit deterministic tie-break:
UTC timestamp, then device id, then event id. A future HLC can replace this order
key without changing the resolver interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
from typing import Any, Mapping

from core.nexus.event_store import ClockRelation, SyncEvent, compare_vector_clocks
from core.nexus.merge_policy import (
    MergePolicyKind,
    MergePolicyRegistry,
    MergePolicySpec,
)


class MergeAction(str, Enum):
    SKIP = "skip"
    APPLY = "apply"
    MERGE = "merge"
    REJECT = "reject"


@dataclass(frozen=True)
class EntitySnapshot:
    entity_id: str
    entity_type: str
    value: Any
    vclock: Mapping[str, int]
    last_event_id: str
    last_device: str
    timestamp: str
    tombstone: bool = False

    def __post_init__(self) -> None:
        if not self.entity_id or not self.entity_type:
            raise ValueError("snapshot requires entity identity and type")
        if not self.last_event_id or not self.last_device:
            raise ValueError("snapshot requires last event identity")
        _parse_ts(self.timestamp)
        _json_copy(self.value)
        object.__setattr__(self, "vclock", dict(self.vclock))
        object.__setattr__(self, "value", _json_copy(self.value))


@dataclass(frozen=True)
class MergeEvidence:
    relation: ClockRelation
    policy: MergePolicyKind
    reason: str
    winner_event_id: str | None = None
    loser_event_id: str | None = None


@dataclass(frozen=True)
class MergeDecision:
    action: MergeAction
    snapshot: EntitySnapshot
    evidence: MergeEvidence


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _parse_ts(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None:
        raise ValueError("timestamp must include timezone")
    return parsed.astimezone(timezone.utc)


def _join_clocks(left: Mapping[str, int], right: Mapping[str, int]) -> dict[str, int]:
    keys = set(left) | set(right)
    return {key: max(left.get(key, 0), right.get(key, 0)) for key in sorted(keys)}


def _event_order_key(timestamp: str, device: str, event_id: str) -> tuple:
    return (_parse_ts(timestamp), device, event_id)


def _snapshot_order_key(snapshot: EntitySnapshot) -> tuple:
    return _event_order_key(snapshot.timestamp, snapshot.last_device, snapshot.last_event_id)


def _remote_order_key(event: SyncEvent) -> tuple:
    return _event_order_key(event.timestamp, event.device, event.id)


def _require_value(event: SyncEvent) -> Any:
    payload = event.to_dict()["payload"]
    if "value" not in payload:
        raise ValueError(f"{event.type} requires payload.value")
    return payload["value"]


def _remote_snapshot(
    event: SyncEvent,
    *,
    value: Any | None = None,
    vclock: Mapping[str, int] | None = None,
    tombstone: bool | None = None,
) -> EntitySnapshot:
    if value is None and not event.tombstone:
        value = _require_value(event)
    return EntitySnapshot(
        entity_id=event.entity_id,
        entity_type=event.entity_type,
        value=None if (event.tombstone if tombstone is None else tombstone) else value,
        vclock=dict(vclock or event.vclock),
        last_event_id=event.id,
        last_device=event.device,
        timestamp=event.timestamp,
        tombstone=event.tombstone if tombstone is None else tombstone,
    )


def materialize_orset(snapshot: EntitySnapshot) -> tuple[str, ...]:
    state = snapshot.value or {"adds": {}, "removes": []}
    adds = state.get("adds", {})
    removes = set(state.get("removes", []))
    visible = [
        element
        for element, tags in adds.items()
        if any(tag not in removes for tag in tags)
    ]
    return tuple(sorted(visible))


def materialize_pn_counter(snapshot: EntitySnapshot) -> int:
    state = snapshot.value or {"p": {}, "n": {}}
    return sum(state.get("p", {}).values()) - sum(state.get("n", {}).values())


class ConflictResolver:
    def __init__(self, registry: MergePolicyRegistry | None = None):
        self._registry = registry or MergePolicyRegistry()

    def resolve(
        self, local: EntitySnapshot | None, remote: SyncEvent
    ) -> MergeDecision:
        if not isinstance(remote, SyncEvent):
            raise TypeError("remote must be SyncEvent")
        spec = self._registry.get(remote.entity_type)

        if local is None:
            snapshot = self._apply_policy(None, remote, spec, ClockRelation.AFTER)
            return MergeDecision(
                MergeAction.APPLY,
                snapshot,
                MergeEvidence(
                    ClockRelation.AFTER,
                    spec.kind,
                    "No local entity exists; apply the first validated event.",
                    winner_event_id=remote.id,
                ),
            )

        if local.entity_id != remote.entity_id or local.entity_type != remote.entity_type:
            raise ValueError("local snapshot and remote event must address the same entity")

        relation = compare_vector_clocks(remote.vclock, local.vclock)

        if relation is ClockRelation.BEFORE:
            return MergeDecision(
                MergeAction.SKIP,
                local,
                MergeEvidence(
                    relation,
                    spec.kind,
                    "Remote event is causally older than the materialized entity.",
                    winner_event_id=local.last_event_id,
                    loser_event_id=remote.id,
                ),
            )

        if relation is ClockRelation.EQUAL:
            if remote.id == local.last_event_id:
                return MergeDecision(
                    MergeAction.SKIP,
                    local,
                    MergeEvidence(
                        relation,
                        spec.kind,
                        "Remote event is the already-materialized event.",
                        winner_event_id=local.last_event_id,
                    ),
                )
            return MergeDecision(
                MergeAction.REJECT,
                local,
                MergeEvidence(
                    relation,
                    spec.kind,
                    "Different event ids claim the same vector clock; reject as an integrity conflict.",
                    winner_event_id=local.last_event_id,
                    loser_event_id=remote.id,
                ),
            )

        snapshot = self._apply_policy(local, remote, spec, relation)
        action = MergeAction.APPLY if relation is ClockRelation.AFTER else MergeAction.MERGE
        return MergeDecision(
            action,
            snapshot,
            self._evidence_for(local, remote, snapshot, spec, relation),
        )

    def _apply_policy(
        self,
        local: EntitySnapshot | None,
        remote: SyncEvent,
        spec: MergePolicySpec,
        relation: ClockRelation,
    ) -> EntitySnapshot:
        if spec.kind is MergePolicyKind.LWW_REGISTER:
            return self._lww(local, remote, spec, relation)
        if spec.kind is MergePolicyKind.OR_SET:
            return self._or_set(local, remote, spec, relation)
        if spec.kind is MergePolicyKind.LWW_MAP:
            return self._lww_map(local, remote, spec, relation)
        if spec.kind is MergePolicyKind.PN_COUNTER:
            return self._pn_counter(local, remote, spec, relation)
        raise ValueError(f"Unsupported merge policy {spec.kind}")

    def _lww(
        self,
        local: EntitySnapshot | None,
        remote: SyncEvent,
        spec: MergePolicySpec,
        relation: ClockRelation,
    ) -> EntitySnapshot:
        if local is None or relation is ClockRelation.AFTER:
            return _remote_snapshot(remote)

        joined = _join_clocks(local.vclock, remote.vclock)
        if relation is not ClockRelation.CONCURRENT:
            return local

        if spec.delete_wins and local.tombstone != remote.tombstone:
            if remote.tombstone:
                return _remote_snapshot(remote, vclock=joined)
            return EntitySnapshot(
                **{**local.__dict__, "vclock": joined}
            )

        if _remote_order_key(remote) > _snapshot_order_key(local):
            return _remote_snapshot(remote, vclock=joined)

        return EntitySnapshot(**{**local.__dict__, "vclock": joined})

    def _or_set(
        self,
        local: EntitySnapshot | None,
        remote: SyncEvent,
        spec: MergePolicySpec,
        relation: ClockRelation,
    ) -> EntitySnapshot:
        if local is not None and relation is ClockRelation.CONCURRENT:
            if spec.delete_wins and local.tombstone != remote.tombstone:
                if local.tombstone:
                    return EntitySnapshot(
                        **{**local.__dict__, "vclock": _join_clocks(local.vclock, remote.vclock)}
                    )
                if remote.tombstone:
                    return _remote_snapshot(
                        remote,
                        value={"adds": {}, "removes": []},
                        vclock=_join_clocks(local.vclock, remote.vclock),
                    )

        if remote.tombstone:
            return _remote_snapshot(
                remote,
                value={"adds": {}, "removes": []},
                vclock=_join_clocks(local.vclock, remote.vclock) if local else remote.vclock,
            )

        state = _json_copy(local.value) if local and not local.tombstone else {"adds": {}, "removes": []}
        state.setdefault("adds", {})
        state.setdefault("removes", [])
        payload = remote.to_dict()["payload"]
        element = payload.get("element")
        if not isinstance(element, str) or not element:
            raise ValueError("OR-Set operations require non-empty payload.element")

        if remote.type.endswith(".add"):
            tags = set(state["adds"].get(element, []))
            tags.add(remote.id)
            state["adds"][element] = sorted(tags)
        elif remote.type.endswith(".remove"):
            observed = payload.get("observed_tags")
            if not isinstance(observed, list) or any(not isinstance(x, str) for x in observed):
                raise ValueError("OR-Set remove requires payload.observed_tags string list")
            state["removes"] = sorted(set(state["removes"]) | set(observed))
        else:
            raise ValueError("OR-Set event type must end with .add or .remove")

        joined = _join_clocks(local.vclock, remote.vclock) if local else dict(remote.vclock)
        return EntitySnapshot(
            remote.entity_id,
            remote.entity_type,
            state,
            joined,
            remote.id,
            remote.device,
            remote.timestamp,
            False,
        )

    def _lww_map(
        self,
        local: EntitySnapshot | None,
        remote: SyncEvent,
        spec: MergePolicySpec,
        relation: ClockRelation,
    ) -> EntitySnapshot:
        if remote.tombstone:
            if (
                local is not None
                and relation is ClockRelation.CONCURRENT
                and not spec.delete_wins
                and _remote_order_key(remote) <= _snapshot_order_key(local)
            ):
                return EntitySnapshot(
                    **{**local.__dict__, "vclock": _join_clocks(local.vclock, remote.vclock)}
                )
            return _remote_snapshot(
                remote,
                value={"entries": {}},
                vclock=_join_clocks(local.vclock, remote.vclock) if local else remote.vclock,
            )

        state = _json_copy(local.value) if local and not local.tombstone else {"entries": {}}
        entries = state.setdefault("entries", {})
        payload = remote.to_dict()["payload"]
        key = payload.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError("LWW-Map operations require non-empty payload.key")

        deleting = remote.type.endswith(".delete")
        if not deleting and not remote.type.endswith(".set"):
            raise ValueError("LWW-Map event type must end with .set or .delete")
        if not deleting and "value" not in payload:
            raise ValueError("LWW-Map set requires payload.value")

        existing = entries.get(key)
        should_write = relation is ClockRelation.AFTER or existing is None
        if relation is ClockRelation.CONCURRENT and existing is not None:
            existing_key = _event_order_key(
                existing["timestamp"], existing["device"], existing["event_id"]
            )
            should_write = _remote_order_key(remote) > existing_key

        if should_write:
            entries[key] = {
                "value": None if deleting else payload.get("value"),
                "tombstone": deleting,
                "timestamp": remote.timestamp,
                "device": remote.device,
                "event_id": remote.id,
            }

        joined = _join_clocks(local.vclock, remote.vclock) if local else dict(remote.vclock)
        return EntitySnapshot(
            remote.entity_id,
            remote.entity_type,
            state,
            joined,
            remote.id if should_write else (local.last_event_id if local else remote.id),
            remote.device if should_write else (local.last_device if local else remote.device),
            remote.timestamp if should_write else (local.timestamp if local else remote.timestamp),
            False,
        )

    def _pn_counter(
        self,
        local: EntitySnapshot | None,
        remote: SyncEvent,
        spec: MergePolicySpec,
        relation: ClockRelation,
    ) -> EntitySnapshot:
        if remote.tombstone:
            raise ValueError("PN-counter entities do not accept whole-entity tombstones")
        payload = remote.to_dict()["payload"]
        value = payload.get("value")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("PN-counter component value must be a non-negative integer")

        state = _json_copy(local.value) if local else {"p": {}, "n": {}}
        state.setdefault("p", {})
        state.setdefault("n", {})
        if remote.type.endswith(".p"):
            component = "p"
        elif remote.type.endswith(".n"):
            component = "n"
        else:
            raise ValueError("PN-counter event type must end with .p or .n")

        state[component][remote.device] = max(
            int(state[component].get(remote.device, 0)), value
        )
        joined = _join_clocks(local.vclock, remote.vclock) if local else dict(remote.vclock)
        return EntitySnapshot(
            remote.entity_id,
            remote.entity_type,
            state,
            joined,
            remote.id,
            remote.device,
            remote.timestamp,
            False,
        )

    @staticmethod
    def _evidence_for(
        local: EntitySnapshot,
        remote: SyncEvent,
        result: EntitySnapshot,
        spec: MergePolicySpec,
        relation: ClockRelation,
    ) -> MergeEvidence:
        remote_won = result.last_event_id == remote.id
        if relation is ClockRelation.AFTER:
            reason = "Remote event is a causal successor."
        elif spec.kind is MergePolicyKind.LWW_REGISTER:
            reason = "Concurrent register writes resolved by delete policy then deterministic order key."
        else:
            reason = f"Concurrent operations merged under {spec.kind.value} semantics."
        return MergeEvidence(
            relation,
            spec.kind,
            reason,
            winner_event_id=result.last_event_id,
            loser_event_id=local.last_event_id if remote_won else remote.id,
        )
