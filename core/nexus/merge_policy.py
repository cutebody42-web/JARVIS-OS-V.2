"""Registry of merge policies for NEXUS semantic entities.

ConflictResolver consumes this registry; it does not infer policy from entity_id.
Unknown entity types fail closed so a new memory class cannot silently inherit
unsafe merge semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping


class MergePolicyKind(str, Enum):
    LWW_REGISTER = "lww_register"
    OR_SET = "or_set"
    LWW_MAP = "lww_map"
    PN_COUNTER = "pn_counter"


@dataclass(frozen=True)
class MergePolicySpec:
    kind: MergePolicyKind
    delete_wins: bool = True


DEFAULT_POLICIES: Mapping[str, MergePolicySpec] = MappingProxyType(
    {
        "user.preference": MergePolicySpec(MergePolicyKind.LWW_REGISTER),
        "device.last_seen": MergePolicySpec(MergePolicyKind.LWW_REGISTER, delete_wins=False),
        "memory.fact": MergePolicySpec(MergePolicyKind.LWW_REGISTER),
        "memory.handoff": MergePolicySpec(MergePolicyKind.LWW_REGISTER),
        "memory.project": MergePolicySpec(MergePolicyKind.LWW_MAP),
        "skills.learned": MergePolicySpec(MergePolicyKind.OR_SET),
        "tags": MergePolicySpec(MergePolicyKind.OR_SET),
        "devices.trusted": MergePolicySpec(MergePolicyKind.OR_SET),
        "config.ui_settings": MergePolicySpec(MergePolicyKind.LWW_MAP),
        "usage.stats": MergePolicySpec(MergePolicyKind.PN_COUNTER, delete_wins=False),
    }
)


class UnknownMergePolicyError(KeyError):
    pass


class MergePolicyRegistry:
    def __init__(self, policies: Mapping[str, MergePolicySpec] | None = None):
        source = policies or DEFAULT_POLICIES
        normalized: dict[str, MergePolicySpec] = {}
        for entity_type, spec in source.items():
            if not isinstance(entity_type, str) or not entity_type.strip():
                raise ValueError("entity_type registry keys must be non-empty strings")
            if not isinstance(spec, MergePolicySpec):
                raise TypeError("merge policy registry values must be MergePolicySpec")
            normalized[entity_type] = spec
        self._policies = MappingProxyType(normalized)

    def get(self, entity_type: str) -> MergePolicySpec:
        try:
            return self._policies[entity_type]
        except KeyError:
            raise UnknownMergePolicyError(
                f"No merge policy is registered for entity type {entity_type!r}."
            ) from None

    def as_dict(self) -> dict[str, MergePolicySpec]:
        return dict(self._policies)
