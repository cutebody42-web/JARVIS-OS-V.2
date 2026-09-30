"""Immutable persona contracts for the NEXUS runtime.

PersonaSpec never grants authority. Tool allowlists only restrict what a persona
may ask the action kernel to do; OwnerPolicy remains the final application-owned
authorization boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from core.model_router import PersonaRoutingProfile, TaskKind


@dataclass(frozen=True)
class PersonaSpec:
    name: str
    display_name: str
    system_instruction: str
    routing: PersonaRoutingProfile
    tool_allowlist: frozenset[str]
    default_task: TaskKind
    allowed_tasks: frozenset[TaskKind]
    context_namespace: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("name", self.name),
            ("display_name", self.display_name),
            ("system_instruction", self.system_instruction),
            ("context_namespace", self.context_namespace),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if not isinstance(self.routing, PersonaRoutingProfile):
            raise TypeError("routing must be PersonaRoutingProfile")
        if self.routing.name != self.name:
            raise ValueError("PersonaSpec name must match routing profile name")
        if not isinstance(self.default_task, TaskKind):
            raise TypeError("default_task must be TaskKind")
        if not self.allowed_tasks or any(
            not isinstance(task, TaskKind) for task in self.allowed_tasks
        ):
            raise TypeError("allowed_tasks must be a non-empty set of TaskKind")
        if self.default_task not in self.allowed_tasks:
            raise ValueError("default_task must be included in allowed_tasks")
        if any(not isinstance(tool, str) or not tool.strip() for tool in self.tool_allowlist):
            raise ValueError("tool_allowlist must contain non-empty tool names")
        object.__setattr__(self, "tool_allowlist", frozenset(self.tool_allowlist))
        object.__setattr__(self, "allowed_tasks", frozenset(self.allowed_tasks))

    def allows_task(self, task: TaskKind) -> bool:
        return task in self.allowed_tasks

    def allows_tool(self, tool: str) -> bool:
        return tool in self.tool_allowlist


class PersonaRegistry:
    def __init__(self, personas: Mapping[str, PersonaSpec]):
        normalized: dict[str, PersonaSpec] = {}
        for key, spec in personas.items():
            if not isinstance(spec, PersonaSpec):
                raise TypeError("PersonaRegistry values must be PersonaSpec")
            if key != spec.name:
                raise ValueError("PersonaRegistry keys must match PersonaSpec.name")
            normalized[key] = spec
        if not normalized:
            raise ValueError("PersonaRegistry requires at least one persona")
        self._personas = MappingProxyType(normalized)

    def get(self, name: str) -> PersonaSpec:
        try:
            return self._personas[name]
        except KeyError:
            raise KeyError(f"Unknown persona: {name}") from None

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._personas))

    def as_dict(self) -> dict[str, PersonaSpec]:
        return dict(self._personas)
