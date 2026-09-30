"""Typed persona configuration for the NEXUS agent runtime.

PersonaSpec describes identity, routing preferences, task scope, context namespace
and a capability allowlist. It cannot grant capabilities: OwnerPolicy remains the
authoritative action boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import FrozenSet, Mapping

from core.model_router import PersonaRoutingProfile, TaskKind


@dataclass(frozen=True)
class PersonaSpec:
    name: str
    system_instruction: str
    routing: PersonaRoutingProfile
    tool_allowlist: FrozenSet[str]
    default_task: TaskKind
    allowed_tasks: FrozenSet[TaskKind]
    context_namespace: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("persona name must be non-empty")
        if not isinstance(self.system_instruction, str) or not self.system_instruction.strip():
            raise ValueError("persona system_instruction must be non-empty")
        if not isinstance(self.routing, PersonaRoutingProfile):
            raise TypeError("routing must be PersonaRoutingProfile")
        if self.routing.name != self.name:
            raise ValueError("persona and routing profile names must match")
        if not isinstance(self.default_task, TaskKind):
            raise TypeError("default_task must be TaskKind")
        if self.default_task not in self.allowed_tasks:
            raise ValueError("default_task must be included in allowed_tasks")
        if not isinstance(self.context_namespace, str) or not self.context_namespace.strip():
            raise ValueError("context_namespace must be non-empty")
        if any(not isinstance(tool, str) or not tool.strip() for tool in self.tool_allowlist):
            raise ValueError("tool_allowlist must contain non-empty strings")
        object.__setattr__(self, "tool_allowlist", frozenset(self.tool_allowlist))
        object.__setattr__(self, "allowed_tasks", frozenset(self.allowed_tasks))

    def resolve_task(self, requested: TaskKind | None = None) -> TaskKind:
        task = self.default_task if requested is None else requested
        if not isinstance(task, TaskKind):
            raise TypeError("requested task must be TaskKind")
        if task not in self.allowed_tasks:
            raise ValueError(
                f"Task {task.value!r} is not allowed for persona {self.name!r}"
            )
        return task

    def allows_tool(self, tool: str) -> bool:
        return isinstance(tool, str) and tool in self.tool_allowlist


class PersonaRegistry:
    def __init__(self, personas: Mapping[str, PersonaSpec]):
        normalized: dict[str, PersonaSpec] = {}
        for name, persona in personas.items():
            if not isinstance(persona, PersonaSpec):
                raise TypeError("registry values must be PersonaSpec")
            if name != persona.name:
                raise ValueError("registry key must match persona.name")
            if name in normalized:
                raise ValueError(f"duplicate persona {name!r}")
            normalized[name] = persona
        if not normalized:
            raise ValueError("persona registry cannot be empty")
        self._personas = MappingProxyType(normalized)

    def get(self, name: str) -> PersonaSpec:
        try:
            return self._personas[name]
        except KeyError:
            raise KeyError(f"Unknown persona {name!r}") from None

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._personas))
