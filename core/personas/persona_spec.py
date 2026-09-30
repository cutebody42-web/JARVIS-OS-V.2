"""Typed persona contracts for the NEXUS agent runtime."""

from __future__ import annotations

from dataclasses import dataclass
import re

from core.model_router import PersonaRoutingProfile, TaskKind


_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_NAMESPACE_RE = re.compile(r"^[a-z][a-z0-9_.:-]{0,63}$")


@dataclass(frozen=True)
class PersonaSpec:
    """One behavioral/routing/tool policy slice under the TABY identity.

    OwnerPolicy remains the final security boundary. tool_allowlist can only
    reduce what a persona may request; it can never grant a capability that
    OwnerPolicy does not admit.
    """

    name: str
    display_name: str
    system_instruction: str
    routing: PersonaRoutingProfile
    tool_allowlist: frozenset[str]
    default_task: TaskKind
    allowed_tasks: frozenset[TaskKind]
    context_namespace: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _NAME_RE.fullmatch(self.name):
            raise ValueError("persona name must be a safe lowercase identifier")
        if not isinstance(self.display_name, str) or not self.display_name.strip():
            raise ValueError("display_name must be non-empty")
        if not isinstance(self.system_instruction, str) or not self.system_instruction.strip():
            raise ValueError("system_instruction must be non-empty")
        if not isinstance(self.routing, PersonaRoutingProfile):
            raise TypeError("routing must be PersonaRoutingProfile")
        if self.routing.name != self.name:
            raise ValueError("routing profile name must match persona name")
        tools = frozenset(self.tool_allowlist)
        if any(not isinstance(tool, str) or not tool.strip() for tool in tools):
            raise ValueError("tool_allowlist entries must be non-empty strings")
        tasks = frozenset(self.allowed_tasks)
        if not tasks:
            raise ValueError("allowed_tasks cannot be empty")
        if any(not isinstance(task, TaskKind) for task in tasks):
            raise TypeError("allowed_tasks must contain TaskKind values")
        if self.default_task not in tasks:
            raise ValueError("default_task must be included in allowed_tasks")
        if not isinstance(self.context_namespace, str) or not _NAMESPACE_RE.fullmatch(
            self.context_namespace
        ):
            raise ValueError("context_namespace must be a safe identifier")
        object.__setattr__(self, "tool_allowlist", tools)
        object.__setattr__(self, "allowed_tasks", tasks)

    def resolve_task(self, task: TaskKind | None = None) -> TaskKind:
        resolved = self.default_task if task is None else task
        if not isinstance(resolved, TaskKind):
            raise TypeError("task must be TaskKind")
        if resolved not in self.allowed_tasks:
            raise ValueError(
                f"Persona {self.name!r} does not allow task kind {resolved.value!r}."
            )
        return resolved

    def allows_tool(self, tool: str) -> bool:
        return isinstance(tool, str) and tool in self.tool_allowlist
