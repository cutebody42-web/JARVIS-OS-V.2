"""Trusted persona configuration for the NEXUS cognition layer.

Personas can only *restrict* behavior. They do not grant action authority.
OwnerPolicy/OwnerKernel remains the authoritative capability boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from core.model_router import PersonaRoutingProfile, TaskKind


_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_TOOL_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,120}$")


@dataclass(frozen=True)
class PersonaSpec:
    name: str
    system_instruction: str
    routing: PersonaRoutingProfile
    tool_allowlist: frozenset[str]
    default_task: TaskKind
    allowed_tasks: frozenset[TaskKind]
    context_namespace: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _NAME_RE.fullmatch(self.name):
            raise ValueError("persona name must be a safe lowercase identifier")
        if not isinstance(self.system_instruction, str) or not self.system_instruction.strip():
            raise ValueError("persona system_instruction must be non-empty")
        if len(self.system_instruction) > 12000:
            raise ValueError("persona system_instruction exceeds the configured budget")
        if not isinstance(self.routing, PersonaRoutingProfile):
            raise TypeError("routing must be PersonaRoutingProfile")
        if self.routing.name != self.name:
            raise ValueError("persona and routing profile names must match")

        allowlist = frozenset(self.tool_allowlist)
        if any(not isinstance(tool, str) or not _TOOL_RE.fullmatch(tool) for tool in allowlist):
            raise ValueError("persona tool allowlist contains an invalid tool id")
        object.__setattr__(self, "tool_allowlist", allowlist)

        tasks = frozenset(self.allowed_tasks)
        if not tasks or any(not isinstance(task, TaskKind) for task in tasks):
            raise ValueError("allowed_tasks must contain TaskKind values")
        if self.default_task not in tasks:
            raise ValueError("default_task must be included in allowed_tasks")
        object.__setattr__(self, "allowed_tasks", tasks)

        if not isinstance(self.context_namespace, str) or not _NAME_RE.fullmatch(self.context_namespace):
            raise ValueError("context_namespace must be a safe lowercase identifier")

    def allows_tool(self, tool: str) -> bool:
        return tool in self.tool_allowlist

    def validate_task(self, task: TaskKind | None = None) -> TaskKind:
        resolved = self.default_task if task is None else task
        if not isinstance(resolved, TaskKind) or resolved not in self.allowed_tasks:
            raise ValueError(f"Task kind is not admitted by persona {self.name!r}")
        return resolved
