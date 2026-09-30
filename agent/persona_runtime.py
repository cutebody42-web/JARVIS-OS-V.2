"""One agent runtime, multiple NEXUS personas.

TABY/FRIDAY/JARVIS are PersonaSpec instances, not duplicated agent stacks.
Switching persona changes routing, system instruction, tool ceiling and context
namespace without restarting the runtime.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable

from agent.executor import AgentExecutor
from core.adaptive_provider import AdaptivePersonaProvider
from core.hardware_profile import HardwareProfiler
from core.model_router import ModelRouter, TaskKind
from core.model_runtime import ModelRuntime
from core.personas import DEFAULT_PERSONAS, PersonaRegistry, PersonaSpec


@dataclass(frozen=True)
class PersonaExecution:
    persona: str
    task: TaskKind
    message: str
    provider: str | None
    model: str | None


class PersonaContextStore:
    """Small per-persona session context; durable memory remains a separate layer."""

    def __init__(self, max_entries: int = 6):
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        self._max_entries = max_entries
        self._items = defaultdict(lambda: deque(maxlen=self._max_entries))

    def append(self, namespace: str, goal: str, result: str) -> None:
        self._items[namespace].append((str(goal), str(result)))

    def render(self, namespace: str) -> str:
        items = tuple(self._items.get(namespace, ()))
        if not items:
            return ""
        return "\n".join(
            f"Previous owner goal: {goal}\nPrevious verified/runtime result: {result}"
            for goal, result in items
        )

    def clear(self, namespace: str) -> None:
        self._items.pop(namespace, None)


class PersonaAgentRuntime:
    def __init__(
        self,
        *,
        personas: PersonaRegistry = DEFAULT_PERSONAS,
        router: ModelRouter | None = None,
        hardware_profiler: HardwareProfiler | None = None,
        model_runtime: ModelRuntime | None = None,
        context_store: PersonaContextStore | None = None,
        provider_builder=None,
        initial_persona: str = "taby",
    ):
        self.personas = personas
        self.router = router or ModelRouter()
        self.hardware_profiler = hardware_profiler or HardwareProfiler()
        self.model_runtime = model_runtime or ModelRuntime()
        self.context_store = context_store or PersonaContextStore()
        self.provider_builder = provider_builder
        self._active_persona = self.personas.get(initial_persona)
        self.last_executor: AgentExecutor | None = None
        self.last_execution: PersonaExecution | None = None

    @property
    def active_persona(self) -> PersonaSpec:
        return self._active_persona

    def switch_persona(self, name: str) -> PersonaSpec:
        self._active_persona = self.personas.get(name)
        return self._active_persona

    def execute(
        self,
        goal: str,
        *,
        persona: str | None = None,
        task: TaskKind | None = None,
        speak: Callable | None = None,
        cancel_flag=None,
    ) -> PersonaExecution:
        spec = self.personas.get(persona) if persona is not None else self._active_persona
        selected_task = task or spec.default_task
        if not spec.allows_task(selected_task):
            raise ValueError(
                f"Persona {spec.name!r} does not allow task kind {selected_task.value!r}"
            )

        provider = AdaptivePersonaProvider(
            persona=spec,
            task=selected_task,
            router=self.router,
            hardware_profiler=self.hardware_profiler,
            model_runtime=self.model_runtime,
            provider_builder=self.provider_builder,
        )
        context = self.context_store.render(spec.context_namespace)
        executor = AgentExecutor(
            provider=provider,
            allowed_tools=spec.tool_allowlist,
            context=context,
        )
        self.last_executor = executor
        message = executor.execute(
            str(goal),
            speak=speak,
            cancel_flag=cancel_flag,
        )
        self.context_store.append(spec.context_namespace, str(goal), message)

        choice = provider.last_choice
        execution = PersonaExecution(
            persona=spec.name,
            task=selected_task,
            message=message,
            provider=choice.provider.value if choice is not None else None,
            model=choice.model if choice is not None else None,
        )
        self.last_execution = execution
        return execution
