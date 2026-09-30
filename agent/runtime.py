"""Persona-aware NEXUS agent runtime.

TABY remains the visible identity while FRIDAY/JARVIS are selectable operational
profiles. Persona switching changes routing, system guidance, tool scope and
session context without restarting the process.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable

from agent.executor import AgentExecutor
from core.adaptive_provider import AdaptivePersonaProvider
from core.hardware_profile import HardwareProfiler
from core.model_router import ModelRouter, TaskKind
from core.model_runtime import ModelRuntime
from core.personas import BUILTIN_PERSONAS, PersonaRegistry, PersonaSpec


@dataclass(frozen=True)
class RuntimeTurn:
    goal: str
    result: str


class PersonaContextStore:
    def __init__(self, *, max_turns: int = 8, max_chars: int = 4000):
        if max_turns <= 0 or max_chars <= 0:
            raise ValueError("context bounds must be positive")
        self._max_turns = max_turns
        self._max_chars = max_chars
        self._turns: dict[str, deque[RuntimeTurn]] = {}

    def append(self, namespace: str, goal: str, result: str) -> None:
        turns = self._turns.setdefault(namespace, deque(maxlen=self._max_turns))
        turns.append(RuntimeTurn(str(goal), str(result)))

    def render(self, namespace: str) -> str:
        turns = self._turns.get(namespace)
        if not turns:
            return ""
        text = "\n".join(
            f"User goal: {turn.goal}\nVerified runtime result: {turn.result}"
            for turn in turns
        )
        return text[-self._max_chars :]

    def clear(self, namespace: str) -> None:
        self._turns.pop(namespace, None)


class AgentRuntime:
    def __init__(
        self,
        *,
        hardware_profiler: HardwareProfiler,
        model_runtime: ModelRuntime,
        router: ModelRouter | None = None,
        personas: PersonaRegistry = BUILTIN_PERSONAS,
        awareness=None,
        context_store: PersonaContextStore | None = None,
        provider_factory: Callable[..., AdaptivePersonaProvider] | None = None,
    ):
        self._hardware = hardware_profiler
        self._model_runtime = model_runtime
        self._router = router or ModelRouter()
        self._personas = personas
        self._awareness = awareness
        self._contexts = context_store or PersonaContextStore()
        self._provider_factory = provider_factory or AdaptivePersonaProvider
        self._active_persona = "taby"

    @property
    def active_persona(self) -> PersonaSpec:
        return self._personas.get(self._active_persona)

    def switch_persona(self, name: str) -> PersonaSpec:
        persona = self._personas.get(name)
        self._active_persona = persona.name
        return persona

    def build_executor(
        self,
        *,
        persona_name: str | None = None,
        task: TaskKind | None = None,
        owner_runtime=None,
    ) -> AgentExecutor:
        """Build an isolated executor for queue/background use."""
        persona = (
            self._personas.get(persona_name)
            if persona_name is not None
            else self.active_persona
        )
        resolved_task = persona.resolve_task(task)
        provider = self._provider_factory(
            persona=persona,
            task=resolved_task,
            hardware_profiler=self._hardware,
            model_runtime=self._model_runtime,
            router=self._router,
        )
        return AgentExecutor(
            awareness=self._awareness,
            provider=provider,
            persona=persona,
            owner_runtime=owner_runtime,
        )

    def execute(
        self,
        goal: str,
        *,
        persona_name: str | None = None,
        task: TaskKind | None = None,
        speak=None,
        cancel_flag=None,
    ) -> str:
        persona = (
            self._personas.get(persona_name)
            if persona_name is not None
            else self.active_persona
        )
        resolved_task = persona.resolve_task(task)
        executor = self.build_executor(
            persona_name=persona.name,
            task=resolved_task,
        )
        context = self._contexts.render(persona.context_namespace)
        result = executor.execute(
            goal,
            speak=speak,
            cancel_flag=cancel_flag,
            context=context,
        )
        self._contexts.append(persona.context_namespace, goal, result)
        return result
