"""Adaptive provider composition for a trusted PersonaSpec.

Implements the existing ModelProvider protocol so Planner/Executor can adopt
hardware-aware routing without importing provider SDK details.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import threading
from typing import Callable

from core.hardware_profile import HardwareProfiler
from core.model_provider import ModelProvider, ModelRequest, ModelResponse, ModelTier
from core.model_router import ModelRouter, ProviderChoice, ProviderKind, RoutePlan, TaskKind
from core.model_runtime import ModelRuntime, ModelRuntimeError, RuntimeStatus
from core.personas.persona_spec import PersonaSpec


class RoutedProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class RouteAttempt:
    provider: ProviderKind
    model: str
    reason: str
    outcome: str


ProviderFactory = Callable[[ProviderChoice], ModelProvider]


class RoutedModelProvider:
    def __init__(
        self,
        persona: PersonaSpec,
        *,
        task: TaskKind | None = None,
        profiler: HardwareProfiler | None = None,
        runtime: ModelRuntime | None = None,
        router: ModelRouter | None = None,
        ollama_factory: ProviderFactory | None = None,
        gemini_factory: ProviderFactory | None = None,
        ollama_base_url: str | None = None,
        allow_cloud: bool = True,
    ):
        if not isinstance(persona, PersonaSpec):
            raise TypeError("persona must be PersonaSpec")
        self.persona = persona
        self.task = persona.validate_task(task)
        self._ollama_base_url = (
            ollama_base_url
            or os.environ.get("NEXUS_OLLAMA_URL")
            or os.environ.get("OLLAMA_HOST")
            or "http://127.0.0.1:11434"
        )
        self.profiler = profiler or HardwareProfiler(ollama_base_url=self._ollama_base_url)
        self.runtime = runtime or ModelRuntime(ollama_base_url=self._ollama_base_url)
        self.router = router or ModelRouter()
        self._ollama_factory = ollama_factory or self._default_ollama_factory
        self._gemini_factory = gemini_factory or self._default_gemini_factory
        self.allow_cloud = bool(allow_cloud)
        self._lock = threading.Lock()
        self._last_attempts: tuple[RouteAttempt, ...] = ()

    @property
    def last_attempts(self) -> tuple[RouteAttempt, ...]:
        with self._lock:
            return self._last_attempts

    def _record_attempts(self, attempts: list[RouteAttempt]) -> None:
        with self._lock:
            self._last_attempts = tuple(attempts)

    def _default_ollama_factory(self, choice: ProviderChoice) -> ModelProvider:
        from core.providers.ollama import OllamaProvider
        return OllamaProvider(
            choice.model,
            base_url=self._ollama_base_url,
            keep_alive=choice.keep_alive,
        )

    @staticmethod
    def _default_gemini_factory(choice: ProviderChoice) -> ModelProvider:
        from core.providers.gemini import GeminiProvider
        return GeminiProvider(fast_model=choice.model, standard_model=choice.model)

    def _with_persona_instruction(self, request: ModelRequest) -> ModelRequest:
        application = request.system_instruction.strip()
        persona = self.persona.system_instruction.strip()
        combined = (
            "APPLICATION CONTRACT (must be preserved):\n"
            + application
            + "\n\nTRUSTED PERSONA PROFILE:\n"
            + persona
            if application
            else persona
        )
        return ModelRequest(
            prompt=request.prompt,
            system_instruction=combined,
            tier=request.tier,
            json_output=request.json_output,
        )

    def _cloud_only_plan(self, request: ModelRequest) -> RoutePlan:
        if not self.allow_cloud:
            raise RoutedProviderError(
                "Local hardware state is unavailable and cloud routing is disabled."
            )
        model = (
            self.persona.routing.cloud_fast_model
            if request.tier is ModelTier.FAST
            else self.persona.routing.cloud_standard_model
        )
        return RoutePlan(
            ProviderChoice(
                provider=ProviderKind.GEMINI,
                model=model,
                reason="hardware_snapshot_unavailable",
            )
        )

    def _plan(self, request: ModelRequest) -> RoutePlan:
        try:
            hw = self.profiler.capture()
        except Exception:
            return self._cloud_only_plan(request)

        try:
            runtime_status = self.runtime.get_status()
        except ModelRuntimeError:
            runtime_status = RuntimeStatus(
                models={},
                warnings=("runtime_status_unavailable",),
            )

        return self.router.route(
            request,
            hw,
            runtime_status,
            self.persona.routing,
            task=self.task,
        )

    def generate(self, request: ModelRequest) -> ModelResponse:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be ModelRequest")

        routed_request = self._with_persona_instruction(request)
        plan = self._plan(routed_request)
        attempts: list[RouteAttempt] = []

        for choice in plan.choices:
            if choice.provider is ProviderKind.GEMINI and not self.allow_cloud:
                attempts.append(
                    RouteAttempt(
                        choice.provider,
                        choice.model,
                        choice.reason,
                        "skipped:cloud_disabled",
                    )
                )
                continue
            try:
                if choice.provider is ProviderKind.OLLAMA:
                    if choice.requires_ensure:
                        self.runtime.ensure(
                            choice.model,
                            choice.ensure_priority,
                            keep_alive=choice.keep_alive,
                        )
                    provider = self._ollama_factory(choice)
                elif choice.provider is ProviderKind.GEMINI:
                    provider = self._gemini_factory(choice)
                else:
                    raise RoutedProviderError("Unsupported routed provider")

                response = provider.generate(routed_request)
                if not isinstance(response, ModelResponse):
                    raise RoutedProviderError("Provider returned an invalid response")
                attempts.append(RouteAttempt(choice.provider, choice.model, choice.reason, "succeeded"))
                self._record_attempts(attempts)
                return response
            except Exception as exc:
                attempts.append(
                    RouteAttempt(
                        choice.provider,
                        choice.model,
                        choice.reason,
                        f"failed:{type(exc).__name__}",
                    )
                )

        self._record_attempts(attempts)
        if not self.allow_cloud:
            raise RoutedProviderError(
                "No local JARVIS Brain route completed the request; cloud routing is disabled."
            )
        raise RoutedProviderError("No routed model provider completed the request.")
