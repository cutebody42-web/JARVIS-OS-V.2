"""Persona-aware adaptive provider execution for NEXUS.

The ModelRouter decides; this layer executes its RoutePlan in order. It does
not authorize tools or actions. Local lifecycle operations are delegated to
ModelRuntime, and cloud/local provider failures fall through to the next
precomputed choice without rerouting on stale hardware data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from core.hardware_profile import HardwareProfiler
from core.model_provider import ModelProvider, ModelRequest, ModelResponse, ModelTier
from core.model_router import (
    ModelRouter,
    ProviderChoice,
    ProviderKind,
    RoutePlan,
    TaskKind,
)
from core.model_runtime import ModelRuntime, ModelRuntimeError, RuntimeStatus
from core.personas.persona_spec import PersonaSpec
from core.providers.gemini import GeminiProvider
from core.providers.ollama import OllamaProvider


class RoutedGenerationError(RuntimeError):
    """All choices in a precomputed route plan failed safely."""


@dataclass(frozen=True)
class RouteAttempt:
    provider: ProviderKind
    model: str
    success: bool
    error_code: str = ""


ProviderFactory = Callable[[ProviderChoice], ModelProvider]


class PersonaModelProvider:
    """ModelProvider-compatible executor for one persona/task invocation."""

    def __init__(
        self,
        *,
        persona: PersonaSpec,
        task: TaskKind,
        hardware_profiler: HardwareProfiler,
        model_runtime: ModelRuntime,
        router: ModelRouter | None = None,
        provider_factory: ProviderFactory | None = None,
    ):
        if not isinstance(persona, PersonaSpec):
            raise TypeError("persona must be PersonaSpec")
        self.persona = persona
        self.task = persona.resolve_task(task)
        self._profiler = hardware_profiler
        self._runtime = model_runtime
        self._router = router or ModelRouter()
        self._provider_factory = provider_factory or self._default_provider_factory
        self.last_plan: RoutePlan | None = None
        self.last_choice: ProviderChoice | None = None
        self.last_attempts: tuple[RouteAttempt, ...] = ()

    @staticmethod
    def _default_provider_factory(choice: ProviderChoice) -> ModelProvider:
        if choice.provider is ProviderKind.OLLAMA:
            return OllamaProvider(
                fast_model=choice.model,
                standard_model=choice.model,
                keep_alive=choice.keep_alive,
            )
        if choice.provider is ProviderKind.GEMINI:
            return GeminiProvider(
                fast_model=choice.model,
                standard_model=choice.model,
            )
        raise ValueError(f"Unsupported provider kind: {choice.provider}")

    def _effective_request(self, request: ModelRequest) -> ModelRequest:
        system = self.persona.system_instruction.strip()
        if request.system_instruction.strip():
            system = f"{system}\n\n{request.system_instruction.strip()}"
        return ModelRequest(
            prompt=request.prompt,
            system_instruction=system,
            tier=request.tier,
            json_output=request.json_output,
        )

    def _cloud_only_plan(self, request: ModelRequest, reason: str) -> RoutePlan:
        model = (
            self.persona.routing.cloud_fast_model
            if request.tier is ModelTier.FAST
            else self.persona.routing.cloud_standard_model
        )
        return RoutePlan(
            primary=ProviderChoice(
                provider=ProviderKind.GEMINI,
                model=model,
                reason=reason,
            )
        )

    def _build_plan(self, request: ModelRequest) -> RoutePlan:
        try:
            hw = self._profiler.capture()
        except Exception:
            return self._cloud_only_plan(request, "hardware_profile_unavailable")

        try:
            status = self._runtime.get_status()
        except ModelRuntimeError:
            status = RuntimeStatus(models={}, warnings=("runtime_status_unavailable",))

        return self._router.route(
            request,
            hw,
            status,
            self.persona.routing,
            task=self.task,
        )

    def generate(self, request: ModelRequest) -> ModelResponse:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be ModelRequest")

        plan = self._build_plan(request)
        self.last_plan = plan
        effective = self._effective_request(request)
        attempts: list[RouteAttempt] = []

        for choice in plan.choices:
            try:
                if choice.provider is ProviderKind.OLLAMA and choice.requires_ensure:
                    self._runtime.ensure(
                        choice.model,
                        choice.ensure_priority,
                        keep_alive=choice.keep_alive,
                    )
                provider = self._provider_factory(choice)
                response = provider.generate(effective)
                attempts.append(RouteAttempt(choice.provider, choice.model, True))
                self.last_choice = choice
                self.last_attempts = tuple(attempts)
                return response
            except Exception as exc:
                attempts.append(
                    RouteAttempt(
                        choice.provider,
                        choice.model,
                        False,
                        type(exc).__name__,
                    )
                )

        self.last_choice = None
        self.last_attempts = tuple(attempts)
        raise RoutedGenerationError("All routed model choices failed.")
