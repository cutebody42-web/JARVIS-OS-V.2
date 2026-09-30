"""Offline integration tests for NEXUS personas and AgentRuntime."""

from datetime import datetime, timezone
import json
import unittest

from agent.executor import AgentExecutor
from agent.runtime import AgentRuntime
from core.action_contracts import ActionStatus
from core.hardware_profile import (
    GPUMemoryKind,
    HardwareSnapshot,
    PowerSource,
)
from core.model_provider import ModelRequest, ModelResponse, ModelTier
from core.model_router import ProviderKind, TaskKind
from core.model_runtime import ModelRuntimeError, RuntimeStatus
from core.personas import BUILTIN_PERSONAS, FRIDAY, JARVIS, TABY
from core.routed_provider import PersonaModelProvider


def snapshot():
    return HardwareSnapshot(
        device_id="test-node",
        total_ram_gb=16.0,
        available_ram_gb=8.0,
        cpu_percent=20.0,
        cpu_count=12,
        gpu_vram_gb=None,
        gpu_memory_kind=GPUMemoryKind.SHARED,
        power_source=PowerSource.AC,
        battery_pct=90,
        system_pressure=0.25,
        loaded_models={},
        timestamp=datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc),
    )


class FakeProfiler:
    def __init__(self, *, fail=False):
        self.fail = fail

    def capture(self):
        if self.fail:
            raise RuntimeError("probe failed")
        return snapshot()


class FakeRuntime:
    def __init__(self, *, fail_ensure=False, fail_status=False):
        self.fail_ensure = fail_ensure
        self.fail_status = fail_status
        self.ensure_calls = []

    def get_status(self):
        if self.fail_status:
            raise ModelRuntimeError("status unavailable")
        return RuntimeStatus(models={}, warnings=())

    def ensure(self, model, priority, *, keep_alive=None):
        self.ensure_calls.append((model, priority, keep_alive))
        if self.fail_ensure:
            raise ModelRuntimeError("load failed")
        return object()


class ChoiceProvider:
    def __init__(self, choice, *, plan_tool=None):
        self.choice = choice
        self.plan_tool = plan_tool
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        if self.plan_tool is not None and request.json_output:
            return ModelResponse(
                json.dumps(
                    {
                        "goal": "test",
                        "steps": [
                            {
                                "step": 1,
                                "tool": self.plan_tool,
                                "description": "test tool",
                                "parameters": {"city": "Cairo"}
                                if self.plan_tool == "weather_report"
                                else {},
                                "critical": True,
                            }
                        ],
                    }
                ),
                self.choice.provider.value,
                self.choice.model,
            )
        return ModelResponse(
            self.choice.model,
            self.choice.provider.value,
            self.choice.model,
        )


class PersonaRuntimeTests(unittest.TestCase):
    def test_builtin_personas_are_consistent(self):
        self.assertEqual(set(BUILTIN_PERSONAS), {"taby", "friday", "jarvis"})
        self.assertEqual(TABY.routing.name, TABY.name)
        self.assertEqual(FRIDAY.routing.name, FRIDAY.name)
        self.assertEqual(JARVIS.routing.name, JARVIS.name)
        self.assertEqual(FRIDAY.default_task, TaskKind.REALTIME)
        self.assertEqual(JARVIS.default_task, TaskKind.CODING)

    def test_same_request_routes_friday_and_jarvis_to_different_models(self):
        request = ModelRequest("design a solution", tier=ModelTier.FAST)

        friday = PersonaModelProvider(
            persona=FRIDAY,
            task=FRIDAY.default_task,
            hardware_profiler=FakeProfiler(),
            model_runtime=FakeRuntime(),
            provider_factory=lambda choice: ChoiceProvider(choice),
        )
        jarvis = PersonaModelProvider(
            persona=JARVIS,
            task=JARVIS.default_task,
            hardware_profiler=FakeProfiler(),
            model_runtime=FakeRuntime(),
            provider_factory=lambda choice: ChoiceProvider(choice),
        )

        friday_response = friday.generate(request)
        jarvis_response = jarvis.generate(request)

        self.assertEqual(friday_response.model, "qwen3.5:4b")
        self.assertEqual(jarvis_response.model, "qwen2.5-coder:7b")
        self.assertNotEqual(friday_response.model, jarvis_response.model)

    def test_persona_instruction_is_composed_with_caller_instruction(self):
        captured = []

        class CapturingProvider:
            def __init__(self, choice):
                self.choice = choice

            def generate(self, request):
                captured.append(request)
                return ModelResponse("ok", self.choice.provider.value, self.choice.model)

        provider = PersonaModelProvider(
            persona=FRIDAY,
            task=TaskKind.REALTIME,
            hardware_profiler=FakeProfiler(),
            model_runtime=FakeRuntime(),
            provider_factory=CapturingProvider,
        )
        provider.generate(
            ModelRequest(
                "hello",
                system_instruction="Planner contract.",
                tier=ModelTier.FAST,
            )
        )

        system = captured[0].system_instruction
        self.assertIn(FRIDAY.system_instruction, system)
        self.assertIn("Planner contract.", system)

    def test_local_load_failures_use_precomputed_cloud_fallback(self):
        runtime = FakeRuntime(fail_ensure=True)
        provider = PersonaModelProvider(
            persona=JARVIS,
            task=TaskKind.CODING,
            hardware_profiler=FakeProfiler(),
            model_runtime=runtime,
            provider_factory=lambda choice: ChoiceProvider(choice),
        )

        response = provider.generate(
            ModelRequest("code", tier=ModelTier.STANDARD)
        )

        self.assertEqual(provider.last_choice.provider, ProviderKind.GEMINI)
        self.assertEqual(response.provider, "gemini")
        self.assertGreaterEqual(len(runtime.ensure_calls), 1)
        self.assertTrue(any(not attempt.success for attempt in provider.last_attempts))

    def test_hardware_probe_failure_fails_safe_to_cloud(self):
        provider = PersonaModelProvider(
            persona=TABY,
            task=TaskKind.GENERAL,
            hardware_profiler=FakeProfiler(fail=True),
            model_runtime=FakeRuntime(),
            provider_factory=lambda choice: ChoiceProvider(choice),
        )

        response = provider.generate(ModelRequest("hello", tier=ModelTier.FAST))

        self.assertEqual(response.provider, "gemini")
        self.assertEqual(provider.last_plan.primary.reason, "hardware_profile_unavailable")

    def test_persona_allowlist_can_only_reduce_owner_policy(self):
        class PlanProvider:
            def generate(self, request):
                return ModelResponse(
                    json.dumps(
                        {
                            "goal": "weather",
                            "steps": [
                                {
                                    "step": 1,
                                    "tool": "weather_report",
                                    "description": "read weather",
                                    "parameters": {"city": "Cairo"},
                                    "critical": True,
                                }
                            ],
                        }
                    ),
                    "fake",
                    "fake",
                )

        executor = AgentExecutor(provider=PlanProvider(), persona=JARVIS)
        message = executor.execute("tell me the weather in Cairo")

        self.assertEqual(executor.last_status, ActionStatus.DENIED)
        self.assertEqual(
            executor.last_action_receipts[-1].result.error_code,
            "persona_denied",
        )
        self.assertIn("denied", message.lower())

    def test_agent_runtime_switches_persona_without_restart(self):
        runtime = AgentRuntime(
            hardware_profiler=FakeProfiler(),
            model_runtime=FakeRuntime(),
            provider_factory=lambda choice: ChoiceProvider(choice),
        )
        self.assertEqual(runtime.active_persona.name, "taby")
        self.assertEqual(runtime.switch_persona("friday").name, "friday")
        self.assertEqual(runtime.active_persona.name, "friday")
        self.assertEqual(runtime.switch_persona("jarvis").name, "jarvis")

    def test_persona_rejects_disallowed_task_kind(self):
        with self.assertRaises(ValueError):
            FRIDAY.resolve_task(TaskKind.CODING)


if __name__ == "__main__":
    unittest.main()
