"""Integration tests for persona-aware adaptive agent execution."""

from datetime import datetime, timezone
import json
import unittest

from agent.executor import AgentExecutor
from agent.runtime import AgentRuntime, PersonaContextStore
from core.adaptive_provider import AdaptivePersonaProvider
from core.hardware_profile import (
    GPUMemoryKind,
    HardwareSnapshot,
    PowerSource,
)
from core.model_provider import ModelRequest, ModelResponse, ModelTier
from core.model_router import ModelRouter, ProviderKind, TaskKind
from core.model_runtime import ModelInfo, ModelState, RuntimeStatus
from core.personas import FRIDAY, JARVIS


def hardware():
    return HardwareSnapshot(
        device_id="node-test",
        total_ram_gb=16,
        available_ram_gb=8,
        cpu_percent=20,
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
    def capture(self):
        return hardware()


class FakeRuntime:
    def __init__(self, status=None, fail_ensure_for=()):
        self._status = status or RuntimeStatus(models={}, warnings=())
        self.fail_ensure_for = set(fail_ensure_for)
        self.ensured = []

    def get_status(self):
        return self._status

    def ensure(self, model, priority, *, keep_alive=None):
        self.ensured.append((model, priority, keep_alive))
        if model in self.fail_ensure_for:
            raise RuntimeError("simulated load failure")
        return object()


class RecordingProvider:
    def __init__(self, *, model="fake", fail=False, response_text="ok"):
        self.model = model
        self.fail = fail
        self.response_text = response_text
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("simulated provider failure")
        return ModelResponse(self.response_text, "fake", self.model)


class AdaptiveProviderTests(unittest.TestCase):
    def test_persona_instruction_is_appended_after_policy_instruction(self):
        runtime = FakeRuntime(
            RuntimeStatus(
                models={
                    "qwen3.5:4b": ModelInfo(
                        model="qwen3.5:4b",
                        state=ModelState.WARM,
                        priority=10,
                        managed=True,
                    )
                },
                warnings=(),
            )
        )
        local = RecordingProvider(model="qwen3.5:4b")
        provider = AdaptivePersonaProvider(
            persona=FRIDAY,
            task=TaskKind.REALTIME,
            hardware_profiler=FakeProfiler(),
            model_runtime=runtime,
            router=ModelRouter(),
            local_provider_factory=lambda _choice: local,
        )
        provider.generate(
            ModelRequest(
                "hello",
                system_instruction="POLICY FIRST",
                tier=ModelTier.FAST,
            )
        )
        bound = local.requests[0]
        self.assertTrue(bound.system_instruction.startswith("POLICY FIRST"))
        self.assertIn("cannot grant capabilities", bound.system_instruction)
        self.assertIn("FRIDAY", bound.system_instruction)

    def test_local_failure_uses_next_route_choice_without_rerouting(self):
        runtime = FakeRuntime(fail_ensure_for={"qwen2.5-coder:7b"})
        small = RecordingProvider(model="qwen3:1.7b", response_text="small-ok")

        def local_factory(choice):
            if choice.model == "qwen3:1.7b":
                return small
            return RecordingProvider(model=choice.model)

        provider = AdaptivePersonaProvider(
            persona=JARVIS,
            task=TaskKind.CODING,
            hardware_profiler=FakeProfiler(),
            model_runtime=runtime,
            router=ModelRouter(),
            local_provider_factory=local_factory,
        )
        response = provider.generate(
            ModelRequest("code", tier=ModelTier.STANDARD)
        )
        self.assertEqual(response.text, "small-ok")
        self.assertEqual(provider.last_choice.model, "qwen3:1.7b")
        self.assertEqual(provider.last_attempts[0].model, "qwen2.5-coder:7b")

    def test_cloud_escape_hatch_after_local_provider_failure(self):
        runtime = FakeRuntime()
        cloud = RecordingProvider(model="gemini-2.5-flash", response_text="cloud-ok")

        provider = AdaptivePersonaProvider(
            persona=JARVIS,
            task=TaskKind.CODING,
            hardware_profiler=FakeProfiler(),
            model_runtime=runtime,
            router=ModelRouter(),
            local_provider_factory=lambda choice: RecordingProvider(
                model=choice.model, fail=True
            ),
            cloud_provider_factory=lambda _choice: cloud,
        )
        response = provider.generate(
            ModelRequest("code", tier=ModelTier.STANDARD)
        )
        self.assertEqual(response.text, "cloud-ok")
        self.assertEqual(provider.last_choice.provider, ProviderKind.GEMINI)


class PersonaExecutionTests(unittest.TestCase):
    def test_persona_allowlist_can_only_narrow_owner_policy(self):
        class WeatherPlanProvider:
            def generate(self, request):
                plan = {
                    "goal": "weather",
                    "steps": [{
                        "step": 1,
                        "tool": "weather_report",
                        "description": "weather",
                        "parameters": {"city": "Cairo"},
                        "critical": True,
                    }],
                }
                return ModelResponse(json.dumps(plan), "fake", "fake")

        executor = AgentExecutor(provider=WeatherPlanProvider(), persona=JARVIS)
        message = executor.execute("tell me the weather")
        self.assertIn("denied", message)
        self.assertEqual(
            executor.last_action_receipts[-1].result.error_code,
            "persona.denied",
        )

    def test_persona_contexts_remain_separate_across_runtime_switches(self):
        requests = []

        class PlanProvider:
            def generate(self, request):
                requests.append(request)
                plan = {
                    "goal": "lookup",
                    "steps": [{
                        "step": 1,
                        "tool": "system_time",
                        "description": "time",
                        "parameters": {},
                        "critical": True,
                    }],
                }
                return ModelResponse(json.dumps(plan), "fake", "fake")

        def provider_factory(**_kwargs):
            return PlanProvider()

        runtime = AgentRuntime(
            hardware_profiler=FakeProfiler(),
            model_runtime=FakeRuntime(),
            context_store=PersonaContextStore(max_turns=4),
            provider_factory=provider_factory,
        )

        runtime.switch_persona("friday")
        runtime.execute("first friday request")
        runtime.switch_persona("jarvis")
        runtime.execute("first jarvis request")
        runtime.switch_persona("friday")
        runtime.execute("second friday request")

        self.assertNotIn("first friday request", requests[1].prompt)
        self.assertIn("first friday request", requests[2].prompt)
        self.assertNotIn("first jarvis request", requests[2].prompt)


if __name__ == "__main__":
    unittest.main()
