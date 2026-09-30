"""Offline persona/routed-provider integration tests."""

from datetime import datetime, timezone
import json
import unittest

from agent.executor import AgentExecutor
from core.action_contracts import ActionStatus
from core.hardware_profile import GPUMemoryKind, HardwareSnapshot, PowerSource
from core.model_provider import ModelRequest, ModelResponse, ModelTier
from core.model_router import ModelRouter, ProviderKind, TaskKind
from core.model_runtime import RuntimeStatus
from core.persona_runtime import PersonaAgentRuntime
from core.personas import FRIDAY, JARVIS, TABY, PersonaSpec
from core.routed_model_provider import RoutedModelProvider


def hardware(*, available=8.0, pressure=0.30):
    return HardwareSnapshot(
        device_id="node-test",
        total_ram_gb=16.0,
        available_ram_gb=available,
        cpu_percent=pressure * 100,
        cpu_count=12,
        gpu_vram_gb=None,
        gpu_memory_kind=GPUMemoryKind.SHARED,
        power_source=PowerSource.AC,
        battery_pct=90,
        system_pressure=pressure,
        loaded_models={},
        timestamp=datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc),
    )


class FakeProfiler:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.calls = 0

    def capture(self):
        self.calls += 1
        return self.snapshot


class FakeRuntime:
    def __init__(self, status=None):
        self.status = status or RuntimeStatus(models={}, warnings=())
        self.ensure_calls = []

    def get_status(self):
        return self.status

    def ensure(self, model, priority, *, keep_alive=None):
        self.ensure_calls.append((model, priority, keep_alive))


class RecordingProvider:
    def __init__(self, provider, model, *, fail=False):
        self.provider = provider
        self.model = model
        self.fail = fail
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("offline test failure")
        return ModelResponse("ok", self.provider, self.model)


class PlanProvider:
    def __init__(self, tool):
        self.tool = tool

    def generate(self, request):
        return ModelResponse(
            json.dumps({
                "steps": [{
                    "step": 1,
                    "tool": self.tool,
                    "description": "persona test",
                    "parameters": {},
                    "critical": True,
                }]
            }),
            "fake",
            "offline",
        )


class PersonaContractTests(unittest.TestCase):
    def test_builtin_personas_have_distinct_default_tasks_and_namespaces(self):
        self.assertEqual(TABY.default_task, TaskKind.GENERAL)
        self.assertEqual(FRIDAY.default_task, TaskKind.REALTIME)
        self.assertEqual(JARVIS.default_task, TaskKind.CODING)
        self.assertEqual(
            {TABY.context_namespace, FRIDAY.context_namespace, JARVIS.context_namespace},
            {"taby", "friday", "jarvis"},
        )

    def test_invalid_task_is_rejected_by_persona(self):
        with self.assertRaises(ValueError):
            FRIDAY.validate_task(TaskKind.CODING)

    def test_same_hardware_routes_friday_and_jarvis_to_different_models(self):
        router = ModelRouter()
        snap = hardware()
        status = RuntimeStatus(models={}, warnings=())

        friday = router.route(
            ModelRequest("respond", tier=ModelTier.FAST),
            snap,
            status,
            FRIDAY.routing,
            task=FRIDAY.default_task,
        )
        jarvis = router.route(
            ModelRequest("code", tier=ModelTier.STANDARD),
            snap,
            status,
            JARVIS.routing,
            task=JARVIS.default_task,
        )

        self.assertEqual(friday.primary.provider, ProviderKind.OLLAMA)
        self.assertEqual(friday.primary.model, "qwen3.5:4b")
        self.assertEqual(jarvis.primary.provider, ProviderKind.OLLAMA)
        self.assertEqual(jarvis.primary.model, "qwen2.5-coder:7b")


class RoutedProviderTests(unittest.TestCase):
    def test_local_failure_uses_route_plan_cloud_fallback(self):
        profiler = FakeProfiler(hardware())
        runtime = FakeRuntime()
        local_providers = []
        cloud_providers = []

        def local_factory(choice):
            provider = RecordingProvider("ollama", choice.model, fail=True)
            local_providers.append(provider)
            return provider

        def cloud_factory(choice):
            provider = RecordingProvider("gemini", choice.model)
            cloud_providers.append(provider)
            return provider

        routed = RoutedModelProvider(
            JARVIS,
            profiler=profiler,
            runtime=runtime,
            ollama_factory=local_factory,
            gemini_factory=cloud_factory,
        )
        response = routed.generate(
            ModelRequest("write code", "planner contract", ModelTier.STANDARD)
        )

        self.assertEqual(response.provider, "gemini")
        self.assertTrue(runtime.ensure_calls)
        self.assertGreaterEqual(len(local_providers), 1)
        self.assertEqual(len(cloud_providers), 1)
        self.assertTrue(routed.last_attempts[-1].outcome == "succeeded")
        sent = cloud_providers[0].requests[0].system_instruction
        self.assertIn("APPLICATION CONTRACT", sent)
        self.assertIn("planner contract", sent)
        self.assertIn("JARVIS", sent)

    def test_hardware_probe_failure_fails_over_to_cloud_without_local_ensure(self):
        class BrokenProfiler:
            def capture(self):
                raise OSError("no metrics")

        runtime = FakeRuntime()
        cloud = RecordingProvider("gemini", "gemini-2.5-flash-lite")
        routed = RoutedModelProvider(
            FRIDAY,
            profiler=BrokenProfiler(),
            runtime=runtime,
            ollama_factory=lambda choice: self.fail("local should not be created"),
            gemini_factory=lambda choice: cloud,
        )

        response = routed.generate(ModelRequest("hello", tier=ModelTier.FAST))
        self.assertEqual(response.provider, "gemini")
        self.assertEqual(runtime.ensure_calls, [])
        self.assertEqual(routed.last_attempts[0].reason, "hardware_snapshot_unavailable")


class PersonaActionBoundaryTests(unittest.TestCase):
    def test_persona_can_restrict_an_owner_policy_allowed_tool(self):
        restricted = PersonaSpec(
            name="restricted",
            system_instruction="Restricted test persona.",
            routing=TABY.routing.__class__(
                name="restricted",
                local_candidates=TABY.routing.local_candidates,
                cloud_fast_model=TABY.routing.cloud_fast_model,
                cloud_standard_model=TABY.routing.cloud_standard_model,
            ),
            tool_allowlist=frozenset(),
            default_task=TaskKind.GENERAL,
            allowed_tasks=frozenset({TaskKind.GENERAL}),
            context_namespace="restricted",
        )
        executor = AgentExecutor(
            provider=PlanProvider("system_time"),
            tool_allowlist=restricted.tool_allowlist,
        )
        result = executor.execute("read clock")
        self.assertIn("denied", result)
        self.assertEqual(executor.last_status, ActionStatus.DENIED)
        self.assertEqual(executor.last_action_receipts[0].result.error_code, "persona_denied")

    def test_persona_allowlist_cannot_expand_owner_policy(self):
        executor = AgentExecutor(
            provider=PlanProvider("generated_code"),
            tool_allowlist=frozenset({"generated_code"}),
        )
        result = executor.execute("run generated code")
        self.assertIn("denied", result)
        self.assertEqual(executor.last_status, ActionStatus.DENIED)
        self.assertEqual(executor.last_action_receipts[0].result.error_code, "policy_denied")


class PersonaRuntimeTests(unittest.TestCase):
    def test_switching_persona_reuses_hardware_and_model_runtime(self):
        profiler = FakeProfiler(hardware())
        runtime = FakeRuntime()
        agent = PersonaAgentRuntime(
            FRIDAY,
            profiler=profiler,
            model_runtime=runtime,
            ollama_factory=lambda choice: RecordingProvider("ollama", choice.model),
            gemini_factory=lambda choice: RecordingProvider("gemini", choice.model),
        )
        first_provider = agent.provider
        agent.switch_persona("jarvis")
        second_provider = agent.provider

        self.assertEqual(agent.persona.name, "jarvis")
        self.assertEqual(agent.task, TaskKind.CODING)
        self.assertIs(first_provider.runtime, runtime)
        self.assertIs(second_provider.runtime, runtime)
        self.assertIs(first_provider.profiler, profiler)
        self.assertIs(second_provider.profiler, profiler)


if __name__ == "__main__":
    unittest.main()
