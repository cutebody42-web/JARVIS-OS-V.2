"""Integration tests for NEXUS persona-aware adaptive cognition."""

from datetime import datetime, timezone
import json
import unittest
from unittest.mock import Mock, patch

from agent.action_kernel import run_action
from agent.persona_runtime import PersonaAgentRuntime, PersonaContextStore
from core.action_contracts import ActionStatus
from core.hardware_profile import (
    GPUMemoryKind,
    HardwareProfiler,
    HardwareSnapshot,
    PowerSource,
)
from core.model_provider import ModelResponse, ModelTier
from core.model_router import (
    LocalModelCandidate,
    PersonaRoutingProfile,
    TaskKind,
)
from core.model_runtime import ModelHandle, ModelRuntime, ModelState, RuntimeStatus
from core.personas import DEFAULT_PERSONAS
from core.personas.persona_spec import PersonaRegistry, PersonaSpec


class FakeHardwareProfiler(HardwareProfiler):
    def __init__(self, snapshot):
        self.snapshot = snapshot

    def capture(self):
        return self.snapshot


class FakeModelRuntime(ModelRuntime):
    def __init__(self, status=None):
        self.status = status or RuntimeStatus(models={}, warnings=())
        self.ensure_calls = []

    def get_status(self):
        return self.status

    def ensure(self, model, priority, *, keep_alive=None):
        self.ensure_calls.append((model, priority, keep_alive))
        return ModelHandle(
            model=model,
            state=ModelState.WARM,
            priority=priority,
            keep_alive=keep_alive,
            last_used=datetime.now(timezone.utc),
        )


class PlanProvider:
    def __init__(self, choice, calls, *, fail=False, tool="system_time", parameters=None):
        self.choice = choice
        self.calls = calls
        self.fail = fail
        self.tool = tool
        self.parameters = parameters or {}

    def generate(self, request):
        self.calls.append((self.choice, request))
        if self.fail:
            raise RuntimeError("provider failed")
        return ModelResponse(
            json.dumps({
                "goal": "test",
                "steps": [{
                    "step": 1,
                    "tool": self.tool,
                    "description": "test",
                    "parameters": self.parameters,
                    "critical": True,
                }],
            }),
            self.choice.provider.value,
            self.choice.model,
        )


def healthy_hw():
    return HardwareSnapshot(
        device_id="node-test",
        total_ram_gb=16.0,
        available_ram_gb=10.0,
        cpu_percent=20.0,
        cpu_count=12,
        gpu_vram_gb=None,
        gpu_memory_kind=GPUMemoryKind.SHARED,
        power_source=PowerSource.AC,
        battery_pct=90,
        system_pressure=0.30,
        loaded_models={},
        timestamp=datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc),
    )


class PersonaRuntimeTests(unittest.TestCase):
    def test_same_goal_routes_friday_and_jarvis_to_different_models(self):
        calls = []

        def builder(choice):
            return PlanProvider(choice, calls)

        runtime = PersonaAgentRuntime(
            hardware_profiler=FakeHardwareProfiler(healthy_hw()),
            model_runtime=FakeModelRuntime(),
            provider_builder=builder,
        )

        friday = runtime.execute(
            "Please report this machine's current clock reading",
            persona="friday",
        )
        jarvis = runtime.execute(
            "Please report this machine's current clock reading",
            persona="jarvis",
        )

        self.assertEqual(friday.persona, "friday")
        self.assertEqual(friday.model, "qwen3.5:4b")
        self.assertEqual(jarvis.persona, "jarvis")
        self.assertEqual(jarvis.model, "qwen2.5-coder:3b")
        self.assertNotEqual(friday.model, jarvis.model)
        self.assertIn("Local time:", friday.message)
        self.assertIn("Local time:", jarvis.message)

    def test_switch_persona_changes_mode_without_runtime_restart(self):
        calls = []
        runtime = PersonaAgentRuntime(
            hardware_profiler=FakeHardwareProfiler(healthy_hw()),
            model_runtime=FakeModelRuntime(),
            provider_builder=lambda choice: PlanProvider(choice, calls),
        )

        self.assertEqual(runtime.active_persona.name, "taby")
        switched = runtime.switch_persona("jarvis")
        self.assertEqual(switched.name, "jarvis")

        result = runtime.execute("Please report this machine's current clock reading")
        self.assertEqual(result.persona, "jarvis")
        self.assertEqual(result.task, TaskKind.CODING)

    def test_persona_tool_ceiling_denies_before_dispatch(self):
        receipt = run_action(
            "weather_report",
            {"city": "Cairo"},
            task_id="task",
            step_id="1",
            route="model",
            allowed_tools=frozenset({"system_time"}),
        )
        self.assertEqual(receipt.result.status, ActionStatus.DENIED)
        self.assertEqual(receipt.result.error_code, "persona_denied")

    def test_persona_allowlist_never_expands_owner_policy(self):
        with patch("agent.action_kernel._dispatch") as dispatch:
            receipt = run_action(
                "shell",
                {"command": "whoami"},
                task_id="task",
                step_id="1",
                route="model",
                allowed_tools=frozenset({"shell"}),
            )
        self.assertEqual(receipt.result.status, ActionStatus.DENIED)
        self.assertEqual(receipt.result.error_code, "policy_denied")
        dispatch.assert_not_called()

    def test_provider_failure_uses_route_plan_fallback(self):
        calls = []
        runtime_model = FakeModelRuntime()
        attempts = {"count": 0}

        def builder(choice):
            attempts["count"] += 1
            return PlanProvider(
                choice,
                calls,
                fail=attempts["count"] == 1,
            )

        runtime = PersonaAgentRuntime(
            hardware_profiler=FakeHardwareProfiler(healthy_hw()),
            model_runtime=runtime_model,
            provider_builder=builder,
        )
        result = runtime.execute(
            "Please report this machine's current clock reading",
            persona="taby",
        )

        self.assertGreaterEqual(len(calls), 2)
        self.assertIn("Local time:", result.message)
        self.assertNotEqual(result.model, calls[0][0].model)

    def test_context_is_separate_per_persona_namespace(self):
        store = PersonaContextStore()
        calls = []
        runtime = PersonaAgentRuntime(
            hardware_profiler=FakeHardwareProfiler(healthy_hw()),
            model_runtime=FakeModelRuntime(),
            context_store=store,
            provider_builder=lambda choice: PlanProvider(choice, calls),
        )
        runtime.execute(
            "Please report this machine's current clock reading",
            persona="taby",
        )

        self.assertIn("Previous owner goal", store.render("persona:taby"))
        self.assertEqual(store.render("persona:jarvis"), "")

    def test_disallowed_task_fails_before_model_call(self):
        calls = []
        runtime = PersonaAgentRuntime(
            hardware_profiler=FakeHardwareProfiler(healthy_hw()),
            model_runtime=FakeModelRuntime(),
            provider_builder=lambda choice: PlanProvider(choice, calls),
        )
        with self.assertRaises(ValueError):
            runtime.execute(
                "code",
                persona="friday",
                task=TaskKind.CODING,
            )
        self.assertEqual(calls, [])

    def test_persona_spec_requires_routing_identity_match(self):
        routing = PersonaRoutingProfile(
            name="other",
            local_candidates=(
                LocalModelCandidate(
                    model="small",
                    min_available_ram_gb=1,
                    priority=1,
                    keep_alive=60,
                    tiers=frozenset({ModelTier.FAST}),
                    tasks=frozenset({TaskKind.GENERAL}),
                ),
            ),
            cloud_fast_model="cloud-fast",
            cloud_standard_model="cloud-standard",
        )
        with self.assertRaises(ValueError):
            PersonaSpec(
                name="custom",
                display_name="Custom",
                system_instruction="test",
                routing=routing,
                tool_allowlist=frozenset(),
                default_task=TaskKind.GENERAL,
                allowed_tasks=frozenset({TaskKind.GENERAL}),
                context_namespace="persona:custom",
            )

    def test_registry_contains_single_visible_identity_and_modes(self):
        self.assertEqual(
            DEFAULT_PERSONAS.names(),
            ("friday", "jarvis", "taby"),
        )


if __name__ == "__main__":
    unittest.main()
