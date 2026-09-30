"""Pure offline tests for NEXUS adaptive model routing."""

from datetime import datetime, timezone
import unittest

from core.hardware_profile import (
    GPUMemoryKind,
    HardwareSnapshot,
    PowerSource,
)
from core.model_provider import ModelRequest, ModelTier
from core.model_router import (
    LocalModelCandidate,
    ModelRouter,
    PersonaRoutingProfile,
    ProviderKind,
    RouterPolicy,
    TaskKind,
)
from core.model_runtime import ModelInfo, ModelState, RuntimeStatus


def hw(
    *,
    total=16.0,
    available=8.0,
    pressure=0.40,
    power=PowerSource.AC,
    device_id="node-a",
):
    return HardwareSnapshot(
        device_id=device_id,
        total_ram_gb=total,
        available_ram_gb=available,
        cpu_percent=pressure * 100,
        cpu_count=12,
        gpu_vram_gb=None,
        gpu_memory_kind=GPUMemoryKind.SHARED,
        power_source=power,
        battery_pct=80 if power is not PowerSource.UNKNOWN else None,
        system_pressure=pressure,
        loaded_models={},
        timestamp=datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc),
    )


def status(*warm_models):
    return RuntimeStatus(
        models={
            model: ModelInfo(
                model=model,
                state=ModelState.WARM,
                priority=0,
                managed=False,
                reported_size_gb=4.0,
            )
            for model in warm_models
        },
        warnings=(),
    )


def profile():
    return PersonaRoutingProfile(
        name="test-persona",
        local_candidates=(
            LocalModelCandidate(
                model="coder:7b",
                min_available_ram_gb=6.0,
                priority=5,
                keep_alive=300,
                tiers=frozenset({ModelTier.STANDARD}),
                tasks=frozenset({TaskKind.CODING}),
            ),
            LocalModelCandidate(
                model="friday:4b",
                min_available_ram_gb=4.0,
                priority=10,
                keep_alive=300,
                tiers=frozenset({ModelTier.FAST, ModelTier.STANDARD}),
                tasks=frozenset({TaskKind.GENERAL, TaskKind.REALTIME}),
            ),
            LocalModelCandidate(
                model="small:1.7b",
                min_available_ram_gb=2.0,
                priority=8,
                keep_alive=60,
                tiers=frozenset({ModelTier.FAST, ModelTier.STANDARD}),
                tasks=frozenset({
                    TaskKind.GENERAL,
                    TaskKind.REALTIME,
                    TaskKind.CODING,
                }),
            ),
        ),
        cloud_fast_model="gemini-fast",
        cloud_standard_model="gemini-standard",
    )


class ModelRouterTests(unittest.TestCase):
    def setUp(self):
        self.router = ModelRouter()
        self.persona = profile()

    def test_warm_compatible_model_is_preferred_without_ensure(self):
        plan = self.router.route(
            ModelRequest("hello", tier=ModelTier.FAST),
            hw(available=2.5, pressure=0.75),
            status("friday:4b"),
            self.persona,
            task=TaskKind.GENERAL,
        )

        self.assertEqual(plan.primary.provider, ProviderKind.OLLAMA)
        self.assertEqual(plan.primary.model, "friday:4b")
        self.assertFalse(plan.primary.requires_ensure)
        self.assertEqual(plan.primary.reason, "compatible_model_already_warm")
        self.assertEqual(plan.fallbacks[-1].provider, ProviderKind.GEMINI)

    def test_high_ram_ac_coding_selects_coder_and_requires_ensure(self):
        plan = self.router.route(
            ModelRequest("write code", tier=ModelTier.STANDARD),
            hw(total=16, available=8, pressure=0.30, power=PowerSource.AC),
            status(),
            self.persona,
            task=TaskKind.CODING,
        )

        self.assertEqual(plan.primary.model, "coder:7b")
        self.assertTrue(plan.primary.requires_ensure)
        self.assertEqual(plan.primary.ensure_priority, 5)
        self.assertEqual(plan.primary.provider, ProviderKind.OLLAMA)

    def test_coding_falls_to_small_local_when_7b_does_not_fit(self):
        plan = self.router.route(
            ModelRequest("write code", tier=ModelTier.STANDARD),
            hw(total=8, available=3, pressure=0.60, power=PowerSource.AC),
            status(),
            self.persona,
            task=TaskKind.CODING,
        )

        self.assertEqual(plan.primary.model, "small:1.7b")
        self.assertTrue(plan.primary.requires_ensure)

    def test_battery_low_resource_policy_goes_cloud(self):
        plan = self.router.route(
            ModelRequest("hello", tier=ModelTier.FAST),
            hw(
                total=8,
                available=1.5,
                pressure=0.82,
                power=PowerSource.BATTERY,
            ),
            status(),
            self.persona,
            task=TaskKind.GENERAL,
        )

        self.assertEqual(plan.primary.provider, ProviderKind.GEMINI)
        self.assertEqual(plan.primary.model, "gemini-fast")
        self.assertEqual(plan.primary.reason, "local_resource_policy_cloud_fallback")

    def test_low_memory_device_does_not_cold_load_second_model(self):
        plan = self.router.route(
            ModelRequest("hello", tier=ModelTier.FAST),
            hw(total=8, available=5, pressure=0.35),
            status("external-user-model"),
            self.persona,
            task=TaskKind.GENERAL,
        )

        self.assertEqual(plan.primary.provider, ProviderKind.GEMINI)
        self.assertEqual(plan.primary.reason, "local_resource_policy_cloud_fallback")

    def test_exact_external_warm_candidate_can_be_used_without_taking_ownership(self):
        plan = self.router.route(
            ModelRequest("hello", tier=ModelTier.FAST),
            hw(total=8, available=3, pressure=0.50),
            status("small:1.7b"),
            self.persona,
            task=TaskKind.GENERAL,
        )

        self.assertEqual(plan.primary.model, "small:1.7b")
        self.assertFalse(plan.primary.requires_ensure)

    def test_critical_pressure_avoids_even_warm_local_inference(self):
        plan = self.router.route(
            ModelRequest("hello", tier=ModelTier.FAST),
            hw(total=16, available=1, pressure=0.98),
            status("friday:4b"),
            self.persona,
            task=TaskKind.GENERAL,
        )

        self.assertEqual(plan.primary.provider, ProviderKind.GEMINI)
        self.assertEqual(plan.primary.reason, "critical_pressure_cloud_fallback")

    def test_no_compatible_local_task_goes_cloud(self):
        limited = PersonaRoutingProfile(
            name="limited",
            local_candidates=(
                LocalModelCandidate(
                    model="chat:1b",
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

        plan = self.router.route(
            ModelRequest("code", tier=ModelTier.STANDARD),
            hw(),
            status(),
            limited,
            task=TaskKind.CODING,
        )

        self.assertEqual(plan.primary.provider, ProviderKind.GEMINI)
        self.assertEqual(plan.primary.reason, "no_compatible_local_candidate")

    def test_device_id_does_not_change_routing_for_identical_capability(self):
        request = ModelRequest("hello", tier=ModelTier.FAST)
        first = self.router.route(
            request,
            hw(device_id="node-one"),
            status(),
            self.persona,
        )
        second = self.router.route(
            request,
            hw(device_id="node-two"),
            status(),
            self.persona,
        )

        self.assertEqual(first, second)

    def test_route_plan_contains_cloud_escape_hatch_after_local_choices(self):
        plan = self.router.route(
            ModelRequest("hello", tier=ModelTier.FAST),
            hw(total=16, available=8, pressure=0.25),
            status(),
            self.persona,
        )

        self.assertEqual(plan.primary.provider, ProviderKind.OLLAMA)
        self.assertEqual(plan.fallbacks[-1].provider, ProviderKind.GEMINI)
        self.assertEqual(plan.fallbacks[-1].model, "gemini-fast")

    def test_policy_validation(self):
        with self.assertRaises(ValueError):
            RouterPolicy(new_load_pressure_limit=1.2)
        with self.assertRaises(ValueError):
            RouterPolicy(low_memory_total_gb=24, high_memory_total_gb=10)


if __name__ == "__main__":
    unittest.main()
