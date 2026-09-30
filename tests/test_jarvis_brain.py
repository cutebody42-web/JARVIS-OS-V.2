"""Contracts for the single-identity JARVIS Brain."""

from datetime import datetime, timezone
import unittest

from core.hardware_profile import GPUMemoryKind, HardwareSnapshot, PowerSource
from core.jarvis_brain import (
    BrainIntent,
    BrainLane,
    JARVIS_CONTEXT_NAMESPACE,
    JARVIS_ENGINEERING,
    JARVIS_GENERAL,
    JARVIS_IDENTITY,
    JARVIS_REALTIME,
    classify_intent,
    classify_lane,
)
from core.model_provider import ModelRequest, ModelResponse, ModelTier
from core.model_runtime import RuntimeStatus
from core.routed_model_provider import RoutedModelProvider, RoutedProviderError


def hardware(*, total=16.0, available=8.0, pressure=0.3):
    return HardwareSnapshot(
        device_id="node",
        total_ram_gb=total,
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

    def capture(self):
        return self.snapshot


class FakeRuntime:
    def __init__(self):
        self.ensure_calls = []

    def get_status(self):
        return RuntimeStatus(models={}, warnings=())

    def ensure(self, model, priority, *, keep_alive=None):
        self.ensure_calls.append((model, priority, keep_alive))


class FailingProvider:
    def generate(self, request):
        raise RuntimeError("offline local test failure")


class SuccessProvider:
    def __init__(self, provider, model):
        self.provider = provider
        self.model = model

    def generate(self, request):
        return ModelResponse("ok", self.provider, self.model)


class JarvisIdentityTests(unittest.TestCase):
    def test_all_internal_lanes_share_one_user_identity_and_memory_namespace(self):
        self.assertEqual(JARVIS_IDENTITY, "JARVIS")
        for profile in (JARVIS_GENERAL, JARVIS_REALTIME, JARVIS_ENGINEERING):
            self.assertEqual(profile.context_namespace, JARVIS_CONTEXT_NAMESPACE)
            self.assertIn("You are JARVIS", profile.system_instruction)
            self.assertNotIn("You are FRIDAY", profile.system_instruction)
            self.assertNotIn("You are TABY", profile.system_instruction)

    def test_local_aliases_hide_vendor_identity_from_routing_surface(self):
        aliases = {
            candidate.model
            for profile in (JARVIS_GENERAL, JARVIS_REALTIME, JARVIS_ENGINEERING)
            for candidate in profile.routing.local_candidates
        }
        self.assertEqual(
            aliases,
            {
                "jarvis-core-1b",
                "jarvis-brain-fast",
                "jarvis-brain-engineering",
                "jarvis-brain-lite",
            },
        )

    def test_deterministic_lane_classifier(self):
        self.assertIs(classify_lane("fix this GitHub CI traceback"), BrainLane.ENGINEERING)
        self.assertIs(classify_lane("set the volume to 40"), BrainLane.REALTIME)
        self.assertIs(classify_lane("help me plan tomorrow"), BrainLane.GENERAL)

    def test_ambiguous_language_stays_chat_and_explicit_command_routes_action(self):
        self.assertIs(classify_intent("help me plan tomorrow"), BrainIntent.CHAT)
        self.assertIs(classify_intent("explain how volume control works"), BrainIntent.CHAT)
        self.assertIs(classify_intent("what is the time"), BrainIntent.ACTION)
        self.assertIs(classify_intent("set the volume to 40"), BrainIntent.ACTION)



class ConversationContinuityTests(unittest.TestCase):
    def test_respond_uses_direct_model_path_and_preserves_history_across_lanes(self):
        captured = []

        class Provider:
            def generate(self, request):
                captured.append(request)
                return ModelResponse(
                    "first" if len(captured) == 1 else "second",
                    "ollama",
                    "jarvis-brain-fast",
                )

        runtime = FakeRuntime()
        from core.jarvis_brain import JarvisBrain
        brain = JarvisBrain(
            profiler=FakeProfiler(hardware()),
            model_runtime=runtime,
        )

        # Swap provider factory after construction by replacing routed factories
        # on the shared current runtime; the test remains fully offline.
        brain._runtime._ollama_factory = lambda choice: Provider()
        brain._runtime._gemini_factory = lambda choice: Provider()
        brain._runtime._rebuild()

        self.assertEqual(brain.respond("help me plan tomorrow"), "first")
        self.assertEqual(brain.respond("explain this code bug", task=None), "second")

        self.assertEqual(brain.context_namespace, "jarvis")
        self.assertEqual(brain.identity, "JARVIS")
        self.assertIn("Owner: help me plan tomorrow", captured[1].prompt)
        self.assertIn("JARVIS: first", captured[1].prompt)

class LocalFirstRoutingTests(unittest.TestCase):
    def test_cloud_is_skipped_when_owner_disables_it(self):
        runtime = FakeRuntime()
        routed = RoutedModelProvider(
            JARVIS_ENGINEERING,
            profiler=FakeProfiler(hardware()),
            runtime=runtime,
            allow_cloud=False,
            ollama_factory=lambda choice: FailingProvider(),
            gemini_factory=lambda choice: SuccessProvider("gemini", choice.model),
        )

        with self.assertRaises(RoutedProviderError):
            routed.generate(ModelRequest("debug code", tier=ModelTier.STANDARD))

        self.assertTrue(any(item.outcome.startswith("failed:") for item in routed.last_attempts))
        self.assertEqual(routed.last_attempts[-1].outcome, "skipped:cloud_disabled")

    def test_cloud_boost_can_be_explicitly_enabled(self):
        runtime = FakeRuntime()
        routed = RoutedModelProvider(
            JARVIS_ENGINEERING,
            profiler=FakeProfiler(hardware()),
            runtime=runtime,
            allow_cloud=True,
            ollama_factory=lambda choice: FailingProvider(),
            gemini_factory=lambda choice: SuccessProvider("gemini", choice.model),
        )

        response = routed.generate(ModelRequest("debug code", tier=ModelTier.STANDARD))

        self.assertEqual(response.provider, "gemini")
        self.assertEqual(routed.last_attempts[-1].outcome, "succeeded")


if __name__ == "__main__":
    unittest.main()
