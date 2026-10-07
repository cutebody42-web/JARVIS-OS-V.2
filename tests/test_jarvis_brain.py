"""Contracts for the single-identity JARVIS Brain."""

from datetime import datetime, timezone
import unittest
from unittest.mock import Mock

from core.hardware_profile import GPUMemoryKind, HardwareSnapshot, PowerSource
from core.jarvis_brain import (
    BrainPolicy,
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
from core.model_provider import (
    CloudDisclosureError,
    ContextClassification,
    ModelContext,
    ModelRequest,
    ModelResponse,
    ModelTier,
)
from core.model_router import ProviderChoice, ProviderKind, RoutePlan
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

    def test_discussing_commands_does_not_execute_them(self):
        for message in (
            "Explain how to open a browser",
            "How do I stop a running process?",
            "I am reading about how to start an application",
            'Translate the phrase "open my browser"',
            "Tell me why pause and resume work differently",
        ):
            with self.subTest(message=message):
                self.assertIs(classify_intent(message), BrainIntent.CHAT)

    def test_polite_direct_commands_still_execute(self):
        for message in (
            "Please open my browser",
            "Could you please mute the speakers",
            "JARVIS, set the volume to 40",
            "Can you search the web for Python documentation",
        ):
            with self.subTest(message=message):
                self.assertIs(classify_intent(message), BrainIntent.ACTION)


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
        for request in captured:
            self.assertIn("personal or situational claims must be grounded", request.system_instruction)
            self.assertIn("task lists", request.system_instruction)
            self.assertIn("numeric and time constraints", request.system_instruction)


    def test_memory_history_and_council_stay_local_during_cloud_fallback(self):
        private_memory = "OWNER MEMORY: launch code is heliotrope-71"
        private_council = "COUNCIL NOTES: owner has an undisclosed constraint"
        public_context = "The public project codename is Aurora."
        local_requests = []
        cloud_requests = []

        class Memory:
            def continuity_context(self, *, turn_limit):
                self.turn_limit = turn_limit
                return private_memory

            @staticmethod
            def current_handoff():
                return {}

            @staticmethod
            def append_turn(*args, **kwargs):
                return None

        class Council:
            @staticmethod
            def consult(message, task):
                return object()

            @staticmethod
            def context(result):
                return private_council

        class LocalFailure:
            def generate(self, request):
                local_requests.append(request)
                raise RuntimeError("offline local test failure")

        class CloudSuccess:
            def __init__(self, model):
                self.model = model

            def generate(self, request):
                cloud_requests.append(request)
                return ModelResponse("cloud answer", "gemini", self.model)

        from core.jarvis_brain import JarvisBrain
        brain = JarvisBrain(
            policy=BrainPolicy(allow_cloud=True),
            memory=Memory(),
            profiler=FakeProfiler(hardware()),
            model_runtime=FakeRuntime(),
            ollama_factory=lambda choice: LocalFailure(),
            gemini_factory=lambda choice: CloudSuccess(choice.model),
            council=Council(),
        )

        self.assertEqual(brain.respond("First cloud-eligible request"), "cloud answer")
        local_count_after_first_turn = len(local_requests)
        result = brain.respond(
            "Help me plan the next milestone",
            cloud_shareable_context=public_context,
        )

        self.assertEqual(result, "cloud answer")
        second_turn_local_requests = local_requests[local_count_after_first_turn:]
        self.assertTrue(second_turn_local_requests)
        self.assertEqual(len(cloud_requests), 2)
        self.assertTrue(all(
            private_memory in item.prompt
            and private_council in item.prompt
            and "First cloud-eligible request" in item.prompt
            and "cloud answer" in item.prompt
            for item in second_turn_local_requests
        ))
        sent_to_cloud = cloud_requests[-1]
        self.assertIn("Help me plan the next milestone", sent_to_cloud.prompt)
        self.assertIn(public_context, sent_to_cloud.prompt)
        self.assertNotIn(private_memory, sent_to_cloud.prompt)
        self.assertNotIn(private_council, sent_to_cloud.prompt)
        self.assertNotIn("First cloud-eligible request", sent_to_cloud.prompt)
        self.assertNotIn("cloud answer", sent_to_cloud.prompt)
        self.assertEqual(sent_to_cloud.context, ())
        self.assertIs(brain.last_attempts[-1].provider, ProviderKind.GEMINI)

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

    def test_context_defaults_local_only_and_cloud_projection_is_fail_closed(self):
        runtime = FakeRuntime()
        local_requests = []
        cloud_requests = []

        class LocalFailure:
            def generate(self, request):
                local_requests.append(request)
                raise RuntimeError("offline local test failure")

        class CloudSuccess:
            def __init__(self, model):
                self.model = model

            def generate(self, request):
                cloud_requests.append(request)
                return ModelResponse("ok", "gemini", self.model)

        private = ModelContext("private owner preference", label="owner memory")
        shareable = ModelContext(
            "public issue description",
            label="owner-approved context",
            classification=ContextClassification.CLOUD_SHAREABLE,
        )
        routed = RoutedModelProvider(
            JARVIS_ENGINEERING,
            profiler=FakeProfiler(hardware()),
            runtime=runtime,
            allow_cloud=True,
            ollama_factory=lambda choice: LocalFailure(),
            gemini_factory=lambda choice: CloudSuccess(choice.model),
        )

        response = routed.generate(ModelRequest(
            "debug the current failure",
            tier=ModelTier.STANDARD,
            context=(private, shareable),
        ))

        self.assertEqual(response.provider, "gemini")
        self.assertTrue(local_requests)
        self.assertTrue(all("private owner preference" in item.prompt for item in local_requests))
        self.assertEqual(len(cloud_requests), 1)
        cloud_request = cloud_requests[0]
        self.assertIn("debug the current failure", cloud_request.prompt)
        self.assertIn("public issue description", cloud_request.prompt)
        self.assertNotIn("private owner preference", cloud_request.prompt)
        self.assertEqual(cloud_request.context, ())
        self.assertIs(routed.last_attempts[-1].provider, ProviderKind.GEMINI)

    def test_cloud_secret_block_falls_through_to_local_without_mutating_request(self):
        secret = "api_key=owner-private-credential-123456"
        plan = RoutePlan(
            primary=ProviderChoice(
                ProviderKind.GEMINI,
                "gemini-test",
                "adversarial_cloud_primary",
            ),
            fallbacks=(ProviderChoice(
                ProviderKind.OLLAMA,
                "jarvis-local-test",
                "local_privacy_fallback",
                ensure_priority=1,
            ),),
        )
        router = Mock()
        router.route.return_value = plan
        cloud_calls = []
        local_calls = []

        class CloudSpy:
            def generate(self, request):
                cloud_calls.append(request)
                return ModelResponse("unsafe", "gemini", "gemini-test")

        class LocalSuccess:
            def generate(self, request):
                local_calls.append(request)
                return ModelResponse("local answer", "ollama", "jarvis-local-test")

        routed = RoutedModelProvider(
            JARVIS_ENGINEERING,
            profiler=FakeProfiler(hardware()),
            runtime=FakeRuntime(),
            router=router,
            allow_cloud=True,
            ollama_factory=lambda choice: LocalSuccess(),
            gemini_factory=lambda choice: CloudSpy(),
        )

        response = routed.generate(ModelRequest(secret))

        self.assertEqual(response.provider, "ollama")
        self.assertEqual(cloud_calls, [])
        self.assertEqual(len(local_calls), 1)
        self.assertIn(secret, local_calls[0].prompt)
        self.assertEqual(
            [attempt.outcome for attempt in routed.last_attempts],
            ["blocked:sensitive_content", "succeeded"],
        )

    def test_local_failure_never_promotes_secret_bearing_request_to_cloud(self):
        secret = "Authorization: Bearer ownerPrivateToken123456789"
        cloud_calls = []

        class CloudSpy:
            def generate(self, request):
                cloud_calls.append(request)
                return ModelResponse("unsafe", "gemini", "gemini-test")

        routed = RoutedModelProvider(
            JARVIS_ENGINEERING,
            profiler=FakeProfiler(hardware()),
            runtime=FakeRuntime(),
            allow_cloud=True,
            ollama_factory=lambda choice: FailingProvider(),
            gemini_factory=lambda choice: CloudSpy(),
        )

        with self.assertRaises(RoutedProviderError):
            routed.generate(ModelRequest(secret, tier=ModelTier.STANDARD))

        self.assertEqual(cloud_calls, [])
        self.assertEqual(routed.last_attempts[-1].outcome, "blocked:sensitive_content")

    def test_cloud_projection_scans_only_content_that_would_leave_machine(self):
        private_secret = "password=private-local-value-12345"
        request = ModelRequest(
            "safe current request",
            context=(
                ModelContext(private_secret, label="private memory"),
                ModelContext(
                    "public architecture summary",
                    label="approved public context",
                    classification=ContextClassification.CLOUD_SHAREABLE,
                ),
            ),
        )

        cloud = request.for_cloud_provider()
        local = request.for_local_provider()

        self.assertNotIn(private_secret, cloud.prompt)
        self.assertIn("public architecture summary", cloud.prompt)
        self.assertIn(private_secret, local.prompt)

    def test_cloud_projection_rejects_credentials_in_each_egress_channel(self):
        requests = (
            ModelRequest("GEMINI_API_KEY=owner-private-value-123456"),
            ModelRequest(
                "safe prompt",
                system_instruction="Authorization: Bearer ownerPrivateToken123456789",
            ),
            ModelRequest(
                "safe prompt",
                context=(ModelContext(
                    "github_pat_0123456789abcdef0123456789abcdef",
                    label="mistakenly approved",
                    classification=ContextClassification.CLOUD_SHAREABLE,
                ),),
            ),
        )

        for request in requests:
            with self.subTest(request=request):
                with self.assertRaises(CloudDisclosureError) as caught:
                    request.for_cloud_provider()
                self.assertNotIn("owner-private", str(caught.exception))
                self.assertNotIn("github_pat_", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
