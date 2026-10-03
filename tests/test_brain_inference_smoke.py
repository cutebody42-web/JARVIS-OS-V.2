"""Fail-closed evidence checks for the real-inference smoke workflow.

These fixtures exercise validation only. They do not run a model or provide
evidence that inference succeeded on the current machine.
"""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from core.jarvis_council import CORE_MODEL, CouncilNote, CouncilResult
from core.model_router import ProviderKind
from core.routed_model_provider import RouteAttempt


_SCRIPT = Path(__file__).resolve().parents[1] / "packaging" / "smoke" / "brain_inference.py"
_SPEC = importlib.util.spec_from_file_location("jarvis_brain_inference_smoke", _SCRIPT)
_SMOKE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SMOKE
_SPEC.loader.exec_module(_SMOKE)


def council_result(*, brief="Owner requests a short factual answer.", notes=None, models=None):
    if notes is None:
        notes = (CouncilNote("jarvis-brain-fast", "Use only stated facts."),)
    if models is None:
        models = (CORE_MODEL, "jarvis-brain-fast")
    return CouncilResult(brief, tuple(notes), tuple(models))


def route_attempt(provider=ProviderKind.OLLAMA, *, model="jarvis-brain-fast", outcome="succeeded"):
    return RouteAttempt(provider, model, "smoke_test_fixture", outcome)


class InferenceEvidenceTests(unittest.TestCase):
    def validate(self, *, response="The response contains an answer.", councils=None, routes=None):
        if councils is None:
            councils = [council_result()]
        if routes is None:
            routes = [route_attempt()]
        return _SMOKE.validate_inference(response, councils, routes)

    def test_accepts_complete_local_evidence_and_returns_json_evidence(self):
        evidence = self.validate(response="  The response contains an answer.\n")
        self.assertIsInstance(evidence, dict)
        self.assertEqual(evidence["response"], "The response contains an answer.")
        self.assertEqual(evidence["core_brief"], "Owner requests a short factual answer.")
        self.assertEqual(evidence["council_models"], [CORE_MODEL, "jarvis-brain-fast"])
        self.assertEqual(evidence["expert_notes"], [{"model": "jarvis-brain-fast", "text": "Use only stated facts."}])
        self.assertIs(evidence["core_and_expert_completed"], True)
        self.assertEqual(evidence["route_attempts"][-1]["provider"], "ollama")
        json.dumps(evidence)

    def test_failed_local_candidate_does_not_hide_successful_local_retry(self):
        evidence = self.validate(routes=[
            route_attempt(model=CORE_MODEL, outcome="failed:TimeoutError"),
            route_attempt(),
        ])
        self.assertIsInstance(evidence, dict)
        self.assertTrue(evidence)

    def test_rejects_empty_or_whitespace_final_response(self):
        for response in (None, 42, "", " \n\t"):
            with self.subTest(response=response), self.assertRaises(RuntimeError):
                self.validate(response=response)

    def test_requires_exactly_one_council_call(self):
        for councils in ([], [council_result(), council_result()]):
            with self.subTest(count=len(councils)), self.assertRaises(RuntimeError):
                self.validate(councils=councils)

    def test_rejects_failed_council_result(self):
        with self.assertRaises(RuntimeError):
            self.validate(councils=[None])

    def test_rejects_empty_or_whitespace_core_brief(self):
        for brief in (None, 42, "", " \n\t"):
            with self.subTest(brief=brief), self.assertRaises(RuntimeError):
                self.validate(councils=[council_result(brief=brief)])

    def test_requires_core_model_in_council_evidence(self):
        with self.assertRaises(RuntimeError):
            self.validate(councils=[council_result(models=("jarvis-brain-fast",))])

    def test_requires_fast_expert_note(self):
        for notes in ((), (CouncilNote("jarvis-brain-lite", "Some other expert succeeded."),)):
            with self.subTest(notes=notes), self.assertRaises(RuntimeError):
                self.validate(councils=[council_result(notes=notes)])

    def test_requires_fast_expert_model_attribution(self):
        with self.assertRaises(RuntimeError):
            self.validate(councils=[council_result(models=(CORE_MODEL,))])

    def test_rejects_empty_or_whitespace_fast_expert_note(self):
        for text in (None, 42, "", " \n\t"):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                self.validate(councils=[council_result(notes=(CouncilNote("jarvis-brain-fast", text),))])

    def test_rejects_routes_without_successful_local_inference(self):
        for routes in ([], [route_attempt(outcome="failed:RuntimeError")], [route_attempt(outcome="skipped:cloud_disabled")]):
            with self.subTest(routes=routes), self.assertRaises(RuntimeError):
                self.validate(routes=routes)

    def test_rejects_successful_cloud_fallback(self):
        cloud = route_attempt(ProviderKind.GEMINI, model="gemini-test")
        for routes in ([cloud], [route_attempt(), cloud], [cloud, route_attempt()]):
            with self.subTest(routes=routes), self.assertRaises(RuntimeError):
                self.validate(routes=routes)

    def test_failed_final_route_preserves_completed_council_and_failed_attempts(self):
        result = council_result()
        attempts = [route_attempt(outcome="failed:LocalProviderError"),
                    route_attempt(ProviderKind.GEMINI, outcome="skipped:cloud_disabled")]

        class FailingBrain:
            identity = "JARVIS"
            lane = SimpleNamespace(value="general")
            last_attempts = attempts

            def __init__(self, **kwargs):
                self.council = kwargs["council"]

            def respond(self, prompt, *, task):
                self.council.consult(prompt, task)
                raise RuntimeError("final local route unavailable")

        snapshot = SimpleNamespace(total_ram_gb=16.0, available_ram_gb=8.0,
                                   system_pressure=0.2, cpu_count=4, gpu_vram_gb=None, warnings=())
        report = {}
        with patch("core.hardware_profile.HardwareProfiler.capture", return_value=snapshot), \
             patch("core.jarvis_brain.JarvisBrain", FailingBrain), \
             patch("core.jarvis_council.JarvisCouncil.consult", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "final local route unavailable"):
                _SMOKE.run_inference("http://127.0.0.1:11435", report)
        self.assertTrue(report["council_calls"][0]["completed"])
        self.assertEqual(report["council_results"][0]["core_brief"], result.core_brief)
        self.assertEqual(report["council_results"][0]["notes"][0]["model"], "jarvis-brain-fast")
        self.assertEqual([item["outcome"] for item in report["route_attempts"]],
                         ["failed:LocalProviderError", "skipped:cloud_disabled"])
        self.assertEqual(report["identity"], "JARVIS")
        self.assertNotIn("core_and_expert_completed", report)
        json.dumps(report)


if __name__ == "__main__":
    unittest.main()
