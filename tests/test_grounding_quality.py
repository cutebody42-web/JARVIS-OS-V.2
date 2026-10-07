"""Deterministic grounding contracts for the production JARVIS planning probe."""

import importlib.util
from pathlib import Path
import sys
import unittest

from core.jarvis_council import CouncilNote, CouncilResult, JarvisCouncil


_SCRIPT = Path(__file__).resolve().parents[1] / "packaging" / "smoke" / "brain_inference.py"
_SPEC = importlib.util.spec_from_file_location("jarvis_grounding_quality_smoke", _SCRIPT)
_SMOKE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SMOKE
_SPEC.loader.exec_module(_SMOKE)


class GroundingContractTests(unittest.TestCase):
    def test_hidden_council_context_carries_synthesis_grounding_contract(self):
        context = JarvisCouncil.context(CouncilResult(
            "Owner has thirty minutes tomorrow; other details are UNKNOWN.",
            (CouncilNote("jarvis-brain-fast", "Offer a neutral three-step plan."),),
            ("jarvis-core-1b", "jarvis-brain-fast"),
        ))
        self.assertIn("analysis, not evidence", context)
        self.assertIn("durable synchronized memory", context)
        self.assertIn("recent owner messages", context)
        self.assertIn("Never convert a plausible scenario", context)
        self.assertIn("meeting", context)
        self.assertIn("presentation", context)
        self.assertIn("task list", context)
        self.assertIn("project", context)
        self.assertIn("goal", context)
        self.assertIn("priority", context)
        self.assertIn("session", context)
        self.assertIn("material", context)
        self.assertIn("document", context)
        self.assertIn("workspace", context)
        self.assertIn("resource", context)
        self.assertIn("numeric and time budgets", context)
        self.assertIn("never allocate more time", context)

    def test_core_brief_gate_accepts_normalized_authoritative_owner_contract(self):
        quality = _SMOKE.validate_core_brief_quality(
            "GOAL: Create a neutral preparation plan.\n"
            "STATED_CONSTRAINTS: AUTHORITATIVE_OWNER_REQUEST: I have thirty minutes to prepare "
            "for tomorrow. Return exactly three short numbered planning steps. Do not assume my "
            "calendar, events or preferences. Keep the final answer under eighty words.\n"
            "UNKNOWNS: The preparation subject and personal schedule are UNKNOWN.\n"
            "VERIFY: Three steps fit within the thirty-minute budget and add no personal facts."
        )
        self.assertTrue(quality["certified"])
        self.assertEqual(
            quality["labels_seen"],
            ["GOAL", "STATED_CONSTRAINTS", "UNKNOWNS", "VERIFY"],
        )
        self.assertTrue(quality["authoritative_request_preserved"])
        self.assertTrue(quality["time_budget_preserved"])
        self.assertTrue(quality["output_shape_preserved"])
        self.assertTrue(quality["word_budget_preserved"])
        self.assertTrue(quality["grounding_prohibition_preserved"])
        self.assertEqual(quality["refusal_markers"], [])

    def test_core_brief_gate_rejects_previous_real_refusal_regression(self):
        quality = _SMOKE.validate_core_brief_quality(
            "I cannot provide a response that assumes your calendar, events, or preferences. "
            "Can I help you with something else?"
        )
        self.assertFalse(quality["certified"])
        self.assertIn("core_brief_protocol_mismatch", quality["failures"])
        self.assertIn("core_brief_refusal", quality["failures"])
        self.assertIn("core_brief_missing_authoritative_request", quality["failures"])
        self.assertIn("core_brief_omitted_time_budget", quality["failures"])

    def test_core_brief_gate_rejects_authoritative_brief_that_drops_time_budget(self):
        quality = _SMOKE.validate_core_brief_quality(
            "GOAL: Create a neutral preparation plan.\n"
            "STATED_CONSTRAINTS: AUTHORITATIVE_OWNER_REQUEST: Return exactly three numbered "
            "planning steps. Do not assume my calendar, events or preferences. Keep the final "
            "answer under eighty words.\n"
            "UNKNOWNS: The preparation subject is UNKNOWN.\n"
            "VERIFY: Do not invent personal facts."
        )
        self.assertFalse(quality["certified"])
        self.assertIn("core_brief_omitted_time_budget", quality["failures"])
        self.assertNotIn("core_brief_missing_authoritative_request", quality["failures"])

    def test_quality_gate_accepts_grounded_three_step_response(self):
        quality = _SMOKE.validate_grounding_quality(
            "1. Pick one focus for the thirty minutes. "
            "2. Spend most of the time preparing it. "
            "3. Use the final minutes to review and note the next action."
        )
        self.assertTrue(quality["certified"])
        self.assertEqual(quality["numbered_steps_seen"], ["1", "2", "3"])
        self.assertEqual(quality["unsupported_assumption_markers"], [])
        self.assertEqual(quality["time_budget_violations"], [])

    def test_quality_gate_accepts_consistent_per_step_time(self):
        quality = _SMOKE.validate_grounding_quality(
            "1. Pick a focus. 2. Work on it. 3. Review it. Spend ten minutes per step."
        )
        self.assertTrue(quality["certified"])
        self.assertEqual(quality["time_budget_evidence"][0]["implied_total_minutes"], 30)

    def test_quality_gate_accepts_summed_explicit_step_allocations_within_budget(self):
        quality = _SMOKE.validate_grounding_quality(
            "1. Spend 10 minutes choosing a focus. "
            "2. Allocate 15 minutes to work on it. "
            "3. Use 5 minutes to review."
        )
        self.assertTrue(quality["certified"])
        self.assertEqual(quality["explicit_step_total_minutes"], 30)
        self.assertEqual(len(quality["explicit_step_time_allocations"]), 3)

    def test_quality_gate_rejects_invented_personal_events(self):
        for response, marker in (
            (
                "1. Review your presentation. 2. Rehearse it. 3. Prepare for tomorrow's meeting.",
                "claimed_personal_event",
            ),
            (
                "1. List priorities. 2. Prepare your slides. 3. Review the result.",
                "invented_presentation_work",
            ),
            (
                "1. Review tomorrow's task list. 2. Draft your projects. 3. Rank your priorities.",
                "claimed_task_inventory",
            ),
            (
                "1. Prepare materials for tomorrow's session. "
                "2. Review relevant documents beforehand. "
                "3. Ensure workspace is ready.",
                "claimed_preparation_context",
            ),
        ):
            with self.subTest(response=response):
                quality = _SMOKE.validate_grounding_quality(response)
                self.assertFalse(quality["certified"])
                self.assertIn(marker, quality["unsupported_assumption_markers"])
                self.assertIn("unsupported_personal_assumption", quality["failures"])

    def test_quality_gate_rejects_previous_real_task_list_regression(self):
        quality = _SMOKE.validate_grounding_quality(
            "1. Review tomorrow's task list to identify highest priority items. "
            "2. Allocate 15 minutes to draft outlines for those tasks. "
            "3. Reserve remaining time for final review and resource preparation."
        )
        self.assertFalse(quality["certified"])
        self.assertIn("claimed_task_inventory", quality["unsupported_assumption_markers"])

    def test_quality_gate_rejects_previous_real_artifact_regression(self):
        quality = _SMOKE.validate_grounding_quality(
            "1. Prepare materials for tomorrow's session. "
            "2. Review relevant documents beforehand. "
            "3. Ensure workspace is ready. "
            "Total time: 30 minutes (flexible allocation)."
        )
        self.assertFalse(quality["certified"])
        self.assertIn("claimed_preparation_context", quality["unsupported_assumption_markers"])

    def test_quality_gate_rejects_latest_real_core_tasks_regression(self):
        quality = _SMOKE.validate_grounding_quality(
            "1. Allocate twenty minutes to review your core tasks. "
            "2. Reserve ten minutes for contingency planning. "
            "3. Ensure total time equals exactly thirty minutes. "
            "This plan strictly adheres to your thirty-minute constraint."
        )
        self.assertFalse(quality["certified"])
        self.assertIn("claimed_task_inventory", quality["unsupported_assumption_markers"])
        self.assertIn("claimed_work_object", quality["unsupported_assumption_markers"])

    def test_quality_gate_rejects_impossible_per_step_time(self):
        for phrase in ("twenty minutes per action", "20 minutes for each step"):
            with self.subTest(phrase=phrase):
                quality = _SMOKE.validate_grounding_quality(
                    f"1. Pick a focus. 2. Work on it. 3. Review it. Spend {phrase}."
                )
                self.assertFalse(quality["certified"])
                self.assertIn("time_budget_inconsistent", quality["failures"])
                self.assertGreater(
                    quality["time_budget_violations"][0]["implied_total_minutes"],
                    quality["time_budget_minutes"],
                )

    def test_quality_gate_rejects_summed_explicit_step_allocations_over_budget(self):
        quality = _SMOKE.validate_grounding_quality(
            "1. Spend 15 minutes choosing a focus. "
            "2. Allocate 10 minutes to work on it. "
            "3. Use 10 minutes to review."
        )
        self.assertFalse(quality["certified"])
        self.assertEqual(quality["explicit_step_total_minutes"], 35)
        self.assertIn("time_budget_inconsistent", quality["failures"])
        explicit_violation = next(
            item for item in quality["time_budget_violations"]
            if item.get("kind") == "explicit_step_allocations"
        )
        self.assertEqual(explicit_violation["total_minutes"], 35)
        self.assertEqual(explicit_violation["budget_minutes"], 30)

    def test_quality_gate_rejects_ordinal_step_allocations_over_budget(self):
        quality = _SMOKE.validate_grounding_quality(
            "1. Work on a chosen focus for the first fifteen minutes. "
            "2. Continue for the next ten minutes. "
            "3. Review for the final ten minutes."
        )
        self.assertFalse(quality["certified"])
        self.assertEqual(quality["explicit_step_total_minutes"], 35)
        self.assertIn("time_budget_inconsistent", quality["failures"])

    def test_quality_gate_requires_exact_three_numbered_steps(self):
        quality = _SMOKE.validate_grounding_quality(
            "Choose one focus, work on it, then review what remains."
        )
        self.assertFalse(quality["certified"])
        self.assertIn("missing_exact_three_numbered_steps", quality["failures"])

    def test_quality_gate_rejects_duplicate_numbered_steps(self):
        quality = _SMOKE.validate_grounding_quality(
            "1. Choose a focus. 1. Check it. 2. Work on it. 3. Review it."
        )
        self.assertFalse(quality["certified"])
        self.assertEqual(quality["numbered_steps_seen"], ["1", "1", "2", "3"])
        self.assertIn("missing_exact_three_numbered_steps", quality["failures"])

    def test_quality_gate_enforces_word_budget(self):
        response = "1. " + "focus " * 81 + "2. review 3. finish"
        quality = _SMOKE.validate_grounding_quality(response)
        self.assertFalse(quality["certified"])
        self.assertIn("word_limit_exceeded", quality["failures"])

    def test_real_probe_explicitly_requests_machine_checkable_shape(self):
        self.assertIn("exactly three short numbered planning steps", _SMOKE.PROMPT)
        self.assertIn("Do not assume my calendar, events or preferences", _SMOKE.PROMPT)
        self.assertIn("total no more than thirty minutes", _SMOKE.PROMPT)
        self.assertIn("under eighty words", _SMOKE.PROMPT)


if __name__ == "__main__":
    unittest.main()
