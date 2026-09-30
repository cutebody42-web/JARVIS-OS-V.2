"""Product-runtime tests for bounded automatic self-healing."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from api.jarvis_local_server import LocalBrainHost
from core.self_heal import (
    IncidentKind,
    RepairJournal,
    RepairState,
    SelfHealController,
)


class FlakyBrain:
    def __init__(self, *, succeeds_after=1):
        self.calls = 0
        self.succeeds_after = succeeds_after

    def handle(self, message, *, task=None):
        self.calls += 1
        if self.calls <= self.succeeds_after:
            raise RuntimeError("simulated model runtime failure")
        return "Recovered response"


class LocalBrainSelfHealTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.host = LocalBrainHost.__new__(LocalBrainHost)
        self.host.repair_journal = RepairJournal(Path(self.tmp.name) / "repair.db")
        self.host.self_heal = SelfHealController(journal=self.host.repair_journal)
        self.host.owner_runtime = SimpleNamespace(
            gateway=SimpleNamespace(receipts=[]),
        )
        self.host.self_heal.register_runtime_healer(
            IncidentKind.MODEL_FAILURE,
            lambda incident: True,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_chat_failure_heals_and_retries_once(self):
        brain = FlakyBrain(succeeds_after=1)
        self.host.brain = brain

        result = self.host.handle_message("hello JARVIS")

        self.assertEqual(result, "Recovered response")
        self.assertEqual(brain.calls, 2)
        recent = self.host.repair_journal.recent()
        self.assertEqual(recent[0]["state"], RepairState.HEALED.value)
        self.assertEqual(recent[1]["state"], RepairState.DETECTED.value)

    def test_action_failure_is_never_blindly_replayed(self):
        brain = FlakyBrain(succeeds_after=10)
        self.host.brain = brain

        with self.assertRaises(RuntimeError):
            self.host.handle_message("open calculator")

        self.assertEqual(brain.calls, 1)
        recent = self.host.repair_journal.recent()
        self.assertEqual(recent[0]["state"], RepairState.HEALED.value)
        self.assertEqual(recent[1]["state"], RepairState.DETECTED.value)

    def test_recent_repair_journal_is_bounded_and_validated(self):
        for index in range(5):
            self.host._record_and_heal(
                IncidentKind.MODEL_FAILURE,
                f"failure-{index}",
                "model_runtime",
            )
        self.assertEqual(len(self.host.repair_journal.recent(limit=3)), 3)
        with self.assertRaises(ValueError):
            self.host.repair_journal.recent(limit=0)


if __name__ == "__main__":
    unittest.main()
