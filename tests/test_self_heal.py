"""Bounded autonomous repair policy tests."""

import hashlib
from pathlib import Path
import tempfile
import unittest

from core.self_heal import (
    FileEdit,
    Incident,
    IncidentKind,
    RepairDisposition,
    RepairJournal,
    RepairPatch,
    RepairState,
    SelfHealController,
    SelfHealPolicy,
    VerificationResult,
)


class Engine:
    def __init__(self, patch):
        self.patch = patch

    def propose(self, incident, repository_context):
        return self.patch


class Sandbox:
    def __init__(self, files, *, verify=True):
        self.files = dict(files)
        self.verify_ok = verify
        self.applied = False
        self.published = False
        self.rolled_back = False
        self.snapshot = None

    def create_checkpoint(self, incident):
        self.snapshot = dict(self.files)
        return "checkpoint-1"

    def read_text(self, path):
        return self.files[path]

    def apply(self, patch):
        for edit in patch.edits:
            self.files[edit.path] = edit.replacement
        self.applied = True

    def verify(self, patch):
        return VerificationResult(self.verify_ok, ("unit",), "")

    def publish(self, patch, checkpoint_id):
        self.published = True

    def rollback(self, checkpoint_id):
        if self.snapshot is not None:
            self.files = dict(self.snapshot)
        self.rolled_back = True


def edit(path, old, new):
    return FileEdit(
        path,
        hashlib.sha256(old.encode()).hexdigest(),
        new,
    )


class SelfHealTests(unittest.TestCase):
    def test_runtime_heal_is_automatic(self):
        incident = Incident.create(
            IncidentKind.MODEL_FAILURE,
            "model stalled",
            "model_runtime",
        )
        controller = SelfHealController()
        controller.register_runtime_healer(
            IncidentKind.MODEL_FAILURE,
            lambda item: item.id == incident.id,
        )
        outcome = controller.heal_runtime(incident)
        self.assertEqual(outcome.state, RepairState.HEALED)
        self.assertEqual(outcome.disposition, RepairDisposition.AUTO_APPLY)

    def test_low_risk_patch_can_verify_and_publish_automatically(self):
        incident = Incident.create(
            IncidentKind.CODE_FAILURE,
            "format helper failed",
            "core/helper.py",
        )
        patch = RepairPatch(
            incident.id,
            (edit("core/helper.py", "old", "new"),),
            "repair helper",
            ("unit",),
        )
        sandbox = Sandbox({"core/helper.py": "old"})
        outcome = SelfHealController().repair_code(
            incident,
            engine=Engine(patch),
            sandbox=sandbox,
            repository_context="test",
        )
        self.assertEqual(outcome.state, RepairState.APPLIED)
        self.assertTrue(sandbox.published)
        self.assertEqual(sandbox.files["core/helper.py"], "new")

    def test_security_path_requires_owner_and_cannot_self_approve(self):
        incident = Incident.create(
            IncidentKind.CODE_FAILURE,
            "gateway change",
            "core/action_gateway.py",
        )
        patch = RepairPatch(
            incident.id,
            (edit("core/action_gateway.py", "old", "new"),),
            "change authority",
            ("unit",),
        )
        sandbox = Sandbox({"core/action_gateway.py": "old"})
        outcome = SelfHealController().repair_code(
            incident,
            engine=Engine(patch),
            sandbox=sandbox,
            repository_context="test",
            owner_approved=False,
        )
        self.assertEqual(outcome.state, RepairState.AWAITING_OWNER)
        self.assertFalse(sandbox.applied)

    def test_failed_verification_rolls_back(self):
        incident = Incident.create(
            IncidentKind.TEST_FAILURE,
            "web regression",
            "web/view.tsx",
        )
        patch = RepairPatch(
            incident.id,
            (edit("web/view.tsx", "old", "new"),),
            "repair view",
            ("frontend",),
        )
        sandbox = Sandbox({"web/view.tsx": "old"}, verify=False)
        outcome = SelfHealController().repair_code(
            incident,
            engine=Engine(patch),
            sandbox=sandbox,
            repository_context="test",
        )
        self.assertEqual(outcome.state, RepairState.ROLLED_BACK)
        self.assertTrue(sandbox.rolled_back)
        self.assertEqual(sandbox.files["web/view.tsx"], "old")

    def test_stale_patch_is_rejected_and_rolled_back(self):
        incident = Incident.create(
            IncidentKind.CODE_FAILURE,
            "stale",
            "core/helper.py",
        )
        patch = RepairPatch(
            incident.id,
            (edit("core/helper.py", "expected", "new"),),
            "stale repair",
            ("unit",),
        )
        sandbox = Sandbox({"core/helper.py": "changed already"})
        outcome = SelfHealController().repair_code(
            incident,
            engine=Engine(patch),
            sandbox=sandbox,
            repository_context="test",
        )
        self.assertEqual(outcome.state, RepairState.FAILED)
        self.assertTrue(sandbox.rolled_back)

    def test_forbidden_secret_surface_is_rejected(self):
        incident = Incident.create(
            IncidentKind.CODE_FAILURE,
            "bad",
            "config/secrets.json",
        )
        patch = RepairPatch(
            incident.id,
            (edit("config/secrets.json", "old", "new"),),
            "bad repair",
            ("unit",),
        )
        self.assertEqual(
            SelfHealPolicy().disposition(patch),
            RepairDisposition.REJECT,
        )

    def test_repair_journal_is_durable(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = RepairJournal(Path(tmp) / "repair.db")
            incident = Incident.create(
                IncidentKind.RUNTIME_CRASH,
                "crash",
                "ui",
            )
            journal.record(incident.id, RepairState.DETECTED, "seen")
            reopened = RepairJournal(Path(tmp) / "repair.db")
            self.assertEqual(reopened.history(incident.id)[0]["state"], "detected")


if __name__ == "__main__":
    unittest.main()
