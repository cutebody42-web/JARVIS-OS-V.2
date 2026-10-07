"""Major update approval and rollback contracts."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from core.nexus.event_store import EventStore
from core.nexus.owner_approval import OwnerApprovalManager
from core.nexus.peer_auth import DeviceSigner, PeerRegistry
from core.update_manager import (
    UpdateApprovalGate,
    UpdateClass,
    UpdateManager,
    UpdatePlan,
    UpdatePolicy,
    UpdateState,
)


NOW = datetime(2026, 9, 30, 14, 0, tzinfo=timezone.utc)


class Installer:
    def __init__(self, *, verify=True, fail_apply=False):
        self.verify_ok = verify
        self.fail_apply = fail_apply
        self.staged = False
        self.applied = False
        self.rolled_back = False

    def stage(self, plan):
        self.staged = True
        return "checkpoint"

    def apply(self, plan, checkpoint_id):
        if self.fail_apply:
            raise RuntimeError("install failed")
        self.applied = True

    def verify(self, plan):
        return self.verify_ok

    def rollback(self, checkpoint_id):
        self.rolled_back = True


def plan(update_class=UpdateClass.MAJOR, paths=("core/owner_kernel.py",)):
    return UpdatePlan(
        version="2.1.0",
        commit_sha="a" * 40,
        artifact_sha256="b" * 64,
        changed_paths=tuple(paths),
        notes="test update",
        update_class=update_class,
    )


class UpdateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = EventStore(Path(self.tmp.name) / "state", "desktop")
        self.registry = PeerRegistry(self.store)
        self.phone_signer = DeviceSigner.generate()
        self.registry.trust_peer(
            "phone",
            self.phone_signer.public_b64,
            "http://phone.tailnet.ts.net:8765",
        )
        self.gate = UpdateApprovalGate(
            self.registry,
            clock=lambda: NOW,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_major_update_waits_for_companion_biometric_approval(self):
        installer = Installer()
        manager = UpdateManager(installer, approval_gate=self.gate)
        result = manager.apply(plan())
        self.assertEqual(result.state, UpdateState.AWAITING_APPROVAL)
        self.assertTrue(installer.staged)
        self.assertFalse(installer.applied)

    def test_valid_biometric_approval_is_one_time_and_applies_major_update(self):
        update = plan()
        challenge = self.gate.create_challenge(update)
        approval = UpdateApprovalGate.sign_on_companion(
            "phone",
            challenge,
            self.phone_signer,
            user_verified=True,
        )
        installer = Installer()
        manager = UpdateManager(installer, approval_gate=self.gate)

        result = manager.apply(update, approval=approval)

        self.assertEqual(result.state, UpdateState.APPLIED)
        self.assertTrue(installer.applied)
        with self.assertRaises(PermissionError):
            self.gate.verify_and_consume(update, approval)

    def test_major_update_can_use_generic_phone_biometric_approval_queue(self):
        approvals = OwnerApprovalManager(self.store, clock=lambda: NOW)
        installer = Installer()
        manager = UpdateManager(installer, owner_approvals=approvals)
        update = plan()

        waiting = manager.apply(update)
        self.assertEqual(waiting.state, UpdateState.AWAITING_APPROVAL)
        self.assertIsNotNone(waiting.approval_id)
        self.assertTrue(installer.staged)

        approvals.decide(
            waiting.approval_id,
            peer_id="phone",
            approved=True,
            user_verified=True,
        )
        applied = manager.apply(update, approval_id=waiting.approval_id)
        self.assertEqual(applied.state, UpdateState.APPLIED)
        self.assertTrue(installer.applied)

    def test_unverified_mobile_approval_is_rejected_before_signature(self):
        update = plan()
        challenge = self.gate.create_challenge(update)
        with self.assertRaises(PermissionError):
            UpdateApprovalGate.sign_on_companion(
                "phone",
                challenge,
                self.phone_signer,
                user_verified=False,
            )

    def test_wrong_phone_key_is_rejected(self):
        update = plan()
        challenge = self.gate.create_challenge(update)
        rogue = DeviceSigner.generate()
        approval = UpdateApprovalGate.sign_on_companion(
            "phone",
            challenge,
            rogue,
            user_verified=True,
        )
        with self.assertRaises(PermissionError):
            self.gate.verify_and_consume(update, approval)

    def test_expired_challenge_is_rejected(self):
        update = plan()
        challenge = self.gate.create_challenge(update, ttl_seconds=10)
        approval = UpdateApprovalGate.sign_on_companion(
            "phone",
            challenge,
            self.phone_signer,
            user_verified=True,
        )
        late_gate = UpdateApprovalGate(
            self.registry,
            clock=lambda: NOW + timedelta(seconds=11),
        )
        with self.assertRaises(PermissionError):
            late_gate.verify_and_consume(update, approval)

    def test_underclassified_security_update_is_rejected(self):
        bad = plan(UpdateClass.PATCH, ("core/owner_kernel.py",))
        with self.assertRaises(PermissionError):
            UpdatePolicy().validate_declared_class(bad)

    def test_patch_and_minor_updates_wait_for_explicit_owner_approval(self):
        cases = (
            plan(UpdateClass.PATCH, ("docs/readme.md",)),
            plan(UpdateClass.MINOR, ("actions/helper.py",)),
        )
        for update in cases:
            with self.subTest(update_class=update.update_class):
                installer = Installer()
                result = UpdateManager(installer).apply(update)
                self.assertEqual(result.state, UpdateState.AWAITING_APPROVAL)
                self.assertTrue(installer.staged)
                self.assertFalse(installer.applied)

    def test_failed_verification_rolls_back(self):
        update = plan(UpdateClass.PATCH, ("web/view.tsx",))
        challenge = self.gate.create_challenge(update)
        approval = UpdateApprovalGate.sign_on_companion(
            "phone", challenge, self.phone_signer, user_verified=True,
        )
        installer = Installer(verify=False)
        result = UpdateManager(installer, approval_gate=self.gate).apply(
            update, approval=approval,
        )
        self.assertEqual(result.state, UpdateState.ROLLED_BACK)
        self.assertTrue(installer.rolled_back)

    def test_injected_policy_cannot_bypass_patch_approval(self):
        class UnsafePolicy(UpdatePolicy):
            def requires_owner_approval(self, plan):
                return False

        update = plan(UpdateClass.PATCH, ("docs/readme.md",))
        installer = Installer()
        result = UpdateManager(installer, policy=UnsafePolicy()).apply(update)
        self.assertEqual(result.state, UpdateState.AWAITING_APPROVAL)
        self.assertFalse(installer.applied)

    def test_all_version_and_path_categories_require_approval_under_unsafe_policy(self):
        class UnsafePolicy(UpdatePolicy):
            def requires_owner_approval(self, plan):
                return False

        cases = (
            (UpdateClass.MAJOR, "core/helper.py"),
            (UpdateClass.MAJOR, "models/router.yaml"),
            (UpdateClass.MINOR, "skills/personal/SKILL.md"),
            (UpdateClass.MINOR, "config/brain-policy.yaml"),
            (UpdateClass.MINOR, "policy/authority.yaml"),
            (UpdateClass.MAJOR, "packaging/windows/setup.nsi"),
            (UpdateClass.PATCH, "tests/test_helper.py"),
            (UpdateClass.PATCH, "docs/runtime.md"),
        )
        for update_class, path in cases:
            with self.subTest(update_class=update_class, path=path):
                update = plan(update_class, (path,))
                installer = Installer()
                result = UpdateManager(installer, policy=UnsafePolicy()).apply(update)
                self.assertEqual(result.state, UpdateState.AWAITING_APPROVAL)
                self.assertTrue(installer.staged)
                self.assertFalse(installer.applied)

    def test_owner_approval_subclass_is_rejected_at_install_boundary(self):
        class UnsafeApprovals(OwnerApprovalManager):
            def _row(self, row):
                value = OwnerApprovalManager._row(row)
                return replace(value, state="approved", decided_by="forged")

        approvals = UnsafeApprovals(self.store, clock=lambda: NOW)
        installer = Installer()
        with self.assertRaises(TypeError):
            UpdateManager(installer, owner_approvals=approvals)
        self.assertFalse(installer.applied)

    def test_signature_gate_subclass_is_rejected_at_install_boundary(self):
        class UnsafeGate(UpdateApprovalGate):
            def verify_and_consume(self, plan, approval):
                return True

        gate = UnsafeGate(self.registry, clock=lambda: NOW)
        installer = Installer()
        with self.assertRaises(TypeError):
            UpdateManager(installer, approval_gate=gate)
        self.assertFalse(installer.applied)

    def test_duck_typed_approval_verifiers_are_rejected(self):
        class Noop:
            def verify_and_consume(self, *args, **kwargs):
                return True

            def consume(self, *args, **kwargs):
                return None

        with self.assertRaises(TypeError):
            UpdateManager(Installer(), approval_gate=Noop())
        with self.assertRaises(TypeError):
            UpdateManager(Installer(), owner_approvals=Noop())

    def test_update_plan_subclass_cannot_override_the_approval_digest(self):
        class ForgedPlan(UpdatePlan):
            def digest(self):
                return "0" * 64

        update = ForgedPlan(
            version="2.1.0",
            commit_sha="a" * 40,
            artifact_sha256="b" * 64,
            changed_paths=("docs/readme.md",),
            notes="forged digest",
            update_class=UpdateClass.PATCH,
        )
        installer = Installer()
        with self.assertRaises(TypeError):
            UpdateManager(installer).apply(update)
        self.assertFalse(installer.staged)
        self.assertFalse(installer.applied)

    def test_patch_approval_is_bound_to_exact_update_digest(self):
        approvals = OwnerApprovalManager(self.store, clock=lambda: NOW)
        installer = Installer()
        manager = UpdateManager(installer, owner_approvals=approvals)
        original = plan(UpdateClass.PATCH, ("docs/readme.md",))
        waiting = manager.apply(original)
        approvals.decide(
            waiting.approval_id, peer_id="phone", approved=True, user_verified=True,
        )
        altered = UpdatePlan.from_dict({**original.to_dict(), "notes": "altered after approval"})
        with self.assertRaises(PermissionError):
            manager.apply(altered, approval_id=waiting.approval_id)
        self.assertFalse(approvals.get(waiting.approval_id).consumed)
        self.assertFalse(installer.applied)

    def test_major_candidate_stage_failure_creates_no_approval(self):
        class BrokenInstaller(Installer):
            def stage(self, plan):
                raise RuntimeError("candidate integrity failed")
        approvals = OwnerApprovalManager(self.store, clock=lambda: NOW)
        installer = BrokenInstaller()
        result = UpdateManager(installer, owner_approvals=approvals).apply(plan())
        self.assertEqual(result.state, UpdateState.FAILED)
        self.assertFalse(installer.applied)
        self.assertEqual(approvals.pending(), ())

    def test_noncanonical_cross_platform_paths_are_rejected(self):
        for path in ("core/../docs/readme.md", "core\\owner_kernel.py", "C:/jarvis.py", ".", "docs//a.md", "docs/CON.txt"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                plan(UpdateClass.MAJOR, (path,))

    def test_security_classification_cannot_be_bypassed_by_case(self):
        self.assertEqual(UpdatePolicy().classify_paths(("CORE/UPDATE_MANAGER.PY",)), UpdateClass.MAJOR)

    def test_scratch_core_receipts_and_training_are_major_updates(self):
        for path in ("core/scratch_activation.py", "core/ollama_bootstrap.py", "training/activate.py"):
            with self.subTest(path=path):
                self.assertEqual(UpdatePolicy().classify_paths((path,)), UpdateClass.MAJOR)

    def test_release_metadata_is_bound_to_approval_digest(self):
        original = plan()
        modified = UpdatePlan.from_dict({**original.to_dict(), "release_tag": "v2.1.0"})
        self.assertNotEqual(original.digest(), modified.digest())

    def test_exact_expiration_boundary_is_not_a_valid_approval(self):
        update = plan()
        challenge = self.gate.create_challenge(update, ttl_seconds=10)
        approval = UpdateApprovalGate.sign_on_companion("phone", challenge, self.phone_signer, user_verified=True)
        boundary = UpdateApprovalGate(self.registry, clock=lambda: NOW + timedelta(seconds=10))
        with self.assertRaises(PermissionError):
            boundary.verify_and_consume(update, approval)


if __name__ == "__main__":
    unittest.main()
