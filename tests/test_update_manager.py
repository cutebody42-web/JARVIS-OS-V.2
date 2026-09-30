"""Major update approval and rollback contracts."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from core.nexus.event_store import EventStore
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
        self.assertFalse(installer.staged)

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

    def test_patch_update_can_apply_without_biometric_gate(self):
        update = plan(UpdateClass.PATCH, ("docs/readme.md",))
        installer = Installer()
        result = UpdateManager(installer).apply(update)
        self.assertEqual(result.state, UpdateState.APPLIED)

    def test_failed_verification_rolls_back(self):
        update = plan(UpdateClass.PATCH, ("web/view.tsx",))
        installer = Installer(verify=False)
        result = UpdateManager(installer).apply(update)
        self.assertEqual(result.state, UpdateState.ROLLED_BACK)
        self.assertTrue(installer.rolled_back)


if __name__ == "__main__":
    unittest.main()
