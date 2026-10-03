"""Desktop update controls with real one-time approval storage and no installer."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from api.jarvis_local_server import LocalBrainHost, create_local_brain_app
from core.local_update_service import LocalUpdateService
from core.nexus.event_store import EventStore
from core.nexus.owner_approval import OwnerApprovalManager
from core.update_manager import UpdateClass, UpdateOutcome, UpdatePlan, UpdateState


PLAN = UpdatePlan(
    version="2.0.1", commit_sha="a" * 40, artifact_sha256="b" * 64,
    changed_paths=("core/owner_kernel.py",), notes="Verified release",
    update_class=UpdateClass.MAJOR, release_tag="v2.0.1", artifact_name="JARVIS-Setup.exe",
)


class FakeCoordinator:
    def __init__(self, approvals):
        self.approvals = approvals
        self.installed_commit_sha = "0" * 40
        self.source = SimpleNamespace(discover=Mock(side_effect=self.discover))
        self.plan = PLAN
        self.discover_error = None
        self.stage_error = None
        self.discover_entered = threading.Event()
        self.discover_release = threading.Event()
        self.discover_release.set()
        self.stage_entered = threading.Event()
        self.stage_release = threading.Event()
        self.stage_release.set()
        self.prepared = []
        self.handoffs = []

    def discover(self):
        self.discover_entered.set()
        if not self.discover_release.wait(timeout=3):
            raise RuntimeError("test discovery was not released")
        if self.discover_error:
            raise self.discover_error
        return self.plan

    def prepare(self, plan):
        self.prepared.append(plan)
        self.stage_entered.set()
        if not self.stage_release.wait(timeout=3):
            raise RuntimeError("test staging was not released")
        if self.stage_error:
            raise self.stage_error
        request = self.approvals.create("Install verified update", plan.digest(), ttl_seconds=15)
        return UpdateOutcome(
            UpdateState.AWAITING_APPROVAL, "Use your phone fingerprint", f"{len(self.prepared):032x}",
            request.approval_id,
        )

    def authorize_handoff(self, checkpoint_id, *, approval_id, parent_pid):
        # Exercise the real exact-digest, one-time approval boundary. This fake
        # records a handoff and never starts a process or downloads an artifact.
        self.approvals.consume(approval_id, action_digest=PLAN.digest())
        self.handoffs.append((checkpoint_id, approval_id, parent_pid))
        return UpdateOutcome(UpdateState.APPLYING, "Ready to close desktop", checkpoint_id)

    def history(self):
        return ()


class LocalUpdateServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.now = datetime(2026, 10, 3, tzinfo=timezone.utc)
        self.approvals = OwnerApprovalManager(EventStore(self.root / "nexus", "desktop"), clock=lambda: self.now)
        self.coordinator = FakeCoordinator(self.approvals)
        self.service = LocalUpdateService(
            state_dir=self.root, approvals=self.approvals, parent_pid=42, coordinator=self.coordinator,
        )
        self.addCleanup(self.coordinator.discover_release.set)
        self.addCleanup(self.coordinator.stage_release.set)

    def idle_status(self):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            result = self.service.status()
            if not result["busy"]:
                return result
            threading.Event().wait(0.001)
        self.fail("update worker did not finish")

    def staged(self):
        self.service.check()
        self.assertEqual(self.idle_status()["phase"], "available")
        self.service.prepare()
        result = self.idle_status()
        self.assertEqual(result["phase"], "awaiting_approval")
        return result

    def approve(self, approval_id):
        return self.approvals.decide(approval_id, peer_id="phone", approved=True, user_verified=True)

    def test_release_check_returns_while_discovery_is_still_blocked(self):
        self.coordinator.discover_release.clear()
        try:
            result = self.service.check()
            self.assertTrue(self.coordinator.discover_entered.wait(timeout=1))
            self.assertTrue(result["busy"])
            self.assertEqual(result["phase"], "checking")
            self.assertEqual(self.coordinator.handoffs, [])
            with self.assertRaises(RuntimeError):
                self.service.check()
        finally:
            self.coordinator.discover_release.set()
        self.assertEqual(self.idle_status()["phase"], "available")

    def test_staging_is_background_work_and_does_not_start_installation(self):
        self.service.check()
        self.idle_status()
        self.coordinator.stage_release.clear()
        try:
            result = self.service.prepare()
            self.assertTrue(self.coordinator.stage_entered.wait(timeout=1))
            self.assertEqual(result["phase"], "staging")
            self.assertTrue(result["busy"])
            self.assertEqual(self.approvals.pending(), ())
            with self.assertRaises(ValueError):
                self.service.apply(checkpoint_id="0" * 32, approval_id="guess")
        finally:
            self.coordinator.stage_release.set()
        result = self.idle_status()
        self.assertEqual(result["approval_state"], "pending")
        self.assertEqual(self.coordinator.handoffs, [])

    def test_pending_and_approved_stages_cannot_be_replaced_by_another_check(self):
        stage = self.staged()
        for approve in (False, True):
            with self.subTest(approved=approve):
                if approve:
                    self.approve(stage["approval_id"])
                with self.assertRaises(RuntimeError):
                    self.service.check()
                self.assertEqual(self.service.status()["checkpoint_id"], stage["checkpoint_id"])
        self.coordinator.source.discover.assert_called_once()

    def test_wrong_checkpoint_or_approval_never_reaches_handoff(self):
        stage = self.staged()
        self.approve(stage["approval_id"])
        for checkpoint, approval in (
            ("f" * 32, stage["approval_id"]), (stage["checkpoint_id"], "wrong-approval"),
        ):
            with self.subTest(checkpoint=checkpoint, approval=approval), self.assertRaises(PermissionError):
                self.service.apply(checkpoint_id=checkpoint, approval_id=approval)
        self.assertEqual(self.coordinator.handoffs, [])
        self.assertFalse(self.approvals.get(stage["approval_id"]).consumed)

    def test_pending_approval_cannot_install_and_verified_approval_is_used_once(self):
        stage = self.staged()
        with self.assertRaises(PermissionError):
            self.service.apply(checkpoint_id=stage["checkpoint_id"], approval_id=stage["approval_id"])
        self.assertEqual(self.coordinator.handoffs, [])
        self.approve(stage["approval_id"])
        result = self.service.apply(checkpoint_id=stage["checkpoint_id"], approval_id=stage["approval_id"])
        self.assertEqual(result["phase"], "applying")
        self.assertEqual(result["approval_state"], "consumed")
        self.assertEqual(self.coordinator.handoffs, [(stage["checkpoint_id"], stage["approval_id"], 42)])
        with self.assertRaises(ValueError):
            self.service.apply(checkpoint_id=stage["checkpoint_id"], approval_id=stage["approval_id"])
        with self.assertRaises(RuntimeError):
            self.service.check()

    def test_expired_approval_can_be_checked_and_staged_again(self):
        stage = self.staged()
        self.now += timedelta(seconds=16)
        self.assertEqual(self.service.status()["approval_state"], "expired")
        self.service.check()
        self.assertEqual(self.idle_status()["phase"], "available")
        self.service.prepare()
        fresh = self.idle_status()
        self.assertNotEqual(fresh["approval_id"], stage["approval_id"])
        self.assertNotEqual(fresh["checkpoint_id"], stage["checkpoint_id"])
        self.assertEqual(fresh["approval_state"], "pending")

    def test_rejected_stage_can_be_prepared_again_without_losing_plan(self):
        stage = self.staged()
        self.approvals.decide(stage["approval_id"], peer_id="phone", approved=False, user_verified=True)
        self.service.prepare()
        fresh = self.idle_status()
        self.assertEqual(fresh["phase"], "awaiting_approval")
        self.assertNotEqual(fresh["approval_id"], stage["approval_id"])
        self.assertEqual(self.coordinator.prepared, [PLAN, PLAN])

    def test_discovery_and_staging_errors_clear_busy_and_allow_recovery(self):
        self.coordinator.discover_error = OSError("release metadata unavailable")
        self.service.check()
        failed = self.idle_status()
        self.assertEqual(failed["phase"], "failed")
        self.assertIn("release metadata unavailable", failed["message"])
        self.coordinator.discover_error = None
        self.service.check()
        self.assertEqual(self.idle_status()["phase"], "available")
        self.coordinator.stage_error = RuntimeError("candidate hash mismatch")
        self.service.prepare()
        self.assertEqual(self.idle_status()["phase"], "failed")
        self.assertEqual(self.coordinator.handoffs, [])
        self.coordinator.stage_error = None
        self.service.check()
        self.idle_status()
        self.service.prepare()
        self.assertEqual(self.idle_status()["phase"], "awaiting_approval")

    def test_no_release_and_current_commit_do_not_request_approval(self):
        for plan, phase in ((None, "idle"), (PLAN, "current")):
            with self.subTest(phase=phase):
                self.coordinator.plan = plan
                self.coordinator.installed_commit_sha = PLAN.commit_sha
                self.service.check()
                self.assertEqual(self.idle_status()["phase"], phase)
                with self.assertRaises(ValueError):
                    self.service.prepare()
        self.assertEqual(self.approvals.pending(), ())
        self.assertEqual(self.coordinator.prepared, [])

    def test_source_build_has_no_installation_capability_or_network_access(self):
        with patch("core.local_update_service.sys.platform", "linux"), patch("core.local_update_service.sys.frozen", False, create=True), patch("core.local_update_service.GitHubReleaseSource") as source:
            service = LocalUpdateService(state_dir=self.root, approvals=self.approvals, parent_pid=42)
            self.assertFalse(service.status()["supported"])
            with self.assertRaises(ValueError):
                service.check()
            source.assert_not_called()

    def test_missing_packaged_provenance_disables_updater(self):
        with patch("core.local_update_service.sys.platform", "win32"), patch("core.local_update_service.sys.frozen", True, create=True), patch("core.local_update_service.read_build_info", side_effect=ValueError("invalid metadata")), patch("core.local_update_service.WindowsReleaseUpdateCoordinator") as coordinator:
            service = LocalUpdateService(state_dir=self.root, approvals=self.approvals, parent_pid=42)
            result = service.status()
            self.assertFalse(result["supported"])
            self.assertIn("provenance", result["message"])
            coordinator.assert_not_called()


class LocalUpdateHTTPTests(unittest.TestCase):
    def setUp(self):
        self.host = LocalBrainHost.__new__(LocalBrainHost)
        self.host.ui_token = "isolated-update-owner-token-" * 2
        self.host.updates = Mock()
        self.host.close = Mock()
        self.client = TestClient(create_local_brain_app(self.host))
        self.headers = {"Authorization": "Bearer " + self.host.ui_token}
        self.request = {"checkpoint_id": "a" * 32, "approval_id": "approval-1"}

    def test_every_update_endpoint_requires_the_exact_owner_token(self):
        for endpoint in ("status", "check", "prepare", "apply"):
            for headers in ({}, {"Authorization": "Bearer wrong-token"}):
                with self.subTest(endpoint=endpoint, headers=headers):
                    response = self.client.get("/v1/update/status", headers=headers) if endpoint == "status" else self.client.post(
                        "/v1/update/" + endpoint, headers=headers,
                        json=self.request if endpoint == "apply" else {},
                    )
                    self.assertEqual(response.status_code, 401)
        self.assertEqual(self.host.updates.mock_calls, [])

    def test_authorized_apply_forwards_only_the_current_binding(self):
        self.host.updates.apply.return_value = {"phase": "applying"}
        response = self.client.post("/v1/update/apply", headers=self.headers, json=self.request)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["phase"], "applying")
        self.host.updates.apply.assert_called_once_with(**self.request)

    def test_permission_failures_are_forbidden_and_state_conflicts_are_retryable(self):
        for error, code in ((PermissionError("fingerprint required"), 403), (ValueError("wrong checkpoint"), 409), (RuntimeError("already staging"), 409)):
            with self.subTest(error=type(error).__name__):
                self.host.updates.apply.side_effect = error
                response = self.client.post("/v1/update/apply", headers=self.headers, json=self.request)
                self.assertEqual(response.status_code, code)
                self.assertEqual(response.json()["detail"], str(error))

    def test_invalid_apply_binding_cannot_reach_the_service(self):
        for payload in ({}, {**self.request, "checkpoint_id": "not-a-checkpoint"}, {**self.request, "approval_id": ""}):
            with self.subTest(payload=payload):
                response = self.client.post("/v1/update/apply", headers=self.headers, json=payload)
                self.assertEqual(response.status_code, 422)
        self.host.updates.apply.assert_not_called()


if __name__ == "__main__":
    unittest.main()
