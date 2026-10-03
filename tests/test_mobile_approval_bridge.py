"""End-to-end exact action resumption after paired-phone biometric approval."""

import os
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from core.action_contracts import ActionStatus
from core.action_gateway import create_runtime
from core.mobile_approval_bridge import MobileApprovalBridge
from core.nexus.event_store import EventStore
from core.nexus.owner_approval import OwnerApprovalManager


class MobileApprovalBridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.workspace = root / "workspace"
        self._qa_workspace = patch.dict(
            os.environ,
            {"JARVIS_QA_WORKSPACE": str(self.workspace)},
            clear=False,
        )
        self._qa_workspace.start()
        self.runtime = create_runtime(
            owner_id="local-owner",
            workspace_root=self.workspace,
            environment="cloud",
        )
        self.store = EventStore(root / "nexus", "desktop")
        self.approvals = OwnerApprovalManager(self.store)
        self.bridge = MobileApprovalBridge(self.approvals, self.runtime)

    def tearDown(self):
        self.runtime.gateway.close()
        self._qa_workspace.stop()
        self.tmp.cleanup()

    def _pending_create(self, name="approved.txt", content="hello"):
        receipt = self.runtime.gateway.run_tool(
            "file_controller",
            {"action": "create_file", "path": name, "content": content},
            route="owner",
            runtime=self.runtime,
        )
        self.assertEqual(receipt.result.status, ActionStatus.REQUIRE_CONFIRMATION)
        self.assertFalse((self.workspace / name).exists())
        bridged = self.bridge.queue_receipt(receipt)
        return receipt, bridged

    def test_fingerprint_approved_exact_request_executes_once(self):
        receipt, bridged = self._pending_create()
        self.approvals.decide(
            bridged.approval_id,
            peer_id="phone",
            approved=True,
            user_verified=True,
        )

        outcomes = self.bridge.reconcile()

        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0].state, "succeeded")
        self.assertEqual((self.workspace / "approved.txt").read_text("utf-8"), "hello")
        self.assertTrue(self.approvals.get(bridged.approval_id).consumed)
        self.assertEqual(self.bridge.reconcile(), ())

    def test_rejected_fingerprint_request_never_executes(self):
        _, bridged = self._pending_create("rejected.txt", "never")
        self.approvals.decide(
            bridged.approval_id,
            peer_id="phone",
            approved=False,
            user_verified=True,
        )

        outcomes = self.bridge.reconcile()

        self.assertEqual(outcomes[0].state, "rejected")
        self.assertFalse((self.workspace / "rejected.txt").exists())
        self.assertEqual(self.runtime.owner.pending(), ())

    def test_phone_approval_cannot_be_rebound_to_different_digest(self):
        receipt, bridged = self._pending_create("bound.txt", "exact")
        self.approvals.decide(
            bridged.approval_id,
            peer_id="phone",
            approved=True,
            user_verified=True,
        )

        # Tampering with the process-local bridge binding fails closed because
        # the durable phone decision is bound to the original receipt digest.
        object.__setattr__(bridged, "action_digest", "0" * 64)
        outcomes = self.bridge.reconcile()

        self.assertEqual(outcomes[0].state, "failed")
        self.assertFalse((self.workspace / "bound.txt").exists())
        self.assertEqual(receipt.normalized_argument_digest != "0" * 64, True)

    def test_same_pending_request_does_not_create_duplicate_phone_prompts(self):
        receipt, first = self._pending_create("one.txt", "1")
        second = self.bridge.queue_receipt(receipt)
        self.assertEqual(first.approval_id, second.approval_id)
        self.assertEqual(len(self.approvals.pending()), 1)

    def test_approved_request_cannot_execute_after_approval_deadline(self):
        now = [datetime.now(timezone.utc)]
        self.approvals._clock = lambda: now[0]
        _, bridged = self._pending_create("expired.txt", "never")
        self.approvals.decide(
            bridged.approval_id, peer_id="phone", approved=True, user_verified=True,
        )
        now[0] += timedelta(seconds=181)
        outcomes = self.bridge.reconcile()
        self.assertEqual(outcomes[0].state, "expired")
        self.assertFalse((self.workspace / "expired.txt").exists())
        self.assertEqual(self.runtime.owner.pending(), ())

    def test_concurrent_reconciliation_cannot_cancel_the_authorized_execution(self):
        _, bridged = self._pending_create("concurrent.txt", "once")
        self.approvals.decide(
            bridged.approval_id, peer_id="phone", approved=True, user_verified=True,
        )
        execute = self.runtime.execute_approved
        entered = threading.Event()
        release = threading.Event()

        def held_execution(*args, **kwargs):
            entered.set()
            if not release.wait(timeout=3):
                raise RuntimeError("test execution was not released")
            return execute(*args, **kwargs)

        with patch.object(self.runtime, "execute_approved", side_effect=held_execution):
            with ThreadPoolExecutor(max_workers=3) as pool:
                first = pool.submit(self.bridge.reconcile)
                self.assertTrue(entered.wait(timeout=3))
                followers = [pool.submit(self.bridge.reconcile) for _ in range(2)]
                try:
                    # A second reconciler must wait, rather than consuming the
                    # same approval and cancelling the first in-flight action.
                    done, _ = wait(followers, timeout=0.1)
                    self.assertFalse(done)
                finally:
                    release.set()
                self.assertEqual(first.result(timeout=3)[0].state, "succeeded")
                for follower in followers:
                    self.assertEqual(follower.result(timeout=3), ())
        self.assertEqual((self.workspace / "concurrent.txt").read_text("utf-8"), "once")
        self.assertEqual(len(self.bridge.recent), 1)


if __name__ == "__main__":
    unittest.main()
