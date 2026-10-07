"""Signed HTTP contracts for phone biometric approvals."""

from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from uuid import uuid4

from fastapi.testclient import TestClient

from api.nexus_sync_server import create_sync_app
from core.nexus.owner_approval import OwnerApprovalManager
from core.nexus.peer_auth import PeerRole
from core.nexus.signed_transport import MEDIA_TYPE
from core.nexus.sync_node import NexusSyncNode
from core.secret_store import NoopSecretStore


class ApprovalHTTPTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.desktop = NexusSyncNode(root / "desktop", "desktop", secret_store=NoopSecretStore())
        self.phone = NexusSyncNode(root / "phone", "phone", secret_store=NoopSecretStore())

        self.desktop.registry.trust_companion("phone", self.phone.identity.public_key)
        self.phone.trust_peer(
            "desktop",
            self.desktop.identity.public_key,
            "http://desktop.tailnet.ts.net:8765",
        )
        self.manager = OwnerApprovalManager(self.desktop.store)
        self.digest = hashlib.sha256(b"sensitive action").hexdigest()
        self.approval = self.manager.create("Apply sensitive JARVIS action", self.digest)
        self.app = create_sync_app(
            self.desktop,
            approvals=self.manager,
            run_scheduler=False,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _post_signed(self, path, envelope):
        with TestClient(self.app) as client:
            return client.post(
                path,
                content=envelope,
                headers={"content-type": MEDIA_TYPE, "accept": MEDIA_TYPE},
            )

    def test_companion_lists_and_approves_with_verified_signed_decision(self):
        request_id = uuid4().hex
        listing = self.phone.authenticator.sign(
            "approval.list",
            "desktop",
            {"version": 1, "request_id": request_id},
            message_id="approval-list:" + request_id,
        )
        response = self._post_signed("/nexus/approval/v1/pending", listing)
        self.assertEqual(response.status_code, 200)

        peer, payload, message_id = self.phone.authenticator.verify(
            response.content,
            expected_kind="approval.pending",
            expected_sender="desktop",
        )
        self.assertEqual(peer.peer_id, "desktop")
        self.assertEqual(message_id, "approval-pending:" + request_id)
        self.assertEqual(payload["pending"][0]["approval_id"], self.approval.approval_id)

        decision_id = uuid4().hex
        decision = self.phone.authenticator.sign(
            "approval.decision",
            "desktop",
            {
                "version": 1,
                "request_id": decision_id,
                "approval_id": self.approval.approval_id,
                "action_digest": self.digest,
                "approved": True,
                "user_verified": True,
                "biometry_type": "fingerprint",
            },
            message_id="approval-decision:" + decision_id,
        )
        receipt = self._post_signed("/nexus/approval/v1/decision", decision)
        self.assertEqual(receipt.status_code, 200)
        _, receipt_payload, receipt_message = self.phone.authenticator.verify(
            receipt.content,
            expected_kind="approval.receipt",
            expected_sender="desktop",
        )
        self.assertEqual(receipt_message, "approval-receipt:" + decision_id)
        self.assertEqual(receipt_payload["state"], "approved")
        self.assertEqual(receipt_payload["action_digest"], self.digest)
        self.assertEqual(self.manager.get(self.approval.approval_id).decided_by, "phone")

    def test_unverified_decision_is_rejected(self):
        request_id = uuid4().hex
        decision = self.phone.authenticator.sign(
            "approval.decision",
            "desktop",
            {
                "version": 1,
                "request_id": request_id,
                "approval_id": self.approval.approval_id,
                "action_digest": self.digest,
                "approved": True,
                "user_verified": False,
                "biometry_type": "fingerprint",
            },
            message_id="approval-decision:" + request_id,
        )
        response = self._post_signed("/nexus/approval/v1/decision", decision)
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.manager.get(self.approval.approval_id).state, "pending")


    def test_non_fingerprint_biometry_is_rejected(self):
        request_id = uuid4().hex
        decision = self.phone.authenticator.sign(
            "approval.decision",
            "desktop",
            {
                "version": 1,
                "request_id": request_id,
                "approval_id": self.approval.approval_id,
                "action_digest": self.digest,
                "approved": True,
                "user_verified": True,
                "biometry_type": "face",
            },
            message_id="approval-decision:" + request_id,
        )
        response = self._post_signed("/nexus/approval/v1/decision", decision)
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.manager.get(self.approval.approval_id).state, "pending")

    def test_signed_digest_substitution_is_rejected_before_decision(self):
        request_id = uuid4().hex
        substituted = hashlib.sha256(b"different action").hexdigest()
        decision = self.phone.authenticator.sign(
            "approval.decision",
            "desktop",
            {
                "version": 1,
                "request_id": request_id,
                "approval_id": self.approval.approval_id,
                "action_digest": substituted,
                "approved": True,
                "user_verified": True,
                "biometry_type": "fingerprint",
            },
            message_id="approval-decision:" + request_id,
        )
        response = self._post_signed("/nexus/approval/v1/decision", decision)
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.manager.get(self.approval.approval_id).state, "pending")

    def test_signature_rejects_tampering_with_approval_id_and_digest(self):
        request_id = uuid4().hex
        decision = self.phone.authenticator.sign(
            "approval.decision",
            "desktop",
            {
                "version": 1,
                "request_id": request_id,
                "approval_id": self.approval.approval_id,
                "action_digest": self.digest,
                "approved": True,
                "user_verified": True,
                "biometry_type": "fingerprint",
            },
            message_id="approval-decision:" + request_id,
        )
        other_digest = hashlib.sha256(b"substituted action").hexdigest()
        other = self.manager.create("Substituted action", other_digest)
        tampered = json.loads(decision)
        tampered["payload"]["approval_id"] = other.approval_id
        tampered["payload"]["action_digest"] = other_digest

        response = self._post_signed(
            "/nexus/approval/v1/decision",
            json.dumps(tampered, separators=(",", ":")).encode("utf-8"),
        )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.manager.get(self.approval.approval_id).state, "pending")
        self.assertEqual(self.manager.get(other.approval_id).state, "pending")

    def test_signed_approval_decision_cannot_be_replayed(self):
        request_id = uuid4().hex
        decision = self.phone.authenticator.sign(
            "approval.decision",
            "desktop",
            {
                "version": 1,
                "request_id": request_id,
                "approval_id": self.approval.approval_id,
                "action_digest": self.digest,
                "approved": True,
                "user_verified": True,
                "biometry_type": "fingerprint",
            },
            message_id="approval-decision:" + request_id,
        )
        self.assertEqual(
            self._post_signed("/nexus/approval/v1/decision", decision).status_code,
            200,
        )
        self.assertEqual(
            self._post_signed("/nexus/approval/v1/decision", decision).status_code,
            403,
        )
        value = self.manager.get(self.approval.approval_id)
        self.assertEqual(value.state, "approved")
        self.assertFalse(value.consumed)

    def test_node_role_cannot_masquerade_as_companion_approver(self):
        rogue = NexusSyncNode(
            Path(self.tmp.name) / "rogue",
            "rogue",
            secret_store=NoopSecretStore(),
        )
        self.desktop.trust_peer(
            "rogue",
            rogue.identity.public_key,
            "http://rogue.tailnet.ts.net:8765",
        )
        rogue.trust_peer(
            "desktop",
            self.desktop.identity.public_key,
            "http://desktop.tailnet.ts.net:8765",
        )
        request_id = uuid4().hex
        listing = rogue.authenticator.sign(
            "approval.list",
            "desktop",
            {"version": 1, "request_id": request_id},
            message_id="approval-list:" + request_id,
        )
        response = self._post_signed("/nexus/approval/v1/pending", listing)
        self.assertEqual(response.status_code, 403)


if __name__ == "__main__":
    unittest.main()
