"""Authenticated HTTP consent channel, using real auth and an isolated test DB."""
from dataclasses import replace
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import delete

from api.database import SessionLocal, init_db
from api.models import User
from api.security import create_access_token, hash_password
from api.server import app, live_sessions
from core.action_gateway import OwnerRuntime
from core.authority_contracts import PolicyContext


class OwnerApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def setUp(self):
        # Exercise production consent semantics with only isolated test files.
        # QA's independent additional restrictions have their own gateway tests.
        env = patch.dict(os.environ, {"JARVIS_QA_MODE": "0"})
        env.start()
        self.addCleanup(env.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.user_id = str(uuid.uuid4())
        self.other_id = str(uuid.uuid4())
        with SessionLocal.begin() as db:
            for uid in (self.user_id, self.other_id):
                db.add(User(id=uid, email=f"{uid}@example.com", display_name="Test owner",
                            password_hash=hash_password("owner-api-test-password")))
        self.headers = {"Authorization": "Bearer " + create_access_token(self.user_id)}
        self.other_headers = {"Authorization": "Bearer " + create_access_token(self.other_id)}
        self.context = PolicyContext(self.user_id, str(uuid.uuid4()), Path(self.tmp.name) / "workspace")
        self.runtime = OwnerRuntime(self.context)
        self.engine = SimpleNamespace(owner_runtime=self.runtime)
        live_sessions._sessions[self.user_id] = (self.engine, None, SimpleNamespace(done=lambda: False))
        self.receipt = self.runtime.gateway.run_tool("file_controller", {
            "action": "write", "path": "review.txt", "content": "exact content",
        }, runtime=self.runtime)
        self.approval = {"session_id": self.context.session_id,
                         "expected_digest": self.receipt.normalized_argument_digest, "ttl_seconds": 120}
        self.base = f"/owner/actions/{self.receipt.request_id}"

    def tearDown(self):
        live_sessions._sessions.pop(self.user_id, None)
        live_sessions._sessions.pop(self.other_id, None)
        self.runtime.gateway.close()
        self.tmp.cleanup()
        with SessionLocal.begin() as db:
            db.execute(delete(User).where(User.id.in_([self.user_id, self.other_id])))

    def approve(self):
        response = self.client.post(self.base + "/approve", headers=self.headers, json=self.approval)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["ticket_id"]

    def execute(self, ticket):
        return self.client.post(self.base + "/execute", headers=self.headers,
                                json={"session_id": self.context.session_id, "ticket_id": ticket})

    def test_all_owner_endpoints_require_authentication(self):
        for method, url, kwargs in (
            ("get", "/owner/actions/pending", {}),
            ("post", self.base + "/approve", {"json": self.approval}),
            ("post", self.base + "/execute", {"json": {"session_id": self.context.session_id, "ticket_id": "fake"}}),
            ("delete", "/owner/consents/fake", {"params": {"session_id": self.context.session_id}}),
            ("delete", self.base, {"params": {"session_id": self.context.session_id}}),
        ):
            self.assertEqual(getattr(self.client, method)(url, **kwargs).status_code, 401)

    def test_owner_reviews_exact_snapshot_and_executes_once(self):
        response = self.client.get("/owner/actions/pending", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        pending = response.json()
        self.assertEqual(pending["session_id"], self.context.session_id)
        self.assertEqual(pending["requests"][0]["arguments"], {"path": "review.txt", "content": "exact content"})
        ticket = self.approve()
        self.assertFalse(self.context.workspace_root.exists())
        response = self.execute(ticket)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["result"]["status"], "succeeded")
        self.assertEqual(response.json()["confirmation_ticket_id"], ticket)
        self.assertEqual((self.context.workspace_root / "review.txt").read_text(), "exact content")
        self.assertEqual(self.execute(ticket).status_code, 409)

    def test_other_authenticated_user_cannot_approve_or_execute(self):
        other = OwnerRuntime(replace(self.context, owner_id=self.other_id, session_id="other"))
        live_sessions._sessions[self.other_id] = (SimpleNamespace(owner_runtime=other), None, SimpleNamespace(done=lambda: False))
        self.assertEqual(self.client.get("/owner/actions/pending", headers=self.other_headers).json()["requests"], [])
        response = self.client.post(self.base + "/approve", headers=self.other_headers, json=self.approval)
        self.assertEqual(response.status_code, 409)
        payload = {**self.approval, "session_id": "other"}
        self.assertEqual(self.client.post(self.base + "/approve", headers=self.other_headers, json=payload).status_code, 409)
        ticket = self.approve()
        response = self.client.post(self.base + "/execute", headers=self.other_headers,
                                    json={"session_id": "other", "ticket_id": ticket})
        self.assertEqual(response.status_code, 409)
        other.gateway.close()

    def test_stale_session_and_changed_digest_are_rejected(self):
        for patch in ({"session_id": "stale"}, {"expected_digest": "0" * 64}):
            response = self.client.post(self.base + "/approve", headers=self.headers, json={**self.approval, **patch})
            self.assertEqual(response.status_code, 409)

    def test_approval_payload_cannot_widen_permission(self):
        for field in ("arguments", "capability_id", "workspace_root", "owner_id", "approved"):
            response = self.client.post(self.base + "/approve", headers=self.headers,
                                        json={**self.approval, field: "override"})
            self.assertEqual(response.status_code, 422)
        for ttl in (True, 0, 301, "120"):
            response = self.client.post(self.base + "/approve", headers=self.headers, json={**self.approval, "ttl_seconds": ttl})
            self.assertEqual(response.status_code, 422)

    def test_execution_does_not_accept_changed_arguments(self):
        ticket = self.approve()
        response = self.client.post(self.base + "/execute", headers=self.headers, json={
            "session_id": self.context.session_id, "ticket_id": ticket,
            "arguments": {"path": "elsewhere.txt", "content": "changed"},
        })
        self.assertEqual(response.status_code, 422)
        self.assertFalse(self.context.workspace_root.exists())

    def test_revocation_prevents_execution(self):
        ticket = self.approve()
        response = self.client.delete(f"/owner/consents/{ticket}", headers=self.headers,
                                     params={"session_id": self.context.session_id})
        self.assertTrue(response.json()["revoked"])
        response = self.execute(ticket)
        self.assertEqual(response.json()["authorization_decision"], "DENY")
        self.assertFalse(self.context.workspace_root.exists())

    def test_cancel_removes_pending_request(self):
        ticket = self.approve()
        response = self.client.delete(self.base, headers=self.headers, params={"session_id": self.context.session_id})
        self.assertTrue(response.json()["cancelled"])
        self.assertEqual(self.execute(ticket).status_code, 409)

    def test_no_live_session_means_no_owner_authority(self):
        live_sessions._sessions.pop(self.user_id)
        self.assertEqual(self.client.post(self.base + "/approve", headers=self.headers, json=self.approval).status_code, 409)


if __name__ == "__main__":
    unittest.main()
