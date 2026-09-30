"""Real authenticated API with isolated SQLite mission state and no Live session."""
import os
import tempfile
import unittest
from uuid import uuid4
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import delete

from agent.missions import MissionService
from api.database import SessionLocal, init_db
from api.models import User
from api.security import create_access_token, hash_password
from api.server import app


class MissionApiTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"JARVIS_QA_MODE": "0", "NEXUS_STATE_DIR": ""})
        env.start()
        self.addCleanup(env.stop)
        init_db()
        self.client = TestClient(app)
        self.client.__enter__()
        self.tmp = tempfile.TemporaryDirectory()
        self.host = MissionService(self.tmp.name)
        app.state.missions = self.host
        self.uid, self.other = str(uuid4()), str(uuid4())
        with SessionLocal.begin() as db:
            for uid in (self.uid, self.other):
                db.add(User(id=uid, email=f"{uid}@example.com", display_name="Mission owner",
                            password_hash=hash_password("mission-test-password")))
        self.headers = {"Authorization": "Bearer " + create_access_token(self.uid)}
        self.other_headers = {"Authorization": "Bearer " + create_access_token(self.other)}

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.tmp.cleanup()
        with SessionLocal.begin() as db:
            db.execute(delete(User).where(User.id.in_([self.uid, self.other])))

    def create(self):
        response = self.client.post("/missions/actions", headers=self.headers, json={"tool": "file_controller", "arguments": {"action": "write", "path": "artifact.txt", "content": "private"}})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def run_mission(self, m, **extra):
        return self.client.post(f"/missions/{m['id']}/run", headers=self.headers,
                                json={"session_id": m["session_id"], "version": m["version"], **extra})

    def test_auth_all_routes_and_owner_isolation(self):
        m = self.create()
        base = "/missions/" + m["id"]
        for path in ("/missions", base):
            self.assertEqual(self.client.get(path).status_code, 401)
        for path in ("/missions", "/missions/actions", base + "/run", base + "/approve", base + "/cancel", base + "/revoke"):
            self.assertEqual(self.client.post(path, json={}).status_code, 401)
        self.assertEqual(self.client.get(base, headers=self.other_headers).status_code, 404)
        self.assertEqual(self.client.get("/missions", headers=self.other_headers).json(), [])
        self.assertEqual(self.client.post(base + "/run", headers=self.other_headers, json={"session_id": m["session_id"], "version": 0}).status_code, 404)

    def test_complete_owner_flow_without_live_or_model(self):
        m = self.run_mission(self.create()).json()
        self.assertEqual(m["state"], "waiting_confirmation")
        approval = self.client.post(f"/missions/{m['id']}/approve", headers=self.headers,
            json={"session_id": m["session_id"], "expected_digest": m["digest"]})
        self.assertEqual(approval.status_code, 200, approval.text)
        final = self.run_mission(m, ticket_id=approval.json()["ticket_id"])
        self.assertEqual(final.status_code, 200, final.text)
        self.assertEqual(final.json()["state"], "succeeded")
        self.assertEqual(self.run_mission(final.json(), ticket_id=approval.json()["ticket_id"]).status_code, 409)

    def test_payload_cannot_change_authority_or_arguments(self):
        m = self.create()
        for field in ("owner_id", "arguments", "capability_id", "approved", "workspace_root"):
            self.assertEqual(self.run_mission(m, **{field: "injected"}).status_code, 422)
        self.assertEqual(self.run_mission(m, version=True).status_code, 422)
        waiting = self.run_mission(m).json()
        self.assertEqual(self.client.post(f"/missions/{m['id']}/approve", headers=self.headers,
            json={"session_id": m["session_id"], "expected_digest": "0" * 64}).status_code, 409)
        self.assertEqual(self.run_mission(waiting, session_id="stale").status_code, 409)

    def test_revoke_cancel_and_stale_version(self):
        m = self.run_mission(self.create()).json()
        base = f"/missions/{m['id']}"
        t = self.client.post(base + "/approve", headers=self.headers, json={"session_id": m["session_id"], "expected_digest": m["digest"]}).json()["ticket_id"]
        revoked = self.client.post(base + "/revoke", headers=self.headers, json={"session_id": m["session_id"], "ticket_id": t})
        self.assertTrue(revoked.json()["revoked"])
        self.assertEqual(self.run_mission(m, version=0).status_code, 409)
        cancelled = self.client.post(base + "/cancel", headers=self.headers, json={"session_id": m["session_id"], "version": m["version"]})
        self.assertEqual(cancelled.json()["state"], "cancelled")
        self.assertEqual(self.run_mission(cancelled.json(), ticket_id=t).status_code, 409)

    def test_disabled_host_is_explicit(self):
        app.state.missions = None
        try:
            self.assertEqual(self.client.get("/missions", headers=self.headers).status_code, 503)
        finally:
            app.state.missions = self.host

    def test_model_fields_never_configure_local_endpoint(self):
        response = self.client.post("/missions", headers=self.headers, json={"goal": "what time is it", "base_url": "https://remote.invalid"})
        self.assertEqual(response.status_code, 422)
        m = self.client.post("/missions", headers=self.headers, json={"goal": "what time is it"}).json()
        self.assertEqual(self.run_mission(m).json()["state"], "succeeded")
