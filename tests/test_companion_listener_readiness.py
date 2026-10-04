"""The pairing UI follows a real local listener, including startup/bind failure."""

import asyncio
from contextlib import asynccontextmanager
import http.client
from pathlib import Path
import socket
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
import uvicorn

from api.jarvis_local_server import LocalBrainHost, SetupState, create_local_brain_app
import brain_sidecar
from core.nexus.peer_auth import DeviceSigner
from core.nexus.sync_node import NexusSyncNode


_UVICORN_SERVER = uvicorn.Server


class CompanionListenerReadinessTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        host = LocalBrainHost.__new__(LocalBrainHost)
        host.ui_token = "gateway-listener-test-owner-token-" * 2
        # The advertisement is never contacted. Only 127.0.0.1 is bound below.
        host.companion_endpoint = "http://100.64.0.10:8765"
        host._companion_listener = None
        host._guard = threading.RLock()
        host.snapshot = None
        host.setup = SetupState("setup_required")
        host.model_store = None
        host.manual_model = None
        host.ollama_base_url = "http://127.0.0.1:11435"
        host.local_brain_ready = lambda: False
        host.available_models = lambda: ()
        host.council = SimpleNamespace(max_parallel_experts=2)
        host.owner_face = SimpleNamespace(
            enrolled=False, recognized=False, last_score=0.0, engine="test",
        )
        host.voice = SimpleNamespace(status=lambda: {}, stop=lambda: None)
        host.approvals = SimpleNamespace(pending=lambda: ())
        host.approval_bridge = SimpleNamespace(status=lambda: {})
        host.repair_journal = SimpleNamespace(recent=lambda limit: ())
        host.node = NexusSyncNode(
            Path(directory.name) / "node", "gateway-listener-test",
            signer=DeviceSigner.generate(),
        )
        host._approval_stop = threading.Event()
        host._approval_thread = Mock()
        self.host = host
        self.client = TestClient(create_local_brain_app(host))
        self.addCleanup(self.client.close)
        self.headers = {"Authorization": "Bearer " + host.ui_token}

    def wait_for(self, predicate):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.01)
        self.fail("Local companion listener did not reach the expected state")

    def available(self):
        response = self.client.get("/v1/status", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        return response.json()["companion"]["available"]

    def offer(self, **payload):
        return self.client.post(
            "/v1/pair/offer", headers=self.headers,
            json={"ttl_seconds": 30, **payload},
        )

    def start_server(self, app, port=0):
        captured = {}
        errors = []

        class CapturingServer(_UVICORN_SERVER):
            def __init__(self, config):
                super().__init__(config)
                captured["server"] = self

        def run():
            try:
                brain_sidecar._serve_companion(app, "127.0.0.1", port, self.host)
            except BaseException as error:
                errors.append(error)

        factory = patch("brain_sidecar.uvicorn.Server", CapturingServer)
        factory.start()
        thread = threading.Thread(target=run, daemon=True)
        thread.start()

        def cleanup():
            server = captured.get("server")
            if server is not None:
                server.should_exit = True
            thread.join(timeout=5)
            factory.stop()
            self.assertFalse(thread.is_alive(), "Companion test listener did not stop")

        self.addCleanup(cleanup)
        self.wait_for(lambda: "server" in captured)
        return captured["server"], thread, errors

    def test_configured_endpoint_does_not_enable_status_or_manual_qr(self):
        self.assertFalse(self.available())
        with patch.object(self.host.node.pairing, "create_offer") as create:
            response = self.offer(endpoint="http://100.64.0.11:8765")
            self.assertEqual(response.status_code, 409)
            create.assert_not_called()

    def test_listener_observation_error_fails_closed(self):
        self.host.set_companion_listener(Mock(side_effect=RuntimeError("listener closed")))
        self.assertFalse(self.available())
        self.assertEqual(self.offer().status_code, 409)

    def test_actual_listener_startup_closed_socket_and_shutdown_control_status_and_qr(self):
        startup_entered = threading.Event()
        startup_release = threading.Event()
        loop = {}

        @asynccontextmanager
        async def lifespan(app):
            loop["value"] = asyncio.get_running_loop()
            startup_entered.set()
            while not startup_release.is_set():
                await asyncio.sleep(0.01)
            yield

        app = FastAPI(lifespan=lifespan)

        @app.get("/health")
        def health():
            return {"ok": True}

        server, thread, errors = self.start_server(app)
        # Cleanup must release a blocked startup before joining the server.
        self.addCleanup(startup_release.set)
        self.assertTrue(startup_entered.wait(timeout=5))
        self.assertFalse(self.available())
        self.assertEqual(self.offer().status_code, 409)

        startup_release.set()
        self.wait_for(self.host.companion_available)
        self.assertTrue(self.available())
        port = server.servers[0].sockets[0].getsockname()[1]
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
        try:
            connection.request("GET", "/health")
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(response.read(), b'{"ok":true}')
        finally:
            connection.close()
        invitation = self.offer()
        self.assertEqual(invitation.status_code, 200)
        self.assertTrue(invitation.json()["deep_link"].startswith("jarvis://pair?offer="))

        # A closed listening socket must be unavailable even if Uvicorn's
        # historical started flag is still true and its main loop is alive.
        for listener in server.servers:
            loop["value"].call_soon_threadsafe(listener.close)
        self.wait_for(lambda: not self.host.companion_available())
        self.assertTrue(server.started)
        self.assertTrue(thread.is_alive())
        self.assertFalse(self.available())
        with patch.object(self.host.node.pairing, "create_offer") as create:
            self.assertEqual(self.offer().status_code, 409)
            create.assert_not_called()

        server.should_exit = True
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertFalse(self.available())
        self.assertIsNone(self.host._companion_listener)
        self.assertEqual(errors, [])

    def test_failed_port_bind_never_enables_status_or_pairing(self):
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            port = occupied.getsockname()[1]
            with patch("uvicorn.server.logger.error"):
                server, thread, errors = self.start_server(FastAPI(), port)
                thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertFalse(server.started)
            self.assertTrue(any(isinstance(error, SystemExit) for error in errors))
            self.assertIsNone(self.host._companion_listener)
            self.assertFalse(self.available())
            with patch.object(self.host.node.pairing, "create_offer") as create:
                self.assertEqual(self.offer().status_code, 409)
                create.assert_not_called()

    def test_host_shutdown_with_a_live_listener_revokes_availability(self):
        server, thread, errors = self.start_server(FastAPI())
        self.wait_for(self.host.companion_available)
        self.assertTrue(self.available())
        self.host.close()
        self.assertFalse(self.available())
        self.assertEqual(self.offer().status_code, 409)
        # A delayed server-start callback cannot re-advertise a closed host.
        self.host.set_companion_listener(lambda: brain_sidecar._companion_server_listening(server))
        self.assertFalse(self.available())
        self.assertIsNone(self.host._companion_listener)
        server.should_exit = True
        thread.join(timeout=5)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
