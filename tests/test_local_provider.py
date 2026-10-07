"""Real loopback HTTP transport against a fake Ollama server, no model/hardware."""
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading
import unittest
from unittest.mock import Mock, patch

from core.model_provider import (
    ContextClassification,
    ModelContext,
    ModelProvider,
    ModelRequest,
    default_provider,
)
from core.providers.ollama import LocalProviderError, OllamaProvider


class LocalProviderTests(unittest.TestCase):
    def setUp(self):
        self.responses = deque()
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                outer.requests.append((self.path, json.loads(raw), self.client_address[1]))
                status, body = outer.responses.popleft()
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Location", "http://example.invalid/private")
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.provider = OllamaProvider("fixture-model", base_url=self.url, timeout=1)
        self.addCleanup(self.cleanup)

    def cleanup(self):
        self.provider.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def reply(self, value=None, status=200):
        value = value if value is not None else {"done": True, "message": {"role": "assistant", "content": "hello"}}
        self.responses.append((status, json.dumps(value).encode()))

    def test_protocol_bounds_persistent_connection_and_no_proxy(self):
        self.reply()
        self.reply()
        with patch.dict(os.environ, {"HTTP_PROXY": "http://127.0.0.1:1", "HTTPS_PROXY": "http://127.0.0.1:1"}):
            for _ in range(2):
                result = self.provider.generate(ModelRequest("مرحبا", "system", json_output=True))
                self.assertEqual((result.provider, result.text), ("ollama", "hello"))
        self.assertIsInstance(self.provider, ModelProvider)
        self.assertEqual(self.requests[0][2], self.requests[1][2])
        path, body, _ = self.requests[0]
        self.assertEqual(path, "/api/chat")
        self.assertEqual(body["messages"][1]["content"], "مرحبا")
        self.assertEqual(body["format"], "json")
        self.assertFalse(body["stream"])
        self.assertIs(body["think"], False)
        self.assertEqual(body["keep_alive"], "5m")
        self.assertEqual(body["options"], {"num_ctx": 4096, "num_predict": 512})
        self.assertNotIn("tools", body)

    def test_direct_local_adapter_receives_private_and_shareable_context(self):
        self.reply()
        private_secret = "api_key=owner-private-value-123456"
        public_context = "public project architecture"

        self.provider.generate(ModelRequest(
            "review this locally",
            context=(
                ModelContext(private_secret, label="private memory"),
                ModelContext(
                    public_context,
                    label="public context",
                    classification=ContextClassification.CLOUD_SHAREABLE,
                ),
            ),
        ))

        prompt = self.requests[-1][1]["messages"][1]["content"]
        self.assertIn(private_secret, prompt)
        self.assertIn(public_context, prompt)
        self.assertIn("review this locally", prompt)

    def test_redirect_and_errors_never_follow_or_leak(self):
        for status in (302, 500):
            self.reply({"error": "secret private prompt"}, status)
            with self.assertRaises(LocalProviderError) as error:
                self.provider.generate(ModelRequest("input"))
            self.assertNotIn("secret", str(error.exception))
        self.assertEqual(len(self.requests), 2)

    def test_invalid_incomplete_or_tool_response_fails_closed(self):
        for value in (
            {"done": False},
            [],
            {"done": True, "message": {"role": "assistant", "content": ""}},
            {"done": True, "message": {"role": "assistant", "content": " \n\t"}},
            {"done": True, "message": {"role": "assistant", "content": "ok", "tool_calls": [{"function": "shell"}]}},
        ):
            self.reply(value)
            with self.assertRaises(LocalProviderError):
                self.provider.generate(ModelRequest("input"))

    def test_blank_local_response_uses_next_local_route(self):
        from core.model_router import ProviderChoice, ProviderKind, RoutePlan
        from core.personas import TABY
        from core.routed_model_provider import RoutedModelProvider

        fallback = OllamaProvider("fallback-model", base_url=self.url, timeout=1)
        self.addCleanup(fallback.close)
        providers = {self.provider.model: self.provider, fallback.model: fallback}
        plan = RoutePlan(
            primary=ProviderChoice(ProviderKind.OLLAMA, self.provider.model, "primary", ensure_priority=1),
            fallbacks=(ProviderChoice(ProviderKind.OLLAMA, fallback.model, "fallback", ensure_priority=1),),
        )
        router = Mock()
        router.route.return_value = plan
        routed = RoutedModelProvider(
            TABY,
            profiler=Mock(),
            runtime=Mock(),
            router=router,
            ollama_factory=lambda choice: providers[choice.model],
            gemini_factory=lambda choice: self.fail("Local fallback must not use cloud"),
            allow_cloud=False,
        )
        self.reply({"done": True, "message": {"role": "assistant", "content": " \n\t"}})
        self.reply()

        response = routed.generate(ModelRequest("hello"))

        self.assertEqual((response.model, response.text), (fallback.model, "hello"))
        self.assertEqual([body["model"] for _, body, _ in self.requests], list(providers))
        self.assertEqual(
            [attempt.outcome for attempt in routed.last_attempts],
            ["failed:LocalProviderError", "succeeded"],
        )
        router.route.assert_called_once()

    def test_request_limit_prevents_http(self):
        with self.assertRaises(ValueError):
            self.provider.generate(ModelRequest("x" * 65536))
        self.assertEqual(self.requests, [])

    def test_response_limit_and_invalid_json_disconnect(self):
        for body in (b"x" * 262145, b"not JSON"):
            self.responses.append((200, body))
            with self.assertRaises(LocalProviderError):
                self.provider.generate(ModelRequest("input"))

    def test_remote_ambiguous_and_credential_origins_rejected(self):
        for url in ("http://localhost:11434", "http://example.com", "https://127.0.0.1", "http://127.0.0.1/path", "http://user:pass@127.0.0.1", "http://127.0.0.1?redirect=x"):
            with self.assertRaises(ValueError):
                OllamaProvider("fixture", base_url=url)

    def test_opt_in_provider_selection_no_fallback(self):
        with patch.dict(os.environ, {"NEXUS_MODEL_PROVIDER": "ollama", "NEXUS_OLLAMA_MODEL": "fixture", "NEXUS_OLLAMA_URL": self.url}):
            self.assertIsInstance(default_provider(), OllamaProvider)
        with patch.dict(os.environ, {"NEXUS_MODEL_PROVIDER": "unknown"}):
            with self.assertRaises(ValueError):
                default_provider()

    def test_loopback_provider_to_mission_to_verified_receipt(self):
        import tempfile
        from agent.missions import MissionService
        plan = {"steps": [{"step": 1, "tool": "system_time", "parameters": {}}]}
        self.reply({"done": True, "message": {"role": "assistant", "content": json.dumps(plan)}})
        with tempfile.TemporaryDirectory() as directory:
            host = MissionService(directory, provider=self.provider)
            try:
                mission = host.create_goal("owner", "Read the native system clock via a planned action")
                result = host.run("owner", mission["id"], mission["session_id"], mission["version"])
                self.assertEqual(result["state"], "succeeded")
                self.assertEqual(result["receipt"]["route"], "model")
                self.assertEqual(result["receipt"]["verifier_result"]["status"], "pass")
                self.assertEqual(len(self.requests), 1)
            finally:
                host.close()
