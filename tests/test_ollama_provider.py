"""Contract tests for the hardened tier-aware local Ollama provider."""

import json
import unittest
from unittest.mock import Mock, patch

from core.model_provider import ModelProvider, ModelRequest, ModelTier
from core.providers.ollama import (
    LocalProviderError,
    OllamaProvider,
    OllamaProviderError,
)


class FakeStreamResponse:
    status = 200

    def __init__(self, lines):
        self._lines = list(lines)

    def readline(self, _limit):
        return self._lines.pop(0) if self._lines else b""


class FakeStreamConnection:
    def __init__(self, *_args, **_kwargs):
        self.request_args = None
        self.closed = False
        self.response = FakeStreamResponse([
            json.dumps({"message": {"role": "assistant", "content": "Hel"}}).encode() + b"\n",
            json.dumps({"message": {"role": "assistant", "content": "lo"}}).encode() + b"\n",
            json.dumps({"done": True, "message": {"role": "assistant", "content": ""}}).encode() + b"\n",
        ])

    def request(self, *args, **kwargs):
        self.request_args = (args, kwargs)

    def getresponse(self):
        return self.response

    def close(self):
        self.closed = True


class OllamaProviderTests(unittest.TestCase):
    def test_structurally_implements_model_provider(self):
        self.assertIsInstance(OllamaProvider(), ModelProvider)
        self.assertIs(OllamaProviderError, LocalProviderError)

    def test_payload_maps_neutral_tiers_without_tools(self):
        provider = OllamaProvider(
            fast_model="fast-local",
            standard_model="coder-local",
            keep_alive="3m",
        )
        payload = provider._payload(
            ModelRequest(
                prompt="solve",
                system_instruction="You are Jarvis.",
                tier=ModelTier.STANDARD,
                json_output=True,
            ),
            stream=False,
        )
        self.assertEqual(payload["model"], "coder-local")
        self.assertEqual(
            payload["messages"],
            [
                {"role": "system", "content": "You are Jarvis."},
                {"role": "user", "content": "solve"},
            ],
        )
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["format"], "json")
        self.assertEqual(payload["keep_alive"], "3m")
        self.assertEqual(payload["options"], {"num_ctx": 4096, "num_predict": 512})
        self.assertNotIn("tools", payload)

    def test_fast_tier_omits_empty_system_and_json_format(self):
        provider = OllamaProvider(fast_model="friday-local")
        payload = provider._payload(ModelRequest("hi"), stream=False)
        self.assertEqual(payload["model"], "friday-local")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "hi"}])
        self.assertNotIn("format", payload)

    def test_stream_yields_bounded_text_chunks(self):
        connection = FakeStreamConnection()
        with patch(
            "core.providers.ollama.http.client.HTTPConnection",
            return_value=connection,
        ):
            provider = OllamaProvider(
                base_url="http://127.0.0.1:11434",
                keep_alive=60,
            )
            chunks = list(provider.stream(ModelRequest("say hello")))
        self.assertEqual(chunks, ["Hel", "lo"])
        self.assertTrue(connection.closed)
        sent = json.loads(connection.request_args[0][2])
        self.assertTrue(sent["stream"])
        self.assertEqual(sent["keep_alive"], 60)

    def test_strict_loopback_origin_and_safe_bind_normalization(self):
        provider = OllamaProvider(base_url="127.0.0.1:1234")
        self.assertEqual(
            provider._endpoint.chat_url,
            "http://127.0.0.1:1234/api/chat",
        )
        bind = OllamaProvider(base_url="0.0.0.0:11434")
        self.assertEqual(
            bind._endpoint.chat_url,
            "http://127.0.0.1:11434/api/chat",
        )
        for url in (
            "http://localhost:11434",
            "http://example.com",
            "https://127.0.0.1",
            "http://127.0.0.1/path",
            "http://user:pass@127.0.0.1",
            "http://127.0.0.1?redirect=x",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                OllamaProvider(base_url=url)

    def test_constructor_validation(self):
        with self.assertRaises(ValueError):
            OllamaProvider(timeout_seconds=0)
        with self.assertRaises(ValueError):
            OllamaProvider(timeout_seconds=61)
        with self.assertRaises(ValueError):
            OllamaProvider(fast_model="")
        with self.assertRaises(ValueError):
            OllamaProvider(model="bad model name")

    def test_request_size_limit_is_checked_before_network(self):
        provider = OllamaProvider()
        with self.assertRaises(ValueError):
            provider.generate(ModelRequest("x" * 65536))

    def test_complete_response_rejects_tool_calls_and_incomplete_values(self):
        for value in (
            {"done": False},
            [],
            {
                "done": True,
                "message": {
                    "role": "assistant",
                    "content": "ok",
                    "tool_calls": [{"function": "shell"}],
                },
            },
        ):
            with self.subTest(value=value):
                with self.assertRaises(LocalProviderError):
                    provider = OllamaProvider()
                    provider._decode_complete(
                        json.dumps(value).encode(),
                        "model",
                    )


if __name__ == "__main__":
    unittest.main()
