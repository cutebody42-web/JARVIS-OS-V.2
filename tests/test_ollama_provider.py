"""Offline contract tests for the local Ollama provider adapter."""

import json
import unittest
from unittest.mock import Mock, patch

from core.model_provider import ModelProvider, ModelRequest, ModelTier
from core.providers.ollama import OllamaProvider, OllamaProviderError


class OllamaProviderTests(unittest.TestCase):
    def test_structurally_implements_model_provider(self):
        self.assertIsInstance(OllamaProvider(), ModelProvider)

    def test_generate_translates_neutral_request_without_actions(self):
        provider = OllamaProvider(
            fast_model="fast-local",
            standard_model="coder-local",
            base_url="http://127.0.0.1:11434/",
            keep_alive="3m",
        )
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "model": "coder-local",
            "message": {"role": "assistant", "content": "  result text  "},
        }

        with patch("requests.post", return_value=response) as post:
            result = provider.generate(
                ModelRequest(
                    prompt="solve",
                    system_instruction="You are Jarvis.",
                    tier=ModelTier.STANDARD,
                    json_output=True,
                )
            )

        self.assertEqual(result.text, "result text")
        self.assertEqual(result.provider, "ollama")
        self.assertEqual(result.model, "coder-local")
        call = post.call_args
        self.assertEqual(call.args[0], "http://127.0.0.1:11434/api/chat")
        self.assertEqual(call.kwargs["timeout"], 120.0)
        payload = call.kwargs["json"]
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
        self.assertNotIn("tools", payload)

    def test_fast_tier_omits_empty_system_and_json_format(self):
        provider = OllamaProvider(fast_model="friday-local")
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"message": {"content": "hello"}}

        with patch("requests.post", return_value=response) as post:
            result = provider.generate(ModelRequest("hi"))

        self.assertEqual(result.model, "friday-local")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["messages"], [{"role": "user", "content": "hi"}])
        self.assertNotIn("format", payload)

    def test_stream_yields_content_chunks(self):
        provider = OllamaProvider(base_url="http://localhost:11434", keep_alive=60)
        response = Mock()
        response.raise_for_status.return_value = None
        response.iter_lines.return_value = [
            json.dumps({"message": {"content": "Hel"}}),
            "",
            json.dumps({"message": {"content": "lo"}}),
            json.dumps({"done": True, "message": {"content": ""}}),
        ]
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)

        with patch("requests.post", return_value=response) as post:
            chunks = list(provider.stream(ModelRequest("say hello")))

        self.assertEqual(chunks, ["Hel", "lo"])
        self.assertTrue(post.call_args.kwargs["stream"])
        self.assertEqual(post.call_args.kwargs["json"]["keep_alive"], 60)

    def test_network_errors_fail_closed_without_raw_details(self):
        import requests

        provider = OllamaProvider()
        secretish = "http://token@example.invalid/private"
        with patch(
            "requests.post",
            side_effect=requests.ConnectionError(secretish),
        ):
            with self.assertRaises(OllamaProviderError) as caught:
                provider.generate(ModelRequest("hello"))
        self.assertNotIn("token", str(caught.exception))
        self.assertIn("not reachable", str(caught.exception))

    def test_invalid_or_empty_responses_fail_closed(self):
        provider = OllamaProvider()
        invalid_json = Mock()
        invalid_json.raise_for_status.return_value = None
        invalid_json.json.side_effect = ValueError("bad JSON")

        empty = Mock()
        empty.raise_for_status.return_value = None
        empty.json.return_value = {"message": {"content": "   "}}

        for response in (invalid_json, empty):
            with self.subTest(response=response), patch("requests.post", return_value=response):
                with self.assertRaises(OllamaProviderError):
                    provider.generate(ModelRequest("hello"))

    def test_scheme_less_ollama_host_is_accepted(self):
        provider = OllamaProvider(base_url="127.0.0.1:1234")
        self.assertEqual(provider._endpoint.chat_url, "http://127.0.0.1:1234/api/chat")

    def test_unspecified_bind_host_becomes_connectable_loopback(self):
        provider = OllamaProvider(base_url="0.0.0.0:11434")
        self.assertEqual(provider._endpoint.chat_url, "http://127.0.0.1:11434/api/chat")

    def test_constructor_validation(self):
        with self.assertRaises(ValueError):
            OllamaProvider(base_url="ftp://example.com")
        with self.assertRaises(ValueError):
            OllamaProvider(timeout_seconds=0)
        with self.assertRaises(ValueError):
            OllamaProvider(fast_model="")


if __name__ == "__main__":
    unittest.main()
