"""Privacy contracts at Gemini Live's actual outbound boundary."""

import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import main


class LivePrivacyTests(unittest.TestCase):
    def jarvis(self):
        jarvis = main.JarvisLive.__new__(main.JarvisLive)
        jarvis.voice_name = "charon"
        jarvis.cloud_safe = False
        jarvis.ui = Mock(operational_ready=True, muted=False)
        jarvis._pending_self_quit = False
        jarvis._shutdown_requested = Mock()
        jarvis._shutdown_requested.is_set.return_value = False
        return jarvis

    def test_live_config_never_loads_or_formats_durable_memory(self):
        jarvis = self.jarvis()
        with (
            patch("memory.memory_manager.load_memory", side_effect=AssertionError("memory crossed boundary")) as load,
            patch("memory.memory_manager.format_memory_for_prompt", side_effect=AssertionError("memory crossed boundary")) as format_memory,
            patch.object(main, "_load_system_prompt", return_value="public system policy"),
        ):
            config = jarvis._build_config()
        load.assert_not_called()
        format_memory.assert_not_called()
        self.assertNotIn("durable owner memory", config.system_instruction)
        self.assertIn("public system policy", config.system_instruction)

    def test_startup_greeting_contains_no_stored_identity(self):
        jarvis = self.jarvis()
        jarvis.session = Mock(send_client_content=AsyncMock())
        asyncio.run(jarvis._announce_startup())
        payload = jarvis.session.send_client_content.await_args.kwargs
        self.assertEqual(
            payload["turns"]["parts"][0]["text"],
            "Jarvis is online. What would you like to accomplish today?",
        )

    def test_typed_credential_is_blocked_before_live_transport(self):
        jarvis = self.jarvis()
        jarvis.session = Mock(send_client_content=AsyncMock())
        jarvis._handle_reflex_text = Mock(return_value=False)
        sent = asyncio.run(jarvis.send_text("api_key=AIza" + "A" * 35))
        self.assertFalse(sent)
        jarvis.session.send_client_content.assert_not_awaited()
        self.assertIn("blocked", jarvis.ui.write_log.call_args.args[0].lower())

    @staticmethod
    def receipt(capability, message, *, status="succeeded"):
        return SimpleNamespace(
            action_id="action-1",
            request_id="request-1",
            capability_id=capability,
            normalized_argument_digest="d" * 64,
            authorization_decision=SimpleNamespace(value="ALLOW"),
            result=SimpleNamespace(status=SimpleNamespace(value=status), message=message),
            to_dict=Mock(side_effect=AssertionError("full receipt crossed cloud boundary")),
        )

    def execute(self, receipt, *, tool="file_controller"):
        jarvis = self.jarvis()
        gateway = Mock()
        gateway.run_tool.return_value = receipt
        jarvis.owner_runtime = SimpleNamespace(gateway=gateway)
        call = SimpleNamespace(id="call-1", name=tool, args={"action": "read", "path": "private.txt"})
        return asyncio.run(jarvis._execute_tool(call))

    def test_local_tool_output_and_full_receipt_never_return_to_cloud_model(self):
        receipt = self.receipt("workspace.read", "PRIVATE FILE CONTENT")
        response = self.execute(receipt).response
        self.assertNotIn("PRIVATE FILE CONTENT", str(response))
        self.assertIn("withheld", response["result"])
        self.assertEqual(set(response["receipt"]), {
            "action_id", "request_id", "status", "capability_id",
            "authorization_decision", "normalized_argument_digest",
        })
        receipt.to_dict.assert_not_called()

    def test_shareable_public_result_still_passes_credential_gate(self):
        public = self.receipt("web.search", "Public search result")
        self.assertEqual(self.execute(public, tool="web_search").response["result"], "Public search result")

        secret = self.receipt("web.search", "Bearer ownerPrivateToken123456789")
        blocked = self.execute(secret, tool="web_search").response["result"]
        self.assertNotIn("ownerPrivateToken", blocked)
        self.assertIn("withheld", blocked)


if __name__ == "__main__":
    unittest.main()
