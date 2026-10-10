"""Privacy contracts at Gemini Live's actual outbound boundary."""

import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import main


class LivePrivacyTests(unittest.TestCase):
    def jarvis(self):
        jarvis = main.JarvisLive.__new__(main.JarvisLive)
        jarvis.voice_name = "charon"
        jarvis.cloud_safe = False
        jarvis.allow_cloud_microphone = False
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

    def test_speak_keeps_local_derived_text_off_live_transport_by_default(self):
        jarvis = self.jarvis()
        jarvis._loop = Mock()
        jarvis.session = Mock(send_client_content=AsyncMock())

        sent = jarvis.speak("PRIVATE result derived from a local file")

        self.assertFalse(sent)
        jarvis.session.send_client_content.assert_not_called()
        jarvis.ui.show_subtitle.assert_called_once_with(
            "PRIVATE result derived from a local file"
        )
        self.assertIn("kept local", jarvis.ui.write_log.call_args.args[0].lower())

    def test_vision_result_uses_local_notification_without_internal_directive(self):
        jarvis = self.jarvis()
        jarvis._loop = Mock()
        jarvis.session = Mock(send_client_content=AsyncMock())

        sent = jarvis._speak_vision_result("  private   visual detail  ")

        self.assertFalse(sent)
        jarvis.session.send_client_content.assert_not_called()
        jarvis.ui.show_subtitle.assert_called_once_with("private visual detail")

    def test_speak_explicit_cloud_disclosure_still_applies_credential_gate(self):
        jarvis = self.jarvis()
        jarvis._loop = Mock()
        jarvis.session = Mock(send_client_content=AsyncMock())

        sent = jarvis.speak(
            "Bearer ownerPrivateToken123456789",
            cloud_shareable=True,
        )

        self.assertFalse(sent)
        jarvis.session.send_client_content.assert_not_called()
        self.assertIn("blocked", jarvis.ui.write_log.call_args.args[0].lower())

    def test_speak_sends_only_when_explicitly_marked_cloud_shareable(self):
        jarvis = self.jarvis()
        jarvis._loop = Mock()
        jarvis.session = Mock(send_client_content=AsyncMock())

        scheduled = []

        def capture(coro, loop):
            scheduled.append((coro, loop))
            coro.close()

        with patch.object(main.asyncio, "run_coroutine_threadsafe", side_effect=capture):
            sent = jarvis.speak("Public status update", cloud_shareable=True)

        self.assertTrue(sent)
        self.assertEqual(len(scheduled), 1)
        self.assertIs(scheduled[0][1], jarvis._loop)
        payload = jarvis.session.send_client_content.call_args.kwargs
        self.assertEqual(payload["turns"]["parts"][0]["text"], "Public status update")
        self.assertIn("explicitly cloud-shareable", jarvis.ui.write_log.call_args.args[0])

    def test_microphone_streaming_is_fail_closed_without_explicit_opt_in(self):
        jarvis = self.jarvis()
        with patch.object(
            main,
            "_require_sounddevice",
            side_effect=AssertionError("microphone opened without opt-in"),
        ) as require_audio:
            asyncio.run(jarvis._listen_audio())

        require_audio.assert_not_called()
        self.assertIn("streaming is off", jarvis.ui.write_log.call_args.args[0])

    def test_cloud_microphone_defaults_off_and_environment_requires_truthy_opt_in(self):
        client = Mock()
        jarvis = main.JarvisLive(client)
        self.assertFalse(jarvis.allow_cloud_microphone)

        with patch.dict(main.os.environ, {}, clear=True):
            self.assertFalse(main._cloud_microphone_env_opted_in())
        with patch.dict(
            main.os.environ,
            {main.CLOUD_MICROPHONE_OPT_IN_ENV: "true"},
            clear=True,
        ):
            self.assertTrue(main._cloud_microphone_env_opted_in())

    def test_microphone_opt_in_is_disclosed_before_capture_starts(self):
        jarvis = self.jarvis()
        jarvis.allow_cloud_microphone = True
        jarvis._shutdown_requested.is_set.return_value = True
        audio = MagicMock()

        with (
            patch.object(main, "_require_sounddevice", return_value=audio),
            patch("builtins.print"),
        ):
            asyncio.run(jarvis._listen_audio())

        audio.InputStream.assert_called_once()
        first_log = jarvis.ui.write_log.call_args_list[0].args[0]
        self.assertIn("streaming to Gemini Live", first_log)

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
