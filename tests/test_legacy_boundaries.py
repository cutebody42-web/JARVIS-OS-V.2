"""Public legacy entrypoints remain subject to owner policy, even in unit QA."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from actions import email_control, file_controller, media_control, screen_processor, send_message
from core.action_gateway import OwnerRuntime, runtime_scope
from core.authority_contracts import AuthorizationDecision as A, PolicyContext


class LegacyBoundaryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.runtime = OwnerRuntime(PolicyContext("test-owner", "test-session", self.root / "workspace"))
        self.addCleanup(self.runtime.gateway.close)
        self.addCleanup(email_control._clear_pending_email)
        self.addCleanup(send_message._clear_pending_message)

    def test_ambient_legacy_mutations_never_invoke_adapter(self):
        cases = (
            (file_controller.file_controller, {"action": "open", "path": "file.txt"}),
            (media_control.media_control, {"action": "play", "query": "song"}),
            (screen_processor.screen_process, {"angle": "screen", "text": "read secrets"}),
            (send_message.send_message, {"action": "approve", "platform": "Instagram"}),
            (send_message.prepare_message_reply, {"action": "approve"}),
            (email_control.email_control, {"action": "approve", "provider": "gmail"}),
        )
        with patch.dict(os.environ, {"JARVIS_QA_MODE": "0"}), runtime_scope(self.runtime), \
             patch("core.action_adapters.invoke") as invoke:
            for entry, arguments in cases:
                with self.subTest(tool=entry.__name__):
                    self.assertTrue(entry(arguments).startswith("denied:"))
                    receipt = self.runtime.gateway.receipts[-1]
                    self.assertIs(receipt.authorization_decision, A.DENY)
                    self.assertTrue(receipt.authorization_evidence)
            invoke.assert_not_called()

    def test_pending_draft_is_not_owner_authority(self):
        email_control._set_pending_email({"to": "recipient@example.test", "body": "hello"})
        send_message._set_pending_message("iMessage", "Recipient", "hello")
        with patch.dict(os.environ, {"JARVIS_QA_MODE": "0"}), runtime_scope(self.runtime), \
             patch("core.action_adapters.invoke") as invoke:
            for entry in (email_control.email_control, send_message.prepare_message_reply):
                self.assertTrue(entry({"action": "approve"}).startswith("denied:"))
            invoke.assert_not_called()

    def test_public_file_mutation_requires_ticket_and_has_no_effect(self):
        with patch.dict(os.environ, {"JARVIS_QA_MODE": "0"}), runtime_scope(self.runtime):
            result = file_controller.file_controller({"action": "write", "path": "new.txt", "content": "test"})
        self.assertTrue(result.startswith("require_confirmation:"))
        self.assertFalse((self.runtime.gateway.context.workspace_root / "new.txt").exists())
        self.assertIs(self.runtime.gateway.receipts[-1].authorization_decision, A.REQUIRE_CONFIRMATION)

    def test_sensitive_email_read_requires_owner_confirmation(self):
        with patch.dict(os.environ, {"JARVIS_QA_MODE": "0"}), runtime_scope(self.runtime), \
             patch("core.action_adapters.invoke") as invoke:
            result = email_control.email_control({"action": "read", "message_id": "example"})
        self.assertTrue(result.startswith("require_confirmation:"))
        invoke.assert_not_called()

    def test_qa_workspace_paths_use_owner_root_and_preserve_consent_digest(self):
        with patch.dict(os.environ, {"JARVIS_QA_MODE": "1", "JARVIS_QA_WORKSPACE": str(self.root)}):
            proposal = self.runtime.gateway.run_tool("file_controller", {
                "action": "write", "path": "allowed.txt", "content": "approved",
            }, runtime=self.runtime)
            self.assertIs(proposal.authorization_decision, A.REQUIRE_CONFIRMATION)
            ticket = self.runtime.owner.approve(proposal.request_id, proposal.normalized_argument_digest)
            result = self.runtime.execute_approved(proposal.request_id, ticket.ticket_id)
            self.assertEqual(result.result.status.value, "succeeded")
            self.assertEqual(result.normalized_argument_digest, proposal.normalized_argument_digest)
            self.assertEqual(result.to_dict()["parameters"]["path"], "allowed.txt")

    def test_qa_restriction_is_rechecked_at_approved_resumption(self):
        with patch.dict(os.environ, {"JARVIS_QA_MODE": "0"}):
            proposal = self.runtime.gateway.run_tool("file_controller", {
                "action": "write", "path": "denied.txt", "content": "must not write",
            }, runtime=self.runtime)
            ticket = self.runtime.owner.approve(proposal.request_id, proposal.normalized_argument_digest)
        with patch.dict(os.environ, {"JARVIS_QA_MODE": "1", "JARVIS_QA_WORKSPACE": str(self.root / "different")}):
            result = self.runtime.execute_approved(proposal.request_id, ticket.ticket_id)
        self.assertIs(result.authorization_decision, A.DENY)
        self.assertFalse((self.runtime.gateway.context.workspace_root / "denied.txt").exists())

    def test_screen_helpers_import_when_native_audio_is_unavailable(self):
        code = '''
import importlib.abc, sys
class NoAudio(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "sounddevice":
            raise OSError("PortAudio library not found")
sys.meta_path.insert(0, NoAudio())
from actions.screen_processor import _compress
from actions.tts_engine import PROVIDER_VOICES
assert callable(_compress)
assert PROVIDER_VOICES["gemini"]
assert "sounddevice" not in sys.modules
'''
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
