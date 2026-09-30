"""Exercise the real Live text entry methods with an inert audio import only.

These tests do not validate audio hardware, PyQt rendering or a live model service.
The substitute module is scoped to import, not installed into the existing suite.
"""

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch


class NexusTextRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "nexus_text_entry_under_test", Path(__file__).resolve().parents[1] / "main.py")
        cls.main = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {"sounddevice": SimpleNamespace()}):
            spec.loader.exec_module(cls.main)

    def setUp(self):
        self.client = SimpleNamespace(operational_ready=True, write_log=Mock(), show_subtitle=Mock())
        self.engine = self.main.JarvisLive(self.client)

    def test_desktop_callback_works_offline_without_provider(self):
        with patch("agent.planner.default_provider", side_effect=AssertionError("provider created")):
            self.engine._on_text_command("الساعة كام؟")
        self.client.show_subtitle.assert_called_once()
        self.assertEqual(self.engine.last_action_receipts[0].route, "reflex")

    def test_async_text_works_offline(self):
        self.assertTrue(asyncio.run(self.engine.send_text("What time is it?")))
        self.client.write_log.assert_called_once()
        self.assertTrue(self.client.write_log.call_args.args[0].startswith("Local time:"))

    def test_online_reflex_never_sends_a_live_model_turn(self):
        self.engine.session = SimpleNamespace(send_client_content=AsyncMock())
        self.assertTrue(asyncio.run(self.engine.send_text("time")))
        self.engine.session.send_client_content.assert_not_awaited()

    def test_nonmatching_text_keeps_existing_live_route(self):
        self.engine.session = SimpleNamespace(send_client_content=AsyncMock())
        self.assertTrue(asyncio.run(self.engine.send_text("Explain orbital mechanics")))
        self.engine.session.send_client_content.assert_awaited_once()
        self.client.show_subtitle.assert_not_called()

    def test_startup_and_shutdown_gates_prevent_reflex_dispatch(self):
        self.client.operational_ready = False
        self.engine._on_text_command("time")
        self.client.operational_ready = True
        self.engine._shutdown_requested.set()
        self.engine._on_text_command("time")
        self.client.show_subtitle.assert_not_called()


if __name__ == "__main__":
    unittest.main()
