import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.browser_semantics import BrowserSemanticObserver
from core.nexus.integration_hub import NEXUSIntegrationHub


class NEXUSBrowserHubTests(unittest.TestCase):
    def test_browser_semantics_is_first_class_when_playwright_is_available(self):
        with tempfile.TemporaryDirectory() as td:
            hub = NEXUSIntegrationHub(data_root=Path(td), environment={}, platform_name="Linux")
            with patch.object(hub, "_module_available", side_effect=lambda name: name == "playwright"):
                states = {item.name: item for item in hub.status()}
                self.assertTrue(states["browser_semantics"].available)
                self.assertEqual(states["browser_semantics"].provider, "playwright-dom")
                observer = hub.browser_semantics()
                self.assertIsInstance(observer, BrowserSemanticObserver)
                self.assertTrue({item.name: item for item in hub.status()}["browser_semantics"].active)

    def test_browser_semantics_fails_closed_without_playwright(self):
        with tempfile.TemporaryDirectory() as td:
            hub = NEXUSIntegrationHub(data_root=Path(td), environment={}, platform_name="Linux")
            with patch.object(hub, "_module_available", return_value=False):
                with self.assertRaises(RuntimeError):
                    hub.browser_semantics()


if __name__ == "__main__":
    unittest.main()
