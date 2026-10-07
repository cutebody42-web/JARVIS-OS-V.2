import asyncio
import unittest

from core.browser_semantics import BrowserSemanticError, BrowserSemanticObserver


class _Page:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.calls = []

    async def evaluate(self, script, args):
        self.calls.append((script, args))
        if self.error:
            raise self.error
        return self.payload


class BrowserSemanticTests(unittest.TestCase):
    def test_semantic_snapshot_is_bounded_and_sanitized(self):
        page = _Page({
            "url": "https://example.com/path",
            "title": "  Example   Page  ",
            "nodes": [
                {
                    "tag": "BUTTON", "role": "button", "name": " Save  file ",
                    "text": "  Save\nnow ", "inputType": "", "href": "",
                    "disabled": False, "visible": True,
                    "rect": {"x": 1, "y": 2, "width": 100, "height": 40},
                },
                {
                    "tag": "a", "role": "", "name": "Docs", "text": "Documentation",
                    "inputType": "", "href": "https://example.com/docs",
                    "disabled": False, "visible": True,
                    "rect": {"x": 2, "y": 50, "width": 200, "height": 20},
                },
            ],
        })
        result = asyncio.run(BrowserSemanticObserver().snapshot(page, max_nodes=1))
        self.assertEqual(result.url, "https://example.com/path")
        self.assertEqual(result.title, "Example Page")
        self.assertTrue(result.truncated)
        self.assertEqual(len(result.elements), 1)
        self.assertEqual(result.elements[0].tag, "button")
        self.assertEqual(result.elements[0].name, "Save file")
        self.assertEqual(result.elements[0].rectangle.width, 100)
        self.assertEqual(page.calls[0][1], {"maxNodes": 1, "includeHidden": False})

    def test_observer_rejects_unbounded_or_invalid_inputs(self):
        observer = BrowserSemanticObserver()
        page = _Page({"nodes": []})
        with self.assertRaises(ValueError):
            asyncio.run(observer.snapshot(page, max_nodes=0))
        with self.assertRaises(ValueError):
            asyncio.run(observer.snapshot(page, max_nodes=801))
        with self.assertRaises(ValueError):
            asyncio.run(observer.snapshot(None))

    def test_observer_wraps_page_errors_without_leaking_details(self):
        page = _Page(error=RuntimeError("secret browser details"))
        with self.assertRaises(BrowserSemanticError) as caught:
            asyncio.run(BrowserSemanticObserver().snapshot(page))
        self.assertIn("RuntimeError", str(caught.exception))
        self.assertNotIn("secret browser details", str(caught.exception))

    def test_observation_script_is_fixed_and_read_only(self):
        script = BrowserSemanticObserver().observation_script().lower()
        self.assertIn("queryselectorall", script)
        self.assertNotIn(".click(", script)
        self.assertNotIn(".fill(", script)
        self.assertNotIn(".type(", script)
        self.assertNotIn("window.location =", script)


if __name__ == "__main__":
    unittest.main()
