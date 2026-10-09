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

    def test_scan_budget_truncation_and_oversized_fields_are_preserved_safely(self):
        page = _Page({
            "url": "u" * 5000,
            "title": "t" * 800,
            "truncated": True,
            "nodes": [{
                "tag": "BUTTON" * 20,
                "role": "r" * 200,
                "name": "n" * 800,
                "text": "x" * 5000,
                "inputType": "i" * 100,
                "href": "h" * 3000,
                "disabled": True,
                "visible": True,
                "rect": {
                    "x": float("nan"),
                    "y": float("inf"),
                    "width": float("-inf"),
                    "height": 10**20,
                },
            }],
        })
        result = asyncio.run(BrowserSemanticObserver().snapshot(page, max_nodes=1))
        self.assertTrue(result.truncated)
        self.assertEqual(len(result.url), 4096)
        self.assertEqual(len(result.title), 500)
        self.assertEqual(len(result.elements[0].tag), 40)
        self.assertEqual(len(result.elements[0].role), 100)
        self.assertEqual(len(result.elements[0].name), 500)
        self.assertEqual(len(result.elements[0].text), 1000)
        self.assertEqual(len(result.elements[0].input_type), 80)
        self.assertEqual(len(result.elements[0].href), 2048)
        self.assertEqual(result.elements[0].rectangle.x, 0.0)
        self.assertEqual(result.elements[0].rectangle.y, 0.0)
        self.assertEqual(result.elements[0].rectangle.width, 0.0)
        self.assertEqual(result.elements[0].rectangle.height, 1_000_000.0)

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
        self.assertIn("createtreewalker", script)
        self.assertIn("visited < scanlimit", script)
        self.assertIn("out.length <= maxnodes", script)
        self.assertIn("value.slice(0, limit)", script)
        self.assertNotIn("array.from", script)
        self.assertNotIn("queryselectorall", script)
        self.assertIn("boundedtext", script)
        self.assertIn("text: boundedtext(", script)
        self.assertNotIn("innertext", script)
        self.assertNotIn(".textcontent", script)
        self.assertIn("url: clip(", script)
        self.assertIn("title: clip(", script)
        self.assertNotIn(".click(", script)
        self.assertNotIn(".fill(", script)
        self.assertNotIn(".type(", script)
        self.assertNotIn("window.location =", script)


if __name__ == "__main__":
    unittest.main()
