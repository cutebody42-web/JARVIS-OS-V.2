import asyncio

import pytest

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


def test_semantic_snapshot_is_bounded_and_sanitized():
    page = _Page({
        "url": "https://example.com/path",
        "title": "  Example   Page  ",
        "nodes": [
            {
                "tag": "BUTTON",
                "role": "button",
                "name": " Save  file ",
                "text": "  Save\nnow ",
                "inputType": "",
                "href": "",
                "disabled": False,
                "visible": True,
                "rect": {"x": 1, "y": 2, "width": 100, "height": 40},
            },
            {
                "tag": "a",
                "role": "",
                "name": "Docs",
                "text": "Documentation",
                "inputType": "",
                "href": "https://example.com/docs",
                "disabled": False,
                "visible": True,
                "rect": {"x": 2, "y": 50, "width": 200, "height": 20},
            },
        ],
    })
    result = asyncio.run(BrowserSemanticObserver().snapshot(page, max_nodes=1))

    assert result.url == "https://example.com/path"
    assert result.title == "Example Page"
    assert result.truncated is True
    assert len(result.elements) == 1
    assert result.elements[0].tag == "button"
    assert result.elements[0].name == "Save file"
    assert result.elements[0].rectangle.width == 100
    assert page.calls[0][1] == {"maxNodes": 1, "includeHidden": False}


def test_observer_rejects_unbounded_or_invalid_inputs():
    observer = BrowserSemanticObserver()
    page = _Page({"nodes": []})
    with pytest.raises(ValueError):
        asyncio.run(observer.snapshot(page, max_nodes=0))
    with pytest.raises(ValueError):
        asyncio.run(observer.snapshot(page, max_nodes=801))
    with pytest.raises(ValueError):
        asyncio.run(observer.snapshot(None))


def test_observer_wraps_page_errors_without_leaking_details():
    page = _Page(error=RuntimeError("secret browser details"))
    with pytest.raises(BrowserSemanticError) as error:
        asyncio.run(BrowserSemanticObserver().snapshot(page))
    assert "RuntimeError" in str(error.value)
    assert "secret browser details" not in str(error.value)


def test_observation_script_is_fixed_and_read_only():
    script = BrowserSemanticObserver().observation_script()
    lowered = script.lower()
    assert "queryselectorall" in lowered
    assert ".click(" not in lowered
    assert ".fill(" not in lowered
    assert ".type(" not in lowered
    assert "window.location =" not in lowered
