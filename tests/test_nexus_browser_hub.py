import pytest

from core.browser_semantics import BrowserSemanticObserver
from core.nexus.integration_hub import NEXUSIntegrationHub


def test_browser_semantics_is_first_class_when_playwright_is_available(monkeypatch, tmp_path):
    hub = NEXUSIntegrationHub(data_root=tmp_path, environment={}, platform_name="Linux")
    monkeypatch.setattr(hub, "_module_available", lambda name: name == "playwright")

    states = {item.name: item for item in hub.status()}
    assert states["browser_semantics"].available is True
    assert states["browser_semantics"].provider == "playwright-dom"

    observer = hub.browser_semantics()
    assert isinstance(observer, BrowserSemanticObserver)
    assert {item.name: item for item in hub.status()}["browser_semantics"].active is True


def test_browser_semantics_fails_closed_without_playwright(monkeypatch, tmp_path):
    hub = NEXUSIntegrationHub(data_root=tmp_path, environment={}, platform_name="Linux")
    monkeypatch.setattr(hub, "_module_available", lambda name: False)
    with pytest.raises(RuntimeError):
        hub.browser_semantics()
