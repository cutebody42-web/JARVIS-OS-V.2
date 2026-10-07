from pathlib import Path

import pytest

from core.local_stt import LocalSTTError
from core.mcp_discovery import MCPDiscoveryError
from core.nexus.integration_hub import NEXUSIntegrationHub


def _states(hub):
    return {item.name: item for item in hub.status()}


def test_hub_exposes_temporal_memory_as_builtin(tmp_path):
    hub = NEXUSIntegrationHub(data_root=tmp_path, environment={}, platform_name="Linux")
    states = _states(hub)

    assert states["temporal_memory"].available is True
    assert states["temporal_memory"].provider == "sqlite-temporal-claims"

    store = hub.temporal_memory()
    claim = store.add_claim(
        subject="jarvis",
        predicate="integration",
        value="temporal-memory",
        source="test:hub",
        verified=True,
    )
    assert store.get(claim.claim_id).value == "temporal-memory"
    assert _states(hub)["temporal_memory"].active is True


def test_hub_requires_explicit_local_stt_model_path(tmp_path):
    hub = NEXUSIntegrationHub(data_root=tmp_path, environment={}, platform_name="Linux")
    with pytest.raises(LocalSTTError):
        hub.local_speech()


def test_hub_reports_configured_stt_path_without_starting_runtime(tmp_path):
    model = tmp_path / "whisper"
    model.mkdir()
    hub = NEXUSIntegrationHub(
        data_root=tmp_path,
        environment={"JARVIS_STT_MODEL_PATH": str(model)},
        platform_name="Linux",
    )
    state = _states(hub)["local_stt"]
    assert state.configured is True
    assert state.active is False


def test_hub_validates_mcp_configuration_only_on_explicit_use(tmp_path):
    hub = NEXUSIntegrationHub(
        data_root=tmp_path,
        environment={"JARVIS_MCP_DISCOVERY_JSON": "not-json"},
        platform_name="Linux",
    )
    assert _states(hub)["mcp_discovery"].configured is True
    with pytest.raises(MCPDiscoveryError):
        hub.mcp_discovery()


def test_hub_builds_local_mcp_registry_without_connecting(tmp_path):
    hub = NEXUSIntegrationHub(
        data_root=tmp_path,
        environment={
            "JARVIS_MCP_DISCOVERY_JSON": '[{"server_id":"local","url":"http://127.0.0.1:9000/mcp"}]'
        },
        platform_name="Linux",
    )
    runtime = hub.mcp_discovery()
    assert runtime.registry.snapshot()[0].server_id == "local"
    assert _states(hub)["mcp_discovery"].active is True


def test_non_windows_uia_fails_closed(tmp_path):
    hub = NEXUSIntegrationHub(data_root=tmp_path, environment={}, platform_name="Linux")
    with pytest.raises(RuntimeError):
        hub.windows_uia()
