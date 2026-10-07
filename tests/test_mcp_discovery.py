from types import SimpleNamespace

import pytest

from core.mcp_discovery import (
    MCPDiscoveryError,
    MCPDiscoveryRegistry,
    MCPDiscoveryRuntime,
    MCPDiscoveryServer,
)


class _FakeClient:
    def __init__(self, url):
        self.url = url

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def list_tools(self):
        return SimpleNamespace(tools=[
            SimpleNamespace(
                name="memory.search",
                description="Search owner-approved memory",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            ),
            SimpleNamespace(name="bad tool name", description="ignored", input_schema={}),
        ])


def test_localhost_discovery_is_allowed():
    server = MCPDiscoveryServer("local-tools", "http://127.0.0.1:8765/mcp")
    registry = MCPDiscoveryRegistry((server,))
    runtime = MCPDiscoveryRuntime(registry, client_factory=_FakeClient)

    tools = runtime.list_tools_sync("local-tools")

    assert len(tools) == 1
    assert tools[0].name == "memory.search"
    assert tools[0].server_id == "local-tools"
    assert tools[0].input_schema["type"] == "object"


def test_remote_requires_explicit_https_opt_in():
    with pytest.raises(ValueError):
        MCPDiscoveryServer("remote", "https://tools.example.com/mcp")
    with pytest.raises(ValueError):
        MCPDiscoveryServer("remote", "http://tools.example.com/mcp", allow_remote=True)

    server = MCPDiscoveryServer("remote", "https://tools.example.com/mcp", allow_remote=True)
    assert server.allow_remote is True


def test_registry_rejects_unknown_server():
    runtime = MCPDiscoveryRuntime(MCPDiscoveryRegistry(), client_factory=_FakeClient)
    with pytest.raises(MCPDiscoveryError):
        runtime.list_tools_sync("missing")


def test_environment_configuration_is_bounded_and_owner_explicit(monkeypatch):
    monkeypatch.setenv(
        "JARVIS_MCP_DISCOVERY_JSON",
        '[{"server_id":"local","url":"http://localhost:9000/mcp"}]',
    )
    registry = MCPDiscoveryRegistry.from_environment()
    assert registry.snapshot()[0].server_id == "local"

    monkeypatch.setenv(
        "JARVIS_MCP_DISCOVERY_JSON",
        '[{"server_id":"remote","url":"https://example.com/mcp"}]',
    )
    with pytest.raises(MCPDiscoveryError):
        MCPDiscoveryRegistry.from_environment()
