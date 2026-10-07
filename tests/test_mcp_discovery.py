import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

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


class MCPDiscoveryTests(unittest.TestCase):
    def test_localhost_discovery_is_allowed(self):
        server = MCPDiscoveryServer("local-tools", "http://127.0.0.1:8765/mcp")
        runtime = MCPDiscoveryRuntime(MCPDiscoveryRegistry((server,)), client_factory=_FakeClient)
        tools = runtime.list_tools_sync("local-tools")
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].name, "memory.search")
        self.assertEqual(tools[0].server_id, "local-tools")
        self.assertEqual(tools[0].input_schema["type"], "object")

    def test_remote_requires_explicit_https_opt_in(self):
        with self.assertRaises(ValueError):
            MCPDiscoveryServer("remote", "https://tools.example.com/mcp")
        with self.assertRaises(ValueError):
            MCPDiscoveryServer("remote", "http://tools.example.com/mcp", allow_remote=True)
        server = MCPDiscoveryServer("remote", "https://tools.example.com/mcp", allow_remote=True)
        self.assertTrue(server.allow_remote)

    def test_registry_rejects_unknown_server(self):
        runtime = MCPDiscoveryRuntime(MCPDiscoveryRegistry(), client_factory=_FakeClient)
        with self.assertRaises(MCPDiscoveryError):
            runtime.list_tools_sync("missing")

    def test_environment_configuration_is_bounded_and_owner_explicit(self):
        with patch.dict(os.environ, {"JARVIS_MCP_DISCOVERY_JSON": '[{"server_id":"local","url":"http://localhost:9000/mcp"}]'}, clear=False):
            registry = MCPDiscoveryRegistry.from_environment()
            self.assertEqual(registry.snapshot()[0].server_id, "local")
        with patch.dict(os.environ, {"JARVIS_MCP_DISCOVERY_JSON": '[{"server_id":"remote","url":"https://example.com/mcp"}]'}, clear=False):
            with self.assertRaises(MCPDiscoveryError):
                MCPDiscoveryRegistry.from_environment()


if __name__ == "__main__":
    unittest.main()
