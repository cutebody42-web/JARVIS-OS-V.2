import asyncio
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
        ])


class _PagedClient:
    def __init__(self, url, pages):
        self.url = url
        self.pages = pages
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def list_tools(self, cursor=None):
        self.calls.append(cursor)
        return self.pages[cursor]


def _tool(name="memory.search", *, description="ok", schema=None):
    return SimpleNamespace(
        name=name,
        description=description,
        input_schema={"type": "object"} if schema is None else schema,
    )


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

    def test_remote_opt_in_is_a_strict_boolean_and_local_hosts_are_literal(self):
        for value in (1, 0, "true", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MCPDiscoveryServer("remote", "https://tools.example.com/mcp", allow_remote=value)
        for url in (
            "http://localhost.example.com/mcp",
            "http://127.0.0.1.example.com/mcp",
            "http://2130706433/mcp",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                MCPDiscoveryServer("not-local", url)
        self.assertEqual(MCPDiscoveryServer("ipv6", "http://[::1]:9000/mcp").url, "http://[::1]:9000/mcp")

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

    def test_registry_parser_supports_injected_config_and_rejects_truthy_opt_in(self):
        registry = MCPDiscoveryRegistry.from_environment(environ={
            "JARVIS_MCP_DISCOVERY_JSON": '[{"server_id":"local","url":"http://localhost:9000/mcp"}]',
        })
        self.assertEqual(registry.snapshot()[0].server_id, "local")
        decoded = [{"server_id": "remote", "url": "https://example.com/mcp", "allow_remote": True}]
        self.assertTrue(MCPDiscoveryRegistry.from_config(decoded).snapshot()[0].allow_remote)
        for value in (1, "true", None):
            raw = '[{"server_id":"remote","url":"https://example.com/mcp","allow_remote":%s}]' % (
                "null" if value is None else str(value).lower() if isinstance(value, int) else '"true"'
            )
            with self.subTest(value=value), self.assertRaises(MCPDiscoveryError):
                MCPDiscoveryRegistry.from_json(raw)

    def test_discovery_paginates_and_deep_copies_immutable_schemas(self):
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string", "enum": ["a", "b"]}},
        }
        pages = {
            None: SimpleNamespace(tools=[_tool("memory.search", schema=schema)], nextCursor="second"),
            "second": SimpleNamespace(tools=[_tool("memory.get")], nextCursor=None),
        }
        client = _PagedClient("", pages)
        server = MCPDiscoveryServer("local", "http://127.0.0.1:8765/mcp")
        runtime = MCPDiscoveryRuntime(MCPDiscoveryRegistry((server,)), client_factory=lambda url: client)
        tools = runtime.list_tools_sync("local")
        self.assertEqual([item.name for item in tools], ["memory.search", "memory.get"])
        self.assertEqual(client.calls, [None, "second"])

        schema["properties"]["query"]["type"] = "number"
        schema["properties"]["query"]["enum"].append("c")
        query = tools[0].input_schema["properties"]["query"]
        self.assertEqual(query["type"], "string")
        self.assertEqual(query["enum"], ("a", "b"))
        with self.assertRaises(TypeError):
            query["type"] = "boolean"

    def test_discovery_fails_closed_instead_of_skipping_or_truncating(self):
        server = MCPDiscoveryServer("local", "http://127.0.0.1:8765/mcp")

        for raw_tool in (
            _tool("bad tool name"),
            _tool(description="x" * 4001),
            _tool(schema={"description": "x" * 70000}),
            _tool(schema={"items": ["x" * 1000] * 70}),
            _tool(schema={"maximum": 1 << 5000}),
        ):
            client = _PagedClient("", {None: SimpleNamespace(tools=[raw_tool])})
            runtime = MCPDiscoveryRuntime(MCPDiscoveryRegistry((server,)), client_factory=lambda url, client=client: client)
            with self.subTest(tool=raw_tool), self.assertRaises(MCPDiscoveryError):
                runtime.list_tools_sync("local")

        deeply_nested = {}
        cursor = deeply_nested
        for _ in range(22):
            cursor["nested"] = {}
            cursor = cursor["nested"]
        client = _PagedClient("", {None: SimpleNamespace(tools=[_tool(schema=deeply_nested)])})
        runtime = MCPDiscoveryRuntime(MCPDiscoveryRegistry((server,)), client_factory=lambda url: client)
        with self.assertRaises(MCPDiscoveryError):
            runtime.list_tools_sync("local")

    def test_discovery_bounds_tool_count_pages_and_total_time(self):
        server = MCPDiscoveryServer("local", "http://127.0.0.1:8765/mcp")
        registry = MCPDiscoveryRegistry((server,))

        too_many = _PagedClient("", {None: SimpleNamespace(tools=[_tool("one"), _tool("two")])})
        runtime = MCPDiscoveryRuntime(registry, client_factory=lambda url: too_many)
        runtime.MAX_TOOLS = 1
        with self.assertRaises(MCPDiscoveryError):
            runtime.list_tools_sync("local")

        pages = {
            None: SimpleNamespace(tools=[_tool("one")], nextCursor="two"),
            "two": SimpleNamespace(tools=[_tool("two")], nextCursor="three"),
        }
        paged = _PagedClient("", pages)
        runtime = MCPDiscoveryRuntime(registry, client_factory=lambda url: paged)
        runtime.MAX_PAGES = 2
        with self.assertRaises(MCPDiscoveryError):
            runtime.list_tools_sync("local")
        self.assertEqual(paged.calls, [None, "two"])

        class SlowClient(_FakeClient):
            async def list_tools(self):
                await asyncio.sleep(1)
                return SimpleNamespace(tools=[])

        runtime = MCPDiscoveryRuntime(registry, client_factory=SlowClient)
        runtime.DISCOVERY_TIMEOUT_SECONDS = 0.001
        with self.assertRaisesRegex(MCPDiscoveryError, "timed out"):
            runtime.list_tools_sync("local")


if __name__ == "__main__":
    unittest.main()
