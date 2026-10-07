"""Read-only MCP capability discovery for NEXUS.

The first MCP integration deliberately supports discovery only. It can inspect
tools exposed by owner-configured Streamable HTTP servers, but it cannot invoke
a tool, mutate server state, or launch subprocesses. This lets JARVIS understand
available MCP capabilities while the Owner Kernel remains the sole authority for
future execution paths.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
import re
from types import MappingProxyType
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit


_SERVER_ID = re.compile(r"[a-z][a-z0-9_.-]{0,63}")
_TOOL_ID = re.compile(r"[A-Za-z0-9_.:/-]{1,160}")


class MCPDiscoveryError(RuntimeError):
    pass


@dataclass(frozen=True)
class MCPDiscoveryServer:
    server_id: str
    url: str
    allow_remote: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.server_id, str) or not _SERVER_ID.fullmatch(self.server_id):
            raise ValueError("Invalid MCP server id.")
        if not isinstance(self.url, str) or len(self.url) > 2048:
            raise ValueError("Invalid MCP server URL.")
        parsed = urlsplit(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("MCP server must use an http(s) URL.")
        if parsed.username or parsed.password:
            raise ValueError("Credentials must not be embedded in MCP URLs.")
        localhost = parsed.hostname.lower() in {"localhost", "127.0.0.1", "::1"}
        if not localhost and not self.allow_remote:
            raise ValueError("Remote MCP discovery requires explicit owner opt-in.")
        if not localhost and parsed.scheme != "https":
            raise ValueError("Remote MCP discovery requires HTTPS.")


@dataclass(frozen=True)
class MCPDiscoveredTool:
    server_id: str
    name: str
    description: str
    input_schema: Mapping[str, Any]


class MCPDiscoveryRegistry:
    def __init__(self, servers: tuple[MCPDiscoveryServer, ...] = ()) -> None:
        mapping: dict[str, MCPDiscoveryServer] = {}
        for server in servers:
            if server.server_id in mapping:
                raise ValueError(f"Duplicate MCP server id: {server.server_id}")
            mapping[server.server_id] = server
        self._servers = MappingProxyType(mapping)

    def get(self, server_id: str) -> MCPDiscoveryServer:
        try:
            return self._servers[server_id]
        except KeyError:
            raise MCPDiscoveryError("Unknown MCP discovery server.") from None

    def snapshot(self) -> tuple[MCPDiscoveryServer, ...]:
        return tuple(self._servers.values())

    @classmethod
    def from_environment(cls, variable: str = "JARVIS_MCP_DISCOVERY_JSON") -> "MCPDiscoveryRegistry":
        raw = os.environ.get(variable, "").strip()
        if not raw:
            return cls()
        if len(raw) > 131072:
            raise MCPDiscoveryError("MCP discovery configuration is too large.")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MCPDiscoveryError("MCP discovery configuration is not valid JSON.") from exc
        if not isinstance(payload, list) or len(payload) > 32:
            raise MCPDiscoveryError("MCP discovery configuration must be a bounded list.")
        servers: list[MCPDiscoveryServer] = []
        for item in payload:
            if not isinstance(item, dict) or set(item) - {"server_id", "url", "allow_remote"}:
                raise MCPDiscoveryError("Invalid MCP discovery server entry.")
            try:
                servers.append(MCPDiscoveryServer(
                    server_id=item["server_id"],
                    url=item["url"],
                    allow_remote=item.get("allow_remote", False) is True,
                ))
            except (KeyError, TypeError, ValueError):
                raise MCPDiscoveryError("Invalid MCP discovery server entry.") from None
        return cls(tuple(servers))


class MCPDiscoveryRuntime:
    def __init__(
        self,
        registry: MCPDiscoveryRegistry | None = None,
        *,
        client_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.registry = registry or MCPDiscoveryRegistry.from_environment()
        self._client_factory = client_factory

    def _make_client(self, url: str):
        if self._client_factory is not None:
            return self._client_factory(url)
        try:
            from mcp import Client
        except ImportError as exc:
            raise MCPDiscoveryError("The official MCP SDK is not installed.") from exc
        return Client(url, raise_exceptions=False)

    async def list_tools(self, server_id: str) -> tuple[MCPDiscoveredTool, ...]:
        server = self.registry.get(server_id)
        client = self._make_client(server.url)
        try:
            async with client:
                result = await client.list_tools()
        except MCPDiscoveryError:
            raise
        except Exception as exc:
            raise MCPDiscoveryError(f"MCP discovery failed ({type(exc).__name__}).") from None

        rows: list[MCPDiscoveredTool] = []
        for raw in getattr(result, "tools", ()) or ():
            name = getattr(raw, "name", "")
            if not isinstance(name, str) or not _TOOL_ID.fullmatch(name):
                continue
            description = getattr(raw, "description", "") or ""
            if not isinstance(description, str):
                description = ""
            schema = getattr(raw, "input_schema", None)
            if schema is None:
                schema = getattr(raw, "inputSchema", {})
            if not isinstance(schema, Mapping):
                schema = {}
            rows.append(MCPDiscoveredTool(
                server_id=server.server_id,
                name=name,
                description=description[:4000],
                input_schema=MappingProxyType(dict(schema)),
            ))
        return tuple(rows)

    @staticmethod
    def _run_sync(coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        coro.close()
        raise MCPDiscoveryError("Use the async MCP discovery API inside an active event loop.")

    def list_tools_sync(self, server_id: str) -> tuple[MCPDiscoveredTool, ...]:
        return self._run_sync(self.list_tools(server_id))
