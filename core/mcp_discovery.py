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
import math
import os
import re
from types import MappingProxyType
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit


_SERVER_ID = re.compile(r"[a-z][a-z0-9_.-]{0,63}")
_TOOL_ID = re.compile(r"[A-Za-z0-9_.:/-]{1,160}")
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

_MAX_CONFIG_BYTES = 131072
_MAX_SERVERS = 32
_MAX_SCHEMA_BYTES = 65536
_MAX_SCHEMA_DEPTH = 20
_MAX_SCHEMA_NODES = 4096
_MAX_SCHEMA_COLLECTION_ITEMS = 1024


class MCPDiscoveryError(RuntimeError):
    pass


def _copy_json_value(
    value: Any,
    *,
    depth: int,
    active: set[int],
    nodes: list[int],
    byte_budget: list[int],
) -> Any:
    """Copy a JSON value while enforcing limits before it becomes trusted data."""
    if depth > _MAX_SCHEMA_DEPTH:
        raise ValueError("MCP tool schema is too deeply nested.")
    nodes[0] += 1
    if nodes[0] > _MAX_SCHEMA_NODES:
        raise ValueError("MCP tool schema has too many values.")
    byte_budget[0] -= 4
    if byte_budget[0] < 0:
        raise ValueError("MCP tool schema is too large.")

    if isinstance(value, str):
        try:
            size = len(value.encode("utf-8"))
        except UnicodeEncodeError:
            raise ValueError("MCP tool schema contains invalid text.") from None
        byte_budget[0] -= size
        if byte_budget[0] < 0:
            raise ValueError("MCP tool schema is too large.")
        return value
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        if value.bit_length() > 4096:
            raise ValueError("MCP tool schema contains an oversized integer.")
        byte_budget[0] -= len(str(value))
        if byte_budget[0] < 0:
            raise ValueError("MCP tool schema is too large.")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("MCP tool schema contains a non-finite number.")
        byte_budget[0] -= 32
        if byte_budget[0] < 0:
            raise ValueError("MCP tool schema is too large.")
        return value

    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise ValueError("MCP tool schema contains a cycle.")
        if len(value) > _MAX_SCHEMA_COLLECTION_ITEMS:
            raise ValueError("MCP tool schema collection is too large.")
        active.add(identity)
        try:
            copied: dict[str, Any] = {}
            for key, nested in value.items():
                if not isinstance(key, str):
                    raise ValueError("MCP tool schema keys must be strings.")
                try:
                    key_size = len(key.encode("utf-8"))
                except UnicodeEncodeError:
                    raise ValueError("MCP tool schema contains invalid text.") from None
                byte_budget[0] -= key_size
                if byte_budget[0] < 0:
                    raise ValueError("MCP tool schema is too large.")
                copied[key] = _copy_json_value(
                    nested,
                    depth=depth + 1,
                    active=active,
                    nodes=nodes,
                    byte_budget=byte_budget,
                )
            return copied
        finally:
            active.remove(identity)

    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            raise ValueError("MCP tool schema contains a cycle.")
        if len(value) > _MAX_SCHEMA_COLLECTION_ITEMS:
            raise ValueError("MCP tool schema collection is too large.")
        active.add(identity)
        try:
            return [
                _copy_json_value(
                    nested,
                    depth=depth + 1,
                    active=active,
                    nodes=nodes,
                    byte_budget=byte_budget,
                )
                for nested in value
            ]
        finally:
            active.remove(identity)

    raise ValueError("MCP tool schema must contain only JSON values.")


def _freeze_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json_value(nested) for key, nested in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json_value(nested) for nested in value)
    return value


def _bounded_schema(schema: Mapping[str, Any]) -> Mapping[str, Any]:
    copied = _copy_json_value(
        schema,
        depth=0,
        active=set(),
        nodes=[0],
        byte_budget=[_MAX_SCHEMA_BYTES],
    )
    if not isinstance(copied, dict):
        raise ValueError("MCP tool schema must be an object.")
    try:
        encoded = json.dumps(
            copied,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError, UnicodeEncodeError):
        raise ValueError("MCP tool schema is not valid JSON.") from None
    if len(encoded) > _MAX_SCHEMA_BYTES:
        raise ValueError("MCP tool schema is too large.")
    return _freeze_json_value(copied)


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
        if not isinstance(self.allow_remote, bool):
            raise ValueError("allow_remote must be boolean.")
        if any(ord(character) <= 0x20 or ord(character) == 0x7F for character in self.url):
            raise ValueError("Invalid MCP server URL.")
        try:
            parsed = urlsplit(self.url)
            hostname = parsed.hostname
            # Accessing port performs urllib's range and syntax validation.
            parsed.port
        except ValueError:
            raise ValueError("Invalid MCP server URL.") from None
        if parsed.scheme not in {"http", "https"} or not hostname:
            raise ValueError("MCP server must use an http(s) URL.")
        if parsed.username or parsed.password:
            raise ValueError("Credentials must not be embedded in MCP URLs.")
        local = hostname.lower() in _LOCAL_HOSTS
        if not local and not self.allow_remote:
            raise ValueError("Remote MCP discovery requires explicit owner opt-in.")
        if not local and parsed.scheme != "https":
            raise ValueError("Remote MCP discovery requires HTTPS.")


@dataclass(frozen=True)
class MCPDiscoveredTool:
    server_id: str
    name: str
    description: str
    input_schema: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.server_id, str) or not _SERVER_ID.fullmatch(self.server_id):
            raise ValueError("Invalid MCP tool server id.")
        if not isinstance(self.name, str) or not _TOOL_ID.fullmatch(self.name):
            raise ValueError("Invalid MCP tool name.")
        if not isinstance(self.description, str) or len(self.description) > 4000:
            raise ValueError("Invalid MCP tool description.")
        if not isinstance(self.input_schema, Mapping):
            raise ValueError("MCP tool schema must be an object.")
        object.__setattr__(self, "input_schema", _bounded_schema(self.input_schema))


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
    def from_json(cls, raw: str) -> "MCPDiscoveryRegistry":
        """Parse a bounded registry from an explicit JSON value."""
        if not isinstance(raw, str):
            raise MCPDiscoveryError("MCP discovery configuration must be text.")
        raw = raw.strip()
        if not raw:
            return cls()
        if len(raw) > _MAX_CONFIG_BYTES:
            raise MCPDiscoveryError("MCP discovery configuration is too large.")
        try:
            raw_size = len(raw.encode("utf-8"))
        except UnicodeEncodeError:
            raise MCPDiscoveryError("MCP discovery configuration is not valid UTF-8 text.") from None
        if raw_size > _MAX_CONFIG_BYTES:
            raise MCPDiscoveryError("MCP discovery configuration is too large.")
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise MCPDiscoveryError("MCP discovery configuration is not valid JSON.") from exc
        return cls.from_config(payload)

    @classmethod
    def from_config(cls, payload: Any) -> "MCPDiscoveryRegistry":
        """Build a registry from already-decoded, application-provided config."""
        if not isinstance(payload, list) or len(payload) > _MAX_SERVERS:
            raise MCPDiscoveryError("MCP discovery configuration must be a bounded list.")
        servers: list[MCPDiscoveryServer] = []
        for item in payload:
            if not isinstance(item, Mapping) or set(item) - {"server_id", "url", "allow_remote"}:
                raise MCPDiscoveryError("Invalid MCP discovery server entry.")
            allow_remote = item.get("allow_remote", False)
            if not isinstance(allow_remote, bool):
                raise MCPDiscoveryError("Invalid MCP discovery server entry.")
            try:
                servers.append(MCPDiscoveryServer(
                    server_id=item["server_id"],
                    url=item["url"],
                    allow_remote=allow_remote,
                ))
            except (KeyError, TypeError, ValueError):
                raise MCPDiscoveryError("Invalid MCP discovery server entry.") from None
        try:
            return cls(tuple(servers))
        except ValueError as exc:
            raise MCPDiscoveryError(str(exc)) from None

    @classmethod
    def from_environment(
        cls,
        variable: str = "JARVIS_MCP_DISCOVERY_JSON",
        *,
        environ: Mapping[str, str] | None = None,
    ) -> "MCPDiscoveryRegistry":
        source = os.environ if environ is None else environ
        try:
            raw = source.get(variable, "")
        except (AttributeError, TypeError):
            raise MCPDiscoveryError("Invalid MCP discovery environment mapping.") from None
        return cls.from_json(raw)


class MCPDiscoveryRuntime:
    DISCOVERY_TIMEOUT_SECONDS = 10.0
    MAX_PAGES = 16
    MAX_TOOLS = 256
    MAX_CURSOR_LENGTH = 4096

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

        async def discover() -> tuple[MCPDiscoveredTool, ...]:
            rows: list[MCPDiscoveredTool] = []
            names: set[str] = set()
            seen_cursors: set[str] = set()
            cursor: str | None = None

            async with client:
                for page in range(self.MAX_PAGES):
                    if cursor is None:
                        result = await client.list_tools()
                    else:
                        result = await client.list_tools(cursor=cursor)

                    tools = getattr(result, "tools", None)
                    if not isinstance(tools, (list, tuple)):
                        raise MCPDiscoveryError("MCP discovery returned an invalid tool page.")
                    if len(rows) + len(tools) > self.MAX_TOOLS:
                        raise MCPDiscoveryError("MCP discovery returned too many tools.")

                    for raw in tools:
                        name = getattr(raw, "name", None)
                        if not isinstance(name, str) or not _TOOL_ID.fullmatch(name):
                            raise MCPDiscoveryError("MCP discovery returned an invalid tool name.")
                        if name in names:
                            raise MCPDiscoveryError("MCP discovery returned a duplicate tool name.")

                        description = getattr(raw, "description", "")
                        if description is None:
                            description = ""
                        if not isinstance(description, str) or len(description) > 4000:
                            raise MCPDiscoveryError("MCP discovery returned an invalid tool description.")

                        missing = object()
                        schema = getattr(raw, "input_schema", missing)
                        if schema is missing:
                            schema = getattr(raw, "inputSchema", missing)
                        if not isinstance(schema, Mapping):
                            raise MCPDiscoveryError("MCP discovery returned an invalid tool schema.")
                        try:
                            tool = MCPDiscoveredTool(
                                server_id=server.server_id,
                                name=name,
                                description=description,
                                input_schema=schema,
                            )
                        except (TypeError, ValueError, RecursionError):
                            raise MCPDiscoveryError("MCP discovery returned an invalid tool schema.") from None
                        rows.append(tool)
                        names.add(name)

                    next_cursor = getattr(result, "nextCursor", None)
                    if next_cursor is None:
                        next_cursor = getattr(result, "next_cursor", None)
                    if next_cursor is None:
                        return tuple(rows)
                    if (
                        not isinstance(next_cursor, str)
                        or not next_cursor
                        or len(next_cursor) > self.MAX_CURSOR_LENGTH
                        or next_cursor in seen_cursors
                    ):
                        raise MCPDiscoveryError("MCP discovery returned an invalid pagination cursor.")
                    if page + 1 >= self.MAX_PAGES:
                        raise MCPDiscoveryError("MCP discovery returned too many pages.")
                    seen_cursors.add(next_cursor)
                    cursor = next_cursor

            # The loop either returns on the final page or rejects another page.
            raise MCPDiscoveryError("MCP discovery pagination did not terminate.")

        try:
            return await asyncio.wait_for(discover(), timeout=self.DISCOVERY_TIMEOUT_SECONDS)
        except MCPDiscoveryError:
            raise
        except TimeoutError:
            raise MCPDiscoveryError("MCP discovery timed out.") from None
        except Exception as exc:
            raise MCPDiscoveryError(f"MCP discovery failed ({type(exc).__name__}).") from None

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
