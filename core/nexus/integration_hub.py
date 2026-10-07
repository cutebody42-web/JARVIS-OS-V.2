"""First-class integration hub for the NEXUS runtime.

The hub makes external/open-source powered subsystems part of the NEXUS core
without letting imports or model output become authority. Construction is
side-effect free: no network connection, microphone capture, subprocess, or
model download occurs until an explicit subsystem method is called.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import platform
from typing import Mapping

from core.browser_semantics import BrowserSemanticObserver
from core.local_stt import FasterWhisperRuntime, LocalSTTError
from core.mcp_discovery import MCPDiscoveryRegistry, MCPDiscoveryRuntime, MCPDiscoveryError
from core.temporal_memory import TemporalMemoryStore
from core.windows_uia import WindowsUIAObserver


@dataclass(frozen=True)
class IntegrationState:
    name: str
    available: bool
    configured: bool
    active: bool
    provider: str
    detail: str = ""


class NEXUSIntegrationHub:
    """Single reviewed entry point for optional NEXUS subsystems."""

    def __init__(
        self,
        *,
        data_root: str | Path | None = None,
        environment: Mapping[str, str] | None = None,
        platform_name: str | None = None,
    ) -> None:
        root = Path(data_root or os.environ.get("JARVIS_DATA_ROOT", "~/.jarvis")).expanduser()
        self.data_root = root.resolve()
        self._env = dict(os.environ if environment is None else environment)
        self._platform = platform_name or platform.system()
        self._speech = None
        self._mcp = None
        self._uia = None
        self._memory = None
        self._browser = None

    def _module_available(self, name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is not None
        except (ImportError, AttributeError, ValueError):
            return False

    @property
    def stt_model_path(self) -> Path | None:
        value = self._env.get("JARVIS_STT_MODEL_PATH", "").strip()
        return Path(value).expanduser().resolve() if value else None

    @property
    def temporal_memory_path(self) -> Path:
        value = self._env.get("JARVIS_TEMPORAL_MEMORY_PATH", "").strip()
        return (Path(value).expanduser() if value else self.data_root / "memory" / "temporal.db").resolve()

    def status(self) -> tuple[IntegrationState, ...]:
        stt_path = self.stt_model_path
        stt_installed = self._module_available("faster_whisper")
        stt_configured = bool(stt_path and stt_path.exists() and stt_path.is_dir())

        mcp_installed = self._module_available("mcp")
        mcp_configured = bool(self._env.get("JARVIS_MCP_DISCOVERY_JSON", "").strip())

        uia_installed = self._module_available("pywinauto")
        uia_available = self._platform == "Windows" and uia_installed
        playwright_installed = self._module_available("playwright")

        return (
            IntegrationState(
                "local_stt",
                available=stt_installed and stt_configured,
                configured=stt_configured,
                active=self._speech is not None,
                provider="faster-whisper",
                detail=(str(stt_path) if stt_path else "Set JARVIS_STT_MODEL_PATH to an existing local model directory."),
            ),
            IntegrationState(
                "mcp_discovery",
                available=mcp_installed and mcp_configured,
                configured=mcp_configured,
                active=self._mcp is not None,
                provider="modelcontextprotocol/python-sdk",
                detail=("configured" if mcp_configured else "Set JARVIS_MCP_DISCOVERY_JSON."),
            ),
            IntegrationState(
                "windows_uia",
                available=uia_available,
                configured=self._platform == "Windows",
                active=self._uia is not None,
                provider="pywinauto-uia",
                detail=("Windows UIA ready." if uia_available else "Requires Windows and pywinauto."),
            ),
            IntegrationState(
                "browser_semantics",
                available=playwright_installed,
                configured=playwright_installed,
                active=self._browser is not None,
                provider="playwright-dom",
                detail=("Semantic DOM observer ready." if playwright_installed else "Requires Playwright."),
            ),
            IntegrationState(
                "temporal_memory",
                available=True,
                configured=True,
                active=self._memory is not None,
                provider="sqlite-temporal-claims",
                detail=str(self.temporal_memory_path),
            ),
        )

    def local_speech(self, *, device: str = "auto", compute_type: str = "default") -> FasterWhisperRuntime:
        if self._speech is None:
            path = self.stt_model_path
            if path is None:
                raise LocalSTTError("JARVIS_STT_MODEL_PATH is not configured.")
            self._speech = FasterWhisperRuntime(path, device=device, compute_type=compute_type)
        return self._speech

    def mcp_discovery(self) -> MCPDiscoveryRuntime:
        if self._mcp is None:
            raw = self._env.get("JARVIS_MCP_DISCOVERY_JSON", "").strip()
            if not raw:
                raise MCPDiscoveryError("JARVIS_MCP_DISCOVERY_JSON is not configured.")
            import json
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise MCPDiscoveryError("MCP discovery configuration is not valid JSON.") from exc
            if not isinstance(payload, list) or len(payload) > 32:
                raise MCPDiscoveryError("MCP discovery configuration must be a bounded list.")
            from core.mcp_discovery import MCPDiscoveryServer
            servers = []
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
            self._mcp = MCPDiscoveryRuntime(MCPDiscoveryRegistry(tuple(servers)))
        return self._mcp

    def windows_uia(self) -> WindowsUIAObserver:
        if self._uia is None:
            observer = WindowsUIAObserver(platform_name=self._platform)
            if not observer.available:
                raise RuntimeError("Windows UIA integration is not available on this host.")
            self._uia = observer
        return self._uia

    def browser_semantics(self) -> BrowserSemanticObserver:
        if self._browser is None:
            if not self._module_available("playwright"):
                raise RuntimeError("Playwright browser integration is not installed.")
            self._browser = BrowserSemanticObserver()
        return self._browser

    def temporal_memory(self) -> TemporalMemoryStore:
        if self._memory is None:
            self._memory = TemporalMemoryStore(self.temporal_memory_path)
        return self._memory
