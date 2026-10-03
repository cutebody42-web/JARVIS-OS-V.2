"""Testable hardware snapshots for NEXUS adaptive model routing.

The router must consume HardwareSnapshot data, never read WMI/psutil directly.
HardwareProfiler is the OS-facing probe layer and is deliberately best-effort:
missing GPU telemetry or an unavailable Ollama daemon does not make the whole
snapshot fail.

Integrated/shared-memory GPUs are not reported as fake dedicated VRAM. When a
reliable dedicated-VRAM probe is unavailable, gpu_vram_gb remains None.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import os
from pathlib import Path
import platform
import re
import subprocess
from types import MappingProxyType
from typing import Callable, Mapping
from core.ollama_endpoint import normalize_local_ollama_url


_GIB = 1024 ** 3
_DEVICE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class PowerSource(str, Enum):
    AC = "ac"
    BATTERY = "battery"
    UNKNOWN = "unknown"


class GPUMemoryKind(str, Enum):
    DEDICATED = "dedicated"
    SHARED = "shared"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CgroupMemoryBudget:
    limit_bytes: int | None = None
    available_bytes: int | None = None
    warnings: tuple[str, ...] = ()


def _probe_cgroup_memory(
    *,
    read_text: Callable[[Path], str] | None = None,
    platform_name: str | None = None,
) -> CgroupMemoryBudget:
    """Read Linux cgroup hard limits and remaining allocation headroom.

    Usage includes file cache. We deliberately do not subtract presumed
    reclaimable cache: mapped or pinned model pages may still consume the limit.
    Parent limits also apply, including memory consumed by sibling processes.
    """
    if (platform_name or platform.system()) != "Linux":
        return CgroupMemoryBudget()
    read = read_text or (lambda path: path.read_text(encoding="ascii"))
    roots = {"v2": Path("/sys/fs/cgroup"), "v1": Path("/sys/fs/cgroup/memory")}
    relative = {"v2": "", "v1": ""}
    warnings: list[str] = []
    try:
        for line in read(Path("/proc/self/cgroup")).splitlines():
            parts = line.split(":", 2)
            if len(parts) != 3:
                continue
            mode = "v2" if parts[:2] == ["0", ""] else "v1" if "memory" in parts[1].split(",") else None
            if mode and ".." not in Path(parts[2]).parts:
                relative[mode] = parts[2].lstrip("/")
    except (OSError, UnicodeError):
        # Namespace roots often expose the effective limit directly even when
        # /proc is unavailable, so still inspect the mounted roots.
        pass

    limits: list[int] = []
    headrooms: list[int] = []
    for mode, root in roots.items():
        limit_name, usage_name = (("memory.max", "memory.current") if mode == "v2"
                                  else ("memory.limit_in_bytes", "memory.usage_in_bytes"))
        current = root / relative[mode]
        while True:
            try:
                value = read(current / limit_name).strip()
                limit = None if value == "max" else int(value)
                # v1 represents no limit with a page-aligned LONG_MAX sentinel.
                if mode == "v1" and limit is not None and limit >= 2**60:
                    limit = None
                if limit is not None:
                    if limit <= 0:
                        raise ValueError("invalid hard limit")
                    limits.append(limit)
                    try:
                        usage = int(read(current / usage_name).strip())
                        if usage < 0:
                            raise ValueError("invalid usage")
                        headrooms.append(max(0, limit - usage))
                    except (OSError, ValueError, UnicodeError):
                        headrooms.append(0)
                        warnings.append("cgroup_memory_usage_unavailable")
            except FileNotFoundError:
                pass
            except (OSError, ValueError, UnicodeError):
                # A present but unreadable/invalid constraint cannot be treated
                # as unlimited memory for new model allocations.
                headrooms.append(0)
                warnings.append("cgroup_memory_limit_unavailable")
            if current == root:
                break
            current = current.parent
    return CgroupMemoryBudget(
        min(limits) if limits else None,
        min(headrooms) if headrooms else None,
        tuple(dict.fromkeys(warnings)),
    )


@dataclass(frozen=True)
class HardwareSnapshot:
    """Immutable, serializable-enough input for model routing tests."""

    device_id: str
    total_ram_gb: float
    available_ram_gb: float
    cpu_percent: float
    cpu_count: int
    gpu_vram_gb: float | None
    gpu_memory_kind: GPUMemoryKind
    power_source: PowerSource
    battery_pct: int | None
    system_pressure: float
    loaded_models: Mapping[str, float]
    timestamp: datetime
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _DEVICE_ID_RE.fullmatch(self.device_id):
            raise ValueError("device_id must be a safe 1-64 character identifier")
        for name, value in (
            ("total_ram_gb", self.total_ram_gb),
            ("available_ram_gb", self.available_ram_gb),
            ("cpu_percent", self.cpu_percent),
            ("system_pressure", self.system_pressure),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
        if self.total_ram_gb <= 0:
            raise ValueError("total_ram_gb must be positive")
        if not 0 <= self.available_ram_gb <= self.total_ram_gb + 0.05:
            raise ValueError("available_ram_gb must be between zero and total RAM")
        if not 0 <= self.cpu_percent <= 100:
            raise ValueError("cpu_percent must be between 0 and 100")
        if isinstance(self.cpu_count, bool) or not isinstance(self.cpu_count, int) or self.cpu_count <= 0:
            raise ValueError("cpu_count must be a positive integer")
        if self.gpu_vram_gb is not None and self.gpu_vram_gb < 0:
            raise ValueError("gpu_vram_gb cannot be negative")
        if self.battery_pct is not None and not 0 <= self.battery_pct <= 100:
            raise ValueError("battery_pct must be between 0 and 100")
        if not 0 <= self.system_pressure <= 1:
            raise ValueError("system_pressure must be between 0 and 1")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")

        normalized: dict[str, float] = {}
        for model, memory_gb in self.loaded_models.items():
            if not isinstance(model, str) or not model.strip():
                raise ValueError("loaded model names must be non-empty strings")
            if isinstance(memory_gb, bool) or not isinstance(memory_gb, (int, float)) or memory_gb < 0:
                raise ValueError("loaded model memory must be a non-negative number")
            normalized[model] = float(memory_gb)

        object.__setattr__(self, "loaded_models", MappingProxyType(normalized))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    @property
    def memory_pressure(self) -> float:
        return max(0.0, min(1.0, 1.0 - (self.available_ram_gb / self.total_ram_gb)))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


_normalize_base_url = normalize_local_ollama_url

def _configured_or_anonymous_device_id() -> str:
    configured = os.getenv("JARVIS_DEVICE_ID")
    if configured:
        if not _DEVICE_ID_RE.fullmatch(configured):
            raise ValueError("JARVIS_DEVICE_ID must be a safe 1-64 character identifier")
        return configured

    raw = _stable_machine_identifier()
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"node-{digest}"


def _stable_machine_identifier() -> str:
    """Return a local stable identifier; callers expose only its hash."""
    if platform.system() == "Windows":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Cryptography",
            ) as key:
                value, _ = winreg.QueryValueEx(key, "MachineGuid")
                if value:
                    return f"windows:{value}"
        except (OSError, ImportError):
            pass

    for path in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):
        try:
            value = path.read_text(encoding="utf-8").strip()
            if value:
                return f"machine-id:{value}"
        except OSError:
            pass

    node = platform.node().strip()
    if node:
        return f"hostname:{node}"
    return f"fallback:{platform.system()}:{platform.machine()}"


def _probe_nvidia_vram_gb() -> tuple[float | None, GPUMemoryKind]:
    """Best-effort dedicated VRAM probe.

    Deliberately does not use Win32_VideoController.AdapterRAM for integrated
    graphics because shared-memory adapters can produce misleading values.
    """
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None, GPUMemoryKind.UNKNOWN

    values: list[float] = []
    for line in completed.stdout.splitlines():
        try:
            mib = float(line.strip())
        except ValueError:
            continue
        if mib > 0:
            values.append(mib / 1024.0)
    if not values:
        return None, GPUMemoryKind.UNKNOWN
    return sum(values), GPUMemoryKind.DEDICATED


class HardwareProfiler:
    """Capture one current machine snapshot with injectable probes for tests."""

    def __init__(
        self,
        *,
        ollama_base_url: str | None = None,
        device_id: str | None = None,
        psutil_module=None,
        http_get: Callable | None = None,
        gpu_probe: Callable[[], tuple[float | None, GPUMemoryKind]] | None = None,
        memory_probe: Callable[[], CgroupMemoryBudget] | None = None,
        clock: Callable[[], datetime] | None = None,
        ollama_timeout_seconds: float = 1.5,
    ):
        if device_id is not None and not _DEVICE_ID_RE.fullmatch(device_id):
            raise ValueError("device_id must be a safe 1-64 character identifier")
        if ollama_timeout_seconds <= 0:
            raise ValueError("ollama_timeout_seconds must be positive")

        self._device_id = device_id
        self._psutil = psutil_module
        self._http_get = http_get
        self._gpu_probe = gpu_probe or _probe_nvidia_vram_gb
        # Injected psutil fixtures describe their own machine. Only real OS
        # capture gets ambient cgroup limits unless a test supplies its probe.
        self._memory_probe = memory_probe if memory_probe is not None else (
            _probe_cgroup_memory if psutil_module is None else None
        )
        self._clock = clock or _utc_now
        self._ollama_timeout = ollama_timeout_seconds
        self._ollama_base_url = _normalize_base_url(
            ollama_base_url
            or os.getenv("OLLAMA_HOST")
            or "http://127.0.0.1:11434"
        )

    def _psutil_module(self):
        if self._psutil is None:
            import psutil

            self._psutil = psutil
        return self._psutil

    def _get(self):
        if self._http_get is None:
            import requests

            self._http_get = requests.get
        return self._http_get

    def _probe_loaded_models(self) -> tuple[dict[str, float], str | None]:
        """Return model -> reported resident size in GiB from Ollama /api/ps."""
        try:
            response = self._get()(
                f"{self._ollama_base_url}/api/ps",
                timeout=self._ollama_timeout,
            )
            response.raise_for_status()
            body = response.json()
        except Exception:
            return {}, "ollama_ps_unavailable"

        models = body.get("models") if isinstance(body, dict) else None
        if not isinstance(models, list):
            return {}, "ollama_ps_invalid"

        loaded: dict[str, float] = {}
        for item in models:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("model")
            size = item.get("size")
            if not isinstance(name, str) or not name.strip():
                continue
            if isinstance(size, bool) or not isinstance(size, (int, float)) or size < 0:
                continue
            loaded[name] = float(size) / _GIB
        return loaded, None

    def capture(self) -> HardwareSnapshot:
        psutil = self._psutil_module()
        vm = psutil.virtual_memory()
        total_ram_gb = float(vm.total) / _GIB
        available_ram_gb = float(vm.available) / _GIB

        warnings: list[str] = []
        if self._memory_probe is not None:
            try:
                budget = self._memory_probe()
                warnings.extend(budget.warnings)
                host_total, host_available = total_ram_gb, available_ram_gb
                if budget.limit_bytes is not None:
                    total_ram_gb = min(total_ram_gb, budget.limit_bytes / _GIB)
                if budget.available_bytes is not None:
                    available_ram_gb = min(available_ram_gb, budget.available_bytes / _GIB)
                available_ram_gb = max(0.0, min(available_ram_gb, total_ram_gb))
                if total_ram_gb < host_total or available_ram_gb < host_available:
                    warnings.append("cgroup_memory_limit_applied")
            except Exception:
                available_ram_gb = 0.0
                warnings.append("cgroup_memory_probe_failed")

        cpu_percent = float(psutil.cpu_percent(interval=None))
        cpu_count = int(psutil.cpu_count(logical=True) or 1)

        try:
            battery = psutil.sensors_battery()
        except Exception:
            # Battery telemetry may be missing on desktops, containers, or an
            # OS without a supported power-supply interface. RAM/CPU routing
            # must still work; never invent a battery or an AC power reading.
            battery = None
            warnings.append("battery_probe_failed")
        if battery is None:
            power_source = PowerSource.UNKNOWN
            battery_pct = None
        else:
            plugged = battery.power_plugged
            if plugged is True:
                power_source = PowerSource.AC
            elif plugged is False:
                power_source = PowerSource.BATTERY
            else:
                power_source = PowerSource.UNKNOWN
            try:
                battery_pct = int(round(float(battery.percent)))
            except (TypeError, ValueError):
                battery_pct = None
            if battery_pct is not None:
                battery_pct = min(100, max(0, battery_pct))

        try:
            gpu_vram_gb, gpu_memory_kind = self._gpu_probe()
        except Exception:
            gpu_vram_gb, gpu_memory_kind = None, GPUMemoryKind.UNKNOWN
            warnings.append("gpu_probe_failed")

        loaded_models, ollama_warning = self._probe_loaded_models()
        if ollama_warning:
            warnings.append(ollama_warning)

        memory_pressure = 1.0 - (available_ram_gb / total_ram_gb)
        system_pressure = max(
            0.0,
            min(1.0, max(memory_pressure, cpu_percent / 100.0)),
        )

        timestamp = self._clock()
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("HardwareProfiler clock must return timezone-aware datetime")

        return HardwareSnapshot(
            device_id=self._device_id or _configured_or_anonymous_device_id(),
            total_ram_gb=total_ram_gb,
            available_ram_gb=available_ram_gb,
            cpu_percent=cpu_percent,
            cpu_count=cpu_count,
            gpu_vram_gb=gpu_vram_gb,
            gpu_memory_kind=gpu_memory_kind,
            power_source=power_source,
            battery_pct=battery_pct,
            system_pressure=system_pressure,
            loaded_models=loaded_models,
            timestamp=timestamp,
            warnings=tuple(warnings),
        )
