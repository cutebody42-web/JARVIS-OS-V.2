"""Discover and validate the exact network address for JARVIS companion access.

The desktop UI API remains loopback-only. This helper only decides whether the
separate signed NEXUS companion gateway may bind to an explicitly configured
address or to the current Tailscale IPv4 address.

No wildcard/LAN bind is selected automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import os
from pathlib import Path
import shutil
import subprocess


_DEFAULT_PORT = 8765
_TAILSCALE_CGNAT = ipaddress.ip_network("100.64.0.0/10")


@dataclass(frozen=True)
class CompanionGatewayAddress:
    bind_host: str
    port: int
    endpoint: str
    source: str


def _validate_port(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1024 <= value <= 65535:
        raise ValueError("companion gateway port must be in 1024..65535")
    return value


def _literal_bind(value: str) -> str:
    try:
        address = ipaddress.ip_address(value.strip())
    except (AttributeError, ValueError) as exc:
        raise ValueError("companion bind must be a literal IP address") from exc
    if address.is_unspecified or address.is_multicast:
        raise ValueError("companion bind cannot be wildcard or multicast")
    if address.is_loopback:
        raise ValueError("companion gateway requires a non-loopback address")
    return str(address)


def _tailscale_candidates() -> tuple[Path, ...]:
    values: list[Path] = []
    found = shutil.which("tailscale")
    if found:
        values.append(Path(found))
    program_files = os.environ.get("ProgramFiles")
    local_app = os.environ.get("LOCALAPPDATA")
    if program_files:
        values.append(Path(program_files) / "Tailscale" / "tailscale.exe")
    if local_app:
        values.append(Path(local_app) / "Tailscale" / "tailscale.exe")
    # preserve order while deduplicating
    return tuple(dict.fromkeys(values))


def find_tailscale_executable() -> Path | None:
    for candidate in _tailscale_candidates():
        if candidate.is_file():
            return candidate
    return None


def tailscale_ipv4(executable: Path | None = None) -> str | None:
    exe = executable or find_tailscale_executable()
    if exe is None:
        return None
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        completed = subprocess.run(
            [str(exe), "ip", "-4"],
            check=True,
            capture_output=True,
            text=True,
            timeout=4,
            creationflags=creationflags,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    for raw in completed.stdout.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            address = ipaddress.ip_address(raw)
        except ValueError:
            continue
        if isinstance(address, ipaddress.IPv4Address) and address in _TAILSCALE_CGNAT:
            return str(address)
    return None


def resolve_companion_gateway() -> CompanionGatewayAddress | None:
    raw_port = os.environ.get("JARVIS_COMPANION_PORT", str(_DEFAULT_PORT))
    try:
        port = _validate_port(int(raw_port))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid JARVIS_COMPANION_PORT") from exc

    explicit = os.environ.get("JARVIS_COMPANION_BIND")
    if explicit:
        host = _literal_bind(explicit)
        source = "configured"
    else:
        host = tailscale_ipv4()
        if host is None:
            return None
        source = "tailscale"

    return CompanionGatewayAddress(
        bind_host=host,
        port=port,
        endpoint=f"http://{host}:{port}",
        source=source,
    )
