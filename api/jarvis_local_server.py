"""Loopback-only control API for the installed JARVIS desktop shell.

The Tauri desktop process starts this module as an embedded sidecar and passes
a fresh high-entropy UI token. The webview never launches Python directly and
the owner never installs Python/pip.

This API is intentionally separate from the signed remote companion protocol:
- desktop UI -> loopback + bearer token;
- paired phone -> Ed25519 signed NEXUS Brain RPC.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hmac
import threading
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.action_gateway import create_runtime
from core.app_paths import user_data_dir
from core.hardware_profile import HardwareProfiler, HardwareSnapshot
from core.jarvis_brain import BrainIntent, BrainPolicy, JarvisBrain, classify_intent
from core.jarvis_memory import JarvisMemory
from core.model_router import TaskKind
from core.model_runtime import ModelRuntime, ModelRuntimeError
from core.nexus.sync_node import NexusSyncNode
from core.ollama_bootstrap import (
    OllamaBootstrapError,
    bootstrap_local_brain,
    ensure_ollama_service,
    find_ollama_executable,
    installed_model_names,
)
from core.self_heal import (
    Incident,
    IncidentKind,
    RepairJournal,
    RepairState,
    SelfHealController,
)


_REQUIRED_LOCAL_MODELS = frozenset({"jarvis-brain-fast", "jarvis-brain-lite"})


class MessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=16000)
    task: str | None = None


class SetupRequest(BaseModel):
    approved: bool


class PairOfferRequest(BaseModel):
    endpoint: str | None = Field(default=None, min_length=1, max_length=512)
    ttl_seconds: int = Field(default=300, ge=30, le=600)


class PairDecisionRequest(BaseModel):
    pairing_id: str = Field(min_length=1, max_length=128)


@dataclass
class SetupState:
    phase: str = "boot"
    percent: float = 0.0
    message: str = "Starting JARVIS Brain"
    error: str | None = None


class LocalBrainHost:
    def __init__(
        self,
        *,
        ui_token: str,
        state_dir: str | Path | None = None,
        allow_cloud: bool = False,
        companion_endpoint: str | None = None,
    ):
        if not isinstance(ui_token, str) or len(ui_token) < 32:
            raise ValueError("ui_token must contain at least 32 characters")
        self.ui_token = ui_token
        self.companion_endpoint = companion_endpoint
        self.state_dir = Path(state_dir) if state_dir else user_data_dir() / "brain"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._guard = threading.RLock()
        self._setup_guard = threading.Lock()
        self.setup = SetupState()

        self.profiler = HardwareProfiler()
        self.snapshot = self._capture_snapshot()
        device_id = self.snapshot.device_id if self.snapshot is not None else "jarvis-desktop"

        self.node = NexusSyncNode(self.state_dir / "nexus", device_id)
        self.memory = JarvisMemory(self.node.store, self.node.applier)
        self.memory.recover()

        self.owner_runtime = create_runtime(
            owner_id="local-owner",
            workspace_root=self.state_dir / "workspace",
            environment="desktop",
        )
        self.model_runtime = ModelRuntime()
        self.repair_journal = RepairJournal(self.state_dir / "repair.db")
        self.self_heal = SelfHealController(journal=self.repair_journal)
        self.self_heal.register_runtime_healer(
            IncidentKind.MODEL_FAILURE,
            self._heal_model_runtime,
        )
        self.self_heal.register_runtime_healer(
            IncidentKind.SYNC_FAILURE,
            self._heal_sync_runtime,
        )
        self.brain = JarvisBrain(
            policy=BrainPolicy(allow_cloud=bool(allow_cloud)),
            owner_runtime=self.owner_runtime,
            profiler=self.profiler,
            model_runtime=self.model_runtime,
            memory=self.memory,
        )
        self.setup.phase = "ready" if self.local_brain_ready() else "setup_required"
        self.setup.percent = 100.0 if self.setup.phase == "ready" else 0.0
        self.setup.message = (
            "JARVIS Brain ready"
            if self.setup.phase == "ready"
            else "Local JARVIS Brain setup required"
        )

    def _capture_snapshot(self) -> HardwareSnapshot | None:
        try:
            return self.profiler.capture()
        except Exception:
            return None

    def refresh_hardware(self) -> HardwareSnapshot | None:
        snapshot = self._capture_snapshot()
        with self._guard:
            self.snapshot = snapshot
        return snapshot

    def local_brain_ready(self) -> bool:
        executable = find_ollama_executable()
        if executable is None:
            return False
        names = installed_model_names(executable)
        return bool(names & _REQUIRED_LOCAL_MODELS)

    def status(self) -> dict[str, Any]:
        with self._guard:
            snapshot = self.snapshot
            setup = SetupState(
                self.setup.phase,
                self.setup.percent,
                self.setup.message,
                self.setup.error,
            )
        return {
            "identity": "JARVIS",
            "mode": "desktop_local",
            "brain_ready": self.local_brain_ready(),
            "setup": {
                "phase": setup.phase,
                "percent": setup.percent,
                "message": setup.message,
                "error": setup.error,
            },
            "device": {
                "device_id": self.node.identity.device_id,
                "ram_total_gb": round(snapshot.total_ram_gb, 2) if snapshot else None,
                "ram_available_gb": round(snapshot.available_ram_gb, 2) if snapshot else None,
                "system_pressure": round(snapshot.system_pressure, 3) if snapshot else None,
                "power_source": snapshot.power_source.value if snapshot else "unknown",
                "battery_pct": snapshot.battery_pct if snapshot else None,
            },
            "companion": {
                "available": self.companion_endpoint is not None,
                "endpoint": self.companion_endpoint,
            },
            "paired_devices": [
                {
                    "peer_id": peer.peer_id,
                    "endpoint": peer.endpoint,
                    "role": peer.role.value,
                }
                for peer in self.node.registry.active_peers()
            ],
            "self_heal": {
                "recent": list(self.repair_journal.recent(limit=8)),
            },
        }

    def _heal_model_runtime(self, incident: Incident) -> bool:
        snapshot = self.refresh_hardware()
        if snapshot is not None:
            try:
                self.model_runtime.relieve_pressure(snapshot)
            except Exception:
                pass

        executable = find_ollama_executable()
        if executable is None:
            return False
        try:
            ensure_ollama_service(executable)
            self.model_runtime.get_status()
        except Exception:
            return False
        return True

    def _heal_sync_runtime(self, incident: Incident) -> bool:
        try:
            self.node.recover()
            reports = self.node.scheduler.tick()
        except Exception:
            return False
        return all(report.error is None for report in reports)

    def _record_and_heal(self, kind: IncidentKind, summary: str, component: str):
        incident = Incident.create(kind, summary, component)
        self.repair_journal.record(
            incident.id,
            RepairState.DETECTED,
            incident.summary,
        )
        return self.self_heal.heal_runtime(incident)

    def handle_message(self, message: str, *, task: TaskKind | None = None) -> str:
        intent = classify_intent(message)
        try:
            return self.brain.handle(message, task=task)
        except Exception as exc:
            outcome = self._record_and_heal(
                IncidentKind.MODEL_FAILURE,
                f"JARVIS Brain request failed ({type(exc).__name__}).",
                "model_runtime",
            )
            # Never blindly replay operational actions: an external side effect
            # may have happened before an exception was observed.
            if intent is BrainIntent.CHAT and outcome.state is RepairState.HEALED:
                return self.brain.handle(message, task=task)
            raise

    def setup_local_brain(self, *, approved: bool) -> dict[str, Any]:
        if not approved:
            raise PermissionError("Local model installation requires explicit owner approval.")
        if not self._setup_guard.acquire(blocking=False):
            return self.status()
        try:
            with self._guard:
                self.setup = SetupState("starting", 1.0, "Preparing local JARVIS Brain", None)

            snapshot = self.refresh_hardware()

            def progress(phase: str, percent: float, message: str) -> None:
                with self._guard:
                    self.setup = SetupState(phase, float(percent), str(message)[:500], None)

            try:
                bootstrap_local_brain(
                    snapshot=snapshot,
                    allow_install=True,
                    progress=progress,
                )
            except OllamaBootstrapError as exc:
                with self._guard:
                    self.setup = SetupState("failed", 0.0, "Local Brain setup failed", str(exc))
                raise

            with self._guard:
                self.setup = SetupState("ready", 100.0, "JARVIS Brain ready", None)
            return self.status()
        finally:
            self._setup_guard.release()


def create_local_brain_app(host: LocalBrainHost) -> FastAPI:
    if not isinstance(host, LocalBrainHost):
        raise TypeError("host must be LocalBrainHost")

    app = FastAPI(
        title="JARVIS Local Brain",
        version="2.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://tauri.localhost",
            "https://tauri.localhost",
            "tauri://localhost",
            "http://localhost:3000",
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["authorization", "content-type"],
    )

    def require_ui(authorization: str | None = Header(default=None)) -> None:
        expected = "Bearer " + host.ui_token
        if not isinstance(authorization, str) or not hmac.compare_digest(
            authorization, expected
        ):
            raise HTTPException(status_code=401, detail="JARVIS shell authentication failed")

    @app.get("/v1/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "identity": "JARVIS",
            "device_id": host.node.identity.device_id,
        }

    @app.get("/v1/status", dependencies=[Depends(require_ui)])
    def status() -> dict[str, Any]:
        return host.status()

    @app.post("/v1/setup/local-brain", dependencies=[Depends(require_ui)])
    def setup_local_brain(request: SetupRequest) -> dict[str, Any]:
        try:
            return host.setup_local_brain(approved=request.approved)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from None
        except OllamaBootstrapError:
            raise HTTPException(status_code=503, detail="Local Brain setup failed") from None

    @app.post("/v1/message", dependencies=[Depends(require_ui)])
    def message(request: MessageRequest) -> dict[str, Any]:
        if not host.local_brain_ready():
            raise HTTPException(status_code=409, detail="Local JARVIS Brain is not ready")
        task = None
        if request.task is not None:
            try:
                task = TaskKind(request.task)
            except ValueError:
                raise HTTPException(status_code=400, detail="Unsupported task kind") from None
        try:
            text = host.handle_message(request.message, task=task)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"JARVIS Brain request failed ({type(exc).__name__})",
            ) from None
        return {
            "identity": "JARVIS",
            "text": text,
            "lane": host.brain.lane.value,
        }

    @app.post("/v1/pair/offer", dependencies=[Depends(require_ui)])
    def pair_offer(request: PairOfferRequest) -> dict[str, Any]:
        endpoint = request.endpoint or host.companion_endpoint
        if endpoint is None:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Remote companion gateway is unavailable. "
                    "Connect Tailscale or configure JARVIS_COMPANION_BIND."
                ),
            )
        offer = host.node.pairing.create_offer(
            endpoint,
            ttl_seconds=request.ttl_seconds,
        )
        return offer.public_payload()

    @app.get("/v1/pair/pending", dependencies=[Depends(require_ui)])
    def pair_pending() -> dict[str, Any]:
        return {
            "pending": [
                {
                    "pairing_id": item.pairing_id,
                    "candidate_device": item.candidate_device,
                    "candidate_public_key": item.candidate_public_key,
                    "candidate_role": item.candidate_role.value,
                    "candidate_endpoint": item.candidate_endpoint,
                    "created_at": item.created_at,
                }
                for item in host.node.pairing.pending()
            ]
        }

    @app.post("/v1/pair/approve", dependencies=[Depends(require_ui)])
    def pair_approve(request: PairDecisionRequest) -> dict[str, Any]:
        peer = host.node.pairing.approve(request.pairing_id)
        return {
            "approved": True,
            "peer_id": peer.peer_id,
            "endpoint": peer.endpoint,
            "role": peer.role.value,
        }

    @app.post("/v1/pair/cancel", dependencies=[Depends(require_ui)])
    def pair_cancel(request: PairDecisionRequest) -> dict[str, Any]:
        return {
            "cancelled": host.node.pairing.cancel(request.pairing_id),
            "pairing_id": request.pairing_id,
        }

    return app
