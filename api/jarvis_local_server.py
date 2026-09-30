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
import base64
import hmac
import json
import threading
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.action_gateway import create_runtime
from core.app_paths import user_data_dir
from core.hardware_profile import HardwareProfiler, HardwareSnapshot
from core.jarvis_brain import BrainIntent, BrainPolicy, JarvisBrain, classify_intent
from core.jarvis_council import JarvisCouncil
from core.jarvis_memory import JarvisMemory
from core.model_router import TaskKind
from core.model_runtime import ModelRuntime, ModelRuntimeError
from core.mobile_approval_bridge import MobileApprovalBridge
from core.owner_face import FaceIdentityError, OwnerFaceRecognizer
from core.nexus.owner_approval import OwnerApprovalManager
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
from core.voice_runtime import VoiceRuntimeError, WindowsVoiceRuntime


_REQUIRED_LOCAL_MODELS = frozenset({"jarvis-core-1b"})
_JARVIS_OLLAMA_URL = "http://127.0.0.1:11435"


class MessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=16000)
    task: str | None = None


class SetupRequest(BaseModel):
    approved: bool
    model_store: str | None = Field(default=None, max_length=4096)


class ModelSelectionRequest(BaseModel):
    model: str | None = Field(default=None, max_length=128)


class FaceCameraRequest(BaseModel):
    camera_index: int = Field(default=0, ge=0, le=8)


class VoiceListenRequest(BaseModel):
    language: str = Field(default="en-US", min_length=2, max_length=24)
    timeout_seconds: float = Field(default=8.0, ge=1.0, le=30.0)


class VoiceSpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=12000)


class ApprovalCreateRequest(BaseModel):
    summary: str = Field(min_length=1, max_length=500)
    action_digest: str = Field(min_length=64, max_length=64)
    ttl_seconds: int = Field(default=180, ge=15, le=600)


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
        self.settings_path = self.state_dir / "settings.json"
        self.ollama_base_url = _JARVIS_OLLAMA_URL
        self.model_store = self._load_model_store()
        self.manual_model = self._load_manual_model()
        self._guard = threading.RLock()
        self._setup_guard = threading.Lock()
        self.setup = SetupState()

        self.profiler = HardwareProfiler(ollama_base_url=self.ollama_base_url)
        self.snapshot = self._capture_snapshot()
        device_id = self.snapshot.device_id if self.snapshot is not None else "jarvis-desktop"

        self.node = NexusSyncNode(self.state_dir / "nexus", device_id)
        self.approvals = OwnerApprovalManager(self.node.store)
        self.memory = JarvisMemory(self.node.store, self.node.applier)
        self.memory.recover()

        self.owner_face = OwnerFaceRecognizer()
        self.voice = WindowsVoiceRuntime()

        self.owner_runtime = create_runtime(
            owner_id="local-owner",
            workspace_root=self.state_dir / "workspace",
            environment="desktop",
        )
        self.approval_bridge = MobileApprovalBridge(
            self.approvals,
            self.owner_runtime,
        )
        self._approval_stop = threading.Event()
        self._approval_thread = threading.Thread(
            target=self._approval_loop,
            name="jarvis-biometric-approval-bridge",
            daemon=True,
        )
        self._approval_thread.start()
        self.model_runtime = ModelRuntime(ollama_base_url=self.ollama_base_url)
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
        self.council = JarvisCouncil(
            ollama_base_url=self.ollama_base_url,
            profiler=self.profiler,
            runtime=self.model_runtime,
            manual_model=self.manual_model,
        )
        self.brain = JarvisBrain(
            policy=BrainPolicy(allow_cloud=bool(allow_cloud)),
            owner_runtime=self.owner_runtime,
            profiler=self.profiler,
            model_runtime=self.model_runtime,
            memory=self.memory,
            ollama_base_url=self.ollama_base_url,
            council=self.council,
        )
        self.setup.phase = "ready" if self.local_brain_ready() else "setup_required"
        self.setup.percent = 100.0 if self.setup.phase == "ready" else 0.0
        self.setup.message = (
            "JARVIS Brain ready"
            if self.setup.phase == "ready"
            else "Local JARVIS Brain setup required"
        )

    def _load_settings(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.settings_path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _load_model_store(self) -> Path | None:
        payload = self._load_settings()
        raw = payload.get("model_store")
        if not isinstance(raw, str) or not raw.strip():
            return None
        path = Path(raw).expanduser()
        if not path.is_absolute():
            return None
        try:
            resolved = path.resolve()
        except OSError:
            return None
        return resolved if resolved.is_dir() else None

    def _load_manual_model(self) -> str | None:
        raw = self._load_settings().get("manual_model")
        if not isinstance(raw, str) or not raw.strip():
            return None
        return raw.strip()[:128]

    def _write_settings(self) -> None:
        payload = {
            "model_store": str(self.model_store) if self.model_store is not None else None,
            "manual_model": self.manual_model,
        }
        temp = self.settings_path.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
        temp.replace(self.settings_path)

    def _set_model_store(self, raw: str | None) -> Path | None:
        if raw is None or not raw.strip():
            return self.model_store
        path = Path(raw.strip()).expanduser()
        if not path.is_absolute():
            raise ValueError("Local model folder must be an absolute path.")
        path.mkdir(parents=True, exist_ok=True)
        resolved = path.resolve()
        self.model_store = resolved
        self._write_settings()
        return resolved

    def available_models(self) -> tuple[str, ...]:
        executable = find_ollama_executable()
        if executable is None:
            return ()
        return tuple(sorted(installed_model_names(
            executable,
            base_url=self.ollama_base_url,
            model_store=self.model_store,
        )))

    def set_manual_model(self, raw: str | None) -> None:
        model = raw.strip() if isinstance(raw, str) else ""
        if model and model not in self.available_models():
            raise ValueError("Selected local model is not available in the current JARVIS model store.")
        self.manual_model = model or None
        self.council.set_manual_model(self.manual_model)
        self._write_settings()

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
        try:
            ensure_ollama_service(
                executable,
                base_url=self.ollama_base_url,
                model_store=self.model_store,
            )
        except Exception:
            return False
        names = installed_model_names(
            executable,
            base_url=self.ollama_base_url,
            model_store=self.model_store,
        )
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
            "model_store": str(self.model_store) if self.model_store is not None else None,
            "manual_model": self.manual_model,
            "available_models": list(self.available_models()),
            "council": {
                "enabled": True,
                "core_model": "jarvis-core-1b",
                "parallel_experts": self.council.max_parallel_experts,
            },
            "owner_identity": {
                "face_enrolled": self.owner_face.enrolled,
                "face_recognized": self.owner_face.recognized,
                "face_score": round(self.owner_face.last_score, 4),
                "face_engine": self.owner_face.engine,
            },
            "voice": self.voice.status(),
            "approvals": {
                "pending": len(self.approvals.pending()),
                **self.approval_bridge.status(),
            },
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
            ensure_ollama_service(
                executable,
                base_url=self.ollama_base_url,
                model_store=self.model_store,
            )
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

    def _approval_loop(self) -> None:
        while not self._approval_stop.wait(0.5):
            try:
                self.approval_bridge.reconcile()
            except Exception:
                # Approval processing is fail-closed; a loop error must never
                # grant authority or crash the local Brain.
                continue

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
        before = {receipt.action_id for receipt in self.owner_runtime.gateway.receipts}
        try:
            text = self.brain.handle(message, task=task)
            queued = []
            for receipt in self.owner_runtime.gateway.receipts:
                if receipt.action_id in before:
                    continue
                if receipt.result.status.value != "require_confirmation":
                    continue
                queued.append(self.approval_bridge.queue_receipt(receipt))
            if queued:
                if self.node.registry.active_peers():
                    return (
                        text
                        + "\nFingerprint approval sent to your paired phone. "
                        "Only the exact pending action will run after approval."
                    )
                return (
                    text
                    + "\nThis exact action requires owner approval. Pair the JARVIS phone "
                    "companion to approve it with your fingerprint."
                )
            return text
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

    def setup_local_brain(
        self,
        *,
        approved: bool,
        model_store: str | None = None,
    ) -> dict[str, Any]:
        if not approved:
            raise PermissionError("Local model installation requires explicit owner approval.")
        if not self._setup_guard.acquire(blocking=False):
            return self.status()
        try:
            with self._guard:
                self.setup = SetupState("starting", 1.0, "Preparing local JARVIS Brain", None)

            selected_store = self._set_model_store(model_store)
            snapshot = self.refresh_hardware()

            def progress(phase: str, percent: float, message: str) -> None:
                with self._guard:
                    self.setup = SetupState(phase, float(percent), str(message)[:500], None)

            try:
                bootstrap_local_brain(
                    snapshot=snapshot,
                    allow_install=True,
                    progress=progress,
                    base_url=self.ollama_base_url,
                    model_store=selected_store,
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
            return host.setup_local_brain(
                approved=request.approved,
                model_store=request.model_store,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        except OllamaBootstrapError:
            raise HTTPException(status_code=503, detail="Local Brain setup failed") from None

    @app.post("/v1/approval/request", dependencies=[Depends(require_ui)])
    def create_owner_approval(request: ApprovalCreateRequest) -> dict[str, Any]:
        try:
            approval = host.approvals.create(
                request.summary,
                request.action_digest,
                ttl_seconds=request.ttl_seconds,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        return {
            "approval_id": approval.approval_id,
            "summary": approval.summary,
            "action_digest": approval.action_digest,
            "expires_at": approval.expires_at,
            "state": approval.state,
        }

    @app.get("/v1/approval/pending", dependencies=[Depends(require_ui)])
    def pending_owner_approvals() -> dict[str, Any]:
        return {
            "pending": [
                {
                    "approval_id": item.approval_id,
                    "summary": item.summary,
                    "action_digest": item.action_digest,
                    "expires_at": item.expires_at,
                    "state": item.state,
                }
                for item in host.approvals.pending()
            ]
        }

    @app.post("/v1/identity/face/enroll", dependencies=[Depends(require_ui)])
    def enroll_owner_face(request: FaceCameraRequest) -> dict[str, Any]:
        try:
            samples = host.owner_face.enroll_from_camera(camera_index=request.camera_index)
            verified = host.owner_face.verify_from_camera(camera_index=request.camera_index)
        except FaceIdentityError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from None
        return {
            "enrolled": True,
            "samples": samples,
            "recognized": verified.recognized,
            "score": round(verified.score, 4),
        }

    @app.post("/v1/identity/face/verify", dependencies=[Depends(require_ui)])
    def verify_owner_face(request: FaceCameraRequest) -> dict[str, Any]:
        try:
            result = host.owner_face.verify_from_camera(camera_index=request.camera_index)
        except FaceIdentityError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from None
        return {
            "enrolled": result.enrolled,
            "recognized": result.recognized,
            "score": round(result.score, 4),
            "matched_frames": result.matched_frames,
            "total_frames": result.total_frames,
        }

    @app.post("/v1/identity/face/forget", dependencies=[Depends(require_ui)])
    def forget_owner_face() -> dict[str, Any]:
        host.owner_face.forget()
        return {"enrolled": False, "recognized": False}

    @app.get("/v1/voice/status", dependencies=[Depends(require_ui)])
    def voice_status() -> dict[str, Any]:
        return host.voice.status()

    @app.post("/v1/voice/listen", dependencies=[Depends(require_ui)])
    def voice_listen(request: VoiceListenRequest) -> dict[str, Any]:
        try:
            result = host.voice.listen_once(
                language=request.language,
                timeout_seconds=request.timeout_seconds,
            )
        except VoiceRuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        return {
            "text": result.text,
            "confidence": result.confidence,
            "engine": result.engine,
            "state": host.voice.state.value,
        }

    @app.post("/v1/voice/speak", dependencies=[Depends(require_ui)])
    def voice_speak(request: VoiceSpeakRequest) -> dict[str, Any]:
        try:
            host.voice.speak(request.text)
        except VoiceRuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        return {"spoken": True, "state": host.voice.state.value}

    @app.post("/v1/voice/stop", dependencies=[Depends(require_ui)])
    def voice_stop() -> dict[str, Any]:
        return {"stopped": host.voice.stop(), "state": host.voice.state.value}

    @app.post("/v1/models/manual", dependencies=[Depends(require_ui)])
    def select_manual_model(request: ModelSelectionRequest) -> dict[str, Any]:
        try:
            host.set_manual_model(request.model)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        return host.status()

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
        payload = offer.public_payload()
        encoded = base64.urlsafe_b64encode(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).rstrip(b"=").decode("ascii")
        deep_link = f"jarvis://pair?offer={encoded}"
        try:
            import segno
            qr = segno.make(deep_link, error="m", micro=False)
            qr_svg_data_url = qr.svg_data_uri(
                scale=5,
                border=4,
                dark="#0bdcff",
                light="#08131a",
            )
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Pairing QR generation failed ({type(exc).__name__})",
            ) from None
        return {
            **payload,
            "deep_link": deep_link,
            "qr_svg_data_url": qr_svg_data_url,
        }

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
