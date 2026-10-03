"""Nonblocking desktop controls for verified, fingerprint-approved releases."""
from __future__ import annotations

from pathlib import Path
import sys
import threading
from typing import Any

from core.build_info import read_build_info
from core.update_manager import GitHubReleaseSource, WindowsReleaseUpdateCoordinator


class LocalUpdateService:
    def __init__(self, *, state_dir: Path, approvals: Any, parent_pid: int | None,
                 coordinator: Any = None):
        self._guard = threading.RLock()
        self._busy = False
        self._plan = None
        self.approvals = approvals
        self.parent_pid = parent_pid
        self.coordinator = coordinator
        self._status: dict[str, Any] = {
            "supported": False, "phase": "unavailable",
            "message": "Verified installation updates require the packaged Windows desktop app.",
            "plan": None, "checkpoint_id": None, "approval_id": None,
        }
        if coordinator is None and sys.platform == "win32" and getattr(sys, "frozen", False) and parent_pid:
            try:
                info = read_build_info()
                self.coordinator = WindowsReleaseUpdateCoordinator(
                    GitHubReleaseSource(), state_dir=state_dir,
                    installed_version=info["version"], installed_commit_sha=info["commit_sha"],
                    installed_sidecar=Path(sys.executable), owner_approvals=approvals,
                )
            except (OSError, ValueError) as exc:
                self._status["message"] = f"Installed build provenance is unavailable ({type(exc).__name__})."
        if self.coordinator is not None:
            self._status.update(supported=True, phase="idle", message="Check GitHub for a verified JARVIS release.")

    def status(self) -> dict[str, Any]:
        with self._guard:
            value = dict(self._status)
            value["busy"] = self._busy
            value["plan"] = dict(value["plan"]) if value["plan"] else None
            value["history"] = list(self.coordinator.history()) if self.coordinator else []
        value["approval_state"] = None
        if value["approval_id"]:
            try:
                approval = self.approvals.get(value["approval_id"])
                value["approval_state"] = "consumed" if approval.consumed else approval.state
            except KeyError:
                value["approval_state"] = "expired"
        return value

    def _start(self, phase: str, operation: Any) -> dict[str, Any]:
        with self._guard:
            if self.coordinator is None:
                raise ValueError(self._status["message"])
            if self._busy or self._status["phase"] == "applying":
                raise RuntimeError("An update operation is already running.")
            if self._status["phase"] == "awaiting_approval":
                try:
                    current = self.approvals.get(self._status["approval_id"])
                    if not current.consumed and current.state in {"pending", "approved"}:
                        raise RuntimeError("Complete or reject the staged phone approval before checking another release.")
                except KeyError:
                    pass
            self._busy = True
            self._status.update(phase=phase, message="Checking release metadata." if phase == "checking" else "Downloading and verifying installers.")

        def run() -> None:
            try:
                operation()
            except Exception as exc:
                with self._guard:
                    self._status.update(phase="failed", message=f"Update stopped ({type(exc).__name__}): {str(exc)[:300]}")
            finally:
                with self._guard:
                    self._busy = False
        threading.Thread(target=run, name="jarvis-release-update", daemon=True).start()
        return self.status()

    def check(self) -> dict[str, Any]:
        def discover() -> None:
            plan = self.coordinator.source.discover()
            with self._guard:
                self._plan = plan
                if plan is None:
                    self._status.update(phase="idle", plan=None, message="No verified release has been published yet.")
                elif plan.commit_sha == self.coordinator.installed_commit_sha:
                    self._status.update(phase="current", plan=plan.to_dict(), message="This source build is already current.")
                else:
                    self._status.update(phase="available", plan=plan.to_dict(), message="A verified release is available. Stage its installer and recovery copy.")
                self._status.update(checkpoint_id=None, approval_id=None)
        return self._start("checking", discover)

    def prepare(self) -> dict[str, Any]:
        with self._guard:
            if self._status["phase"] == "awaiting_approval" and self._plan is not None:
                try:
                    previous = self.approvals.get(self._status["approval_id"])
                    if previous.state in {"expired", "rejected"}:
                        self._status["phase"] = "available"
                except KeyError:
                    self._status["phase"] = "available"
            if self._status["phase"] != "available" or self._plan is None:
                raise ValueError("Check for an available release before staging.")
            plan = self._plan
        def stage() -> None:
            outcome = self.coordinator.prepare(plan)
            with self._guard:
                self._status.update(phase=outcome.state.value, message=outcome.message,
                                    checkpoint_id=outcome.checkpoint_id, approval_id=outcome.approval_id)
        return self._start("staging", stage)

    def apply(self, *, checkpoint_id: str, approval_id: str) -> dict[str, Any]:
        with self._guard:
            if self._busy or self._status["phase"] != "awaiting_approval":
                raise ValueError("A verified staged update is required.")
            if checkpoint_id != self._status["checkpoint_id"] or approval_id != self._status["approval_id"]:
                raise PermissionError("Approval does not identify the current staged update.")
            outcome = self.coordinator.authorize_handoff(
                checkpoint_id, approval_id=approval_id, parent_pid=self.parent_pid,
            )
            self._status.update(phase=outcome.state.value, message=outcome.message)
        return self.status()
