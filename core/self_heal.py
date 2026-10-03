"""Bounded autonomous self-healing and self-repair contracts for JARVIS.

Runtime recovery is automatic. Code repair may also be automatic only when the
change stays inside explicitly low-risk paths and passes the full configured
verification gate. Security/authority/update paths can never self-approve.

This module does not grant new capabilities and does not implement replication,
privilege escalation or self-preservation behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import PurePosixPath
import sqlite3
from pathlib import Path
from typing import Callable, Mapping, Protocol
from uuid import uuid4

from core.nexus.owner_approval import OwnerApprovalManager


def repository_path(value: str) -> str:
    """Require one portable, canonical path, including on Windows installs."""
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("path must be a canonical repository-relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or str(path) != value:
        raise ValueError("path must be a canonical repository-relative path")
    reserved = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
    for part in path.parts:
        if part in {".", ".."} or part.rstrip(" .") != part or any(ord(c) < 32 for c in part):
            raise ValueError("path must stay inside the repository")
        if part.split(".", 1)[0].casefold() in reserved:
            raise ValueError("path is reserved on Windows")
    return value


class IncidentKind(str, Enum):
    RUNTIME_CRASH = "runtime_crash"
    MODEL_FAILURE = "model_failure"
    SYNC_FAILURE = "sync_failure"
    CONFIG_FAILURE = "config_failure"
    TEST_FAILURE = "test_failure"
    CODE_FAILURE = "code_failure"


class RepairRisk(str, Enum):
    LOW = "low"
    MAJOR = "major"
    FORBIDDEN = "forbidden"


class RepairDisposition(str, Enum):
    AUTO_APPLY = "auto_apply"
    REQUIRE_OWNER = "require_owner"
    REJECT = "reject"


class RepairState(str, Enum):
    DETECTED = "detected"
    HEALED = "healed"
    PATCH_PROPOSED = "patch_proposed"
    VERIFYING = "verifying"
    AWAITING_OWNER = "awaiting_owner"
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass(frozen=True)
class Incident:
    id: str
    kind: IncidentKind
    summary: str
    component: str
    created_at: str
    evidence: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        kind: IncidentKind,
        summary: str,
        component: str,
        evidence: Mapping[str, object] | None = None,
    ) -> "Incident":
        if not isinstance(kind, IncidentKind):
            raise TypeError("kind must be IncidentKind")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("incident summary must be non-empty")
        if not isinstance(component, str) or not component.strip():
            raise ValueError("incident component must be non-empty")
        clean = json.loads(json.dumps(dict(evidence or {}), sort_keys=True, allow_nan=False))
        return cls(
            uuid4().hex,
            kind,
            summary.strip()[:1000],
            component.strip()[:160],
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            clean,
        )


@dataclass(frozen=True)
class FileEdit:
    path: str
    expected_sha256: str
    replacement: str

    def __post_init__(self) -> None:
        repository_path(self.path)
        if not isinstance(self.expected_sha256, str) or len(self.expected_sha256) != 64:
            raise ValueError("expected_sha256 must be a hex SHA-256 digest")
        int(self.expected_sha256, 16)
        if not isinstance(self.replacement, str):
            raise TypeError("replacement must be text")


@dataclass(frozen=True)
class RepairPatch:
    incident_id: str
    edits: tuple[FileEdit, ...]
    rationale: str
    requested_tests: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "edits", tuple(self.edits))
        object.__setattr__(self, "requested_tests", tuple(self.requested_tests))
        if not self.edits:
            raise ValueError("repair patch must contain at least one edit")
        if any(not isinstance(edit, FileEdit) for edit in self.edits):
            raise TypeError("repair edits must be FileEdit values")
        if len({edit.path.casefold() for edit in self.edits}) != len(self.edits):
            raise ValueError("repair patch contains duplicate or case-colliding paths")
        if not self.rationale.strip():
            raise ValueError("repair rationale must be non-empty")

    def digest(self) -> str:
        """Bind an owner decision to content, base hashes, checks and incident."""
        payload = {
            "incident_id": self.incident_id,
            "edits": [{"path": edit.path, "expected_sha256": edit.expected_sha256,
                       "replacement": edit.replacement} for edit in self.edits],
            "rationale": self.rationale,
            "requested_tests": self.requested_tests,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=False, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    checks: tuple[str, ...]
    details: str = ""


@dataclass(frozen=True)
class RepairOutcome:
    incident_id: str
    state: RepairState
    disposition: RepairDisposition
    message: str
    checkpoint_id: str | None = None
    approval_id: str | None = None


class RepairEngine(Protocol):
    def propose(self, incident: Incident, repository_context: str) -> RepairPatch:
        ...


class RepairSandbox(Protocol):
    def create_checkpoint(self, incident: Incident) -> str:
        ...

    def read_text(self, path: str) -> str:
        ...

    def apply(self, patch: RepairPatch) -> None:
        ...

    def verify(self, patch: RepairPatch) -> VerificationResult:
        ...

    def publish(self, patch: RepairPatch, checkpoint_id: str) -> None:
        ...

    def rollback(self, checkpoint_id: str) -> None:
        ...


class SelfHealPolicy:
    """Path-based authority ceiling for automatic code changes."""

    # Any edit here always requires owner approval even if tests pass.
    MAJOR_PREFIXES = (
        "core/owner_kernel.py",
        "core/action_gateway.py",
        "core/authority_contracts.py",
        "core/capability_registry.py",
        "core/windows_broker.py",
        "core/nexus/",
        "core/github_repair.py",
        "core/local_update_service.py",
        "core/build_info.py",
        "brain_sidecar.py",
        "core/secret_store.py",
        "core/owner_face.py",
        "core/self_heal.py",
        "core/update_",
        "api/",
        "packaging/",
        "web/src-tauri/",
        "models/",
        ".github/workflows/",
    )

    # JARVIS may never autonomously edit or introduce these surfaces.
    FORBIDDEN_PATTERNS = (
        ".git/",
        "LICENSE",
        ".github/CODEOWNERS",
        "secrets",
        "credentials",
        "private_key",
    )

    LOW_RISK_PREFIXES = (
        "core/",
        "agent/",
        "actions/",
        "memory/",
        "api/",
        "web/",
        "tests/",
        "docs/",
    )

    def classify(self, patch: RepairPatch) -> RepairRisk:
        paths = tuple(edit.path.replace("\\", "/") for edit in patch.edits)
        lowered = tuple(path.casefold() for path in paths)

        for path in lowered:
            if any(token.casefold() in path for token in self.FORBIDDEN_PATTERNS):
                return RepairRisk.FORBIDDEN

        for path in lowered:
            if any(
                path == prefix.casefold() or path.startswith(prefix.casefold())
                for prefix in self.MAJOR_PREFIXES
            ):
                return RepairRisk.MAJOR

        if all(path.startswith(self.LOW_RISK_PREFIXES) for path in paths):
            return RepairRisk.LOW
        return RepairRisk.MAJOR

    def disposition(self, patch: RepairPatch) -> RepairDisposition:
        risk = self.classify(patch)
        if risk is RepairRisk.FORBIDDEN:
            return RepairDisposition.REJECT
        if risk is RepairRisk.MAJOR:
            return RepairDisposition.REQUIRE_OWNER
        return RepairDisposition.AUTO_APPLY


class RepairJournal:
    """Durable local evidence for automatic repair attempts."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS repair_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    incident_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    message TEXT NOT NULL,
                    checkpoint_id TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )

    def record(
        self,
        incident_id: str,
        state: RepairState,
        message: str,
        checkpoint_id: str | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with sqlite3.connect(self.path) as db:
            db.execute(
                """
                INSERT INTO repair_events(
                    incident_id, state, message, checkpoint_id, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (incident_id, state.value, message[:2000], checkpoint_id, now),
            )

    def history(self, incident_id: str) -> tuple[dict[str, object], ...]:
        with sqlite3.connect(self.path) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(
                """
                SELECT incident_id, state, message, checkpoint_id, created_at
                FROM repair_events
                WHERE incident_id=?
                ORDER BY id
                """,
                (incident_id,),
            ).fetchall()
        return tuple(dict(row) for row in rows)

    def recent(self, limit: int = 20) -> tuple[dict[str, object], ...]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 200:
            raise ValueError("limit must be between 1 and 200")
        with sqlite3.connect(self.path) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(
                """
                SELECT incident_id, state, message, checkpoint_id, created_at
                FROM repair_events
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return tuple(dict(row) for row in rows)


class SelfHealController:
    def __init__(
        self,
        *,
        policy: SelfHealPolicy | None = None,
        journal: RepairJournal | None = None,
        owner_approvals: OwnerApprovalManager | None = None,
    ):
        self.policy = policy or SelfHealPolicy()
        self.journal = journal
        self.owner_approvals = owner_approvals
        self._runtime_handlers: dict[IncidentKind, Callable[[Incident], bool]] = {}

    def register_runtime_healer(
        self,
        kind: IncidentKind,
        handler: Callable[[Incident], bool],
    ) -> None:
        if not isinstance(kind, IncidentKind):
            raise TypeError("kind must be IncidentKind")
        self._runtime_handlers[kind] = handler

    def heal_runtime(self, incident: Incident) -> RepairOutcome:
        handler = self._runtime_handlers.get(incident.kind)
        if handler is None:
            outcome = RepairOutcome(
                incident.id,
                RepairState.FAILED,
                RepairDisposition.REJECT,
                "No bounded runtime healer is registered for this incident.",
            )
        else:
            try:
                ok = bool(handler(incident))
            except Exception as exc:
                ok = False
                message = f"Runtime healer failed ({type(exc).__name__})."
            else:
                message = (
                    "Runtime component recovered automatically."
                    if ok
                    else "Runtime healer could not verify recovery."
                )
            outcome = RepairOutcome(
                incident.id,
                RepairState.HEALED if ok else RepairState.FAILED,
                RepairDisposition.AUTO_APPLY,
                message,
            )
        if self.journal:
            self.journal.record(
                incident.id, outcome.state, outcome.message, outcome.checkpoint_id
            )
        return outcome

    @staticmethod
    def _validate_expected_hashes(
        sandbox: RepairSandbox,
        patch: RepairPatch,
    ) -> None:
        for edit in patch.edits:
            current = sandbox.read_text(edit.path)
            digest = hashlib.sha256(current.encode("utf-8")).hexdigest()
            if digest != edit.expected_sha256:
                raise RuntimeError(
                    f"Repair precondition changed for {edit.path}; refusing stale patch."
                )

    def repair_code(
        self,
        incident: Incident,
        *,
        engine: RepairEngine,
        sandbox: RepairSandbox,
        repository_context: str,
        owner_approved: bool = False,
        approval_id: str | None = None,
    ) -> RepairOutcome:
        patch = engine.propose(incident, repository_context)
        if patch.incident_id != incident.id:
            raise ValueError("repair patch is not bound to the incident")

        disposition = self.policy.disposition(patch)
        if disposition is RepairDisposition.REJECT:
            outcome = RepairOutcome(
                incident.id,
                RepairState.FAILED,
                disposition,
                "Repair touches a forbidden self-modification surface.",
            )
            if self.journal:
                self.journal.record(incident.id, outcome.state, outcome.message)
            return outcome

        # A caller/model supplied boolean is not proof of an owner decision.
        if disposition is RepairDisposition.REQUIRE_OWNER and approval_id is None:
            request_id = None
            if self.owner_approvals is not None:
                request_id = self.owner_approvals.create(
                    f"Apply JARVIS repair: {patch.rationale[:350]}", patch.digest(),
                    ttl_seconds=300,
                ).approval_id
            outcome = RepairOutcome(
                incident.id,
                RepairState.AWAITING_OWNER,
                disposition,
                "Major repair passed policy classification but requires owner approval.",
                approval_id=request_id,
            )
            if self.journal:
                self.journal.record(incident.id, outcome.state, outcome.message)
            return outcome

        if disposition is RepairDisposition.REQUIRE_OWNER:
            if self.owner_approvals is None:
                raise PermissionError("Major repair requires the paired owner approval queue")
            self.owner_approvals.consume(approval_id, action_digest=patch.digest())

        checkpoint = sandbox.create_checkpoint(incident)
        if self.journal:
            self.journal.record(
                incident.id,
                RepairState.VERIFYING,
                "Repair sandbox checkpoint created.",
                checkpoint,
            )

        try:
            self._validate_expected_hashes(sandbox, patch)
            sandbox.apply(patch)
            verification = sandbox.verify(patch)
            if not verification.passed:
                sandbox.rollback(checkpoint)
                outcome = RepairOutcome(
                    incident.id,
                    RepairState.ROLLED_BACK,
                    disposition,
                    "Repair failed verification and was rolled back.",
                    checkpoint,
                )
            else:
                sandbox.publish(patch, checkpoint)
                outcome = RepairOutcome(
                    incident.id,
                    RepairState.APPLIED,
                    disposition,
                    "Repair passed verification and was published.",
                    checkpoint,
                )
        except Exception as exc:
            try:
                sandbox.rollback(checkpoint)
            except Exception:
                pass
            outcome = RepairOutcome(
                incident.id,
                RepairState.FAILED,
                disposition,
                f"Repair pipeline failed ({type(exc).__name__}); rollback attempted.",
                checkpoint,
            )

        if self.journal:
            self.journal.record(
                incident.id,
                outcome.state,
                outcome.message,
                outcome.checkpoint_id,
            )
        return outcome
