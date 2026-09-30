"""Signed update plans and biometric-gated approval for major JARVIS updates.

Minor/patch updates may be automated only when they remain inside low-risk
surfaces and pass verification. Major updates require a one-time approval signed
by an explicitly trusted companion-device key that is expected to be protected
by the mobile OS biometric/keychain boundary.

The biometric sample itself never leaves the phone and is never stored here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import PurePosixPath
import secrets
import sqlite3
from typing import Any, Mapping, Protocol
from uuid import uuid4

from core.nexus.owner_approval import OwnerApprovalManager
from core.nexus.peer_auth import DeviceSigner, PeerRegistry


class UpdateClass(str, Enum):
    PATCH = "patch"
    MINOR = "minor"
    MAJOR = "major"


class UpdateState(str, Enum):
    DISCOVERED = "discovered"
    STAGED = "staged"
    AWAITING_APPROVAL = "awaiting_approval"
    APPLYING = "applying"
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass(frozen=True)
class UpdatePlan:
    version: str
    commit_sha: str
    artifact_sha256: str
    changed_paths: tuple[str, ...]
    notes: str
    update_class: UpdateClass

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("update version must be non-empty")
        if (
            not isinstance(self.commit_sha, str)
            or len(self.commit_sha) != 40
            or any(ch not in "0123456789abcdef" for ch in self.commit_sha.casefold())
        ):
            raise ValueError("commit_sha must be a 40-character Git SHA")
        if (
            not isinstance(self.artifact_sha256, str)
            or len(self.artifact_sha256) != 64
        ):
            raise ValueError("artifact_sha256 must be a SHA-256 digest")
        int(self.artifact_sha256, 16)
        object.__setattr__(self, "changed_paths", tuple(self.changed_paths))
        if any(
            PurePosixPath(path).is_absolute()
            or ".." in PurePosixPath(path).parts
            for path in self.changed_paths
        ):
            raise ValueError("update paths must be repository-relative")
        if not isinstance(self.update_class, UpdateClass):
            raise TypeError("update_class must be UpdateClass")

    def digest(self) -> str:
        payload = {
            "version": self.version,
            "commit_sha": self.commit_sha,
            "artifact_sha256": self.artifact_sha256,
            "changed_paths": list(self.changed_paths),
            "notes": self.notes,
            "update_class": self.update_class.value,
        }
        raw = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


class UpdatePolicy:
    MAJOR_PREFIXES = (
        "core/owner_kernel.py",
        "core/action_gateway.py",
        "core/authority_contracts.py",
        "core/capability_registry.py",
        "core/windows_broker.py",
        "core/self_heal.py",
        "core/update_manager.py",
        "core/nexus/peer_auth.py",
        "core/nexus/pairing.py",
        "core/nexus/signed_transport.py",
        "models/",
        "packaging/",
        ".github/workflows/",
    )

    PATCH_PREFIXES = (
        "docs/",
        "tests/",
        "web/",
    )

    def classify_paths(self, paths: tuple[str, ...]) -> UpdateClass:
        if any(
            any(path == prefix or path.startswith(prefix) for prefix in self.MAJOR_PREFIXES)
            for path in paths
        ):
            return UpdateClass.MAJOR
        if paths and all(path.startswith(self.PATCH_PREFIXES) for path in paths):
            return UpdateClass.PATCH
        return UpdateClass.MINOR

    def validate_declared_class(self, plan: UpdatePlan) -> None:
        required = self.classify_paths(plan.changed_paths)
        order = {
            UpdateClass.PATCH: 0,
            UpdateClass.MINOR: 1,
            UpdateClass.MAJOR: 2,
        }
        if order[plan.update_class] < order[required]:
            raise PermissionError(
                f"Update is under-classified: declared {plan.update_class.value}, "
                f"required at least {required.value}."
            )

    def requires_owner_approval(self, plan: UpdatePlan) -> bool:
        self.validate_declared_class(plan)
        return plan.update_class is UpdateClass.MAJOR


@dataclass(frozen=True)
class ApprovalChallenge:
    challenge_id: str
    plan_digest: str
    nonce: str
    expires_at: str

    def signing_payload(self) -> bytes:
        return json.dumps(
            {
                "challenge_id": self.challenge_id,
                "plan_digest": self.plan_digest,
                "nonce": self.nonce,
                "expires_at": self.expires_at,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")


@dataclass(frozen=True)
class DeviceApproval:
    peer_id: str
    challenge_id: str
    plan_digest: str
    nonce: str
    expires_at: str
    user_verified: bool
    signature: str

    def signing_payload(self) -> bytes:
        return json.dumps(
            {
                "challenge_id": self.challenge_id,
                "plan_digest": self.plan_digest,
                "nonce": self.nonce,
                "expires_at": self.expires_at,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")


class UpdateApprovalGate:
    def __init__(
        self,
        registry: PeerRegistry,
        *,
        clock=lambda: datetime.now(timezone.utc),
    ):
        if not isinstance(registry, PeerRegistry):
            raise TypeError("registry must be PeerRegistry")
        self.registry = registry
        self._clock = clock
        with self.registry._store._connect() as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS nexus_update_challenges (
                    challenge_id TEXT PRIMARY KEY,
                    plan_digest TEXT NOT NULL,
                    nonce TEXT NOT NULL UNIQUE,
                    expires_at TEXT NOT NULL,
                    consumed INTEGER NOT NULL DEFAULT 0 CHECK(consumed IN (0,1))
                )
                """
            )

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("update approval clock must be timezone-aware")
        return value.astimezone(timezone.utc)

    @staticmethod
    def _parse_time(value: str) -> datetime:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("approval timestamp must include timezone")
        return parsed.astimezone(timezone.utc)

    def create_challenge(
        self,
        plan: UpdatePlan,
        *,
        ttl_seconds: int = 180,
    ) -> ApprovalChallenge:
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int):
            raise TypeError("ttl_seconds must be an integer")
        if not 10 <= ttl_seconds <= 600:
            raise ValueError("approval challenge TTL must be 10..600 seconds")
        now = self._now()
        expires = datetime.fromtimestamp(now.timestamp() + ttl_seconds, timezone.utc)
        challenge = ApprovalChallenge(
            uuid4().hex,
            plan.digest(),
            secrets.token_urlsafe(24),
            expires.isoformat().replace("+00:00", "Z"),
        )
        with self.registry._store._connect() as db:
            db.execute(
                """
                INSERT INTO nexus_update_challenges(
                    challenge_id, plan_digest, nonce, expires_at, consumed
                ) VALUES (?, ?, ?, ?, 0)
                """,
                (
                    challenge.challenge_id,
                    challenge.plan_digest,
                    challenge.nonce,
                    challenge.expires_at,
                ),
            )
        return challenge

    @staticmethod
    def sign_on_companion(
        peer_id: str,
        challenge: ApprovalChallenge,
        signer: DeviceSigner,
        *,
        user_verified: bool,
    ) -> DeviceApproval:
        if not user_verified:
            raise PermissionError(
                "Major update approval requires local biometric/user verification."
            )
        if not isinstance(signer, DeviceSigner):
            raise TypeError("signer must be DeviceSigner")
        signature = signer.sign(challenge.signing_payload())
        return DeviceApproval(
            peer_id=peer_id,
            challenge_id=challenge.challenge_id,
            plan_digest=challenge.plan_digest,
            nonce=challenge.nonce,
            expires_at=challenge.expires_at,
            user_verified=True,
            signature=signature,
        )

    def verify_and_consume(
        self,
        plan: UpdatePlan,
        approval: DeviceApproval,
    ) -> bool:
        if not isinstance(approval, DeviceApproval):
            raise TypeError("approval must be DeviceApproval")
        if not approval.user_verified:
            raise PermissionError("approval is missing user verification")
        peer = self.registry.get_peer(approval.peer_id)
        if peer is None:
            raise PermissionError("approval device is not actively trusted")

        with self.registry._store._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                """
                SELECT challenge_id, plan_digest, nonce, expires_at, consumed
                FROM nexus_update_challenges
                WHERE challenge_id=?
                """,
                (approval.challenge_id,),
            ).fetchone()
            if row is None or bool(row["consumed"]):
                db.rollback()
                raise PermissionError("approval challenge is absent or already consumed")
            if (
                row["plan_digest"] != plan.digest()
                or row["plan_digest"] != approval.plan_digest
                or row["nonce"] != approval.nonce
                or row["expires_at"] != approval.expires_at
            ):
                db.rollback()
                raise PermissionError("approval is not bound to this exact update")
            if self._now() > self._parse_time(row["expires_at"]):
                db.rollback()
                raise PermissionError("approval challenge has expired")
            if not DeviceSigner.verify(
                peer.public_key,
                approval.signature,
                approval.signing_payload(),
            ):
                db.rollback()
                raise PermissionError("approval signature is invalid")

            db.execute(
                """
                UPDATE nexus_update_challenges
                SET consumed=1
                WHERE challenge_id=? AND consumed=0
                """,
                (approval.challenge_id,),
            )
            db.commit()
        return True


class UpdateInstaller(Protocol):
    def stage(self, plan: UpdatePlan) -> str:
        ...

    def apply(self, plan: UpdatePlan, checkpoint_id: str) -> None:
        ...

    def verify(self, plan: UpdatePlan) -> bool:
        ...

    def rollback(self, checkpoint_id: str) -> None:
        ...


@dataclass(frozen=True)
class UpdateOutcome:
    state: UpdateState
    message: str
    checkpoint_id: str | None = None
    approval_id: str | None = None


class UpdateManager:
    def __init__(
        self,
        installer: UpdateInstaller,
        *,
        policy: UpdatePolicy | None = None,
        approval_gate: UpdateApprovalGate | None = None,
        owner_approvals: OwnerApprovalManager | None = None,
    ):
        self.installer = installer
        self.policy = policy or UpdatePolicy()
        self.approval_gate = approval_gate
        self.owner_approvals = owner_approvals

    def apply(
        self,
        plan: UpdatePlan,
        *,
        approval: DeviceApproval | None = None,
        approval_id: str | None = None,
    ) -> UpdateOutcome:
        self.policy.validate_declared_class(plan)
        if self.policy.requires_owner_approval(plan):
            if self.owner_approvals is not None:
                if approval_id is None:
                    request = self.owner_approvals.create(
                        f"Install JARVIS major update {plan.version}",
                        plan.digest(),
                        ttl_seconds=300,
                    )
                    return UpdateOutcome(
                        UpdateState.AWAITING_APPROVAL,
                        "Major update requires fingerprint/biometric approval on the paired phone.",
                        approval_id=request.approval_id,
                    )
                self.owner_approvals.consume(
                    approval_id,
                    action_digest=plan.digest(),
                )
            else:
                if self.approval_gate is None or approval is None:
                    return UpdateOutcome(
                        UpdateState.AWAITING_APPROVAL,
                        "Major update is staged but requires companion biometric approval.",
                    )
                self.approval_gate.verify_and_consume(plan, approval)

        checkpoint = self.installer.stage(plan)
        try:
            self.installer.apply(plan, checkpoint)
            if not self.installer.verify(plan):
                self.installer.rollback(checkpoint)
                return UpdateOutcome(
                    UpdateState.ROLLED_BACK,
                    "Update verification failed; the previous version was restored.",
                    checkpoint,
                )
            return UpdateOutcome(
                UpdateState.APPLIED,
                "Update applied and verified.",
                checkpoint,
            )
        except Exception as exc:
            try:
                self.installer.rollback(checkpoint)
            except Exception:
                pass
            return UpdateOutcome(
                UpdateState.FAILED,
                f"Update failed ({type(exc).__name__}); rollback attempted.",
                checkpoint,
            )
