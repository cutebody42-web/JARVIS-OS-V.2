"""Signed update plans and biometric-gated approval for every JARVIS update.

Candidates may be downloaded, staged, and verified automatically. Installing
any candidate is a promotion of executable/source/configuration state and
therefore requires a one-time approval signed by an explicitly trusted
companion-device key that is expected to be protected by the mobile OS
biometric/keychain boundary.

The biometric sample itself never leaves the phone and is never stored here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import sqlite3
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Protocol
from urllib.parse import quote, urljoin, urlsplit
from uuid import uuid4

import requests

from core.nexus.owner_approval import OwnerApprovalManager
from core.nexus.peer_auth import DeviceSigner, PeerRegistry
from core.self_heal import repository_path


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
    release_tag: str = ""
    artifact_name: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9.-]+)?", self.version):
            raise ValueError("update version must be a semantic version")
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
        object.__setattr__(self, "commit_sha", self.commit_sha.casefold())
        object.__setattr__(self, "artifact_sha256", self.artifact_sha256.casefold())
        if not isinstance(self.notes, str) or len(self.notes) > 20_000:
            raise ValueError("update notes must be bounded text")
        object.__setattr__(self, "changed_paths", tuple(self.changed_paths))
        if not self.changed_paths:
            raise ValueError("update must declare changed paths")
        for path in self.changed_paths:
            repository_path(path)
        if len({path.casefold() for path in self.changed_paths}) != len(self.changed_paths):
            raise ValueError("update contains duplicate or case-colliding paths")
        if not isinstance(self.update_class, UpdateClass):
            raise TypeError("update_class must be UpdateClass")
        if not isinstance(self.release_tag, str) or (self.release_tag and not re.fullmatch(r"[A-Za-z0-9._-]{1,160}", self.release_tag)):
            raise ValueError("release_tag must be a simple pinned GitHub tag")
        if not isinstance(self.artifact_name, str) or (self.artifact_name and not re.fullmatch(r"[A-Za-z0-9._-]{1,160}", self.artifact_name)):
            raise ValueError("artifact_name must be a simple release asset name")

    def digest(self) -> str:
        payload = {
            "version": self.version,
            "commit_sha": self.commit_sha,
            "artifact_sha256": self.artifact_sha256,
            "changed_paths": list(self.changed_paths),
            "notes": self.notes,
            "update_class": self.update_class.value,
            "release_tag": self.release_tag,
            "artifact_name": self.artifact_name,
        }
        raw = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version, "commit_sha": self.commit_sha,
            "artifact_sha256": self.artifact_sha256,
            "changed_paths": list(self.changed_paths), "notes": self.notes,
            "update_class": self.update_class.value,
            "release_tag": self.release_tag, "artifact_name": self.artifact_name,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "UpdatePlan":
        clean = dict(value)
        clean["update_class"] = UpdateClass(clean["update_class"])
        return cls(**clean)


class UpdatePolicy:
    MAJOR_PREFIXES = (
        "core/owner_kernel.py",
        "core/action_gateway.py",
        "core/authority_contracts.py",
        "core/capability_registry.py",
        "core/windows_broker.py",
        "core/self_heal.py",
        "core/update_manager.py",
        "core/local_update_service.py",
        "core/build_info.py",
        "core/scratch_activation.py",
        "core/ollama_bootstrap.py",
        "brain_sidecar.py",
        "core/github_repair.py",
        "core/owner_face.py",
        "core/secret_store.py",
        "core/nexus/",
        "api/",
        "models/",
        "training/",
        "packaging/",
        "web/src-tauri/",
        ".github/workflows/",
    )

    PATCH_PREFIXES = (
        "docs/",
        "tests/",
        "web/",
    )

    def classify_paths(self, paths: tuple[str, ...]) -> UpdateClass:
        if any(
            any(path.casefold() == prefix.casefold() or path.casefold().startswith(prefix.casefold()) for prefix in self.MAJOR_PREFIXES)
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
        # Classification determines review depth and release policy, never
        # whether installation may bypass the owner.
        return True


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
        if type(plan) is not UpdatePlan:
            raise TypeError("plan must be an exact UpdatePlan value")
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
        if type(plan) is not UpdatePlan:
            raise TypeError("plan must be an exact UpdatePlan value")
        if type(approval) is not DeviceApproval:
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
            if self._now() >= self._parse_time(row["expires_at"]):
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


class UpdateArtifactError(RuntimeError):
    """Release integrity, staging, or recovery prerequisites failed."""


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(dict(value), sort_keys=True, indent=2), "utf-8")
    os.replace(temporary, path)


class GitHubReleaseSource:
    """Public GitHub release assets pinned to a repository, tag and commit.

    SHA-256 provides artifact integrity; it is not a publisher signature. This
    source trusts HTTPS and the configured repository's release permissions.
    Credentials are deliberately not sent to release/CDN download hosts.
    """

    ALLOWED_HOSTS = {"api.github.com", "github.com", "codeload.github.com",
                     "release-assets.githubusercontent.com", "objects.githubusercontent.com"}
    MANIFEST_NAME = "jarvis-update.json"

    def __init__(self, repository: str = "cutebody42-web/JARVIS-OS-V.2", *,
                 session: requests.Session | None = None, timeout_seconds: float = 30,
                 max_download_bytes: int = 1024 * 1024 * 1024):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise ValueError("repository must be owner/name")
        if timeout_seconds <= 0 or max_download_bytes <= 0:
            raise ValueError("download limits must be positive")
        self.repository = repository
        self.session = session or requests.Session()
        # Use an explicit anonymous auth object to suppress netrc/session tokens
        # while preserving the managed environment's proxy and CA settings.
        class PublicAuth(requests.auth.AuthBase):
            def __call__(self, request):
                request.headers.pop("Authorization", None)
                request.headers.pop("Cookie", None)
                return request
        self.session.auth = PublicAuth()
        self.timeout = timeout_seconds
        self.max_download_bytes = max_download_bytes

    @property
    def api_url(self) -> str:
        return "https://api.github.com/repos/" + self.repository

    def _response(self, url: str):
        for _ in range(6):
            parsed = urlsplit(url)
            if (parsed.scheme != "https" or parsed.hostname not in self.ALLOWED_HOSTS
                    or parsed.username is not None or parsed.password is not None
                    or parsed.port not in {None, 443}):
                raise UpdateArtifactError("Release download redirected outside trusted GitHub HTTPS hosts")
            response = self.session.get(url, timeout=self.timeout, stream=True,
                                        allow_redirects=False,
                                        headers={"Accept": "application/vnd.github+json"})
            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("Location")
                response.close()
                if not location:
                    raise UpdateArtifactError("Release download returned an empty redirect")
                url = urljoin(url, location)
                continue
            if response.status_code != 200:
                response.close()
                raise UpdateArtifactError(f"GitHub release request failed ({response.status_code})")
            return response
        raise UpdateArtifactError("Release download exceeded redirect limit")

    def _json(self, url: str) -> Any:
        response = self._response(url)
        payload = bytearray()
        try:
            for chunk in response.iter_content(64 * 1024):
                payload.extend(chunk)
                if len(payload) > 2 * 1024 * 1024:
                    raise UpdateArtifactError("Release metadata exceeds size limit")
            return json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise UpdateArtifactError("Release metadata is not valid JSON") from exc
        finally:
            response.close()

    def _release_plan(self, release: Mapping[str, Any]) -> UpdatePlan:
        if release.get("draft") is not False:
            raise UpdateArtifactError("Draft releases cannot be installed")
        assets = release.get("assets")
        if not isinstance(assets, list):
            raise UpdateArtifactError("Release assets are missing")
        manifest = next((asset for asset in assets if asset.get("name") == self.MANIFEST_NAME), None)
        if manifest is None:
            raise UpdateArtifactError("Release has no verified update manifest")
        tag = release.get("tag_name")
        if not isinstance(tag, str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,160}", tag):
            raise UpdateArtifactError("Release tag is not a supported pinned tag")
        manifest_url = self._asset_url(manifest, tag)
        try:
            plan = UpdatePlan.from_dict(self._json(manifest_url))
        except (KeyError, TypeError, ValueError) as exc:
            raise UpdateArtifactError("Release update manifest is invalid") from exc
        if plan.release_tag != tag or plan.artifact_name != "JARVIS-Setup.exe" or plan.update_class is not UpdateClass.MAJOR:
            raise UpdateArtifactError("Windows release manifest must declare its tag, installer and major class")
        reference = self._json(self.api_url + "/git/ref/tags/" + quote(tag, safe=""))
        target = reference.get("object", {})
        if target.get("type") == "tag":
            target = self._json(self.api_url + "/git/tags/" + str(target.get("sha"))).get("object", {})
        if target.get("type") != "commit" or target.get("sha") != plan.commit_sha:
            raise UpdateArtifactError("Release tag does not identify the manifest's exact commit")
        return plan

    def _asset_url(self, asset: Mapping[str, Any], tag: str) -> str:
        name = asset.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,160}", name):
            raise UpdateArtifactError("Release asset name is invalid")
        expected = f"https://github.com/{self.repository}/releases/download/{quote(tag, safe='')}/{quote(name, safe='')}"
        if asset.get("browser_download_url") != expected:
            raise UpdateArtifactError("Release asset is not from the configured pinned repository tag")
        return expected

    def discover(self) -> UpdatePlan | None:
        releases = self._json(self.api_url + "/releases?per_page=100")
        if not isinstance(releases, list):
            raise UpdateArtifactError("GitHub returned an invalid release list")
        candidates: list[tuple[datetime, Mapping[str, Any]]] = []
        for release in releases:
            if release.get("draft") is False and any(
                    asset.get("name") == self.MANIFEST_NAME for asset in release.get("assets", [])):
                # GitHub's listing order is not a publication-order guarantee.
                # Compare actual instants, including offsets, so an older build
                # cannot hide a newly published update. Unknown dates must not
                # silently cause a fallback to an older installer.
                value = release.get("published_at")
                if not isinstance(value, str) or not re.fullmatch(
                        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})", value):
                    raise UpdateArtifactError("Release publication timestamp is missing or invalid")
                try:
                    published = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
                except (ValueError, OverflowError) as exc:
                    raise UpdateArtifactError("Release publication timestamp is invalid") from exc
                candidates.append((published, release))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return self._release_plan(candidates[0][1])

    def find_version(self, version: str, *, commit_sha: str | None = None) -> UpdatePlan:
        releases = self._json(self.api_url + "/releases?per_page=100")
        if not isinstance(releases, list):
            raise UpdateArtifactError("GitHub returned an invalid release list")
        for release in releases:
            # The publisher records a full target SHA. Skip unrelated builds
            # before fetching manifests to stay within GitHub's public API rate.
            target = release.get("target_commitish")
            if commit_sha is not None and isinstance(target, str) and re.fullmatch(r"[0-9a-f]{40}", target) and target != commit_sha:
                continue
            if release.get("draft") is False and any(
                    asset.get("name") == self.MANIFEST_NAME for asset in release.get("assets", [])):
                plan = self._release_plan(release)
                if plan.version == version and (commit_sha is None or plan.commit_sha == commit_sha):
                    return plan
        raise UpdateArtifactError("Installed version has no published verified recovery installer")

    def download(self, plan: UpdatePlan, destination: Path) -> None:
        release = self._json(self.api_url + "/releases/tags/" + quote(plan.release_tag, safe=""))
        if self._release_plan(release).digest() != plan.digest():
            raise UpdateArtifactError("Published release changed after its update plan was approved")
        asset = next((asset for asset in release["assets"] if asset.get("name") == plan.artifact_name), None)
        if asset is None:
            raise UpdateArtifactError("Pinned installer asset is missing")
        digest = asset.get("digest")
        if digest is not None and digest != "sha256:" + plan.artifact_sha256:
            raise UpdateArtifactError("GitHub asset digest disagrees with the release manifest")
        response = self._response(self._asset_url(asset, plan.release_tag))
        size = 0
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with destination.open("xb") as target:
                for chunk in response.iter_content(1024 * 1024):
                    size += len(chunk)
                    if size > self.max_download_bytes:
                        raise UpdateArtifactError("Installer download exceeds size limit")
                    target.write(chunk)
            if _file_digest(destination) != plan.artifact_sha256:
                raise UpdateArtifactError("Installer SHA-256 validation failed")
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        finally:
            response.close()


class WindowsReleaseUpdateCoordinator:
    """Stage a new and recovery NSIS installer before biometric approval.

    The existing app keeps running while artifacts are checked. Installation is
    handed to a copy of the frozen sidecar and waits until the desktop exits.
    Successful installation requires a self-test of the newly installed binary;
    failures reinstall and verify the previous, also hash-verified, installer.
    This restores the application, not unrelated system/registry modifications.
    """

    def __init__(self, source: GitHubReleaseSource, *, state_dir: Path,
                 installed_version: str, installed_sidecar: Path,
                 owner_approvals: OwnerApprovalManager | None = None,
                 installed_commit_sha: str | None = None):
        if owner_approvals is not None and type(owner_approvals) is not OwnerApprovalManager:
            raise TypeError("owner_approvals must be OwnerApprovalManager or None")
        self.source = source
        self.directory = Path(state_dir).resolve() / "updates"
        if self.directory.is_symlink() or self.directory.resolve() != self.directory:
            raise UpdateArtifactError("Update state directory must not be a symlink or junction")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.installed_version = installed_version
        self.installed_commit_sha = installed_commit_sha
        self.installed_sidecar = Path(installed_sidecar).resolve()
        self.owner_approvals = owner_approvals

    def _checkpoint(self, checkpoint_id: str) -> Path:
        if not isinstance(checkpoint_id, str) or not re.fullmatch(r"[0-9a-f]{32}", checkpoint_id):
            raise ValueError("Invalid update checkpoint ID")
        checkpoint = self.directory / checkpoint_id
        if checkpoint.is_symlink() or checkpoint.resolve() != checkpoint or not checkpoint.is_dir():
            raise UpdateArtifactError("Update checkpoint is absent or unsafe")
        return checkpoint

    @staticmethod
    def _handoff_digest(
        checkpoint_id: str,
        plan: UpdatePlan,
        previous: UpdatePlan,
        installed_sidecar: Path,
        *,
        helper_sha256: str | None = None,
    ) -> str:
        """Bind approval to this exact staged install and recovery target."""
        if not isinstance(checkpoint_id, str) or not re.fullmatch(r"[0-9a-f]{32}", checkpoint_id):
            raise ValueError("Invalid update checkpoint ID")
        if type(plan) is not UpdatePlan or type(previous) is not UpdatePlan:
            raise TypeError("handoff plans must be exact UpdatePlan values")
        target = Path(installed_sidecar).resolve()
        if helper_sha256 is None:
            helper_sha256 = _file_digest(target)
        if (
            not isinstance(helper_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", helper_sha256)
        ):
            raise ValueError("helper_sha256 must be a lowercase SHA-256 digest")
        payload = {
            "action": "windows.release.install.v1",
            "checkpoint_id": checkpoint_id,
            "plan_digest": UpdatePlan.digest(plan),
            "recovery_digest": UpdatePlan.digest(previous),
            "installed_sidecar": str(target),
            "helper_sha256": helper_sha256,
        }
        return hashlib.sha256(json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")).hexdigest()

    @staticmethod
    def _copy_helper_exclusive(source: Path, destination: Path) -> str:
        """Copy the installed sidecar without following a pre-created target."""
        if source.is_symlink() or not source.is_file():
            raise UpdateArtifactError("Installed update helper source is unsafe")
        created = False
        digest = hashlib.sha256()
        try:
            with source.open("rb") as source_file:
                with destination.open("xb") as helper_file:
                    created = True
                    for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                        digest.update(chunk)
                        helper_file.write(chunk)
                    helper_file.flush()
                    os.fsync(helper_file.fileno())
        except FileExistsError as exc:
            raise UpdateArtifactError("Update helper path already exists") from exc
        except Exception:
            if created:
                destination.unlink(missing_ok=True)
            raise
        copied_digest = digest.hexdigest()
        if destination.is_symlink() or _file_digest(destination) != copied_digest:
            destination.unlink(missing_ok=True)
            raise UpdateArtifactError("Copied update helper failed integrity verification")
        return copied_digest

    @staticmethod
    def _installer(path: Path, plan: UpdatePlan) -> None:
        if path.is_symlink() or not path.is_file() or _file_digest(path) != plan.artifact_sha256:
            raise UpdateArtifactError("Staged installer is missing or has changed")
        with path.open("rb") as artifact:
            header = artifact.read(64)
            if len(header) < 64 or header[:2] != b"MZ":
                raise UpdateArtifactError("Installer is not a Windows PE executable")
            offset = int.from_bytes(header[60:64], "little")
            if offset < 64 or offset > path.stat().st_size - 4:
                raise UpdateArtifactError("Installer has an invalid PE header")
            artifact.seek(offset)
            if artifact.read(4) != b"PE\x00\x00":
                raise UpdateArtifactError("Installer has an invalid PE signature")

    def prepare(self, plan: UpdatePlan) -> UpdateOutcome:
        if type(plan) is not UpdatePlan:
            raise TypeError("plan must be an exact UpdatePlan value")
        if plan.update_class is not UpdateClass.MAJOR or plan.artifact_name != "JARVIS-Setup.exe" or not plan.release_tag:
            raise ValueError("Packaged updates require a major pinned Windows release plan")
        if plan.version == self.installed_version and plan.commit_sha == self.installed_commit_sha:
            return UpdateOutcome(UpdateState.APPLIED, "The installed version is already current.")
        checkpoint_id = uuid4().hex
        checkpoint = self.directory / checkpoint_id
        checkpoint.mkdir()
        try:
            if self.installed_commit_sha is None:
                raise UpdateArtifactError("Installed build has no commit provenance; verified recovery cannot be selected")
            previous = self.source.find_version(self.installed_version, commit_sha=self.installed_commit_sha)
            if previous.commit_sha != self.installed_commit_sha or previous.version != self.installed_version:
                raise UpdateArtifactError("Recovery installer does not match the currently installed build")
            self.source.download(plan, checkpoint / "candidate.exe")
            self.source.download(previous, checkpoint / "recovery.exe")
            self._installer(checkpoint / "candidate.exe", plan)
            self._installer(checkpoint / "recovery.exe", previous)
            metadata = {"plan": plan.to_dict(), "previous": previous.to_dict(),
                        "installed_sidecar": str(self.installed_sidecar), "authorized": False}
            _atomic_json(checkpoint / "stage.json", metadata)
            approval_id = None
            if self.owner_approvals is not None:
                if type(self.owner_approvals) is not OwnerApprovalManager:
                    raise TypeError("owner_approvals must be OwnerApprovalManager or None")
                handoff_digest = self._handoff_digest(
                    checkpoint_id, plan, previous, self.installed_sidecar,
                )
                approval_id = OwnerApprovalManager.create(
                    self.owner_approvals,
                    f"Install JARVIS {plan.version}; recovery {previous.version} is verified",
                    handoff_digest, ttl_seconds=600,
                ).approval_id
                metadata["approval_id"] = approval_id
                _atomic_json(checkpoint / "stage.json", metadata)
            return UpdateOutcome(UpdateState.AWAITING_APPROVAL,
                                 "Installer and previous-version recovery are verified. Fingerprint approval and desktop restart are required.",
                                 checkpoint_id, approval_id)
        except Exception as exc:
            shutil.rmtree(checkpoint, ignore_errors=True)
            return UpdateOutcome(UpdateState.FAILED,
                                 f"Update staging failed ({type(exc).__name__}): {str(exc)[:300]}")

    def authorize_handoff(self, checkpoint_id: str, *, approval_id: str,
                          parent_pid: int, launcher: Callable[[list[str]], Any] | None = None) -> UpdateOutcome:
        if sys.platform != "win32" or not getattr(sys, "frozen", False):
            raise UpdateArtifactError("Packaged installation runs only from the installed Windows JARVIS app")
        if not isinstance(parent_pid, int) or isinstance(parent_pid, bool) or parent_pid <= 0:
            raise ValueError("Desktop parent PID must be positive")
        if self.owner_approvals is None:
            raise PermissionError("An exact companion fingerprint approval is required")
        if type(self.owner_approvals) is not OwnerApprovalManager:
            raise TypeError("owner_approvals must be OwnerApprovalManager or None")
        checkpoint = self._checkpoint(checkpoint_id)
        metadata = json.loads((checkpoint / "stage.json").read_text("utf-8"))
        if metadata.get("authorized"):
            raise PermissionError("Update handoff is already authorized")
        plan = UpdatePlan.from_dict(metadata["plan"])
        previous = UpdatePlan.from_dict(metadata["previous"])
        self._installer(checkpoint / "candidate.exe", plan)
        self._installer(checkpoint / "recovery.exe", previous)
        if Path(metadata["installed_sidecar"]).resolve() != self.installed_sidecar:
            raise UpdateArtifactError("Installed sidecar path changed")
        helper = checkpoint / "jarvis-update-helper.exe"
        helper_sha256 = self._copy_helper_exclusive(self.installed_sidecar, helper)
        # No caller-chosen executable, NSIS flags, shell string, or URL enters the
        # child command; only validated internal state and the desktop PID do.
        command = [str(helper), "--install-update", str(self.directory.parent), checkpoint_id, str(parent_pid)]
        handoff_digest = self._handoff_digest(
            checkpoint_id, plan, previous, self.installed_sidecar,
            helper_sha256=helper_sha256,
        )
        try:
            approval = OwnerApprovalManager.consume(
                self.owner_approvals, approval_id, action_digest=handoff_digest,
            )
        except Exception:
            helper.unlink(missing_ok=True)
            raise
        metadata["authorized"] = True
        metadata["authorization"] = {
            "approval_id": approval.approval_id,
            "action_digest": approval.action_digest,
            "decided_by": approval.decided_by,
            "consumed": approval.consumed,
        }
        metadata["sidecar_pid"] = os.getpid()
        metadata["parent_pid"] = parent_pid
        metadata["helper_sha256"] = helper_sha256
        _atomic_json(checkpoint / "stage.json", metadata)
        try:
            if launcher is None:
                subprocess.Popen(command, shell=False, close_fds=True,
                                 creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
            else:
                launcher(command)
        except Exception as exc:
            metadata["authorized"] = False
            metadata.pop("authorization", None)
            metadata.pop("sidecar_pid", None)
            metadata.pop("parent_pid", None)
            metadata.pop("helper_sha256", None)
            _atomic_json(checkpoint / "stage.json", metadata)
            helper.unlink(missing_ok=True)
            return UpdateOutcome(UpdateState.FAILED,
                                 f"Could not launch update helper ({type(exc).__name__}); request a fresh approval.", checkpoint_id)
        return UpdateOutcome(UpdateState.APPLYING,
                             "Verified installer is waiting for desktop shutdown; the helper will verify or recover the previous version.", checkpoint_id)

    def complete_handoff(self, checkpoint_id: str, *, parent_pid: int,
                         wait_for_exit: Callable[[int], bool],
                         run_installer: Callable[[Path], None],
                         self_test: Callable[[Path, str, str], bool]) -> UpdateOutcome:
        """Trusted helper entry point; fixed OS adapters are supplied by main."""
        checkpoint = self._checkpoint(checkpoint_id)
        metadata = json.loads((checkpoint / "stage.json").read_text("utf-8"))
        if metadata.get("authorized") is not True:
            raise PermissionError("Update helper has no owner-authorized handoff")
        if (checkpoint / "outcome.json").exists():
            raise PermissionError("Update helper has already completed")
        authorized_parent_pid = metadata.get("parent_pid")
        if (
            not isinstance(parent_pid, int)
            or isinstance(parent_pid, bool)
            or parent_pid <= 0
            or authorized_parent_pid != parent_pid
        ):
            raise PermissionError("Update helper parent PID does not match the authorized handoff")
        plan = UpdatePlan.from_dict(metadata["plan"])
        previous = UpdatePlan.from_dict(metadata["previous"])
        target = Path(metadata["installed_sidecar"])
        helper = checkpoint / "jarvis-update-helper.exe"
        helper_sha256 = metadata.get("helper_sha256")
        if (
            not isinstance(helper_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", helper_sha256)
            or helper.is_symlink()
            or not helper.is_file()
            or _file_digest(helper) != helper_sha256
        ):
            raise PermissionError("Update helper executable has changed since authorization")
        handoff_digest = self._handoff_digest(
            checkpoint_id, plan, previous, target,
            helper_sha256=helper_sha256,
        )
        authorization = metadata.get("authorization")
        if not isinstance(authorization, dict) or set(authorization) != {
            "approval_id", "action_digest", "decided_by", "consumed",
        }:
            raise PermissionError("Update helper has no exact owner-approval receipt")
        if (
            not isinstance(authorization["approval_id"], str)
            or authorization["action_digest"] != handoff_digest
            or not isinstance(authorization["decided_by"], str)
            or not authorization["decided_by"]
            or authorization["consumed"] is not True
        ):
            raise PermissionError("Update helper approval is not bound to this exact plan")
        approval_root = self.directory.parent / "nexus"
        approval_db = approval_root / "local.db"
        if (
            approval_root.is_symlink()
            or approval_root.resolve() != approval_root
            or approval_db.is_symlink()
            or not approval_db.is_file()
        ):
            raise PermissionError("Update helper cannot verify the owner-approval ledger")
        db = None
        try:
            db = sqlite3.connect(approval_db.as_uri() + "?mode=ro", uri=True)
            db.row_factory = sqlite3.Row
            row = db.execute(
                "SELECT action_digest, state, decided_by, consumed "
                "FROM nexus_owner_approvals WHERE approval_id=?",
                (authorization["approval_id"],),
            ).fetchone()
        except sqlite3.Error as exc:
            raise PermissionError("Update helper cannot verify the owner-approval ledger") from exc
        finally:
            if db is not None:
                db.close()
        if (
            row is None
            or row["action_digest"] != handoff_digest
            or row["state"] != "approved"
            or row["decided_by"] != authorization["decided_by"]
            or int(row["consumed"]) != 1
        ):
            raise PermissionError("Update helper approval receipt is invalid")
        if target.resolve() != self.installed_sidecar:
            raise UpdateArtifactError("Update helper target changed")
        # A process-exclusive claim prevents a second helper from installing the
        # same approved update concurrently or replaying it after a crash.
        claim = checkpoint / "helper.claim"
        with claim.open("x") as handle:
            handle.write(str(os.getpid()))
        installed = False
        try:
            self._installer(checkpoint / "candidate.exe", plan)
            self._installer(checkpoint / "recovery.exe", previous)
            if not wait_for_exit(parent_pid):
                raise UpdateArtifactError("Desktop did not shut down before the installation deadline")
            sidecar_pid = metadata.get("sidecar_pid")
            if not isinstance(sidecar_pid, int) or isinstance(sidecar_pid, bool) or sidecar_pid <= 0:
                raise UpdateArtifactError("Authorized handoff has no valid original sidecar PID")
            if not wait_for_exit(sidecar_pid):
                raise UpdateArtifactError("Original brain sidecar did not exit before installation")
            installed = True
            run_installer(checkpoint / "candidate.exe")
            if not self_test(target, plan.version, plan.commit_sha):
                raise UpdateArtifactError("Newly installed JARVIS failed its binary self-test")
            outcome = UpdateOutcome(UpdateState.APPLIED, "Packaged update installed and binary self-test passed.", checkpoint_id)
        except Exception as exc:
            if installed:
                try:
                    self._installer(checkpoint / "recovery.exe", previous)
                    run_installer(checkpoint / "recovery.exe")
                    if not self_test(target, previous.version, previous.commit_sha):
                        raise UpdateArtifactError("Previous JARVIS did not pass recovery verification")
                    outcome = UpdateOutcome(UpdateState.ROLLED_BACK,
                                            f"New version failed ({type(exc).__name__}); previous application was reinstalled and verified.", checkpoint_id)
                except Exception as recovery:
                    outcome = UpdateOutcome(UpdateState.FAILED,
                                            f"Update and recovery failed ({type(recovery).__name__}); verified recovery installer remains available.", checkpoint_id)
            else:
                outcome = UpdateOutcome(UpdateState.FAILED, f"Update stopped before installation ({type(exc).__name__}).", checkpoint_id)
        _atomic_json(checkpoint / "outcome.json", {"state": outcome.state.value, "message": outcome.message,
                                                   "checkpoint_id": checkpoint_id, "version": plan.version,
                                                   "previous_version": previous.version})
        return outcome

    def history(self) -> tuple[dict[str, Any], ...]:
        result = []
        for path in sorted(self.directory.glob("*/outcome.json")):
            try:
                self._checkpoint(path.parent.name)
                if path.is_symlink() or path.stat().st_size > 64 * 1024:
                    continue
                result.append(json.loads(path.read_text("utf-8")))
            except (OSError, ValueError, UpdateArtifactError):
                continue
        return tuple(result[-25:])


def run_windows_update_helper(state_dir: str, checkpoint_id: str, parent_pid: int) -> UpdateOutcome:
    """Execute only a staged, authorized NSIS handoff after desktop shutdown."""
    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        raise UpdateArtifactError("Update helper requires the packaged Windows executable")
    root = Path(state_dir).resolve()
    checkpoint = root / "updates" / checkpoint_id
    if not re.fullmatch(r"[0-9a-f]{32}", checkpoint_id):
        raise ValueError("Invalid update checkpoint ID")
    expected_helper = checkpoint / "jarvis-update-helper.exe"
    running_executable = Path(sys.executable)
    if (
        running_executable.is_symlink()
        or not running_executable.is_file()
        or running_executable.resolve() != expected_helper.resolve()
    ):
        raise UpdateArtifactError("Update handoff must run from its staged helper executable")

    import psutil

    metadata = json.loads((checkpoint / "stage.json").read_text("utf-8"))
    coordinator = WindowsReleaseUpdateCoordinator(
        GitHubReleaseSource(), state_dir=root,
        installed_version=metadata["previous"]["version"],
        installed_sidecar=Path(metadata["installed_sidecar"]),
        installed_commit_sha=metadata["previous"]["commit_sha"],
    )

    def wait_for_exit(pid: int) -> bool:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if not psutil.pid_exists(pid):
                return True
            time.sleep(0.5)
        return False

    def install(path: Path) -> None:
        subprocess.run([str(path), "/S"], shell=False, check=True, timeout=600)

    def self_test(path: Path, version: str, commit_sha: str) -> bool:
        if not path.is_file():
            return False
        # PyInstaller's Windows sidecar uses console=False, so stdout is not a
        # reliable IPC channel. The trusted binary writes a bounded result file.
        output = checkpoint / ("self-test-" + uuid4().hex + ".json")
        completed = subprocess.run([str(path), "--update-self-test", str(output)], shell=False,
                                   check=False, timeout=120,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if completed.returncode != 0:
            return False
        try:
            if output.is_symlink() or output.stat().st_size > 64 * 1024:
                return False
            result = json.loads(output.read_text("utf-8"))
        except (ValueError, OSError):
            return False
        finally:
            output.unlink(missing_ok=True)
        return (result.get("ok") is True and result.get("version") == version
                and result.get("commit_sha") == commit_sha)

    return coordinator.complete_handoff(checkpoint_id, parent_pid=parent_pid,
                                        wait_for_exit=wait_for_exit, run_installer=install, self_test=self_test)


class UpdateManager:
    def __init__(
        self,
        installer: UpdateInstaller,
        *,
        policy: UpdatePolicy | None = None,
        approval_gate: UpdateApprovalGate | None = None,
        owner_approvals: OwnerApprovalManager | None = None,
    ):
        if approval_gate is not None and type(approval_gate) is not UpdateApprovalGate:
            raise TypeError("approval_gate must be UpdateApprovalGate or None")
        if owner_approvals is not None and type(owner_approvals) is not OwnerApprovalManager:
            raise TypeError("owner_approvals must be OwnerApprovalManager or None")
        self.installer = installer
        self.policy = policy or UpdatePolicy()
        self.approval_gate = approval_gate
        self.owner_approvals = owner_approvals
        self._staged: dict[str, str] = {}
        self._approval_requests: dict[str, str] = {}

    def prepare(self, plan: UpdatePlan) -> UpdateOutcome:
        """Verify a candidate before asking the owner to approve installation."""
        if type(plan) is not UpdatePlan:
            raise TypeError("plan must be an exact UpdatePlan value")
        # Custom policies may tighten release validation, but cannot weaken the
        # built-in classification floor.
        UpdatePolicy().validate_declared_class(plan)
        self.policy.validate_declared_class(plan)
        digest = plan.digest()
        try:
            if digest not in self._staged:
                self._staged[digest] = self.installer.stage(plan)
        except Exception as exc:
            return UpdateOutcome(UpdateState.FAILED,
                                 f"Update staging failed ({type(exc).__name__}); installation was not started.")
        return UpdateOutcome(UpdateState.STAGED, "Update candidate has been staged and validated.", self._staged[digest])

    def apply(
        self,
        plan: UpdatePlan,
        *,
        approval: DeviceApproval | None = None,
        approval_id: str | None = None,
    ) -> UpdateOutcome:
        prepared = self.prepare(plan)
        if prepared.state is UpdateState.FAILED:
            return prepared
        checkpoint = prepared.checkpoint_id
        if (
            UpdatePolicy().requires_owner_approval(plan)
            or self.policy.requires_owner_approval(plan)
        ):
            if self.owner_approvals is not None:
                if type(self.owner_approvals) is not OwnerApprovalManager:
                    raise TypeError("owner_approvals must be OwnerApprovalManager or None")
                if approval_id is None:
                    request_id = self._approval_requests.get(plan.digest())
                    if (
                        request_id is None
                        or OwnerApprovalManager.get(
                            self.owner_approvals, request_id,
                        ).state != "pending"
                    ):
                        request_id = OwnerApprovalManager.create(
                            self.owner_approvals,
                            f"Install JARVIS update {plan.version}",
                            plan.digest(), ttl_seconds=300,
                        ).approval_id
                        self._approval_requests[plan.digest()] = request_id
                    return UpdateOutcome(
                        UpdateState.AWAITING_APPROVAL,
                        "Update requires explicit fingerprint/biometric approval on the paired phone.",
                        checkpoint_id=checkpoint,
                        approval_id=request_id,
                    )
                OwnerApprovalManager.consume(
                    self.owner_approvals, approval_id,
                    action_digest=plan.digest(),
                )
            else:
                if self.approval_gate is None or approval is None:
                    return UpdateOutcome(
                        UpdateState.AWAITING_APPROVAL,
                        "Update is staged but requires explicit companion biometric approval.",
                        checkpoint,
                    )
                if type(self.approval_gate) is not UpdateApprovalGate:
                    raise TypeError("approval_gate must be UpdateApprovalGate or None")
                # Invoke the concrete verifier so a subclass cannot replace the
                # one-shot signature/challenge checks with a no-op.
                UpdateApprovalGate.verify_and_consume(
                    self.approval_gate, plan, approval,
                )

        self._staged.pop(plan.digest(), None)
        self._approval_requests.pop(plan.digest(), None)
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
