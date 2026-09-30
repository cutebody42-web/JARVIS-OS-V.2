"""GitHub-native repair publishing for bounded JARVIS self-healing.

GitHub remains the source of truth:
- create an isolated repair branch from an expected base SHA;
- apply only policy-admitted text edits through the Contents API;
- open a repair PR;
- let repository CI verify the branch;
- low-risk repairs may auto-merge only after every required check succeeds;
- major/security/update surfaces always stop for owner approval.

This module never edits secrets, branch protection, workflow permissions,
repository settings, or arbitrary git refs outside its repair branch.
"""

from __future__ import annotations

from dataclasses import dataclass
import base64
import hashlib
import re
import time
from typing import Sequence
from urllib.parse import quote

import requests

from core.self_heal import RepairDisposition, RepairPatch, SelfHealPolicy
from core.secret_store import SecretStore, get_secret_store


_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_BRANCH_SAFE = re.compile(r"[^A-Za-z0-9._/-]+")


class GitHubRepairError(RuntimeError):
    pass


@dataclass(frozen=True)
class GitHubRepairPublication:
    branch: str
    pull_number: int
    pull_url: str
    head_sha: str
    disposition: RepairDisposition
    required_workflows: tuple[str, ...]


@dataclass(frozen=True)
class GitHubCheckSummary:
    completed: bool
    successful: bool
    pending: tuple[str, ...]
    failed: tuple[str, ...]
    missing: tuple[str, ...] = ()


class GitHubRepairClient:
    def __init__(
        self,
        repository: str,
        *,
        token: str | None = None,
        secret_store: SecretStore | None = None,
        api_base: str = "https://api.github.com",
        session: requests.Session | None = None,
        timeout_seconds: float = 20.0,
    ):
        if not isinstance(repository, str) or not _REPO_RE.fullmatch(repository):
            raise ValueError("repository must be owner/name")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.repository = repository
        self.secret_store = secret_store or get_secret_store()
        self._token = token
        self.api_base = api_base.rstrip("/")
        self.session = session or requests.Session()
        self.timeout = float(timeout_seconds)

    def _auth_token(self) -> str:
        value = self._token or self.secret_store.get("github.repair_token")
        if not isinstance(value, str) or len(value.strip()) < 20:
            raise GitHubRepairError(
                "GitHub repair token is not configured in the OS secret store."
            )
        return value.strip()

    def _headers(self) -> dict[str, str]:
        return {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self._auth_token()}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "JARVIS-self-heal",
        }

    def _request(self, method: str, path: str, *, json_body=None, params=None):
        try:
            response = self.session.request(
                method,
                self.api_base + path,
                headers=self._headers(),
                json=json_body,
                params=params,
                timeout=self.timeout,
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            raise GitHubRepairError(
                f"GitHub repair transport failed ({type(exc).__name__})."
            ) from None
        if response.status_code >= 300:
            raise GitHubRepairError(
                f"GitHub repair request failed ({response.status_code})."
            )
        try:
            return response.json()
        except ValueError as exc:
            raise GitHubRepairError("GitHub returned invalid JSON.") from exc

    @property
    def _repo_path(self) -> str:
        owner, name = self.repository.split("/", 1)
        return f"/repos/{quote(owner, safe='')}/{quote(name, safe='')}"

    def create_branch(self, branch: str, base_sha: str) -> str:
        if not _SHA_RE.fullmatch(base_sha):
            raise ValueError("base_sha must be a full commit SHA")
        self._request(
            "POST",
            self._repo_path + "/git/refs",
            json_body={"ref": "refs/heads/" + branch, "sha": base_sha},
        )
        return branch

    def file(self, path: str, ref: str) -> tuple[str, str]:
        value = self._request(
            "GET",
            self._repo_path + "/contents/" + quote(path, safe="/"),
            params={"ref": ref},
        )
        blob_sha = value.get("sha")
        content = value.get("content")
        encoding = value.get("encoding")
        if (
            not isinstance(blob_sha, str)
            or not isinstance(content, str)
            or encoding != "base64"
        ):
            raise GitHubRepairError("GitHub returned an invalid file payload.")
        try:
            text = base64.b64decode(content).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise GitHubRepairError("Repair target is not UTF-8 text.") from exc
        return blob_sha, text

    def update_file(
        self,
        *,
        branch: str,
        path: str,
        current_blob_sha: str,
        replacement: str,
        message: str,
    ) -> str:
        value = self._request(
            "PUT",
            self._repo_path + "/contents/" + quote(path, safe="/"),
            json_body={
                "message": message,
                "content": base64.b64encode(
                    replacement.encode("utf-8")
                ).decode("ascii"),
                "sha": current_blob_sha,
                "branch": branch,
            },
        )
        commit_sha = value.get("commit", {}).get("sha")
        if not isinstance(commit_sha, str) or not _SHA_RE.fullmatch(commit_sha):
            raise GitHubRepairError("GitHub returned an invalid repair commit.")
        return commit_sha

    def create_pull_request(
        self,
        *,
        title: str,
        body: str,
        head: str,
        base: str,
        draft: bool,
    ) -> tuple[int, str]:
        value = self._request(
            "POST",
            self._repo_path + "/pulls",
            json_body={
                "title": title,
                "body": body,
                "head": head,
                "base": base,
                "draft": bool(draft),
            },
        )
        number = value.get("number")
        url = value.get("html_url")
        if not isinstance(number, int) or not isinstance(url, str):
            raise GitHubRepairError("GitHub returned an invalid pull request.")
        return number, url

    def workflow_checks(
        self,
        head_sha: str,
        *,
        required_workflows: Sequence[str] = (),
    ) -> GitHubCheckSummary:
        if not _SHA_RE.fullmatch(head_sha):
            raise ValueError("head_sha must be a full commit SHA")
        value = self._request(
            "GET",
            self._repo_path + "/actions/runs",
            params={"head_sha": head_sha, "per_page": 100},
        )
        runs = value.get("workflow_runs")
        if not isinstance(runs, list):
            raise GitHubRepairError("GitHub returned invalid workflow runs.")

        pending: list[str] = []
        failed: list[str] = []
        observed_names: set[str] = set()
        observed = 0
        for run in runs:
            if not isinstance(run, dict):
                continue
            name = str(run.get("name") or "workflow")
            observed_names.add(name)
            status = run.get("status")
            conclusion = run.get("conclusion")
            observed += 1
            if status != "completed":
                pending.append(name)
            elif conclusion not in {"success", "neutral", "skipped"}:
                failed.append(name)

        missing = tuple(
            sorted(set(required_workflows) - observed_names)
        )
        return GitHubCheckSummary(
            completed=observed > 0 and not pending and not missing,
            successful=observed > 0 and not pending and not failed and not missing,
            pending=tuple(sorted(set(pending))),
            failed=tuple(sorted(set(failed))),
            missing=missing,
        )

    def merge_pull_request(
        self,
        pull_number: int,
        expected_head_sha: str,
    ) -> str:
        value = self._request(
            "PUT",
            self._repo_path + f"/pulls/{pull_number}/merge",
            json_body={
                "merge_method": "squash",
                "sha": expected_head_sha,
                "commit_title": "JARVIS self-heal: verified repair",
            },
        )
        if value.get("merged") is not True:
            raise GitHubRepairError("GitHub did not merge the verified repair.")
        sha = value.get("sha")
        if not isinstance(sha, str) or not _SHA_RE.fullmatch(sha):
            raise GitHubRepairError("GitHub returned an invalid merge commit.")
        return sha


class GitHubRepairCoordinator:
    def __init__(
        self,
        client: GitHubRepairClient,
        *,
        policy: SelfHealPolicy | None = None,
        required_workflows: Sequence[str] = (
            "CI",
            "NEXUS architecture contracts",
        ),
    ):
        if not isinstance(client, GitHubRepairClient):
            raise TypeError("client must be GitHubRepairClient")
        self.client = client
        self.policy = policy or SelfHealPolicy()
        self.required_workflows = tuple(required_workflows)

    def _required_for_patch(self, patch: RepairPatch) -> tuple[str, ...]:
        required = set(self.required_workflows)
        paths = tuple(edit.path.replace("\\", "/") for edit in patch.edits)

        if any(path.startswith("web/") for path in paths):
            required.add("JARVIS product shell")
            required.add("Build JARVIS Android companion")

        windows_relevant_prefixes = (
            "brain_sidecar.py",
            "api/",
            "core/",
            "agent/",
            "actions/",
            "memory/",
            "models/",
            "packaging/windows/",
            "web/",
            "requirements.txt",
        )
        if any(
            any(path == prefix or path.startswith(prefix) for prefix in windows_relevant_prefixes)
            for path in paths
        ):
            required.add("Build JARVIS Windows installer")

        return tuple(sorted(required))

    @staticmethod
    def _branch_name(incident_id: str) -> str:
        clean = _BRANCH_SAFE.sub("-", incident_id).strip("-/.")[:48]
        if not clean:
            clean = "incident"
        return "jarvis/self-heal/" + clean

    def publish(
        self,
        patch: RepairPatch,
        *,
        base_branch: str,
        base_sha: str,
    ) -> GitHubRepairPublication:
        disposition = self.policy.disposition(patch)
        if disposition is RepairDisposition.REJECT:
            raise PermissionError("Repair touches a forbidden self-modification surface.")

        branch = self._branch_name(patch.incident_id)
        self.client.create_branch(branch, base_sha)

        head_sha = base_sha
        for edit in patch.edits:
            blob_sha, current = self.client.file(edit.path, branch)
            digest = hashlib.sha256(current.encode("utf-8")).hexdigest()
            if digest != edit.expected_sha256:
                raise GitHubRepairError(
                    f"Repair precondition changed for {edit.path}; refusing stale patch."
                )
            head_sha = self.client.update_file(
                branch=branch,
                path=edit.path,
                current_blob_sha=blob_sha,
                replacement=edit.replacement,
                message=f"JARVIS self-heal: {patch.rationale[:120]}",
            )

        draft = disposition is RepairDisposition.REQUIRE_OWNER
        pull_number, pull_url = self.client.create_pull_request(
            title=f"JARVIS self-heal: {patch.rationale[:100]}",
            body=(
                "Automated JARVIS repair proposal.\n\n"
                f"Incident: {patch.incident_id}\n"
                f"Disposition: {disposition.value}\n"
                f"Requested checks: {', '.join(patch.requested_tests) or 'repository defaults'}\n\n"
                "Security/authority/update changes can never self-approve."
            ),
            head=branch,
            base=base_branch,
            draft=draft,
        )
        return GitHubRepairPublication(
            branch,
            pull_number,
            pull_url,
            head_sha,
            disposition,
            self._required_for_patch(patch),
        )

    def wait_for_ci(
        self,
        publication: GitHubRepairPublication,
        *,
        timeout_seconds: float = 1800,
        poll_seconds: float = 10.0,
    ) -> GitHubCheckSummary:
        deadline = time.monotonic() + timeout_seconds
        latest = GitHubCheckSummary(
            False,
            False,
            (),
            (),
            publication.required_workflows,
        )
        while time.monotonic() < deadline:
            latest = self.client.workflow_checks(
                publication.head_sha,
                required_workflows=publication.required_workflows,
            )
            if latest.completed:
                return latest
            time.sleep(poll_seconds)
        return latest

    def auto_merge_if_safe(
        self,
        publication: GitHubRepairPublication,
        checks: GitHubCheckSummary,
    ) -> str | None:
        if publication.disposition is not RepairDisposition.AUTO_APPLY:
            return None
        if not checks.successful:
            return None
        return self.client.merge_pull_request(
            publication.pull_number,
            publication.head_sha,
        )
