"""GitHub-native candidate publishing for bounded JARVIS self-healing.

GitHub remains the source of truth:
- create an isolated repair branch from an expected base SHA;
- apply only policy-admitted text edits through the Contents API;
- open a repair PR;
- let repository CI verify the branch;
- every repair is published as a draft and stops for owner review;
- no result, including a low-risk patch with green CI, may auto-merge.

This module never edits secrets, branch protection, workflow permissions,
repository settings, or arbitrary git refs outside its repair branch.
"""

from __future__ import annotations

from dataclasses import dataclass
import base64
import hashlib
import json
import re
import time
from typing import Sequence
from urllib.parse import quote

import requests

from core.self_heal import RepairDisposition, RepairPatch, SelfHealPolicy
from core.nexus.owner_approval import OwnerApprovalManager
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
    patch_digest: str = ""
    repository: str = ""
    base_branch: str = ""
    base_sha: str = ""


@dataclass(frozen=True)
class GitHubCheckSummary:
    completed: bool
    successful: bool
    pending: tuple[str, ...]
    failed: tuple[str, ...]
    missing: tuple[str, ...] = ()
    head_sha: str | None = None


@dataclass(frozen=True)
class GitHubPullRequestIdentity:
    number: int
    url: str
    state: str
    draft: bool
    head_ref: str
    head_sha: str
    head_repository: str
    base_ref: str
    base_sha: str
    base_repository: str


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
            if run.get("head_sha") != head_sha:
                continue
            name = str(run.get("name") or "workflow")
            observed_names.add(name)
            status = run.get("status")
            conclusion = run.get("conclusion")
            observed += 1
            if status != "completed":
                pending.append(name)
            elif conclusion != "success":
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
            head_sha=head_sha,
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

    def pull_request_identity(self, pull_number: int) -> GitHubPullRequestIdentity:
        if (
            not isinstance(pull_number, int)
            or isinstance(pull_number, bool)
            or pull_number <= 0
        ):
            raise ValueError("pull_number must be a positive integer")
        value = self._request("GET", self._repo_path + f"/pulls/{pull_number}")
        head = value.get("head")
        base = value.get("base")
        if not isinstance(head, dict) or not isinstance(base, dict):
            raise GitHubRepairError("GitHub returned an invalid pull request identity")
        head_repo = head.get("repo")
        base_repo = base.get("repo")
        fields = (
            value.get("number"),
            value.get("html_url"),
            value.get("state"),
            value.get("draft"),
            head.get("ref"),
            head.get("sha"),
            head_repo.get("full_name") if isinstance(head_repo, dict) else None,
            base.get("ref"),
            base.get("sha"),
            base_repo.get("full_name") if isinstance(base_repo, dict) else None,
        )
        if (
            not isinstance(fields[0], int)
            or isinstance(fields[0], bool)
            or not all(isinstance(item, str) and item for item in fields[1:3])
            or type(fields[3]) is not bool
            or not all(isinstance(item, str) and item for item in fields[4:])
            or not _SHA_RE.fullmatch(fields[5])
            or not _SHA_RE.fullmatch(fields[8])
        ):
            raise GitHubRepairError("GitHub returned an invalid pull request identity")
        return GitHubPullRequestIdentity(*fields)

    def mark_ready_for_review(self, pull_number: int) -> None:
        pull = self._request("GET", self._repo_path + f"/pulls/{pull_number}")
        if pull.get("draft") is not True:
            return
        node_id = pull.get("node_id")
        if not isinstance(node_id, str) or not node_id:
            raise GitHubRepairError("GitHub returned an invalid pull request node")
        result = self._request("POST", "/graphql", json_body={
            "query": "mutation($id: ID!) { markPullRequestReadyForReview(input: {pullRequestId: $id}) { pullRequest { isDraft } } }",
            "variables": {"id": node_id},
        })
        ready = result.get("data", {}).get("markPullRequestReadyForReview", {}).get("pullRequest", {})
        if result.get("errors") or ready.get("isDraft") is not False:
            raise GitHubRepairError("GitHub did not mark the owner-approved repair ready for review")


class GitHubRepairCoordinator:
    BASE_REQUIRED_WORKFLOWS = (
        "CI",
        "NEXUS architecture contracts",
    )

    def __init__(
        self,
        client: GitHubRepairClient,
        *,
        policy: SelfHealPolicy | None = None,
        required_workflows: Sequence[str] = BASE_REQUIRED_WORKFLOWS,
    ):
        if not isinstance(client, GitHubRepairClient):
            raise TypeError("client must be GitHubRepairClient")
        self.client = client
        self.policy = policy or SelfHealPolicy()
        if any(not isinstance(name, str) or not name.strip() for name in required_workflows):
            raise ValueError("required workflow names must be non-empty strings")
        # Configuration may add gates, but may not remove the repository's
        # minimum source-promotion gates.
        self.required_workflows = tuple(sorted({
            *self.BASE_REQUIRED_WORKFLOWS,
            *(name.strip() for name in required_workflows),
        }))

    def _required_for_patch(self, patch: RepairPatch) -> tuple[str, ...]:
        required = set(self.required_workflows)
        paths = tuple(edit.path.replace("\\", "/").casefold() for edit in patch.edits)

        if any(path.startswith("web/") for path in paths):
            required.add("JARVIS product shell")
            required.add("Build JARVIS Android companion")
            required.add("Build JARVIS iOS companion")

        scratch_surfaces = {
            "core/scratch_activation.py", "core/ollama_bootstrap.py",
            "tests/test_scratch_training.py", "tests/test_scratch_core_activation.py",
        }
        if any(path.startswith("training/") or path in scratch_surfaces for path in paths):
            required.add("Scratch model contracts")

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
            "scripts/generate_build_info.py",
        )
        if any(
            any(path == prefix or path.startswith(prefix) for prefix in windows_relevant_prefixes)
            for path in paths
        ):
            required.add("Build JARVIS Windows installer")

        real_runtime_prefixes = (
            "brain_sidecar.py",
            "scripts/generate_build_info.py",
            "api/jarvis_local_server.py",
            "core/",
            "agent/",
            "actions/",
            "memory/",
            "models/",
            "packaging/",
            "web/",
        )
        if any(
            any(path == prefix or path.startswith(prefix) for prefix in real_runtime_prefixes)
            for path in paths
        ):
            required.add("Real JARVIS cloud validation")

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
        if type(patch) is not RepairPatch:
            raise TypeError("patch must be an exact RepairPatch value")
        patch_digest = RepairPatch.digest(patch)
        policy_disposition = self.policy.disposition(patch)
        baseline_disposition = SelfHealPolicy().disposition(patch)
        if (
            policy_disposition is RepairDisposition.REJECT
            or baseline_disposition is RepairDisposition.REJECT
        ):
            raise PermissionError("Repair touches a forbidden self-modification surface.")
        # Policy injection can tighten publication rules, but cannot restore an
        # automatic source-promotion path.
        disposition = RepairDisposition.REQUIRE_OWNER

        # Verify all expected base blobs before making any GitHub mutation.
        validated = []
        for edit in patch.edits:
            blob_sha, current = self.client.file(edit.path, base_sha)
            digest = hashlib.sha256(current.encode("utf-8")).hexdigest()
            if digest != edit.expected_sha256:
                raise GitHubRepairError(
                    f"Repair precondition changed for {edit.path}; refusing stale patch."
                )
            validated.append((edit, blob_sha))
        branch = self._branch_name(patch.incident_id)
        self.client.create_branch(branch, base_sha)
        head_sha = base_sha
        for edit, blob_sha in validated:
            head_sha = self.client.update_file(
                branch=branch,
                path=edit.path,
                current_blob_sha=blob_sha,
                replacement=edit.replacement,
                message=f"JARVIS self-heal: {patch.rationale[:120]}",
            )

        pull_number, pull_url = self.client.create_pull_request(
            title=f"JARVIS self-heal: {patch.rationale[:100]}",
            body=(
                "Automated JARVIS repair proposal.\n\n"
                f"Incident: {patch.incident_id}\n"
                f"Disposition: {disposition.value}\n"
                f"Requested checks: {', '.join(patch.requested_tests) or 'repository defaults'}\n\n"
                "This candidate cannot self-approve or auto-merge. Explicit owner "
                "approval bound to this patch is required after fresh CI."
            ),
            head=branch,
            base=base_branch,
            draft=True,
        )
        return GitHubRepairPublication(
            branch,
            pull_number,
            pull_url,
            head_sha,
            disposition,
            self._required_for_patch(patch),
            patch_digest,
            self.client.repository,
            base_branch,
            base_sha,
        )

    def merge_action_digest(
        self,
        publication: GitHubRepairPublication,
        patch: RepairPatch,
    ) -> str:
        """Bind approval to the exact repository, PR and reviewed head SHA."""
        if type(publication) is not GitHubRepairPublication:
            raise TypeError("publication must be an exact GitHubRepairPublication value")
        if type(patch) is not RepairPatch:
            raise TypeError("patch must be an exact RepairPatch value")
        patch_digest = RepairPatch.digest(patch)
        if patch_digest != publication.patch_digest:
            raise PermissionError("Owner approval is not bound to this published patch")
        if publication.repository != self.client.repository:
            raise PermissionError("Repair publication belongs to a different repository")
        if not _SHA_RE.fullmatch(publication.base_sha) or not _SHA_RE.fullmatch(publication.head_sha):
            raise ValueError("Repair publication must pin full base and head commit SHAs")
        if (
            not isinstance(publication.pull_number, int)
            or isinstance(publication.pull_number, bool)
            or publication.pull_number <= 0
        ):
            raise ValueError("Repair publication must identify a pull request")
        required = self._required_for_patch(patch)
        if publication.required_workflows != required:
            raise PermissionError("Repair publication does not preserve required CI gates")
        payload = {
            "action": "github.repair.merge.v1",
            "repository": publication.repository,
            "base_branch": publication.base_branch,
            "base_sha": publication.base_sha,
            "branch": publication.branch,
            "pull_number": publication.pull_number,
            "pull_url": publication.pull_url,
            "head_sha": publication.head_sha,
            "required_workflows": list(required),
            "patch_digest": patch_digest,
        }
        return hashlib.sha256(json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")).hexdigest()

    def request_merge_approval(
        self,
        publication: GitHubRepairPublication,
        patch: RepairPatch,
        *,
        approvals: OwnerApprovalManager,
        ttl_seconds: int = 300,
    ):
        """Create the only approval shape accepted by the merge boundary."""
        if type(approvals) is not OwnerApprovalManager:
            raise TypeError("approvals must be OwnerApprovalManager")
        digest = self.merge_action_digest(publication, patch)
        summary = (
            f"Merge JARVIS repair {publication.repository}#{publication.pull_number} "
            f"at {publication.head_sha[:12]}: {patch.rationale[:250]}"
        )
        return OwnerApprovalManager.create(
            approvals, summary, digest, ttl_seconds=ttl_seconds,
        )

    def _verify_live_pull_request(
        self,
        publication: GitHubRepairPublication,
    ) -> None:
        live = self.client.pull_request_identity(publication.pull_number)
        if type(live) is not GitHubPullRequestIdentity:
            raise GitHubRepairError("GitHub returned an invalid pull request identity")
        if (
            live.number != publication.pull_number
            or live.url != publication.pull_url
            or live.state != "open"
            or live.head_ref != publication.branch
            or live.head_sha != publication.head_sha
            or live.head_repository.casefold() != publication.repository.casefold()
            or live.base_ref != publication.base_branch
            or live.base_sha != publication.base_sha
            or live.base_repository.casefold() != publication.repository.casefold()
        ):
            raise PermissionError(
                "Repair pull request identity changed after publication or owner review"
            )

    def wait_for_ci(
        self,
        publication: GitHubRepairPublication,
        *,
        timeout_seconds: float = 1800,
        poll_seconds: float = 10.0,
    ) -> GitHubCheckSummary:
        if timeout_seconds <= 0 or not 0 < poll_seconds <= 60:
            raise ValueError("CI wait requires a positive timeout and 0..60 second polling interval")
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
        """Compatibility shim: autonomous source promotion is disabled.

        CI evidence can make a candidate ready for review, but only
        :meth:`merge_with_owner_approval` owns merge authority.
        """
        return None

    def merge_with_owner_approval(
        self, publication: GitHubRepairPublication, patch: RepairPatch, *,
        approvals: OwnerApprovalManager, approval_id: str,
    ) -> str:
        """An exact fingerprint-approved repair still requires fresh green CI."""
        if type(approvals) is not OwnerApprovalManager:
            raise TypeError("approvals must be OwnerApprovalManager")
        if publication.disposition is not RepairDisposition.REQUIRE_OWNER:
            raise ValueError("Owner-approved merge requires an owner-review publication")
        action_digest = self.merge_action_digest(publication, patch)
        latest = self.client.workflow_checks(
            publication.head_sha, required_workflows=publication.required_workflows,
        )
        if not latest.completed or not latest.successful or latest.head_sha != publication.head_sha:
            raise GitHubRepairError("Repair CI has not passed for this exact commit")
        # The PR can be retargeted or its base can advance without changing its
        # head SHA. Re-read every approved identity field at the mutation edge.
        self._verify_live_pull_request(publication)
        # Invoke the concrete verifier, not an override supplied by a subclass.
        OwnerApprovalManager.consume(
            approvals, approval_id, action_digest=action_digest,
        )
        self.client.mark_ready_for_review(publication.pull_number)
        # Marking the draft ready is a separate remote mutation. Recheck before
        # merge while retaining GitHub's expected-head SHA precondition below.
        self._verify_live_pull_request(publication)
        return self.client.merge_pull_request(publication.pull_number, publication.head_sha)
