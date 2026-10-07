"""Explicit orchestration for provider-assisted JARVIS code repair.

The workflow deliberately stops at governance boundaries:
- a configured text provider may propose bounded replacement text;
- the host binds edits to exact file hashes;
- GitHubRepairCoordinator publishes an isolated PR and defines required CI;
- low-risk merge remains the coordinator's exact-head CI decision;
- major/security repairs require a separate owner approval bound to patch digest.

No method in this module grants a model permission to execute code, alter policy,
change repository settings, or approve/merge a major repair by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from core.github_repair import (
    GitHubCheckSummary,
    GitHubRepairCoordinator,
    GitHubRepairPublication,
)
from core.model_provider import ModelProvider, default_provider
from core.model_repair_engine import ModelRepairEngine, build_repair_context
from core.nexus.owner_approval import OwnerApprovalManager
from core.self_heal import Incident, RepairDisposition, RepairPatch


@dataclass(frozen=True)
class ModelRepairProposal:
    patch: RepairPatch
    publication: GitHubRepairPublication

    @property
    def requires_owner(self) -> bool:
        return self.publication.disposition is RepairDisposition.REQUIRE_OWNER


class ModelAssistedRepairWorkflow:
    """Host-owned bridge from a text provider to governed GitHub repair."""

    def __init__(
        self,
        provider: ModelProvider,
        coordinator: GitHubRepairCoordinator,
    ) -> None:
        if not isinstance(provider, ModelProvider):
            raise TypeError("provider must implement ModelProvider")
        if not isinstance(coordinator, GitHubRepairCoordinator):
            raise TypeError("coordinator must be GitHubRepairCoordinator")
        self.engine = ModelRepairEngine(provider)
        self.coordinator = coordinator

    def propose_and_publish(
        self,
        incident: Incident,
        *,
        files: Mapping[str, str],
        base_branch: str,
        base_sha: str,
    ) -> ModelRepairProposal:
        """Generate one proposal and publish it to an isolated repair PR.

        ``files`` must contain the exact text fetched by trusted host code from
        ``base_sha``. The proposal engine recomputes their hashes and the
        coordinator re-fetches and verifies those hashes before any mutation.
        """
        context = build_repair_context(files)
        patch = self.engine.propose(incident, context)
        publication = self.coordinator.publish(
            patch,
            base_branch=base_branch,
            base_sha=base_sha,
        )
        return ModelRepairProposal(patch, publication)

    def verify(
        self,
        proposal: ModelRepairProposal,
        *,
        timeout_seconds: float = 1800,
        poll_seconds: float = 10.0,
    ) -> GitHubCheckSummary:
        if not isinstance(proposal, ModelRepairProposal):
            raise TypeError("proposal must be ModelRepairProposal")
        return self.coordinator.wait_for_ci(
            proposal.publication,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
        )

    def auto_merge_if_safe(
        self,
        proposal: ModelRepairProposal,
        checks: GitHubCheckSummary,
    ) -> str | None:
        """Allow only the coordinator's existing low-risk exact-head policy."""
        if not isinstance(proposal, ModelRepairProposal):
            raise TypeError("proposal must be ModelRepairProposal")
        return self.coordinator.auto_merge_if_safe(proposal.publication, checks)

    def merge_with_owner_approval(
        self,
        proposal: ModelRepairProposal,
        *,
        approvals: OwnerApprovalManager,
        approval_id: str,
    ) -> str:
        """Merge a major proposal only through the existing exact owner gate."""
        if not isinstance(proposal, ModelRepairProposal):
            raise TypeError("proposal must be ModelRepairProposal")
        return self.coordinator.merge_with_owner_approval(
            proposal.publication,
            proposal.patch,
            approvals=approvals,
            approval_id=approval_id,
        )


def configured_model_repair_workflow(
    coordinator: GitHubRepairCoordinator,
) -> ModelAssistedRepairWorkflow:
    """Build the workflow from the explicitly configured neutral provider.

    ``default_provider`` fails closed for unknown provider names. The Gemini
    adapter separately fails if the owner's credentials are absent; Ollama is
    selected only when ``NEXUS_MODEL_PROVIDER=ollama`` is explicit.
    """
    return ModelAssistedRepairWorkflow(default_provider(), coordinator)
