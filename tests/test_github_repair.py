"""Offline contracts for GitHub-native bounded JARVIS self-healing."""

import hashlib
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from core.github_repair import (
    GitHubCheckSummary,
    GitHubPullRequestIdentity,
    GitHubRepairClient,
    GitHubRepairCoordinator,
)
from core.self_heal import FileEdit, RepairDisposition, RepairPatch
from core.nexus.event_store import EventStore
from core.nexus.owner_approval import OwnerApprovalManager


def make_patch(path: str, replacement: str = "new") -> RepairPatch:
    old = "old"
    return RepairPatch(
        incident_id="incident-123",
        edits=(
            FileEdit(
                path=path,
                expected_sha256=hashlib.sha256(old.encode("utf-8")).hexdigest(),
                replacement=replacement,
            ),
        ),
        rationale="repair regression",
        requested_tests=("repository gates",),
    )


class FakeRepairClient(GitHubRepairClient):
    def __init__(self):
        super().__init__(
            "cutebody42-web/JARVIS-OS-V.2",
            token="offline-test-placeholder-value",
        )
        self.created_branch = None
        self.created_pr = None
        self.merged = []
        self.ready = []
        self.head_sha = "b" * 40
        self.live_head_ref = None
        self.live_base_ref = None
        self.live_base_sha = None
        self.live_url = None

    def create_branch(self, branch, base_sha):
        self.created_branch = (branch, base_sha)
        self.live_head_ref = branch
        self.live_base_sha = base_sha
        return branch

    def file(self, path, ref):
        return "blob-sha", "old"

    def update_file(self, **kwargs):
        return self.head_sha

    def create_pull_request(self, **kwargs):
        self.created_pr = kwargs
        self.live_base_ref = kwargs["base"]
        self.live_url = "https://example.invalid/pr/77"
        return 77, self.live_url

    def merge_pull_request(self, pull_number, expected_head_sha):
        self.merged.append((pull_number, expected_head_sha))
        return "c" * 40

    def workflow_checks(self, head_sha, *, required_workflows=()):
        return GitHubCheckSummary(True, True, (), (), (), head_sha)

    def pull_request_identity(self, pull_number):
        return GitHubPullRequestIdentity(
            pull_number,
            self.live_url,
            "open",
            not bool(self.ready),
            self.live_head_ref,
            self.head_sha,
            self.repository,
            self.live_base_ref,
            self.live_base_sha,
            self.repository,
        )

    def mark_ready_for_review(self, pull_number):
        self.ready.append(pull_number)


class StaticRunClient(GitHubRepairClient):
    def __init__(self, runs):
        super().__init__(
            "cutebody42-web/JARVIS-OS-V.2",
            token="offline-test-placeholder-value",
        )
        self.runs = [{"head_sha": "a" * 40, **run} for run in runs]

    def _request(self, method, path, *, json_body=None, params=None):
        return {"workflow_runs": self.runs}


class GitHubRepairGateTests(unittest.TestCase):
    def test_web_repair_requires_all_product_gates(self):
        client = FakeRepairClient()
        coordinator = GitHubRepairCoordinator(client)

        publication = coordinator.publish(
            make_patch("web/src/components/status.tsx"),
            base_branch="product/jarvis-brain",
            base_sha="a" * 40,
        )

        self.assertEqual(publication.disposition, RepairDisposition.REQUIRE_OWNER)
        self.assertEqual(
            set(publication.required_workflows),
            {
                "CI",
                "NEXUS architecture contracts",
                "JARVIS product shell",
                "Build JARVIS Android companion",
                "Build JARVIS iOS companion",
                "Build JARVIS Windows installer",
                "Real JARVIS cloud validation",
            },
        )
        self.assertTrue(client.created_pr["draft"])

    def test_tests_only_repair_requires_core_contract_gates_only(self):
        coordinator = GitHubRepairCoordinator(FakeRepairClient())
        publication = coordinator.publish(
            make_patch("tests/test_example.py"),
            base_branch="product/jarvis-brain",
            base_sha="a" * 40,
        )
        self.assertEqual(
            set(publication.required_workflows),
            {"CI", "NEXUS architecture contracts"},
        )

    def test_scratch_training_repairs_require_real_architecture_contracts(self):
        coordinator = GitHubRepairCoordinator(FakeRepairClient())
        for path in ("training/train.py", "tests/test_scratch_training.py", "tests/test_scratch_core_activation.py"):
            with self.subTest(path=path):
                publication = coordinator.publish(make_patch(path), base_branch="main", base_sha="a" * 40)
                self.assertEqual(set(publication.required_workflows),
                                 {"CI", "NEXUS architecture contracts", "Scratch model contracts"})

    def test_scratch_core_changes_require_architecture_and_actual_runtime_checks(self):
        coordinator = GitHubRepairCoordinator(FakeRepairClient())
        for path in ("core/scratch_activation.py", "core/ollama_bootstrap.py"):
            with self.subTest(path=path):
                publication = coordinator.publish(make_patch(path), base_branch="main", base_sha="a" * 40)
                self.assertEqual(publication.disposition, RepairDisposition.REQUIRE_OWNER)
                self.assertEqual(set(publication.required_workflows),
                                 {"CI", "NEXUS architecture contracts", "Scratch model contracts",
                                  "Build JARVIS Windows installer", "Real JARVIS cloud validation"})

    def test_build_provenance_generator_requires_packaged_and_live_runtime_checks(self):
        coordinator = GitHubRepairCoordinator(FakeRepairClient())
        publication = coordinator.publish(make_patch("scripts/generate_build_info.py"),
                                          base_branch="main", base_sha="a" * 40)
        self.assertEqual(set(publication.required_workflows),
                         {"CI", "NEXUS architecture contracts", "Build JARVIS Windows installer", "Real JARVIS cloud validation"})

    def test_major_repair_is_draft_and_never_auto_merges(self):
        client = FakeRepairClient()
        coordinator = GitHubRepairCoordinator(client)
        publication = coordinator.publish(
            make_patch("core/update_manager.py"),
            base_branch="product/jarvis-brain",
            base_sha="a" * 40,
        )

        self.assertEqual(
            publication.disposition,
            RepairDisposition.REQUIRE_OWNER,
        )
        self.assertTrue(client.created_pr["draft"])

        checks = GitHubCheckSummary(True, True, (), (), ())
        self.assertIsNone(
            coordinator.auto_merge_if_safe(publication, checks)
        )
        self.assertEqual(client.merged, [])

    def test_forbidden_repair_is_rejected_before_github_mutation(self):
        client = FakeRepairClient()
        coordinator = GitHubRepairCoordinator(client)

        with self.assertRaises(PermissionError):
            coordinator.publish(
                make_patch("LICENSE"),
                base_branch="product/jarvis-brain",
                base_sha="a" * 40,
            )

        self.assertIsNone(client.created_branch)
        self.assertIsNone(client.created_pr)

    def test_missing_required_workflow_blocks_success(self):
        client = StaticRunClient(
            [
                {"name": "CI", "status": "completed", "conclusion": "success"},
                {
                    "name": "NEXUS architecture contracts",
                    "status": "completed",
                    "conclusion": "success",
                },
            ]
        )

        summary = client.workflow_checks(
            "a" * 40,
            required_workflows=(
                "CI",
                "NEXUS architecture contracts",
                "Build JARVIS Windows installer",
            ),
        )

        self.assertFalse(summary.completed)
        self.assertFalse(summary.successful)
        self.assertEqual(
            summary.missing,
            ("Build JARVIS Windows installer",),
        )

    def test_failed_or_pending_workflow_blocks_success(self):
        client = StaticRunClient(
            [
                {"name": "CI", "status": "completed", "conclusion": "failure"},
                {
                    "name": "NEXUS architecture contracts",
                    "status": "in_progress",
                    "conclusion": None,
                },
            ]
        )

        summary = client.workflow_checks(
            "a" * 40,
            required_workflows=(
                "CI",
                "NEXUS architecture contracts",
            ),
        )

        self.assertFalse(summary.completed)
        self.assertFalse(summary.successful)
        self.assertEqual(summary.failed, ("CI",))
        self.assertEqual(
            summary.pending,
            ("NEXUS architecture contracts",),
        )
        self.assertEqual(summary.missing, ())

    def test_low_risk_repair_never_auto_merges_even_after_complete_success(self):
        client = FakeRepairClient()
        coordinator = GitHubRepairCoordinator(client)
        publication = coordinator.publish(
            make_patch("docs/runtime.md"),
            base_branch="product/jarvis-brain",
            base_sha="a" * 40,
        )

        blocked = GitHubCheckSummary(
            False,
            False,
            (),
            (),
            ("CI",),
        )
        self.assertIsNone(
            coordinator.auto_merge_if_safe(publication, blocked)
        )

        green = GitHubCheckSummary(True, True, (), (), (), publication.head_sha)
        merged = coordinator.auto_merge_if_safe(publication, green)

        self.assertIsNone(merged)
        self.assertEqual(client.merged, [])

    def test_forged_auto_apply_publication_cannot_restore_auto_merge(self):
        client = FakeRepairClient()
        coordinator = GitHubRepairCoordinator(client)
        publication = coordinator.publish(
            make_patch("docs/runtime.md"), base_branch="main", base_sha="a" * 40,
        )
        forged = type(publication)(
            publication.branch,
            publication.pull_number,
            publication.pull_url,
            publication.head_sha,
            RepairDisposition.AUTO_APPLY,
            publication.required_workflows,
            publication.patch_digest,
        )
        green = GitHubCheckSummary(True, True, (), (), (), forged.head_sha)
        self.assertIsNone(coordinator.auto_merge_if_safe(forged, green))
        self.assertEqual(client.merged, [])

    def test_injected_policy_cannot_publish_a_non_draft_auto_apply_pr(self):
        class UnsafePolicy:
            def disposition(self, patch):
                return RepairDisposition.AUTO_APPLY

        client = FakeRepairClient()
        publication = GitHubRepairCoordinator(client, policy=UnsafePolicy()).publish(
            make_patch("docs/runtime.md"), base_branch="main", base_sha="a" * 40,
        )
        self.assertEqual(publication.disposition, RepairDisposition.REQUIRE_OWNER)
        self.assertTrue(client.created_pr["draft"])

    def test_low_risk_repair_merges_only_after_exact_owner_approval(self):
        with tempfile.TemporaryDirectory() as temp:
            client = FakeRepairClient()
            coordinator = GitHubRepairCoordinator(client)
            patch = make_patch("docs/runtime.md")
            publication = coordinator.publish(
                patch, base_branch="main", base_sha="a" * 40,
            )
            approvals = OwnerApprovalManager(EventStore(Path(temp) / "state", "desktop"))
            grant = coordinator.request_merge_approval(
                publication, patch, approvals=approvals,
            )
            approvals.decide(
                grant.approval_id, peer_id="phone", approved=True, user_verified=True,
            )
            result = coordinator.merge_with_owner_approval(
                publication, patch, approvals=approvals, approval_id=grant.approval_id,
            )
            self.assertEqual(result, "c" * 40)
            self.assertTrue(approvals.get(grant.approval_id).consumed)
            self.assertEqual(client.ready, [publication.pull_number])
            with self.assertRaises(PermissionError):
                coordinator.merge_with_owner_approval(
                    publication, patch, approvals=approvals,
                    approval_id=grant.approval_id,
                )
            self.assertEqual(client.merged, [(publication.pull_number, publication.head_sha)])

    def test_skipped_required_ci_is_not_verification(self):
        client = StaticRunClient([{"name": "CI", "status": "completed", "conclusion": "skipped"}])
        summary = client.workflow_checks("a" * 40, required_workflows=("CI",))
        self.assertFalse(summary.successful)
        self.assertEqual(summary.failed, ("CI",))

    def test_green_checks_from_another_commit_do_not_merge(self):
        client = FakeRepairClient()
        coordinator = GitHubRepairCoordinator(client)
        publication = coordinator.publish(make_patch("docs/runtime.md"), base_branch="main", base_sha="a" * 40)
        stale = GitHubCheckSummary(True, True, (), (), (), "f" * 40)
        self.assertIsNone(coordinator.auto_merge_if_safe(publication, stale))
        self.assertEqual(client.merged, [])

    def test_patch_only_approval_cannot_merge_a_specific_pull_request(self):
        with tempfile.TemporaryDirectory() as temp:
            client = FakeRepairClient()
            coordinator = GitHubRepairCoordinator(client)
            patch = make_patch("docs/runtime.md")
            publication = coordinator.publish(
                patch, base_branch="main", base_sha="a" * 40,
            )
            approvals = OwnerApprovalManager(EventStore(Path(temp) / "state", "desktop"))
            grant = approvals.create("Too broad", patch.digest())
            approvals.decide(
                grant.approval_id, peer_id="phone", approved=True, user_verified=True,
            )
            with self.assertRaises(PermissionError):
                coordinator.merge_with_owner_approval(
                    publication, patch, approvals=approvals,
                    approval_id=grant.approval_id,
                )
            self.assertFalse(approvals.get(grant.approval_id).consumed)
            self.assertEqual(client.merged, [])

    def test_exact_merge_approval_cannot_be_substituted_to_another_head_or_pr(self):
        with tempfile.TemporaryDirectory() as temp:
            client = FakeRepairClient()
            coordinator = GitHubRepairCoordinator(client)
            patch = make_patch("docs/runtime.md")
            publication = coordinator.publish(
                patch, base_branch="main", base_sha="a" * 40,
            )
            approvals = OwnerApprovalManager(EventStore(Path(temp) / "state", "desktop"))
            grant = coordinator.request_merge_approval(
                publication, patch, approvals=approvals,
            )
            approvals.decide(
                grant.approval_id, peer_id="phone", approved=True, user_verified=True,
            )
            for forged in (
                replace(publication, head_sha="d" * 40),
                replace(publication, pull_number=publication.pull_number + 1),
                replace(publication, repository="attacker/other"),
            ):
                with self.subTest(forged=forged), self.assertRaises(PermissionError):
                    coordinator.merge_with_owner_approval(
                        forged, patch, approvals=approvals,
                        approval_id=grant.approval_id,
                    )
                self.assertFalse(approvals.get(grant.approval_id).consumed)
            self.assertEqual(client.merged, [])

    def test_owner_approved_pull_request_cannot_be_retargeted_before_merge(self):
        with tempfile.TemporaryDirectory() as temp:
            client = FakeRepairClient()
            coordinator = GitHubRepairCoordinator(client)
            patch = make_patch("docs/runtime.md")
            publication = coordinator.publish(
                patch, base_branch="main", base_sha="a" * 40,
            )
            approvals = OwnerApprovalManager(EventStore(Path(temp) / "state", "desktop"))
            grant = coordinator.request_merge_approval(
                publication, patch, approvals=approvals,
            )
            approvals.decide(
                grant.approval_id, peer_id="phone", approved=True, user_verified=True,
            )
            client.live_base_ref = "release"

            with self.assertRaises(PermissionError):
                coordinator.merge_with_owner_approval(
                    publication, patch, approvals=approvals,
                    approval_id=grant.approval_id,
                )

            self.assertFalse(approvals.get(grant.approval_id).consumed)
            self.assertEqual(client.ready, [])
            self.assertEqual(client.merged, [])

    def test_owner_approved_pull_request_rejects_an_advanced_base_sha(self):
        with tempfile.TemporaryDirectory() as temp:
            client = FakeRepairClient()
            coordinator = GitHubRepairCoordinator(client)
            patch = make_patch("docs/runtime.md")
            publication = coordinator.publish(
                patch, base_branch="main", base_sha="a" * 40,
            )
            approvals = OwnerApprovalManager(EventStore(Path(temp) / "state", "desktop"))
            grant = coordinator.request_merge_approval(
                publication, patch, approvals=approvals,
            )
            approvals.decide(
                grant.approval_id, peer_id="phone", approved=True, user_verified=True,
            )
            client.live_base_sha = "e" * 40

            with self.assertRaises(PermissionError):
                coordinator.merge_with_owner_approval(
                    publication, patch, approvals=approvals,
                    approval_id=grant.approval_id,
                )

            self.assertFalse(approvals.get(grant.approval_id).consumed)
            self.assertEqual(client.ready, [])
            self.assertEqual(client.merged, [])

    def test_custom_workflow_list_cannot_remove_baseline_merge_gates(self):
        coordinator = GitHubRepairCoordinator(
            FakeRepairClient(), required_workflows=(),
        )
        publication = coordinator.publish(
            make_patch("docs/runtime.md"), base_branch="main", base_sha="a" * 40,
        )
        self.assertEqual(
            set(publication.required_workflows),
            {"CI", "NEXUS architecture contracts"},
        )

    def test_approval_manager_subclass_is_rejected_at_merge_boundaries(self):
        class UnsafeApprovals(OwnerApprovalManager):
            def _row(self, row):
                value = OwnerApprovalManager._row(row)
                return replace(value, state="approved", decided_by="forged")

        with tempfile.TemporaryDirectory() as temp:
            client = FakeRepairClient()
            coordinator = GitHubRepairCoordinator(client)
            patch = make_patch("docs/runtime.md")
            publication = coordinator.publish(
                patch, base_branch="main", base_sha="a" * 40,
            )
            store = EventStore(Path(temp) / "state", "desktop")
            approvals = UnsafeApprovals(store)
            with self.assertRaises(TypeError):
                coordinator.request_merge_approval(
                    publication, patch, approvals=approvals,
                )

            concrete = OwnerApprovalManager(store)
            grant = coordinator.request_merge_approval(
                publication, patch, approvals=concrete,
            )
            concrete.decide(
                grant.approval_id, peer_id="phone", approved=True,
                user_verified=True,
            )
            with self.assertRaises(TypeError):
                coordinator.merge_with_owner_approval(
                    publication, patch, approvals=approvals,
                    approval_id=grant.approval_id,
                )
            self.assertEqual(client.merged, [])

    def test_api_runs_for_another_sha_cannot_satisfy_required_ci(self):
        client = StaticRunClient([{"head_sha": "f" * 40, "name": "CI", "status": "completed", "conclusion": "success"}])
        summary = client.workflow_checks("a" * 40, required_workflows=("CI",))
        self.assertFalse(summary.successful)
        self.assertEqual(summary.missing, ("CI",))

    def test_self_heal_policy_code_is_a_major_repair_surface(self):
        coordinator = GitHubRepairCoordinator(FakeRepairClient())
        publication = coordinator.publish(make_patch("core/github_repair.py"), base_branch="main", base_sha="a" * 40)
        self.assertEqual(publication.disposition, RepairDisposition.REQUIRE_OWNER)

    def test_major_repair_can_merge_only_with_exact_owner_approval_and_fresh_ci(self):
        with tempfile.TemporaryDirectory() as temp:
            client = FakeRepairClient()
            coordinator = GitHubRepairCoordinator(client)
            patch = make_patch("core/update_manager.py")
            publication = coordinator.publish(patch, base_branch="main", base_sha="a" * 40)
            approvals = OwnerApprovalManager(EventStore(Path(temp) / "state", "desktop"))
            grant = coordinator.request_merge_approval(
                publication, patch, approvals=approvals,
            )
            approvals.decide(grant.approval_id, peer_id="phone", approved=True, user_verified=True)
            altered = make_patch("core/update_manager.py", "different")
            with self.assertRaises(PermissionError):
                coordinator.merge_with_owner_approval(publication, altered, approvals=approvals, approval_id=grant.approval_id)
            self.assertFalse(approvals.get(grant.approval_id).consumed)
            result = coordinator.merge_with_owner_approval(publication, patch, approvals=approvals, approval_id=grant.approval_id)
            self.assertEqual(result, "c" * 40)
            self.assertEqual(client.ready, [77])
            self.assertEqual(client.merged, [(77, publication.head_sha)])


if __name__ == "__main__":
    unittest.main()
