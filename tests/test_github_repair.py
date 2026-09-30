"""Offline contracts for GitHub-native bounded JARVIS self-healing."""

import hashlib
import unittest

from core.github_repair import (
    GitHubCheckSummary,
    GitHubRepairClient,
    GitHubRepairCoordinator,
)
from core.self_heal import FileEdit, RepairDisposition, RepairPatch


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
        self.head_sha = "b" * 40

    def create_branch(self, branch, base_sha):
        self.created_branch = (branch, base_sha)
        return branch

    def file(self, path, ref):
        return "blob-sha", "old"

    def update_file(self, **kwargs):
        return self.head_sha

    def create_pull_request(self, **kwargs):
        self.created_pr = kwargs
        return 77, "https://example.invalid/pr/77"

    def merge_pull_request(self, pull_number, expected_head_sha):
        self.merged.append((pull_number, expected_head_sha))
        return "c" * 40


class StaticRunClient(GitHubRepairClient):
    def __init__(self, runs):
        super().__init__(
            "cutebody42-web/JARVIS-OS-V.2",
            token="offline-test-placeholder-value",
        )
        self.runs = runs

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

        self.assertEqual(publication.disposition, RepairDisposition.AUTO_APPLY)
        self.assertEqual(
            set(publication.required_workflows),
            {
                "CI",
                "NEXUS architecture contracts",
                "JARVIS product shell",
                "Build JARVIS Android companion",
                "Build JARVIS Windows installer",
            },
        )
        self.assertFalse(client.created_pr["draft"])

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

    def test_low_risk_repair_merges_only_after_complete_success(self):
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

        green = GitHubCheckSummary(True, True, (), (), ())
        merged = coordinator.auto_merge_if_safe(publication, green)

        self.assertEqual(merged, "c" * 40)
        self.assertEqual(
            client.merged,
            [(77, publication.head_sha)],
        )


if __name__ == "__main__":
    unittest.main()
