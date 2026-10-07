import unittest
from unittest.mock import Mock, patch

from core.github_repair import GitHubCheckSummary, GitHubRepairPublication
from core.model_provider import ModelResponse
from core.model_repair_workflow import ModelAssistedRepairWorkflow, ModelRepairProposal
from core.self_heal import Incident, IncidentKind, RepairDisposition


class FakeProvider:
    def generate(self, request):
        return ModelResponse(
            text='{"schema":"jarvis.repair-proposal.v1","rationale":"Bounded fix.","edits":[{"path":"core/example.py","replacement":"VALUE = 2\\n"}],"requested_tests":["tests.test_example"]}',
            provider="test",
            model="repair-test",
        )


class FakeCoordinator:
    def __init__(self, disposition=RepairDisposition.REQUIRE_OWNER):
        self.disposition = disposition
        self.published = []
        self.waited = []
        self.auto_calls = []
        self.owner_calls = []

    def publish(self, patch, *, base_branch, base_sha):
        self.published.append((patch, base_branch, base_sha))
        return GitHubRepairPublication(
            branch="jarvis/self-heal/test",
            pull_number=99,
            pull_url="https://github.com/example/repo/pull/99",
            head_sha="b" * 40,
            disposition=self.disposition,
            required_workflows=("CI",),
            patch_digest=patch.digest(),
        )

    def wait_for_ci(self, publication, *, timeout_seconds, poll_seconds):
        self.waited.append((publication, timeout_seconds, poll_seconds))
        return GitHubCheckSummary(True, True, (), (), (), publication.head_sha)

    def auto_merge_if_safe(self, publication, checks):
        self.auto_calls.append((publication, checks))
        return "c" * 40 if publication.disposition is RepairDisposition.AUTO_APPLY else None

    def merge_with_owner_approval(self, publication, patch, *, approvals, approval_id):
        self.owner_calls.append((publication, patch, approvals, approval_id))
        return "d" * 40


class ModelRepairWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.incident = Incident.create(
            IncidentKind.CODE_FAILURE,
            "VALUE should be two",
            "core/example",
        )

    @patch("core.model_repair_workflow.GitHubRepairCoordinator", new=FakeCoordinator)
    def test_proposal_is_published_through_coordinator(self):
        coordinator = FakeCoordinator()
        workflow = ModelAssistedRepairWorkflow(FakeProvider(), coordinator)
        proposal = workflow.propose_and_publish(
            self.incident,
            files={"core/example.py": "VALUE = 1\n"},
            base_branch="main",
            base_sha="a" * 40,
        )
        self.assertTrue(proposal.requires_owner)
        self.assertEqual(proposal.publication.pull_number, 99)
        patch, branch, sha = coordinator.published[0]
        self.assertEqual(branch, "main")
        self.assertEqual(sha, "a" * 40)
        self.assertEqual(patch.edits[0].path, "core/example.py")

    @patch("core.model_repair_workflow.GitHubRepairCoordinator", new=FakeCoordinator)
    def test_verify_delegates_to_exact_ci_gate(self):
        coordinator = FakeCoordinator()
        workflow = ModelAssistedRepairWorkflow(FakeProvider(), coordinator)
        proposal = workflow.propose_and_publish(
            self.incident,
            files={"core/example.py": "VALUE = 1\n"},
            base_branch="main",
            base_sha="a" * 40,
        )
        checks = workflow.verify(proposal, timeout_seconds=12, poll_seconds=1)
        self.assertTrue(checks.successful)
        self.assertEqual(coordinator.waited[0][1:], (12, 1))

    @patch("core.model_repair_workflow.GitHubRepairCoordinator", new=FakeCoordinator)
    def test_low_risk_merge_still_uses_coordinator_policy(self):
        coordinator = FakeCoordinator(RepairDisposition.AUTO_APPLY)
        workflow = ModelAssistedRepairWorkflow(FakeProvider(), coordinator)
        proposal = workflow.propose_and_publish(
            self.incident,
            files={"core/example.py": "VALUE = 1\n"},
            base_branch="main",
            base_sha="a" * 40,
        )
        checks = GitHubCheckSummary(True, True, (), (), (), proposal.publication.head_sha)
        self.assertEqual(workflow.auto_merge_if_safe(proposal, checks), "c" * 40)
        self.assertEqual(len(coordinator.auto_calls), 1)

    @patch("core.model_repair_workflow.GitHubRepairCoordinator", new=FakeCoordinator)
    def test_major_merge_requires_explicit_owner_gate(self):
        coordinator = FakeCoordinator(RepairDisposition.REQUIRE_OWNER)
        workflow = ModelAssistedRepairWorkflow(FakeProvider(), coordinator)
        proposal = workflow.propose_and_publish(
            self.incident,
            files={"core/example.py": "VALUE = 1\n"},
            base_branch="main",
            base_sha="a" * 40,
        )
        approvals = Mock()
        result = workflow.merge_with_owner_approval(
            proposal,
            approvals=approvals,
            approval_id="approval-1",
        )
        self.assertEqual(result, "d" * 40)
        self.assertEqual(coordinator.owner_calls[0][3], "approval-1")

    @patch("core.model_repair_workflow.GitHubRepairCoordinator", new=FakeCoordinator)
    def test_workflow_has_no_implicit_publish_or_merge_on_construction(self):
        coordinator = FakeCoordinator()
        ModelAssistedRepairWorkflow(FakeProvider(), coordinator)
        self.assertEqual(coordinator.published, [])
        self.assertEqual(coordinator.auto_calls, [])
        self.assertEqual(coordinator.owner_calls, [])


if __name__ == "__main__":
    unittest.main()
