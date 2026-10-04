"""Tests for bounded evidence-only NEXUS improvement proposals."""

from pathlib import Path
import tempfile
import unittest

from core.improvement_pipeline import (
    CheckConclusion,
    ImprovementLedger,
    ImprovementState,
)


SHA = "a" * 40


class ImprovementLedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ledger = ImprovementLedger(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_candidate_is_immutable_commit_and_starts_evaluating(self):
        proposal = self.ledger.propose(
            objective="Reduce local model cold-start latency",
            candidate_commit=SHA,
            required_checks=("CI", "NEXUS architecture contracts"),
        )
        self.assertEqual(proposal.state, ImprovementState.EVALUATING)
        self.assertEqual(proposal.candidate_commit, SHA)

        for invalid in ("main", "feature/test", "abc123"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.ledger.propose(
                    objective="x",
                    candidate_commit=invalid,
                    required_checks=("CI",),
                )

    def test_all_required_successes_only_make_candidate_ready_for_owner(self):
        proposal = self.ledger.propose(
            objective="Improve routing",
            candidate_commit=SHA,
            required_checks=("CI", "benchmark"),
        )
        self.ledger.record_check(
            proposal.proposal_id,
            check_name="CI",
            conclusion=CheckConclusion.SUCCESS,
            workflow_run_id=100,
        )
        half = self.ledger.get(proposal.proposal_id)
        self.assertEqual(half.state, ImprovementState.EVALUATING)

        done = self.ledger.record_check(
            proposal.proposal_id,
            check_name="benchmark",
            conclusion=CheckConclusion.SUCCESS,
            workflow_run_id=101,
            artifact_digest="b" * 64,
        )
        self.assertEqual(done.state, ImprovementState.READY_FOR_OWNER)
        self.assertEqual(
            self.ledger.release_candidate(proposal.proposal_id),
            SHA,
        )

    def test_any_required_failure_blocks_candidate(self):
        proposal = self.ledger.propose(
            objective="test",
            candidate_commit=SHA,
            required_checks=("CI",),
        )
        failed = self.ledger.record_check(
            proposal.proposal_id,
            check_name="CI",
            conclusion=CheckConclusion.FAILURE,
            workflow_run_id=200,
        )
        self.assertEqual(failed.state, ImprovementState.FAILED)
        with self.assertRaises(PermissionError):
            self.ledger.release_candidate(proposal.proposal_id)

    def test_unregistered_evidence_cannot_promote_candidate(self):
        proposal = self.ledger.propose(
            objective="test",
            candidate_commit=SHA,
            required_checks=("CI",),
        )
        with self.assertRaises(ValueError):
            self.ledger.record_check(
                proposal.proposal_id,
                check_name="untrusted self-test",
                conclusion=CheckConclusion.SUCCESS,
                workflow_run_id=300,
            )

    def test_candidate_and_partial_evidence_survive_reopening(self):
        proposal = self.ledger.propose(
            objective="Evaluate a routing improvement",
            candidate_commit=SHA,
            required_checks=("CI", "benchmark"),
        )
        self.ledger.record_check(
            proposal.proposal_id,
            check_name="CI",
            conclusion=CheckConclusion.SUCCESS,
            workflow_run_id=401,
            artifact_digest="c" * 64,
        )

        reopened = ImprovementLedger(Path(self.tmp.name))
        stored = reopened.get(proposal.proposal_id)
        self.assertEqual(stored.candidate_commit, SHA)
        self.assertEqual(stored.required_checks, ("CI", "benchmark"))
        self.assertEqual(stored.state, ImprovementState.EVALUATING)
        self.assertEqual(stored.evidence["CI"].workflow_run_id, 401)
        self.assertEqual(stored.evidence["CI"].artifact_digest, "c" * 64)
        with self.assertRaises(PermissionError):
            reopened.release_candidate(proposal.proposal_id)


if __name__ == "__main__":
    unittest.main()
