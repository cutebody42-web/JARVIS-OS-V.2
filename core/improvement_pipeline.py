"""Bounded improvement proposal/evaluation ledger.

NEXUS may record an immutable candidate commit and collect CI evidence. This
module deliberately has no merge, checkout, patch-apply, self-update or code
execution primitive. A candidate can become READY_FOR_OWNER, never self-applied.

This is storage only: recorded check evidence is supplied by a caller, not
fetched or authenticated here. Readiness never authorizes an update or merge;
GitHubRepairCoordinator must still verify fresh CI and any owner approval.
The desktop host does not automatically create or apply these proposals.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import re
from pathlib import Path
import sqlite3
from typing import Mapping
from uuid import uuid4


_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_CHECK_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_. /:-]{0,119}$")


class ImprovementState(str, Enum):
    EVALUATING = "evaluating"
    READY_FOR_OWNER = "ready_for_owner"
    FAILED = "failed"


class CheckConclusion(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass(frozen=True)
class CheckEvidence:
    check_name: str
    conclusion: CheckConclusion
    workflow_run_id: int
    artifact_digest: str | None = None


@dataclass(frozen=True)
class ImprovementProposal:
    proposal_id: str
    objective: str
    candidate_commit: str
    required_checks: tuple[str, ...]
    created_at: str
    state: ImprovementState
    evidence: Mapping[str, CheckEvidence]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class ImprovementLedger:
    """Durable evidence ledger with no authority to mutate source control."""

    def __init__(self, directory: str | Path):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        self.db_path = root / "improvements.sqlite"
        self._init_db()

    def _connect(self):
        db = sqlite3.connect(self.db_path, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        return db

    def _init_db(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS improvement_proposals (
                    proposal_id TEXT PRIMARY KEY,
                    objective TEXT NOT NULL,
                    candidate_commit TEXT NOT NULL,
                    required_checks_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS improvement_checks (
                    proposal_id TEXT NOT NULL,
                    check_name TEXT NOT NULL,
                    conclusion TEXT NOT NULL CHECK(conclusion IN ('success','failure')),
                    workflow_run_id INTEGER NOT NULL,
                    artifact_digest TEXT,
                    PRIMARY KEY(proposal_id, check_name),
                    FOREIGN KEY(proposal_id) REFERENCES improvement_proposals(proposal_id)
                );
                """
            )

    @staticmethod
    def _validate_check(name: str) -> str:
        if not isinstance(name, str) or not _CHECK_RE.fullmatch(name):
            raise ValueError("invalid required check name")
        return name

    def propose(
        self,
        *,
        objective: str,
        candidate_commit: str,
        required_checks: tuple[str, ...],
    ) -> ImprovementProposal:
        if not isinstance(objective, str) or not 1 <= len(objective.strip()) <= 4000:
            raise ValueError("objective must contain 1-4000 characters")
        if not isinstance(candidate_commit, str) or not _COMMIT_RE.fullmatch(candidate_commit):
            raise ValueError("candidate_commit must be an immutable 40-character git SHA")
        checks = tuple(dict.fromkeys(self._validate_check(name) for name in required_checks))
        if not checks:
            raise ValueError("at least one independent check is required")

        import json

        proposal_id = uuid4().hex
        created = _now()
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO improvement_proposals(
                    proposal_id, objective, candidate_commit,
                    required_checks_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    proposal_id,
                    objective.strip(),
                    candidate_commit,
                    json.dumps(checks, separators=(",", ":")),
                    created,
                ),
            )
        return self.get(proposal_id)

    def record_check(
        self,
        proposal_id: str,
        *,
        check_name: str,
        conclusion: CheckConclusion,
        workflow_run_id: int,
        artifact_digest: str | None = None,
    ) -> ImprovementProposal:
        name = self._validate_check(check_name)
        if not isinstance(conclusion, CheckConclusion):
            raise TypeError("conclusion must be CheckConclusion")
        if isinstance(workflow_run_id, bool) or not isinstance(workflow_run_id, int) or workflow_run_id <= 0:
            raise ValueError("workflow_run_id must be a positive integer")
        if artifact_digest is not None and (
            not isinstance(artifact_digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", artifact_digest)
        ):
            raise ValueError("artifact_digest must be a sha256 hex digest")

        proposal = self.get(proposal_id)
        if name not in proposal.required_checks:
            raise ValueError("check is not part of this proposal's required evidence")

        with self._connect() as db:
            db.execute(
                """
                INSERT INTO improvement_checks(
                    proposal_id, check_name, conclusion,
                    workflow_run_id, artifact_digest
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(proposal_id, check_name) DO UPDATE SET
                    conclusion=excluded.conclusion,
                    workflow_run_id=excluded.workflow_run_id,
                    artifact_digest=excluded.artifact_digest
                """,
                (
                    proposal_id,
                    name,
                    conclusion.value,
                    workflow_run_id,
                    artifact_digest,
                ),
            )
        return self.get(proposal_id)

    def get(self, proposal_id: str) -> ImprovementProposal:
        import json

        with self._connect() as db:
            proposal = db.execute(
                "SELECT * FROM improvement_proposals WHERE proposal_id=?",
                (proposal_id,),
            ).fetchone()
            if proposal is None:
                raise KeyError("Unknown improvement proposal")
            rows = db.execute(
                "SELECT * FROM improvement_checks WHERE proposal_id=?",
                (proposal_id,),
            ).fetchall()

        required = tuple(json.loads(proposal["required_checks_json"]))
        evidence = {
            row["check_name"]: CheckEvidence(
                row["check_name"],
                CheckConclusion(row["conclusion"]),
                row["workflow_run_id"],
                row["artifact_digest"],
            )
            for row in rows
        }

        if any(item.conclusion is CheckConclusion.FAILURE for item in evidence.values()):
            state = ImprovementState.FAILED
        elif all(
            name in evidence and evidence[name].conclusion is CheckConclusion.SUCCESS
            for name in required
        ):
            state = ImprovementState.READY_FOR_OWNER
        else:
            state = ImprovementState.EVALUATING

        return ImprovementProposal(
            proposal["proposal_id"],
            proposal["objective"],
            proposal["candidate_commit"],
            required,
            proposal["created_at"],
            state,
            evidence,
        )

    def release_candidate(self, proposal_id: str) -> str:
        """Return a reviewed candidate ref; never mutates source control."""
        proposal = self.get(proposal_id)
        if proposal.state is not ImprovementState.READY_FOR_OWNER:
            raise PermissionError("candidate is not ready for owner review")
        return proposal.candidate_commit
