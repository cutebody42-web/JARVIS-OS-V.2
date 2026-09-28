"""Evidence-bearing action results. Model prose is never verification evidence."""

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
import json
import math


class ActionStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DENIED = "denied"
    CANCELLED = "cancelled"
    UNVERIFIED = "unverified"


def _aware_timestamp(value: str) -> None:
    if datetime.fromisoformat(value).utcoffset() is None:
        raise ValueError("Receipt timestamps must include a timezone.")


@dataclass(frozen=True)
class Evidence:
    source: str
    observation: str
    observed_at: str

    def __post_init__(self):
        if not self.source.strip() or not self.observation.strip():
            raise ValueError("Evidence needs a source and an observation.")
        _aware_timestamp(self.observed_at)


@dataclass(frozen=True)
class ToolResult:
    status: ActionStatus
    message: str
    evidence: tuple[Evidence, ...] = ()
    error_code: str = ""

    def __post_init__(self):
        if not isinstance(self.status, ActionStatus):
            raise TypeError("status must be an ActionStatus.")
        if not isinstance(self.evidence, tuple) or any(not isinstance(e, Evidence) for e in self.evidence):
            raise TypeError("Evidence must be an immutable tuple of Evidence records.")
        if self.status is ActionStatus.SUCCEEDED and not self.evidence:
            raise ValueError("Success requires evidence from a trusted action/verifier.")

    def to_dict(self) -> dict:
        return {**asdict(self), "status": self.status.value}


@dataclass(frozen=True)
class ActionReceipt:
    action_id: str
    task_id: str
    step_id: str
    tool: str
    parameters_json: str
    route: str
    started_at: str
    finished_at: str
    duration_ms: float
    result: ToolResult
    schema_version: int = 1

    def __post_init__(self):
        if not self.action_id or not self.task_id or not self.tool:
            raise ValueError("Receipts require action, task, and tool identities.")
        if self.route not in {"reflex", "model"}:
            raise ValueError("Unknown cognition route.")
        _aware_timestamp(self.started_at)
        _aware_timestamp(self.finished_at)
        if not math.isfinite(self.duration_ms) or self.duration_ms < 0:
            raise ValueError("duration_ms must be finite and nonnegative.")
        if not isinstance(json.loads(self.parameters_json), dict):
            raise ValueError("Receipt parameters must be a JSON object.")
        if not isinstance(self.result, ToolResult):
            raise TypeError("Receipts require a ToolResult.")

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["parameters"] = json.loads(payload.pop("parameters_json"))
        payload["result"] = self.result.to_dict()
        return payload
