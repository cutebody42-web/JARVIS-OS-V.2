"""NEXUS Phase 1 executor: model proposals, owner policy, evidence and bounded recovery."""

import threading
from typing import Callable
from uuid import uuid4

from agent.action_kernel import run_action
from agent.planner import create_plan, replan
from agent.reflex import reflex_plan
from core.action_contracts import ActionReceipt, ActionStatus
from core.model_provider import ModelProvider


class AgentExecutor:
    MAX_REPLAN_ATTEMPTS = 2

    def __init__(self, awareness=None, *, provider: ModelProvider | None = None, owner_runtime=None):
        self.awareness = awareness
        self.provider = provider
        from core.action_gateway import current_runtime
        self.owner_runtime = owner_runtime or current_runtime()
        self.last_step_results: dict = {}
        self.last_action_receipts: list[ActionReceipt] = []
        self.last_status = ActionStatus.UNVERIFIED
        self._execution_lock = threading.Lock()

    def _awareness(self, method, *args):
        callback = getattr(self.awareness, method, None)
        if callback:
            try:
                callback(*args)
            except Exception:
                pass  # UI/awareness failure must not alter action authorization.

    def execute(self, goal: str, speak: Callable | None = None,
                cancel_flag: threading.Event | None = None) -> str:
        # Queue uses one executor per task; direct reuse is serialized as well.
        with self._execution_lock:
            self.last_step_results = {}
            self.last_action_receipts = []
            self.last_status = ActionStatus.UNVERIFIED
            self._awareness("set_goal", goal)
            try:
                message = self._execute(goal, cancel_flag)
            finally:
                self._awareness("clear_active_tool")
            if speak:
                speak(message)
            return message

    def _execute(self, goal, cancel_flag):
        task_id = str(uuid4())
        if cancel_flag is not None and cancel_flag.is_set():
            self.last_status = ActionStatus.CANCELLED
            return "Task cancelled before planning."
        local = reflex_plan(goal)
        route = "reflex" if local is not None else "model"
        plan = local if local is not None else create_plan(goal, provider=self.provider)
        completed = []
        for recovery_attempt in range(self.MAX_REPLAN_ATTEMPTS + 1):
            if cancel_flag is not None and cancel_flag.is_set():
                self.last_status = ActionStatus.CANCELLED
                return "Task cancelled."
            if not plan.get("steps"):
                self.last_status = ActionStatus.FAILED
                return "No valid admitted plan is available; the task is not complete."
            failed_step = None
            for step in plan["steps"]:
                self._awareness("set_active_tool", step["tool"], step.get("description", ""))
                receipt = run_action(step["tool"], step["parameters"], task_id=task_id,
                                     step_id=str(step["step"]), route=route, cancel_flag=cancel_flag,
                                     owner_runtime=self.owner_runtime)
                self.last_action_receipts.append(receipt)
                self.last_status = receipt.result.status
                self.last_step_results[len(self.last_action_receipts)] = receipt.result.message
                self._awareness("record_event", f"{receipt.tool}: {receipt.result.status.value}")
                if receipt.result.status is ActionStatus.SUCCEEDED:
                    completed.append(step)
                    continue
                if receipt.result.status is ActionStatus.FAILED and route != "reflex":
                    failed_step = step
                    break
                # A denial, cancellation or unverified outcome cannot be repaired
                # by a model, retried for extra side effects, or described as done.
                return self._summarize()
            if failed_step is None:
                return self._summarize()
            if recovery_attempt < self.MAX_REPLAN_ATTEMPTS:
                if cancel_flag is not None and cancel_flag.is_set():
                    self.last_status = ActionStatus.CANCELLED
                    return "Task cancelled before recovery."
                plan = replan(goal, completed, failed_step,
                              self.last_action_receipts[-1].result.error_code, provider=self.provider)
        return self._summarize()

    def _summarize(self) -> str:
        # Deterministic receipt rendering. No LLM can promote an outcome to success.
        receipt = self.last_action_receipts[-1]
        result = receipt.result
        if result.status is ActionStatus.SUCCEEDED:
            if len(self.last_action_receipts) == 1:
                return result.message
            return "Verified results: " + " | ".join(
                r.result.message for r in self.last_action_receipts
                if r.result.status is ActionStatus.SUCCEEDED
            )
        if result.status is ActionStatus.UNVERIFIED:
            return f"Unverified tool output; completion is not confirmed: {result.message}"
        return f"Task {result.status.value}: {result.message}"
