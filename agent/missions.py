"""Owner-facing durable single-action missions; no model permission channel."""
from dataclasses import asdict
import hashlib
import json
import threading

from agent.planner import create_plan
from agent.reflex import reflex_plan
from core.action_gateway import create_runtime
from core.authority_contracts import ActionRequest, canonical_arguments
from core.capability_registry import resolve_tool
from core.mission_store import MissionConflict, MissionStore


class MissionService:
    ADMITTED = frozenset({"system.time", "workspace.read", "workspace.create_text"})

    def __init__(self, directory, *, provider=None):
        self.store = MissionStore(directory)
        self._provider = provider
        self._runtimes = {}
        self._active = {}
        self._guard = threading.RLock()

    def close(self):
        with self._guard:
            if self._active:
                raise MissionConflict("Cannot release host lease while actions are in flight.")
            for runtime in self._runtimes.values():
                runtime.gateway.close()
            if self._provider is not None and hasattr(self._provider, "close"):
                self._provider.close()
            self.store.close()

    def _runtime(self, owner):
        with self._guard:
            if owner not in self._runtimes:
                if len(self._runtimes) >= 128:
                    raise MissionConflict("Active owner session budget exhausted.")
                root = self.store.directory / "workspaces" / hashlib.sha256(owner.encode()).hexdigest()
                self._runtimes[owner] = create_runtime(owner_id=owner, workspace_root=root)
            return self._runtimes[owner]

    def create_action(self, owner, tool, arguments, *, goal="Explicit owner action", route="owner"):
        if not isinstance(tool, str) or not 0 < len(tool) <= 120:
            raise ValueError("Invalid tool identity.")
        cid, args = resolve_tool(tool, arguments)
        request = ActionRequest(cid or "unknown." + tool, canonical_arguments(args))
        mid = self.store.create(owner, goal, tool, request, route)
        return self.view(owner, mid)

    def create_goal(self, owner, goal):
        if not isinstance(goal, str) or not 0 < len(goal) <= 4000:
            raise ValueError("Mission goal must contain 1–4000 characters.")
        plan = reflex_plan(goal)
        route = "reflex"
        if plan is None:
            route = "model"
            with self._guard:
                if self._provider is None:
                    from core.providers.ollama import OllamaProvider
                    self._provider = OllamaProvider.from_env()
                provider = self._provider
            plan = create_plan(goal, provider=provider)
        if len(plan.get("steps", [])) != 1:
            raise ValueError("This slice requires exactly one independent action; no plan was executed.")
        step = plan["steps"][0]
        return self.create_action(owner, step["tool"], step["parameters"], goal=goal, route=route)

    def view(self, owner, mid):
        row = self.store.get(owner, mid)
        runtime = self._runtime(owner)
        return {key: row[key] for key in ("id", "goal", "tool", "state", "version", "attempts", "created_at", "updated_at", "request_id", "digest", "cancel_requested")} | {
            "capability_id": row["capability"], "arguments": json.loads(row["arguments"]),
            "session_id": runtime.gateway.context.session_id,
            "completion_scope": "single_action_only",
            "goal_completion": "not_evaluated",
            "receipt_scope": "last_completed_attempt",
            "receipt": json.loads(row["receipt"]) if row["receipt"] else None,
            "events": self.store.events(owner, mid),
        }

    def _checked(self, owner, mid, session_id):
        row = self.store.get(owner, mid)  # owner check precedes session creation
        runtime = self._runtime(owner)
        if runtime.gateway.context.session_id != session_id:
            raise MissionConflict("Owner session changed; review the mission again.")
        return row, runtime

    def approve(self, owner, mid, session_id, digest, *, ttl_seconds=120):
        row, runtime = self._checked(owner, mid, session_id)
        if row["state"] != "waiting_confirmation":
            raise MissionConflict("Mission is not waiting for confirmation.")
        return asdict(runtime.owner.approve(row["request_id"], digest, ttl_seconds=ttl_seconds))

    def revoke(self, owner, mid, session_id, ticket_id):
        _, runtime = self._checked(owner, mid, session_id)
        return runtime.owner.revoke(ticket_id)

    def run(self, owner, mid, session_id, version, *, ticket_id=None):
        with self._guard:
            row, runtime = self._checked(owner, mid, session_id)
            # Reconstruct data only; the kernel rechecks current policy and tickets.
            request = ActionRequest(row["capability"], row["arguments"], row["request_id"])
            if request.arguments_digest != row["digest"]:
                raise MissionConflict("Stored arguments failed integrity check.")
            self.store.claim(owner, mid, version)  # durable before any effect
            cancel = threading.Event()
            self._active[mid] = cancel
        try:
            denial = "" if request.capability_id in self.ADMITTED else "Capability is not admitted in the durable mission slice."
            receipt = runtime.gateway.run_request(request, tool=row["tool"], task_id=mid,
                route=row["route"], ticket_id=ticket_id, cancel_flag=cancel,
                additional_denial=denial, runtime=runtime)
            # Failure to commit deliberately leaves RUNNING; recovery is UNKNOWN.
            self.store.complete(owner, mid, receipt)
            return self.view(owner, mid)
        finally:
            with self._guard:
                self._active.pop(mid, None)

    def cancel(self, owner, mid, session_id, version):
        with self._guard:
            row, runtime = self._checked(owner, mid, session_id)
            updated = self.store.cancel(owner, mid, version)
            if mid in self._active:
                self._active[mid].set()
            runtime.owner.cancel(row["request_id"])
            return self.view(owner, updated["id"])
