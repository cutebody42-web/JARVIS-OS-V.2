"""Cloud-safe policy/consent tests. No model, network, desktop or devices."""
from dataclasses import FrozenInstanceError, replace
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from core.authority_contracts import (
    ActionRequest, AuthorizationDecision as A, PolicyContext, canonical_arguments,
)
from core.capability_registry import REGISTRY, resolve_tool, inventory
from core.owner_kernel import OwnerKernel, MAX_PENDING


class OwnerKernelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.now = 100.0
        self.context = PolicyContext("owner-a", "session-a", Path(self.tmp.name))
        self.kernel = OwnerKernel(self.context, clock=lambda: self.now, wall_clock=lambda: 1000 + self.now)
        self.owner = self.kernel._bootstrap_owner_control()

    def request(self, tool="email_control", args=None):
        args = args if args is not None else {"action": "send", "to": "person@example.com", "subject": "Review", "body": "Exact body"}
        cid, normalized = resolve_tool(tool, args)
        return ActionRequest(cid or "unknown", canonical_arguments(normalized))

    def approve(self, request, **kwargs):
        self.assertIs(self.kernel.authorize(request).decision, A.REQUIRE_CONFIRMATION)
        return self.owner.approve(request.request_id, request.arguments_digest, **kwargs)

    def test_unknown_mutation_denied(self):
        request = ActionRequest("unregistered.mutation", canonical_arguments({"policy": "ALLOW"}))
        self.assertIs(self.kernel.authorize(request).decision, A.DENY)

    def test_safe_registered_read_allowed(self):
        self.assertIs(self.kernel.authorize(self.request("system_time", {})).decision, A.ALLOW)

    def test_high_risk_action_requires_confirmation(self):
        decision = self.kernel.authorize(self.request())
        self.assertIs(decision.decision, A.REQUIRE_CONFIRMATION)
        self.assertIsNone(decision.confirmation_ticket_id)

    def test_exact_ticket_grants_only_one_invocation(self):
        request = self.request()
        ticket = self.approve(request)
        self.assertEqual(ticket.grant.owner_id, self.context.owner_id)
        self.assertEqual(ticket.grant.session_id, self.context.session_id)
        self.assertEqual(ticket.grant.capability_id, request.capability_id)
        self.assertEqual(ticket.grant.arguments_digest, request.arguments_digest)
        allowed = self.kernel.authorize(request, ticket_id=ticket.ticket_id)
        self.assertIs(allowed.decision, A.ALLOW)
        self.assertEqual(allowed.confirmation_ticket_id, ticket.ticket_id)
        self.assertIs(self.kernel.authorize(request, ticket_id=ticket.ticket_id).decision, A.DENY)

    def test_changed_arguments_cannot_use_ticket(self):
        request = self.request()
        ticket = self.approve(request)
        for key, value in (("to", "other@example.com"), ("cc", "other@example.com"),
                           ("bcc", "other@example.com"), ("subject", "Different"), ("body", "Exact body ")):
            with self.subTest(key=key):
                changed = replace(request, arguments_json=canonical_arguments({**request.arguments, key: value}))
                self.assertIs(self.kernel.authorize(changed, ticket_id=ticket.ticket_id).decision, A.DENY)

    def test_changed_capability_or_request_cannot_use_ticket(self):
        request = self.request()
        ticket = self.approve(request)
        for changed in (replace(request, request_id="new"), self.request("system_time", {})):
            self.assertIs(self.kernel.authorize(changed, ticket_id=ticket.ticket_id).decision, A.DENY)

    def test_expired_ticket_fails_on_monotonic_boundary(self):
        request = self.request()
        ticket = self.approve(request, ttl_seconds=2)
        self.now += 2
        self.assertIs(self.kernel.authorize(request, ticket_id=ticket.ticket_id).decision, A.DENY)

    def test_revoked_ticket_fails(self):
        request = self.request()
        ticket = self.approve(request)
        self.assertTrue(self.owner.revoke(ticket.ticket_id))
        self.assertIs(self.kernel.authorize(request, ticket_id=ticket.ticket_id).decision, A.DENY)

    def test_forged_ticket_fails(self):
        self.assertIs(self.kernel.authorize(self.request(), ticket_id="owner-approved").decision, A.DENY)

    def test_new_approval_revokes_old_ticket(self):
        request = self.request()
        old = self.approve(request)
        new = self.owner.approve(request.request_id, request.arguments_digest)
        self.assertIs(self.kernel.authorize(request, ticket_id=old.ticket_id).decision, A.DENY)
        self.assertIs(self.kernel.authorize(request, ticket_id=new.ticket_id).decision, A.ALLOW)

    def test_ticket_cannot_cross_owner_or_session(self):
        request = self.request()
        ticket = self.approve(request)
        for ctx in (replace(self.context, owner_id="other"), replace(self.context, session_id="other")):
            other = OwnerKernel(ctx)
            self.assertIs(other.authorize(request, ticket_id=ticket.ticket_id).decision, A.DENY)

    def test_owner_identity_and_digest_are_required(self):
        request = self.request()
        self.kernel.authorize(request)
        with self.assertRaises(PermissionError):
            self.kernel._approve(object(), request.request_id, request.arguments_digest, 120)
        with self.assertRaises(PermissionError):
            self.owner.approve(request.request_id, "changed-digest")

    def test_ttl_cannot_be_unlimited_or_model_coerced(self):
        request = self.request()
        self.kernel.authorize(request)
        for ttl in (0, -1, 301, math.inf, math.nan, True, "120"):
            with self.subTest(ttl=ttl), self.assertRaises(ValueError):
                self.owner.approve(request.request_id, request.arguments_digest, ttl_seconds=ttl)

    def test_expired_pending_request_cannot_be_approved(self):
        request = self.request()
        self.kernel.authorize(request)
        self.now += 300
        with self.assertRaises(PermissionError):
            self.owner.approve(request.request_id, request.arguments_digest)

    def test_close_and_cancel_revoke_outstanding_authority(self):
        request = self.request()
        ticket = self.approve(request)
        self.assertTrue(self.owner.cancel(request.request_id))
        self.assertIs(self.kernel.authorize(request, ticket_id=ticket.ticket_id).decision, A.DENY)
        self.kernel.close()
        self.assertIs(self.kernel.authorize(self.request("system_time", {})).decision, A.DENY)
        with self.assertRaises(PermissionError):
            self.owner.approve(request.request_id, request.arguments_digest)

    def test_policy_and_requests_are_immutable_data(self):
        request = self.request()
        data = request.arguments
        data["body"] = "changed"
        self.assertEqual(request.arguments["body"], "Exact body")
        with self.assertRaises(FrozenInstanceError):
            self.context.owner_id = "attacker"
        with self.assertRaises(TypeError):
            REGISTRY["new"] = REGISTRY["system.time"]
        with self.assertRaises(FrozenInstanceError):
            REGISTRY["system.time"].capability.privilege_requirement = "admin"

    def test_normalization_is_stable_and_preserves_content(self):
        left = self.request()
        args = dict(reversed(list(left.arguments.items())))
        args["to"] = "  person@example.com  "
        right = self.request("email_control", {"action": "send", **args})
        self.assertEqual(left.arguments_digest, right.arguments_digest)
        with self.assertRaises(ValueError):
            self.request("email_control", {"action": "send", "to": ",,,", "subject": "a", "body": "b"})

    def test_pending_ledger_has_capacity_and_expires(self):
        for _ in range(MAX_PENDING):
            self.assertIs(self.kernel.authorize(self.request()).decision, A.REQUIRE_CONFIRMATION)
        self.assertIs(self.kernel.authorize(self.request()).decision, A.DENY)
        self.now += 301
        self.assertIs(self.kernel.authorize(self.request()).decision, A.REQUIRE_CONFIRMATION)

    def test_registry_declares_every_required_field(self):
        required = {"capability_id", "access", "external_side_effect", "sensitive_data_access",
                    "destructive_level", "privilege_requirement", "reversible", "verifier_required",
                    "confirmation_policy", "execution_surface", "resource_lock", "risk"}
        for row in inventory():
            self.assertTrue(required <= row.keys())
        self.assertEqual(REGISTRY["task.cancel"].capability.access, "mutate")

    def test_external_query_egress_also_requires_consent(self):
        for tool, args in (("web_search", {"query": "private text"}), ("weather_report", {"city": "home address"})):
            self.assertIs(self.kernel.authorize(self.request(tool, args)).decision, A.REQUIRE_CONFIRMATION)

    def test_workspace_cannot_be_the_source_tree(self):
        context = replace(self.context, workspace_root=Path(__file__).resolve().parents[1])
        kernel = OwnerKernel(context)
        request = self.request("file_controller", {"action": "write", "path": "config/policy.json", "content": "ALLOW"})
        self.assertIs(kernel.authorize(request).decision, A.DENY)


if __name__ == "__main__":
    unittest.main()
