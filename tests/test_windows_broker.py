"""Offline contracts for the signed Windows broker boundary."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from core.authority_contracts import (
    ActionRequest,
    AuthorizationDecision,
    AuthorizationResult,
    canonical_arguments,
)
from core.nexus.peer_auth import DeviceSigner
from core.windows_broker import (
    BrokerPermitIssuer,
    BrokerReplayStore,
    BrokerStatus,
    SignedBrokerRequest,
    WindowsBrokerService,
)


NOW = datetime(2026, 9, 30, 13, 0, tzinfo=timezone.utc)


def authorized_request(capability_id, arguments):
    request = ActionRequest(capability_id, canonical_arguments(arguments))
    auth = AuthorizationResult(
        AuthorizationDecision.ALLOW,
        "owner.exact_consent",
        "approved",
        request.capability_id,
        request.arguments_digest,
    )
    return request, auth


class WindowsBrokerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.signer = DeviceSigner.generate()
        self.issuer = BrokerPermitIssuer(
            self.signer,
            "owner",
            "session",
            clock=lambda: NOW,
        )
        self.ledger = BrokerReplayStore(Path(self.tmp.name) / "broker.db")

    def tearDown(self):
        self.tmp.cleanup()

    def service(self, adapters=None, *, certified=True, clock=lambda: NOW):
        return WindowsBrokerService(
            self.signer.public_b64,
            self.ledger,
            adapters or {},
            clock=clock,
            hardware_certified=certified,
        )

    def test_typed_volume_capability_executes_once_when_certified(self):
        request, auth = authorized_request(
            "windows.audio.set_volume",
            {"level": 42},
        )
        signed = self.issuer.issue(request, auth)
        adapter = Mock(return_value="volume set to 42")
        service = self.service({"windows.audio.set_volume": adapter})

        first = service.handle(signed)
        second = service.handle(signed)

        self.assertEqual(first.status, BrokerStatus.SUCCEEDED)
        self.assertEqual(second.status, BrokerStatus.SUCCEEDED)
        adapter.assert_called_once_with({"level": 42})

    def test_hardware_gate_blocks_valid_permit_before_adapter(self):
        request, auth = authorized_request(
            "windows.audio.set_volume",
            {"level": 50},
        )
        signed = self.issuer.issue(request, auth)
        adapter = Mock(return_value="should not run")

        result = self.service(
            {"windows.audio.set_volume": adapter},
            certified=False,
        ).handle(signed)

        self.assertEqual(result.status, BrokerStatus.DENIED)
        adapter.assert_not_called()

    def test_tampered_arguments_are_denied(self):
        request, auth = authorized_request(
            "windows.display.set_brightness",
            {"level": 30},
        )
        signed = self.issuer.issue(request, auth)
        tampered = SignedBrokerRequest(
            signed.permit,
            canonical_arguments({"level": 99}),
            signed.signature,
        )
        adapter = Mock(return_value="brightness")

        result = self.service({
            "windows.display.set_brightness": adapter
        }).handle(tampered)

        self.assertEqual(result.status, BrokerStatus.DENIED)
        adapter.assert_not_called()

    def test_expired_permit_is_denied(self):
        request, auth = authorized_request(
            "windows.power.restart",
            {},
        )
        signed = self.issuer.issue(request, auth, ttl_seconds=10)
        adapter = Mock(return_value="restart scheduled")

        result = self.service(
            {"windows.power.restart": adapter},
            clock=lambda: NOW + timedelta(seconds=11),
        ).handle(signed)

        self.assertEqual(result.status, BrokerStatus.DENIED)
        adapter.assert_not_called()

    def test_unregistered_shell_like_capability_cannot_get_permit(self):
        request, auth = authorized_request(
            "windows.shell.execute",
            {"command": "whoami"},
        )
        with self.assertRaises(PermissionError):
            self.issuer.issue(request, auth)

    def test_invalid_volume_schema_is_rejected_before_signing(self):
        request, auth = authorized_request(
            "windows.audio.set_volume",
            {"level": 500},
        )
        with self.assertRaises(ValueError):
            self.issuer.issue(request, auth)

    def test_non_allowed_authorization_cannot_issue_broker_permit(self):
        request = ActionRequest(
            "windows.audio.set_volume",
            canonical_arguments({"level": 20}),
        )
        denied = AuthorizationResult(
            AuthorizationDecision.DENY,
            "policy.denied",
            "denied",
            request.capability_id,
            request.arguments_digest,
        )
        with self.assertRaises(PermissionError):
            self.issuer.issue(request, denied)

    def test_missing_adapter_returns_unavailable_and_replay_is_stable(self):
        request, auth = authorized_request(
            "windows.process.launch_registered",
            {"app_id": "notepad"},
        )
        signed = self.issuer.issue(request, auth)
        service = self.service({})

        first = service.handle(signed)
        second = service.handle(signed)

        self.assertEqual(first.status, BrokerStatus.UNAVAILABLE)
        self.assertEqual(second.status, BrokerStatus.UNAVAILABLE)


if __name__ == "__main__":
    unittest.main()
