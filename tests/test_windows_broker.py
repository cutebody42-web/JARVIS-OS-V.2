"""Cloud tests for the typed fail-closed Windows broker core."""

from datetime import datetime, timedelta, timezone
from dataclasses import replace
import unittest

from core.windows_broker import (
    BrokerAuthorizationError,
    BrokerCapabilitySpec,
    BrokerDisabledError,
    BrokerPermitIssuer,
    BrokerRegistry,
    WindowsBrokerCore,
)


class Clock:
    def __init__(self):
        self.now = datetime(2026, 9, 30, 14, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += timedelta(seconds=seconds)


class Adapter:
    def __init__(self):
        self.calls = []

    def invoke(self, spec, arguments):
        self.calls.append((spec.capability_id, dict(arguments)))
        return "verified adapter result"


class WindowsBrokerTests(unittest.TestCase):
    def setUp(self):
        self.secret = b"x" * 32
        self.clock = Clock()
        self.spec = BrokerCapabilitySpec(
            "system.settings.display.brightness",
            "display_brightness_v1",
            requires_elevation=False,
        )
        self.registry = BrokerRegistry({self.spec.capability_id: self.spec})
        self.adapter = Adapter()
        self.issuer = BrokerPermitIssuer(self.secret, clock=self.clock)
        self.broker = WindowsBrokerCore(
            self.secret,
            self.registry,
            self.adapter,
            clock=self.clock,
        )

    def permit(self):
        return self.issuer.issue(
            request_id="request-1",
            capability_id=self.spec.capability_id,
            arguments={"percent": 70},
            receipt_id="receipt-1",
        )

    def test_valid_typed_permit_executes_registered_adapter(self):
        result = self.broker.execute(self.permit())
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(
            self.adapter.calls,
            [(self.spec.capability_id, {"percent": 70})],
        )

    def test_unknown_capability_is_denied_without_adapter_call(self):
        permit = self.issuer.issue(
            request_id="request-2",
            capability_id="system.shell",
            arguments={"command": "whoami"},
            receipt_id="receipt-2",
        )
        with self.assertRaises(BrokerAuthorizationError):
            self.broker.execute(permit)
        self.assertEqual(self.adapter.calls, [])

    def test_tamper_replay_expiry_and_kill_switch_fail_closed(self):
        permit = self.permit()
        tampered = replace(permit, arguments_json='{"percent":100}')
        with self.assertRaises(BrokerAuthorizationError):
            self.broker.execute(tampered)

        good = self.permit()
        self.broker.execute(good)
        with self.assertRaises(BrokerAuthorizationError):
            self.broker.execute(good)

        expired = self.permit()
        self.clock.advance(31)
        with self.assertRaises(BrokerAuthorizationError):
            self.broker.execute(expired)

        self.broker.set_enabled(False)
        with self.assertRaises(BrokerDisabledError):
            self.broker.execute(self.permit())

    def test_generic_command_adapter_names_are_forbidden(self):
        for adapter_id in ("shell", "powershell_admin", "cmd_exec"):
            with self.subTest(adapter_id=adapter_id), self.assertRaises(ValueError):
                BrokerCapabilitySpec(
                    "danger",
                    adapter_id,
                    requires_elevation=True,
                )


if __name__ == "__main__":
    unittest.main()
