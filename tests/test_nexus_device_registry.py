"""Tests for one-time owner-controlled NEXUS device pairing."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from core.nexus.device_registry import (
    DeviceKind,
    DeviceRegistry,
    DeviceState,
)


class MemorySecrets:
    def __init__(self):
        self.values = {}

    def set_secret(self, peer_id, secret):
        self.values[peer_id] = bytes(secret)

    def delete_secret(self, peer_id):
        self.values.pop(peer_id, None)


class Clock:
    def __init__(self):
        self.now = datetime(2026, 9, 30, 13, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += timedelta(seconds=seconds)


class DeviceRegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.clock = Clock()
        self.registry = DeviceRegistry(Path(self.tmp.name), clock=self.clock)
        self.secrets = MemorySecrets()

    def tearDown(self):
        self.tmp.cleanup()

    def test_pairing_token_is_one_time_and_secret_is_not_stored_in_db(self):
        ticket = self.registry.begin_pairing(DeviceKind.PHONE)
        result = self.registry.complete_pairing(
            ticket.ticket_id,
            ticket.token,
            device_id="phone-1",
            host="phone-1.tailnet.ts.net",
            secret_store=self.secrets,
        )

        self.assertEqual(result.device.state, DeviceState.ACTIVE)
        self.assertEqual(result.device.kind, DeviceKind.PHONE)
        self.assertEqual(self.secrets.values["phone-1"], result.pairwise_secret)
        self.assertNotEqual(result.device.key_fingerprint, result.pairwise_secret.hex())

        raw = self.registry.db_path.read_bytes()
        self.assertNotIn(ticket.token.encode(), raw)
        self.assertNotIn(result.pairwise_secret, raw)

        with self.assertRaises(PermissionError):
            self.registry.complete_pairing(
                ticket.ticket_id,
                ticket.token,
                device_id="phone-2",
                host="phone-2.tailnet.ts.net",
                secret_store=self.secrets,
            )

    def test_expired_or_wrong_token_fails_closed(self):
        ticket = self.registry.begin_pairing(DeviceKind.PHONE, ttl_seconds=30)
        with self.assertRaises(PermissionError):
            self.registry.complete_pairing(
                ticket.ticket_id,
                "wrong",
                device_id="phone-1",
                host="phone.tailnet.ts.net",
                secret_store=self.secrets,
            )

        self.clock.advance(31)
        with self.assertRaises(PermissionError):
            self.registry.complete_pairing(
                ticket.ticket_id,
                ticket.token,
                device_id="phone-1",
                host="phone.tailnet.ts.net",
                secret_store=self.secrets,
            )

    def test_revocation_removes_secret_and_peer_endpoint(self):
        ticket = self.registry.begin_pairing(DeviceKind.DESKTOP)
        self.registry.complete_pairing(
            ticket.ticket_id,
            ticket.token,
            device_id="dell-3420",
            host="dell-3420.tailnet.ts.net",
            secret_store=self.secrets,
        )
        self.assertTrue(self.registry.peer_registry().known("dell-3420"))

        revoked = self.registry.revoke("dell-3420", self.secrets)

        self.assertEqual(revoked.state, DeviceState.REVOKED)
        self.assertNotIn("dell-3420", self.secrets.values)
        self.assertFalse(self.registry.peer_registry().known("dell-3420"))

    def test_repairing_revoked_device_rotates_secret(self):
        first = self.registry.begin_pairing(DeviceKind.PHONE)
        result1 = self.registry.complete_pairing(
            first.ticket_id,
            first.token,
            device_id="phone-1",
            host="phone.tailnet.ts.net",
            secret_store=self.secrets,
        )
        old = result1.pairwise_secret
        self.registry.revoke("phone-1", self.secrets)

        second = self.registry.begin_pairing(DeviceKind.PHONE)
        result2 = self.registry.complete_pairing(
            second.ticket_id,
            second.token,
            device_id="phone-1",
            host="phone.tailnet.ts.net",
            secret_store=self.secrets,
        )
        self.assertNotEqual(old, result2.pairwise_secret)
        self.assertEqual(result2.device.state, DeviceState.ACTIVE)


if __name__ == "__main__":
    unittest.main()
