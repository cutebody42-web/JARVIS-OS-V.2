"""Signing identity must survive restarts and inaccessible secure storage."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from core.nexus.peer_auth import load_or_create_device_signer
from core import secret_store
from core.secret_store import (
    KeyringSecretStore,
    NoopSecretStore,
    SecretStoreUnavailableError,
    UnavailableSecretStore,
)


class SecretStorageTests(unittest.TestCase):
    def test_missing_backend_never_becomes_an_implicit_temporary_identity(self):
        with (
            patch.object(secret_store, "_store", None),
            patch("core.tenant.get_current_user_id", return_value=None),
            patch("keyring.get_keyring", return_value=SimpleNamespace(priority=0)),
        ):
            store = secret_store.get_secret_store()
            self.assertIsInstance(store, UnavailableSecretStore)
            self.assertNotIsInstance(store, NoopSecretStore)
            self.assertIsNone(store.get("gemini_api_key"))
            with self.assertRaisesRegex(SecretStoreUnavailableError, "secure storage"):
                load_or_create_device_signer("desktop", secret_store=store)

    def test_plaintext_backend_is_rejected(self):
        backend = type(
            "PlaintextKeyring", (),
            {"__module__": "keyrings.alt.file", "priority": 1},
        )()
        with patch("keyring.get_keyring", return_value=backend):
            with self.assertRaises(SecretStoreUnavailableError):
                KeyringSecretStore()

    def test_unavailable_backend_inside_chain_is_rejected(self):
        backend = SimpleNamespace(
            priority=10,
            backends=[SimpleNamespace(priority=0)],
        )
        with patch("keyring.get_keyring", return_value=backend):
            with self.assertRaises(SecretStoreUnavailableError):
                KeyringSecretStore()

    def test_locked_identity_read_never_generates_or_overwrites_key(self):
        store = KeyringSecretStore.__new__(KeyringSecretStore)
        store.service = "jarvis"
        store._keyring = Mock()
        store._keyring.get_password.side_effect = RuntimeError("keychain locked")
        with patch("core.nexus.peer_auth.DeviceSigner.generate") as generate:
            with self.assertRaisesRegex(SecretStoreUnavailableError, "restart JARVIS"):
                load_or_create_device_signer("desktop", secret_store=store)
        generate.assert_not_called()
        store._keyring.set_password.assert_not_called()
        # Optional API key reads still allow the setup UI to start.
        self.assertIsNone(store.get("gemini_api_key"))

    def test_secure_write_failure_is_actionable_and_does_not_claim_success(self):
        store = KeyringSecretStore.__new__(KeyringSecretStore)
        store.service = "jarvis"
        store._keyring = Mock()
        store._keyring.set_password.side_effect = RuntimeError("private backend detail")
        with self.assertRaises(SecretStoreUnavailableError) as caught:
            store.set("private-key", "private-value")
        self.assertIn("Secure OS credential storage", str(caught.exception))
        self.assertNotIn("private-value", str(caught.exception))
        self.assertNotIn("private backend detail", str(caught.exception))

    def test_persistent_backend_recovers_same_signer_after_store_restart(self):
        values = {}
        backend = SimpleNamespace(priority=1)

        def save(service, key, value):
            values[(service, key)] = value

        with (
            patch("keyring.get_keyring", return_value=backend),
            patch("keyring.get_password", side_effect=lambda service, key: values.get((service, key))),
            patch("keyring.set_password", side_effect=save) as write,
        ):
            first = load_or_create_device_signer("desktop", secret_store=KeyringSecretStore())
            restarted = load_or_create_device_signer("desktop", secret_store=KeyringSecretStore())
        self.assertEqual(first.public_b64, restarted.public_b64)
        write.assert_called_once()


if __name__ == "__main__":
    unittest.main()
