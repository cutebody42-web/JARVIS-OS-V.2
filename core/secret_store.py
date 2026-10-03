from __future__ import annotations

"""Small abstraction for storing/retrieving secrets locally.

Uses `keyring` (OS keychain). If secure storage is unavailable, reads of
optional settings remain possible but persistent writes fail explicitly.
The in-memory store is only for callers that deliberately inject it.
"""

from dataclasses import dataclass
import base64
import hashlib
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

SERVICE = "jarvis"


class SecretStoreUnavailableError(RuntimeError):
    """Persistent OS credential storage cannot safely complete an operation."""


_UNAVAILABLE_MESSAGE = (
    "Secure OS credential storage is unavailable or locked. "
    "Restore access to Windows Credential Manager, macOS Keychain, or a Linux "
    "Secret Service session, then restart JARVIS. Device pairing and face "
    "enrollment require persistent secure storage."
)


@dataclass
class SecretStore:
    """Key/value secret store."""

    def set(self, key: str, value: str) -> None:
        raise NotImplementedError

    def get(self, key: str) -> Optional[str]:
        raise NotImplementedError

    def get_persistent(self, key: str) -> Optional[str]:
        """Read without treating an inaccessible identity as a missing one."""
        return self.get(key)

    def delete(self, key: str) -> None:
        raise NotImplementedError


class KeyringSecretStore(SecretStore):
    def __init__(self, service: str = SERVICE):
        self.service = service
        import keyring

        backend = keyring.get_keyring()
        if backend.priority <= 0:
            raise SecretStoreUnavailableError(_UNAVAILABLE_MESSAGE)
        candidates = tuple(getattr(backend, "backends", (backend,)))
        if not candidates:
            raise SecretStoreUnavailableError(_UNAVAILABLE_MESSAGE)
        for candidate in candidates:
            module = type(candidate).__module__
            if (
                candidate.priority <= 0
                or module.startswith("keyrings.alt.file")
                or module == "keyring.backends.null"
            ):
                raise SecretStoreUnavailableError(_UNAVAILABLE_MESSAGE)
        self._keyring = keyring

    def set(self, key: str, value: str) -> None:
        # keyring handles encryption on supported platforms.
        try:
            self._keyring.set_password(self.service, key, value)
        except Exception:
            raise SecretStoreUnavailableError(_UNAVAILABLE_MESSAGE) from None

    def get(self, key: str) -> Optional[str]:
        try:
            return self.get_persistent(key)
        except SecretStoreUnavailableError:
            # Optional API-key reads allow setup to validate a session key.
            # Signing identity reads use get_persistent and cannot hide failure.
            return None

    def get_persistent(self, key: str) -> Optional[str]:
        try:
            return self._keyring.get_password(self.service, key)
        except Exception:
            raise SecretStoreUnavailableError(_UNAVAILABLE_MESSAGE) from None

    def delete(self, key: str) -> None:
        try:
            self._keyring.delete_password(self.service, key)
        except Exception:
            # Missing entries and unavailable backends should not crash setup.
            pass


class UnavailableSecretStore(SecretStore):
    """Allow optional reads while refusing to invent a temporary identity."""

    def set(self, key: str, value: str) -> None:
        raise SecretStoreUnavailableError(_UNAVAILABLE_MESSAGE)

    def get(self, key: str) -> Optional[str]:
        return None

    def get_persistent(self, key: str) -> Optional[str]:
        raise SecretStoreUnavailableError(_UNAVAILABLE_MESSAGE)

    def delete(self, key: str) -> None:
        raise SecretStoreUnavailableError(_UNAVAILABLE_MESSAGE)


class NoopSecretStore(SecretStore):
    """Explicitly injected memory-only store for tests and temporary sessions."""

    def __init__(self):
        self._mem: dict[str, str] = {}

    def set(self, key: str, value: str) -> None:
        self._mem[key] = value

    def get(self, key: str) -> Optional[str]:
        return self._mem.get(key)

    def delete(self, key: str) -> None:
        self._mem.pop(key, None)


class SecretCipher:
    """Authenticated encryption for secrets stored outside an OS keychain."""

    def __init__(self, master_secret: str):
        if not isinstance(master_secret, str) or len(master_secret) < 20:
            raise ValueError("The secret encryption key must contain at least 20 characters")
        digest = hashlib.sha256(master_secret.encode("utf-8")).digest()
        self._fernet = Fernet(base64.urlsafe_b64encode(digest))

    def encrypt(self, value: str) -> str:
        if not value:
            raise ValueError("Cannot encrypt an empty secret")
        return self._fernet.encrypt(value.encode("utf-8")).decode("ascii")

    def decrypt(self, token: str) -> str:
        try:
            return self._fernet.decrypt(token.encode("ascii")).decode("utf-8")
        except (InvalidToken, UnicodeError) as exc:
            raise ValueError("Stored secret could not be decrypted") from exc


class TenantDatabaseSecretStore(SecretStore):
    """Encrypted SecretStore view scoped to one hosted user."""

    def __init__(self, user_id: str):
        self.user_id = user_id

    def set(self, key: str, value: str) -> None:
        from api.database import SessionLocal
        from api.secret_service import set_user_secret

        with SessionLocal() as db:
            set_user_secret(db, self.user_id, key, value)

    def get(self, key: str) -> Optional[str]:
        from api.database import SessionLocal
        from api.secret_service import get_user_secret

        with SessionLocal() as db:
            return get_user_secret(db, self.user_id, key)

    def delete(self, key: str) -> None:
        from api.database import SessionLocal
        from api.secret_service import delete_user_secret

        with SessionLocal() as db:
            delete_user_secret(db, self.user_id, key)


_store: SecretStore | None = None


def get_secret_store() -> SecretStore:
    from core.tenant import get_current_user_id

    user_id = get_current_user_id()
    if user_id:
        return TenantDatabaseSecretStore(user_id)
    global _store
    if _store is not None:
        return _store

    try:
        _store = KeyringSecretStore()
    except Exception:
        _store = UnavailableSecretStore()
    return _store
