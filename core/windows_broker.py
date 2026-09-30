"""Typed, fail-closed core for the future elevated Windows capability broker.

This module intentionally contains no shell, PowerShell, subprocess or generic
executable entry point. The elevated Windows transport/service will receive only
short-lived signed permits for capabilities explicitly registered by reviewed
application code.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import secrets
from types import MappingProxyType
from typing import Mapping, Protocol, runtime_checkable
from uuid import uuid4

from core.authority_contracts import canonical_arguments


class BrokerAuthorizationError(RuntimeError):
    pass


class BrokerDisabledError(RuntimeError):
    pass


@dataclass(frozen=True)
class BrokerCapabilitySpec:
    capability_id: str
    adapter_id: str
    requires_elevation: bool
    timeout_seconds: int = 15

    def __post_init__(self) -> None:
        if not self.capability_id or not self.adapter_id:
            raise ValueError("broker capability and adapter ids must be non-empty")
        if any(token in self.adapter_id.lower() for token in ("shell", "powershell", "cmd", "exec")):
            raise ValueError("generic command adapters are forbidden")
        if not 1 <= self.timeout_seconds <= 120:
            raise ValueError("broker timeout must be between 1 and 120 seconds")


class BrokerRegistry:
    def __init__(self, specs: Mapping[str, BrokerCapabilitySpec] | None = None):
        normalized: dict[str, BrokerCapabilitySpec] = {}
        for capability_id, spec in (specs or {}).items():
            if not isinstance(spec, BrokerCapabilitySpec):
                raise TypeError("broker registry values must be BrokerCapabilitySpec")
            if capability_id != spec.capability_id:
                raise ValueError("registry key must match capability_id")
            normalized[capability_id] = spec
        self._specs = MappingProxyType(normalized)

    def get(self, capability_id: str) -> BrokerCapabilitySpec:
        try:
            return self._specs[capability_id]
        except KeyError:
            raise BrokerAuthorizationError("Capability is not admitted by the Windows broker") from None


@dataclass(frozen=True)
class BrokerPermit:
    permit_id: str
    request_id: str
    capability_id: str
    arguments_json: str
    arguments_digest: str
    receipt_id: str
    issued_at: str
    expires_at: str
    nonce: str
    signature: str

    @property
    def arguments(self) -> dict:
        value = json.loads(self.arguments_json)
        if not isinstance(value, dict):
            raise BrokerAuthorizationError("Broker arguments must be an object")
        return value


@dataclass(frozen=True)
class BrokerResult:
    permit_id: str
    capability_id: str
    status: str
    message: str


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise BrokerAuthorizationError("Broker timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _digest_arguments(arguments_json: str) -> str:
    return hashlib.sha256(arguments_json.encode("utf-8")).hexdigest()


def _signing_payload(
    *,
    permit_id: str,
    request_id: str,
    capability_id: str,
    arguments_digest: str,
    receipt_id: str,
    issued_at: str,
    expires_at: str,
    nonce: str,
) -> bytes:
    return "\n".join(
        (
            permit_id,
            request_id,
            capability_id,
            arguments_digest,
            receipt_id,
            issued_at,
            expires_at,
            nonce,
        )
    ).encode("utf-8")


class BrokerPermitIssuer:
    """Create short-lived permits after the owner gateway has authorized work."""

    def __init__(self, secret: bytes, *, clock=None, max_ttl_seconds: int = 60):
        if not isinstance(secret, bytes) or len(secret) < 32:
            raise ValueError("broker signing secret must be at least 32 bytes")
        if not 5 <= max_ttl_seconds <= 300:
            raise ValueError("max broker permit TTL must be between 5 and 300 seconds")
        self._secret = secret
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._max_ttl = max_ttl_seconds

    def issue(
        self,
        *,
        request_id: str,
        capability_id: str,
        arguments: dict,
        receipt_id: str,
        ttl_seconds: int = 30,
    ) -> BrokerPermit:
        if not all(isinstance(v, str) and v for v in (request_id, capability_id, receipt_id)):
            raise ValueError("request, capability and receipt identities are required")
        if not 1 <= ttl_seconds <= self._max_ttl:
            raise ValueError("permit TTL exceeds issuer policy")

        arguments_json = canonical_arguments(arguments)
        arguments_digest = _digest_arguments(arguments_json)
        now = self._clock().astimezone(timezone.utc)
        expires = now + timedelta(seconds=ttl_seconds)
        permit_id = uuid4().hex
        nonce = secrets.token_urlsafe(18)
        issued_at = now.isoformat().replace("+00:00", "Z")
        expires_at = expires.isoformat().replace("+00:00", "Z")
        payload = _signing_payload(
            permit_id=permit_id,
            request_id=request_id,
            capability_id=capability_id,
            arguments_digest=arguments_digest,
            receipt_id=receipt_id,
            issued_at=issued_at,
            expires_at=expires_at,
            nonce=nonce,
        )
        signature = hmac.new(self._secret, payload, hashlib.sha256).hexdigest()
        return BrokerPermit(
            permit_id,
            request_id,
            capability_id,
            arguments_json,
            arguments_digest,
            receipt_id,
            issued_at,
            expires_at,
            nonce,
            signature,
        )


@runtime_checkable
class BrokerAdapter(Protocol):
    def invoke(self, spec: BrokerCapabilitySpec, arguments: dict) -> str:
        ...


class WindowsBrokerCore:
    """Framework-neutral broker validation/execution core.

    The real elevated Windows service/Named Pipe transport is intentionally not
    certified in cloud CI. This core is suitable for that service once physical
    Windows validation is available.
    """

    def __init__(
        self,
        secret: bytes,
        registry: BrokerRegistry,
        adapter: BrokerAdapter,
        *,
        clock=None,
    ):
        if not isinstance(secret, bytes) or len(secret) < 32:
            raise ValueError("broker signing secret must be at least 32 bytes")
        if not isinstance(registry, BrokerRegistry):
            raise TypeError("registry must be BrokerRegistry")
        if not isinstance(adapter, BrokerAdapter):
            raise TypeError("adapter must implement BrokerAdapter")
        self._secret = secret
        self._registry = registry
        self._adapter = adapter
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._used_nonces: set[str] = set()
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, value: bool) -> None:
        if type(value) is not bool:
            raise TypeError("broker enabled state must be boolean")
        self._enabled = value

    def _verify(self, permit: BrokerPermit) -> BrokerCapabilitySpec:
        if not self._enabled:
            raise BrokerDisabledError("Windows broker kill switch is active")
        if not isinstance(permit, BrokerPermit):
            raise TypeError("permit must be BrokerPermit")
        if permit.nonce in self._used_nonces:
            raise BrokerAuthorizationError("Broker permit replay detected")

        issued = _parse_time(permit.issued_at)
        expires = _parse_time(permit.expires_at)
        now = self._clock().astimezone(timezone.utc)
        if expires <= issued or now < issued - timedelta(seconds=5) or now > expires:
            raise BrokerAuthorizationError("Broker permit is outside its validity window")

        canonical = canonical_arguments(permit.arguments)
        if canonical != permit.arguments_json:
            raise BrokerAuthorizationError("Broker arguments are not canonical")
        digest = _digest_arguments(permit.arguments_json)
        if not hmac.compare_digest(digest, permit.arguments_digest):
            raise BrokerAuthorizationError("Broker argument digest mismatch")

        payload = _signing_payload(
            permit_id=permit.permit_id,
            request_id=permit.request_id,
            capability_id=permit.capability_id,
            arguments_digest=permit.arguments_digest,
            receipt_id=permit.receipt_id,
            issued_at=permit.issued_at,
            expires_at=permit.expires_at,
            nonce=permit.nonce,
        )
        expected = hmac.new(self._secret, payload, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, permit.signature):
            raise BrokerAuthorizationError("Invalid broker permit signature")

        spec = self._registry.get(permit.capability_id)
        self._used_nonces.add(permit.nonce)
        return spec

    def execute(self, permit: BrokerPermit) -> BrokerResult:
        spec = self._verify(permit)
        try:
            message = self._adapter.invoke(spec, permit.arguments)
        except Exception as exc:
            return BrokerResult(
                permit.permit_id,
                permit.capability_id,
                "failed",
                f"Broker adapter failed ({type(exc).__name__}).",
            )
        return BrokerResult(
            permit.permit_id,
            permit.capability_id,
            "succeeded",
            str(message),
        )
