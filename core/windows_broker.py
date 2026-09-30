"""Signed, typed contract for a future elevated Windows capability broker.

This module deliberately contains no generic shell/PowerShell/command execution.
The model-facing OwnerKernel never receives broker signing material. A trusted
desktop host may issue a short-lived permit only after an exact ActionRequest has
already been authorized.

Physical Windows execution adapters remain separately certifiable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import Any, Callable, Mapping
from uuid import uuid4

from core.authority_contracts import (
    ActionRequest,
    AuthorizationDecision,
    AuthorizationResult,
    canonical_arguments,
)
from core.nexus.peer_auth import DeviceSigner


BROKER_VERSION = 1
MAX_PERMIT_SECONDS = 120
_CAPABILITY_RE = re.compile(r"^windows\.[a-z][a-z0-9_.-]{0,79}$")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("broker timestamp must include timezone")
    return parsed.astimezone(timezone.utc)


def _no_args(args: dict) -> dict:
    if args:
        raise ValueError("capability accepts no arguments")
    return {}


def _percent(name: str):
    def normalize(args: dict) -> dict:
        if set(args) != {name}:
            raise ValueError(f"capability requires only {name}")
        value = args[name]
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
            raise ValueError(f"{name} must be an integer from 0 to 100")
        return {name: value}
    return normalize


def _registered_app(args: dict) -> dict:
    if set(args) != {"app_id"}:
        raise ValueError("launch requires an exact registered app_id")
    app_id = args["app_id"]
    if not isinstance(app_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", app_id):
        raise ValueError("invalid registered app_id")
    return {"app_id": app_id}


@dataclass(frozen=True)
class BrokerCapabilitySpec:
    capability_id: str
    normalize: Callable[[dict], dict]
    requires_admin: bool
    destructive: bool
    hardware_cert_required: bool = True

    def __post_init__(self) -> None:
        if not _CAPABILITY_RE.fullmatch(self.capability_id):
            raise ValueError("invalid broker capability id")


BROKER_CAPABILITIES: Mapping[str, BrokerCapabilitySpec] = {
    spec.capability_id: spec
    for spec in (
        BrokerCapabilitySpec("windows.audio.set_volume", _percent("level"), False, False),
        BrokerCapabilitySpec("windows.display.set_brightness", _percent("level"), False, False),
        BrokerCapabilitySpec("windows.process.launch_registered", _registered_app, False, False),
        BrokerCapabilitySpec("windows.power.shutdown", _no_args, True, True),
        BrokerCapabilitySpec("windows.power.restart", _no_args, True, True),
    )
}


@dataclass(frozen=True)
class BrokerPermit:
    version: int
    permit_id: str
    request_id: str
    capability_id: str
    arguments_digest: str
    owner_id: str
    session_id: str
    issued_at: str
    expires_at: str
    nonce: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "permit_id": self.permit_id,
            "request_id": self.request_id,
            "capability_id": self.capability_id,
            "arguments_digest": self.arguments_digest,
            "owner_id": self.owner_id,
            "session_id": self.session_id,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "nonce": self.nonce,
        }


@dataclass(frozen=True)
class SignedBrokerRequest:
    permit: BrokerPermit
    arguments_json: str
    signature: str


class BrokerStatus(str, Enum):
    SUCCEEDED = "succeeded"
    DENIED = "denied"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    UNKNOWN_AFTER_CRASH = "unknown_after_crash"


@dataclass(frozen=True)
class BrokerResult:
    permit_id: str
    capability_id: str
    status: BrokerStatus
    message: str


class BrokerPermitIssuer:
    """Trusted host-side signer. Never expose this object to model code."""

    def __init__(
        self,
        signer: DeviceSigner,
        owner_id: str,
        session_id: str,
        *,
        clock=_utc_now,
    ):
        if not isinstance(signer, DeviceSigner):
            raise TypeError("signer must be DeviceSigner")
        if not owner_id or not session_id:
            raise ValueError("owner/session identities are required")
        self.signer = signer
        self.owner_id = owner_id
        self.session_id = session_id
        self._clock = clock

    def issue(
        self,
        request: ActionRequest,
        authorization: AuthorizationResult,
        *,
        ttl_seconds: int = 30,
    ) -> SignedBrokerRequest:
        if not isinstance(request, ActionRequest):
            raise TypeError("request must be ActionRequest")
        if not isinstance(authorization, AuthorizationResult):
            raise TypeError("authorization must be AuthorizationResult")
        if authorization.decision is not AuthorizationDecision.ALLOW:
            raise PermissionError("broker permit requires an already-authorized request")
        if (
            authorization.capability_id != request.capability_id
            or authorization.arguments_digest != request.arguments_digest
        ):
            raise PermissionError("authorization is not bound to this exact request")
        spec = BROKER_CAPABILITIES.get(request.capability_id)
        if spec is None:
            raise PermissionError("capability is not exposed by the Windows broker")
        normalized = spec.normalize(request.arguments)
        if canonical_arguments(normalized) != request.arguments_json:
            raise ValueError("broker request arguments are not normalized")
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int):
            raise TypeError("ttl_seconds must be an integer")
        if not 0 < ttl_seconds <= MAX_PERMIT_SECONDS:
            raise ValueError("broker permit TTL exceeds the allowed bound")

        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("broker clock must be timezone-aware")
        issued = now.astimezone(timezone.utc)
        expires = issued.timestamp() + ttl_seconds
        permit = BrokerPermit(
            BROKER_VERSION,
            uuid4().hex,
            request.request_id,
            request.capability_id,
            request.arguments_digest,
            self.owner_id,
            self.session_id,
            issued.isoformat().replace("+00:00", "Z"),
            datetime.fromtimestamp(expires, timezone.utc).isoformat().replace("+00:00", "Z"),
            uuid4().hex,
        )
        arguments_json = request.arguments_json
        signed = _canonical({
            "permit": permit.to_dict(),
            "arguments_json": arguments_json,
        })
        return SignedBrokerRequest(
            permit,
            arguments_json,
            self.signer.sign(signed),
        )


class BrokerReplayStore:
    """Durable one-shot ledger. Started requests are never blindly replayed."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS broker_permits (
                    permit_id TEXT PRIMARY KEY,
                    nonce TEXT NOT NULL UNIQUE,
                    capability_id TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('started','completed','failed')),
                    result_status TEXT,
                    result_message TEXT,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        db.row_factory = sqlite3.Row
        return db

    def begin(self, permit: BrokerPermit) -> BrokerResult | None:
        now = _utc_now().isoformat().replace("+00:00", "Z")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT * FROM broker_permits WHERE permit_id=? OR nonce=?",
                (permit.permit_id, permit.nonce),
            ).fetchone()
            if existing is not None:
                db.rollback()
                status = existing["result_status"]
                if existing["state"] == "started":
                    return BrokerResult(
                        permit.permit_id,
                        permit.capability_id,
                        BrokerStatus.UNKNOWN_AFTER_CRASH,
                        "A prior broker attempt started but has no verified completion record.",
                    )
                return BrokerResult(
                    permit.permit_id,
                    permit.capability_id,
                    BrokerStatus(status),
                    existing["result_message"] or "",
                )
            db.execute(
                """
                INSERT INTO broker_permits(
                    permit_id, nonce, capability_id, state, updated_at
                ) VALUES (?, ?, ?, 'started', ?)
                """,
                (permit.permit_id, permit.nonce, permit.capability_id, now),
            )
            db.commit()
        return None

    def finish(self, result: BrokerResult) -> None:
        state = "completed" if result.status is BrokerStatus.SUCCEEDED else "failed"
        now = _utc_now().isoformat().replace("+00:00", "Z")
        with self._connect() as db:
            db.execute(
                """
                UPDATE broker_permits
                SET state=?, result_status=?, result_message=?, updated_at=?
                WHERE permit_id=?
                """,
                (state, result.status.value, result.message, now, result.permit_id),
            )


class WindowsBrokerService:
    """Verifier/executor shell around a typed adapter registry."""

    def __init__(
        self,
        pinned_public_key: str,
        replay_store: BrokerReplayStore,
        adapters: Mapping[str, Callable[[dict], str]],
        *,
        clock=_utc_now,
        hardware_certified: bool = False,
    ):
        self.pinned_public_key = pinned_public_key
        self.replay_store = replay_store
        self.adapters = dict(adapters)
        self._clock = clock
        self.hardware_certified = bool(hardware_certified)

    def _verify(self, request: SignedBrokerRequest) -> tuple[BrokerCapabilitySpec, dict]:
        permit = request.permit
        if permit.version != BROKER_VERSION:
            raise PermissionError("unsupported broker protocol version")
        spec = BROKER_CAPABILITIES.get(permit.capability_id)
        if spec is None:
            raise PermissionError("unregistered broker capability")
        if spec.hardware_cert_required and not self.hardware_certified:
            raise PermissionError("Windows hardware certification is not open for this broker")
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("broker clock must be timezone-aware")
        issued = _timestamp(permit.issued_at)
        expires = _timestamp(permit.expires_at)
        current = now.astimezone(timezone.utc)
        if current < issued or current > expires:
            raise PermissionError("broker permit is not currently valid")
        if (expires - issued).total_seconds() > MAX_PERMIT_SECONDS:
            raise PermissionError("broker permit lifetime exceeds policy")

        try:
            arguments = json.loads(request.arguments_json)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid broker arguments JSON") from exc
        normalized = spec.normalize(arguments)
        canonical = canonical_arguments(normalized)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if canonical != request.arguments_json or digest != permit.arguments_digest:
            raise PermissionError("broker arguments do not match the signed permit")

        signed = _canonical({
            "permit": permit.to_dict(),
            "arguments_json": request.arguments_json,
        })
        if not DeviceSigner.verify(
            self.pinned_public_key,
            request.signature,
            signed,
        ):
            raise PermissionError("broker permit signature is invalid")
        return spec, normalized

    def handle(self, request: SignedBrokerRequest) -> BrokerResult:
        try:
            spec, arguments = self._verify(request)
        except (PermissionError, ValueError, TypeError) as exc:
            return BrokerResult(
                request.permit.permit_id,
                request.permit.capability_id,
                BrokerStatus.DENIED,
                str(exc),
            )

        replay = self.replay_store.begin(request.permit)
        if replay is not None:
            return replay

        adapter = self.adapters.get(spec.capability_id)
        if adapter is None:
            result = BrokerResult(
                request.permit.permit_id,
                spec.capability_id,
                BrokerStatus.UNAVAILABLE,
                "No certified broker adapter is installed for this capability.",
            )
            self.replay_store.finish(result)
            return result

        try:
            message = adapter(dict(arguments))
            if not isinstance(message, str) or not message:
                raise RuntimeError("broker adapter returned no evidence")
            result = BrokerResult(
                request.permit.permit_id,
                spec.capability_id,
                BrokerStatus.SUCCEEDED,
                message,
            )
        except Exception as exc:
            result = BrokerResult(
                request.permit.permit_id,
                spec.capability_id,
                BrokerStatus.FAILED,
                f"Broker adapter failed ({type(exc).__name__}).",
            )
        self.replay_store.finish(result)
        return result
