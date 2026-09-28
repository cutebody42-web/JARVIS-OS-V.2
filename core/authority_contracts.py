"""Data contracts for owner authority. None of these are model permissions.

Only application bootstrap constructs PolicyContext. Model-facing entry points
accept a tool name and JSON arguments, never a context, grant or decision.
"""

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from uuid import uuid4


class AuthorizationDecision(str, Enum):
    ALLOW = "ALLOW"
    REQUIRE_CONFIRMATION = "REQUIRE_CONFIRMATION"
    DENY = "DENY"


class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ConfirmationPolicy(str, Enum):
    NONE = "none"
    EXACT = "exact"
    FORBIDDEN = "forbidden"


class DestructiveLevel(str, Enum):
    NONE = "none"
    RECOVERABLE = "recoverable"
    IRREVERSIBLE = "irreversible"


class ExecutionSurface(str, Enum):
    NATIVE = "native"
    NETWORK = "network"
    DOM = "dom"
    DESKTOP = "desktop"
    INTERNAL = "internal"


def canonical_arguments(arguments: dict) -> str:
    """Snapshot strict, bounded JSON. Preserve message text exactly, including spaces."""
    def check(value, depth=0):
        if depth > 12:
            raise ValueError("Arguments are too deeply nested.")
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                raise ValueError("Argument keys must be strings.")
            for child in value.values():
                check(child, depth + 1)
        elif type(value) is list:
            for child in value:
                check(child, depth + 1)
        elif type(value) not in (str, int, float, bool, type(None)):
            raise ValueError("Arguments must be JSON data, never executable objects.")
        elif type(value) is float and not math.isfinite(value):
            raise ValueError("Non-finite arguments are not allowed.")
    if type(arguments) is not dict:
        raise ValueError("Arguments must be an object.")
    check(arguments)
    encoded = json.dumps(arguments, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > 65536:
        raise ValueError("Arguments exceed 64 KiB.")
    return encoded


def argument_digest(encoded: str) -> str:
    # An equality/binding digest, NOT a signature or a tamper-proof audit seal.
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ActionRequest:
    capability_id: str
    arguments_json: str
    request_id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self):
        if not self.capability_id or not self.request_id:
            raise ValueError("Request and capability identities are required.")
        if canonical_arguments(json.loads(self.arguments_json)) != self.arguments_json:
            raise ValueError("ActionRequest requires normalized arguments.")

    @property
    def arguments(self) -> dict:
        return json.loads(self.arguments_json)  # never expose mutable internal state

    @property
    def arguments_digest(self) -> str:
        return argument_digest(self.arguments_json)


@dataclass(frozen=True)
class PolicyContext:
    owner_id: str
    session_id: str
    workspace_root: Path
    environment: str = "cloud"

    def __post_init__(self):
        if not self.owner_id or not self.session_id:
            raise ValueError("Application-authenticated owner/session identities required.")
        if self.environment not in {"cloud", "desktop"}:
            raise ValueError("Unknown execution environment.")
        root = Path(self.workspace_root).absolute()
        if root == Path(root.anchor):
            raise ValueError("The entire filesystem cannot be a workspace.")
        object.__setattr__(self, "workspace_root", root)


@dataclass(frozen=True)
class ToolCapability:
    capability_id: str
    access: str
    risk: RiskLevel
    external_side_effect: bool
    sensitive_data_access: bool
    destructive_level: DestructiveLevel
    privilege_requirement: str
    reversible: bool
    verifier_required: bool
    confirmation_policy: ConfirmationPolicy
    execution_surface: ExecutionSurface
    resource_lock: str | None
    environments: frozenset[str] = frozenset({"cloud", "desktop"})
    disabled_reason: str = ""

    def __post_init__(self):
        if self.access not in {"read", "mutate"}:
            raise ValueError("Capabilities must explicitly declare read or mutate.")
        if not isinstance(self.environments, frozenset):
            raise TypeError("Execution environments must be immutable.")
        if self.privilege_requirement not in {"user", "admin"}:
            raise ValueError("Unknown privilege requirement.")


@dataclass(frozen=True)
class CapabilityGrant:
    grant_id: str
    owner_id: str
    session_id: str
    capability_id: str
    arguments_digest: str
    expires_at: float
    policy_rule: str


@dataclass(frozen=True)
class ConsentTicket:
    ticket_id: str
    request_id: str
    grant: CapabilityGrant
    issued_at: float
    expires_at: float
    policy_version: str


@dataclass(frozen=True)
class AuthorizationResult:
    decision: AuthorizationDecision
    policy_rule: str
    reason: str
    capability_id: str
    arguments_digest: str
    confirmation_ticket_id: str | None = None
