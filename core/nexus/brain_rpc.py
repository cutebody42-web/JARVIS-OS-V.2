"""Signed companion RPC for the single JARVIS Brain.

The phone is a remote face for the desktop brain, not a second assistant.
Requests are authenticated with the same explicit Ed25519 peer identity used
by semantic sync. Underlying model/provider details remain audit-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from core.jarvis_brain import BrainLane, JarvisBrain
from core.model_router import TaskKind
from core.nexus.peer_auth import PeerAuthenticator


BRAIN_RPC_VERSION = 1
MAX_MESSAGE_CHARS = 16000


@dataclass(frozen=True)
class BrainRequest:
    request_id: str
    message: str
    task: TaskKind | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": BRAIN_RPC_VERSION,
            "request_id": self.request_id,
            "message": self.message,
            "task": self.task.value if self.task is not None else None,
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> "BrainRequest":
        if not isinstance(value, Mapping):
            raise TypeError("brain request must be an object")
        if set(value) != {"version", "request_id", "message", "task"}:
            raise ValueError("brain request schema mismatch")
        if value["version"] != BRAIN_RPC_VERSION:
            raise ValueError("unsupported brain RPC version")
        request_id = value["request_id"]
        message = value["message"]
        if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
            raise ValueError("invalid brain request id")
        if not isinstance(message, str) or not message.strip() or len(message) > MAX_MESSAGE_CHARS:
            raise ValueError("invalid brain message")
        task_value = value["task"]
        task = None
        if task_value is not None:
            try:
                task = TaskKind(task_value)
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid brain task kind") from exc
        return cls(request_id, message.strip(), task)


@dataclass(frozen=True)
class BrainResponse:
    request_id: str
    text: str
    lane: BrainLane

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": BRAIN_RPC_VERSION,
            "request_id": self.request_id,
            "identity": "JARVIS",
            "text": self.text,
            "lane": self.lane.value,
        }


class SignedBrainEndpoint:
    """Authenticate a paired device, run the shared brain, sign the response."""

    def __init__(self, brain: JarvisBrain, authenticator: PeerAuthenticator):
        if not isinstance(brain, JarvisBrain):
            raise TypeError("brain must be JarvisBrain")
        if not isinstance(authenticator, PeerAuthenticator):
            raise TypeError("authenticator must be PeerAuthenticator")
        self.brain = brain
        self.authenticator = authenticator

    def handle(self, encoded: bytes) -> bytes:
        peer, payload, message_id = self.authenticator.verify(
            encoded,
            expected_kind="brain.request",
        )
        request = BrainRequest.from_payload(payload)
        if message_id != request.request_id:
            raise PermissionError("signed message id does not match brain request")

        text = self.brain.handle(request.message, task=request.task)
        response = BrainResponse(request.request_id, text, self.brain.lane)
        return self.authenticator.sign(
            "brain.response",
            peer.peer_id,
            response.to_payload(),
            message_id="brain-response:" + request.request_id,
        )
