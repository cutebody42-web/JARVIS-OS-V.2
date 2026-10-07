"""Provider-neutral text generation and privacy boundary.

``ModelRequest.prompt`` is the current request.  Additional conversational or
durable context must be carried as classified ``ModelContext`` values instead
of being concatenated into that prompt.  Context is local-only by default and
therefore cannot cross a cloud route accidentally.
"""

from dataclasses import dataclass, replace
from enum import Enum
import re
from typing import Protocol, runtime_checkable


class ModelTier(str, Enum):
    FAST = "fast"
    STANDARD = "standard"


class ContextClassification(str, Enum):
    """Where an individual context fragment is allowed to be disclosed."""

    LOCAL_ONLY = "local_only"
    CLOUD_SHAREABLE = "cloud_shareable"


class CloudDisclosureError(RuntimeError):
    """Cloud egress was blocked before a network provider saw the request."""


# This is deliberately a high-confidence credential gate rather than a broad
# natural-language DLP classifier.  False negatives remain possible, so callers
# must still classify supplementary context; false positives are limited to
# credential-shaped values and explicit credential assignments.  Never include
# the matching value in an exception or audit record.
_HIGH_CONFIDENCE_CREDENTIAL_PATTERNS = (
    re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----", re.IGNORECASE),
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    re.compile(r"\bsk-(?:ant-|proj-)?[0-9A-Za-z_-]{16,}\b", re.IGNORECASE),
    re.compile(r"\bgh(?:p|o|u|s|r)_[0-9A-Za-z]{20,}\b"),
    re.compile(r"\bgithub_pat_[0-9A-Za-z_]{20,}\b", re.IGNORECASE),
    re.compile(r"\bxox(?:a|b|p|r|s)-[0-9A-Za-z-]{16,}\b", re.IGNORECASE),
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(r"\bhttps?://[^/\s:@]+:[^/\s@]{8,}@", re.IGNORECASE),
    re.compile(
        r"\beyJ[0-9A-Za-z_-]{8,}\.[0-9A-Za-z_-]{8,}\.[0-9A-Za-z_-]{8,}\b"
    ),
    re.compile(r"\bBearer\s+[0-9A-Za-z._~+/=-]{12,}\b", re.IGNORECASE),
    re.compile(
        r"(?<![0-9A-Za-z])(?:[0-9A-Za-z]+[_-])*"
        r"(?:api[_ -]?key|access[_ -]?token|refresh[_ -]?token|"
        r"client[_ -]?secret|private[_ -]?key|secret|credential|password|"
        r"passwd|authorization)\b\s*[:=]\s*"
        r"[\"']?[0-9A-Za-z][0-9A-Za-z_./+=:-]{11,}[\"']?",
        re.IGNORECASE,
    ),
)


def _contains_likely_credential(text: str) -> bool:
    return any(pattern.search(text) for pattern in _HIGH_CONFIDENCE_CREDENTIAL_PATTERNS)


def assert_cloud_safe_text(text: str) -> None:
    """Fail closed before credential-shaped text reaches any cloud transport."""
    if not isinstance(text, str):
        raise TypeError("cloud disclosure candidate must be text")
    if _contains_likely_credential(text):
        raise CloudDisclosureError(
            "Cloud request blocked because it contains likely credential material."
        )


@dataclass(frozen=True)
class ModelContext:
    """A separately classified fragment of non-current-request context.

    The default is deliberately fail-closed.  Callers must name
    ``CLOUD_SHAREABLE`` at the construction site for a fragment to leave the
    machine through a routed cloud provider.
    """

    text: str
    label: str = "context"
    classification: ContextClassification = ContextClassification.LOCAL_ONLY

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("context text must be non-empty text")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("context label must be non-empty text")
        if not isinstance(self.classification, ContextClassification):
            raise TypeError("classification must be ContextClassification")


@dataclass(frozen=True)
class ModelRequest:
    """One inference request with provider-enforceable context boundaries.

    ``prompt`` is always the current request and is eligible for the selected
    provider.  Owner memory, prior turns, retrieved documents, and other
    supplementary data belong in ``context``.  The tuple requirement avoids a
    mutable classification set changing while a route is in progress.
    """

    prompt: str
    system_instruction: str = ""
    tier: ModelTier = ModelTier.FAST
    json_output: bool = False
    context: tuple[ModelContext, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be non-empty text")
        if not isinstance(self.system_instruction, str):
            raise TypeError("system_instruction must be text")
        if not isinstance(self.tier, ModelTier):
            raise TypeError("tier must be ModelTier")
        if not isinstance(self.json_output, bool):
            raise TypeError("json_output must be bool")
        if not isinstance(self.context, tuple):
            raise TypeError("context must be a tuple of ModelContext values")
        if not all(isinstance(item, ModelContext) for item in self.context):
            raise TypeError("context must contain only ModelContext values")

    @staticmethod
    def _render(prompt: str, context: tuple[ModelContext, ...]) -> str:
        if not context:
            return prompt
        sections = [
            f"CONTEXT [{item.label.strip()}]:\n{item.text.strip()}"
            for item in context
        ]
        sections.append("CURRENT REQUEST:\n" + prompt)
        return "\n\n".join(sections)

    def for_local_provider(self) -> "ModelRequest":
        """Materialize every context fragment for an on-device provider."""
        return replace(self, prompt=self._render(self.prompt, self.context), context=())

    def for_cloud_provider(self) -> "ModelRequest":
        """Materialize only explicitly cloud-shareable context.

        Returning a new request with an empty context tuple means the cloud
        adapter never receives local-only text, even if it records or logs the
        request object before making its network call.  Credential-shaped
        content in the current request, system instruction, or explicitly
        shareable context is rejected rather than silently redacted.
        """
        shareable = tuple(
            item
            for item in self.context
            if item.classification is ContextClassification.CLOUD_SHAREABLE
        )
        prompt = self._render(self.prompt, shareable)
        assert_cloud_safe_text(prompt)
        assert_cloud_safe_text(self.system_instruction)
        return replace(self, prompt=prompt, context=())


@dataclass(frozen=True)
class ModelResponse:
    text: str
    provider: str
    model: str


@runtime_checkable
class ModelProvider(Protocol):
    def generate(self, request: ModelRequest) -> ModelResponse:
        """Return text only; providers cannot dispatch actions or change policy."""
        ...


def default_provider() -> ModelProvider:
    # Resolve per invocation: no shared SDK configuration or cross-tenant client.
    import os
    selected = os.environ.get("NEXUS_MODEL_PROVIDER", "gemini")
    if selected == "ollama":
        from core.providers.ollama import OllamaProvider
        return OllamaProvider.from_env()
    if selected != "gemini":
        raise ValueError("Unknown model provider; no fallback permitted.")
    from core.providers.gemini import GeminiProvider

    return GeminiProvider()
