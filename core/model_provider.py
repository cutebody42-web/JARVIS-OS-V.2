"""Provider-neutral text generation boundary. No SDK or credentials at import time."""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


class ModelTier(str, Enum):
    FAST = "fast"
    STANDARD = "standard"


@dataclass(frozen=True)
class ModelRequest:
    prompt: str
    system_instruction: str = ""
    tier: ModelTier = ModelTier.FAST
    json_output: bool = False


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
    from core.providers.gemini import GeminiProvider

    return GeminiProvider()
