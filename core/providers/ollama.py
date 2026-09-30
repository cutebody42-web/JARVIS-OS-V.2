"""Ollama adapter for the neutral text-generation contract.

The adapter is intentionally text-only at the ModelProvider boundary today.
Tool schemas are not accepted until ModelRequest grows a provider-neutral tools
field; action dispatch remains owned by the agent/action kernel.

Streaming is exposed as an optional adapter method for future UI/runtime use,
while generate() preserves the synchronous ModelProvider contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterator
from urllib.parse import urlparse

from core.model_provider import ModelRequest, ModelResponse, ModelTier


class OllamaProviderError(RuntimeError):
    """Safe, user-presentable Ollama adapter failure."""


@dataclass(frozen=True)
class OllamaEndpoint:
    base_url: str

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Ollama base_url must be an absolute http(s) URL.")
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))

    @property
    def chat_url(self) -> str:
        return f"{self.base_url}/api/chat"


class OllamaProvider:
    """Provider-neutral Ollama chat adapter.

    The model mapping is tier-based to match ModelRequest today. Persona/task
    routing belongs in a later model_router layer rather than this adapter.
    """

    def __init__(
        self,
        *,
        fast_model: str = "qwen3.5:4b",
        standard_model: str = "qwen2.5-coder:7b",
        base_url: str | None = None,
        keep_alive: str | int | None = "5m",
        timeout_seconds: float = 120.0,
    ):
        if not fast_model or not standard_model:
            raise ValueError("Ollama model names must be non-empty.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")

        resolved_base = base_url or os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434"
        self._endpoint = OllamaEndpoint(resolved_base)
        self._models = {
            ModelTier.FAST: fast_model,
            ModelTier.STANDARD: standard_model,
        }
        self._keep_alive = keep_alive
        self._timeout = timeout_seconds

    def _payload(self, request: ModelRequest, *, stream: bool) -> dict:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a ModelRequest.")

        messages: list[dict[str, str]] = []
        if request.system_instruction.strip():
            messages.append({"role": "system", "content": request.system_instruction})
        messages.append({"role": "user", "content": request.prompt})

        payload: dict = {
            "model": self._models[request.tier],
            "messages": messages,
            "stream": stream,
        }
        if request.json_output:
            payload["format"] = "json"
        if self._keep_alive is not None:
            payload["keep_alive"] = self._keep_alive
        return payload

    @staticmethod
    def _safe_request_error(exc: Exception) -> OllamaProviderError:
        name = type(exc).__name__
        if name in {"ConnectTimeout", "ReadTimeout", "Timeout"}:
            return OllamaProviderError("Ollama did not respond before the request timeout.")
        if name in {"ConnectionError"}:
            return OllamaProviderError("Ollama is not reachable on the configured local endpoint.")
        return OllamaProviderError("Ollama request failed.")

    def generate(self, request: ModelRequest) -> ModelResponse:
        """Return one complete response through the existing ModelProvider contract."""
        import requests

        payload = self._payload(request, stream=False)
        try:
            response = requests.post(
                self._endpoint.chat_url,
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
            body = response.json()
        except requests.RequestException as exc:
            raise self._safe_request_error(exc) from None
        except ValueError:
            raise OllamaProviderError("Ollama returned an invalid JSON response.") from None

        message = body.get("message") if isinstance(body, dict) else None
        text = message.get("content") if isinstance(message, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise OllamaProviderError("Ollama returned no text.")

        model = body.get("model") if isinstance(body, dict) else None
        if not isinstance(model, str) or not model.strip():
            model = payload["model"]
        return ModelResponse(text.strip(), "ollama", model)

    def stream(self, request: ModelRequest) -> Iterator[str]:
        """Yield text chunks without changing the synchronous ModelProvider protocol."""
        import json
        import requests

        payload = self._payload(request, stream=True)
        try:
            with requests.post(
                self._endpoint.chat_url,
                json=payload,
                timeout=self._timeout,
                stream=True,
            ) as response:
                response.raise_for_status()
                for raw_line in response.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    try:
                        body = json.loads(raw_line)
                    except ValueError:
                        raise OllamaProviderError(
                            "Ollama returned an invalid streaming JSON response."
                        ) from None
                    message = body.get("message") if isinstance(body, dict) else None
                    chunk = message.get("content") if isinstance(message, dict) else None
                    if isinstance(chunk, str) and chunk:
                        yield chunk
        except requests.RequestException as exc:
            raise self._safe_request_error(exc) from None
