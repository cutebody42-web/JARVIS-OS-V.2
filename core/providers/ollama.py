"""Hardened local Ollama adapter for the neutral ModelProvider contract.

The provider is local-only by design:
- literal loopback HTTP origin only (no DNS names, remote hosts, credentials,
  redirects or proxy routing);
- bounded request/response sizes;
- text-only responses (tool calls are rejected);
- persistent connection for synchronous generation;
- optional streaming on a dedicated bounded connection.

It supports both the Phase-3 positional single-model constructor and the newer
FAST/STANDARD tier mapping without weakening the local-provider boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import http.client
import json
import os
import re
import threading
from typing import Iterator
from urllib.parse import urlsplit


class LocalProviderError(RuntimeError):
    """Safe local-provider failure; never contains provider payload text."""


# Newer name retained as a compatibility alias.
OllamaProviderError = LocalProviderError


@dataclass(frozen=True)
class OllamaEndpoint:
    base_url: str

    @property
    def chat_url(self) -> str:
        return f"{self.base_url}/api/chat"


def _normalize_origin(value: str) -> tuple[OllamaEndpoint, str, int]:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("Ollama base_url cannot be empty.")
    if "://" not in raw:
        raw = "http://" + raw

    url = urlsplit(raw)
    host = url.hostname
    if host == "0.0.0.0":
        host = "127.0.0.1"
    elif host == "::":
        host = "::1"

    if (
        url.scheme != "http"
        or host not in {"127.0.0.1", "::1"}
        or url.username is not None
        or url.password is not None
        or url.path not in {"", "/"}
        or url.query
        or url.fragment
    ):
        raise ValueError("Local provider requires a literal loopback HTTP origin.")

    port = url.port or 11434
    display_host = f"[{host}]" if ":" in host else host
    endpoint = OllamaEndpoint(f"http://{display_host}:{port}")
    return endpoint, host, port


class OllamaProvider:
    MAX_REQUEST_BYTES = 65536
    MAX_RESPONSE_BYTES = 262144
    DEFAULT_OPTIONS = {"num_ctx": 4096, "num_predict": 512}

    def __init__(
        self,
        model: str | None = None,
        *,
        fast_model: str = "qwen3.5:4b",
        standard_model: str = "qwen2.5-coder:7b",
        base_url: str | None = None,
        keep_alive: str | int | None = "5m",
        timeout_seconds: float = 60.0,
        timeout: float | None = None,
    ):
        from core.model_provider import ModelTier

        if model is not None:
            fast_model = standard_model = model
        for value in (fast_model, standard_model):
            if not isinstance(value, str) or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}", value
            ):
                raise ValueError("An explicitly configured model identifier is required.")

        if timeout is not None:
            timeout_seconds = timeout
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
            raise ValueError("timeout_seconds must be numeric.")
        # Keep the Phase-3 bounded local-provider timeout.
        if not 0 < float(timeout_seconds) <= 60:
            raise ValueError("Timeout must be in (0, 60] seconds.")

        resolved = (
            base_url
            or os.environ.get("NEXUS_OLLAMA_URL")
            or os.environ.get("OLLAMA_HOST")
            or "http://127.0.0.1:11434"
        )
        self._endpoint, self._host, self._port = _normalize_origin(resolved)
        self._models = {
            ModelTier.FAST: fast_model,
            ModelTier.STANDARD: standard_model,
        }
        self._keep_alive = keep_alive
        self._timeout = float(timeout_seconds)
        self._connection: http.client.HTTPConnection | None = None
        self._lock = threading.Lock()

    @property
    def model(self) -> str:
        """Legacy single-model view; FAST model is the compatibility value."""
        from core.model_provider import ModelTier
        return self._models[ModelTier.FAST]

    @classmethod
    def from_env(cls):
        model = os.environ.get("NEXUS_OLLAMA_MODEL", "")
        return cls(
            model,
            base_url=os.environ.get(
                "NEXUS_OLLAMA_URL",
                os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
            ),
        )

    def close(self) -> None:
        with self._lock:
            self._disconnect()

    def _disconnect(self) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            finally:
                self._connection = None

    def _payload(self, request, *, stream: bool) -> dict:
        from core.model_provider import ModelRequest

        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a ModelRequest.")

        messages: list[dict[str, str]] = []
        if request.system_instruction:
            messages.append({"role": "system", "content": request.system_instruction})
        messages.append({"role": "user", "content": request.prompt})
        payload = {
            "model": self._models[request.tier],
            "messages": messages,
            "stream": stream,
            "options": dict(self.DEFAULT_OPTIONS),
        }
        if request.json_output:
            payload["format"] = "json"
        if self._keep_alive is not None:
            payload["keep_alive"] = self._keep_alive
        return payload

    def _encode(self, payload: dict) -> bytes:
        body = json.dumps(
            payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
        if len(body) > self.MAX_REQUEST_BYTES:
            raise ValueError("Model request exceeds the byte budget.")
        return body

    @staticmethod
    def _decode_complete(data: bytes, configured_model: str):
        from core.model_provider import ModelResponse

        try:
            value = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise LocalProviderError("Local model returned invalid JSON.") from None
        if not isinstance(value, dict) or value.get("done") is not True:
            raise LocalProviderError("Local model returned an incomplete response.")
        message = value.get("message")
        if (
            not isinstance(message, dict)
            or message.get("role") not in {None, "assistant"}
            or not isinstance(message.get("content"), str)
            or not message.get("content", "").strip()
            or message.get("tool_calls")
        ):
            raise LocalProviderError("Local model returned an invalid text response.")
        returned_model = value.get("model")
        model = (
            returned_model.strip()
            if isinstance(returned_model, str) and returned_model.strip()
            else configured_model
        )
        return ModelResponse(message["content"].strip(), "ollama", model)

    def generate(self, request):
        payload = self._payload(request, stream=False)
        body = self._encode(payload)
        with self._lock:
            try:
                if self._connection is None:
                    self._connection = http.client.HTTPConnection(
                        self._host, self._port, timeout=self._timeout
                    )
                self._connection.request(
                    "POST",
                    "/api/chat",
                    body,
                    {
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                )
                response = self._connection.getresponse()
                # Never follow redirects; any non-200 is a local-provider failure.
                if response.status != 200:
                    raise LocalProviderError("Local model endpoint rejected the request.")
                data = response.read(self.MAX_RESPONSE_BYTES + 1)
                if len(data) > self.MAX_RESPONSE_BYTES:
                    raise LocalProviderError("Local model response exceeds the byte budget.")
                return self._decode_complete(data, payload["model"])
            except ValueError:
                self._disconnect()
                raise
            except LocalProviderError:
                self._disconnect()
                raise
            except Exception:
                self._disconnect()
                raise LocalProviderError(
                    "Local model request failed; no fallback was attempted."
                ) from None

    def stream(self, request) -> Iterator[str]:
        """Yield bounded NDJSON text chunks from a dedicated loopback connection."""
        payload = self._payload(request, stream=True)
        body = self._encode(payload)
        connection = http.client.HTTPConnection(
            self._host, self._port, timeout=self._timeout
        )
        total = 0
        try:
            connection.request(
                "POST",
                "/api/chat",
                body,
                {
                    "Content-Type": "application/json",
                    "Accept": "application/x-ndjson",
                },
            )
            response = connection.getresponse()
            if response.status != 200:
                raise LocalProviderError("Local model endpoint rejected the request.")
            while True:
                raw = response.readline(self.MAX_RESPONSE_BYTES + 1)
                if not raw:
                    break
                total += len(raw)
                if len(raw) > self.MAX_RESPONSE_BYTES or total > self.MAX_RESPONSE_BYTES:
                    raise LocalProviderError("Local model response exceeds the byte budget.")
                try:
                    value = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    raise LocalProviderError(
                        "Local model returned invalid streaming JSON."
                    ) from None
                message = value.get("message") if isinstance(value, dict) else None
                if isinstance(message, dict) and message.get("tool_calls"):
                    raise LocalProviderError("Local model attempted an unsupported tool call.")
                chunk = message.get("content") if isinstance(message, dict) else None
                if isinstance(chunk, str) and chunk:
                    yield chunk
                if isinstance(value, dict) and value.get("done") is True:
                    break
        except LocalProviderError:
            raise
        except Exception:
            raise LocalProviderError("Local model streaming request failed.") from None
        finally:
            connection.close()
