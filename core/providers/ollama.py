"""Opt-in local text provider. No downloads, tools, proxies or cloud fallback."""
import http.client
import json
import os
import re
import threading
from urllib.parse import urlsplit

from core.model_provider import ModelRequest, ModelResponse
from core.ollama_endpoint import normalize_local_ollama_url


class LocalProviderError(RuntimeError):
    pass


class OllamaProvider:
    MAX_REQUEST_BYTES = 65536
    MAX_RESPONSE_BYTES = 262144

    def __init__(self, model: str, *, base_url="http://127.0.0.1:11434",
                 timeout=60, keep_alive="5m"):
        normalized = normalize_local_ollama_url(base_url)
        url = urlsplit(normalized)
        if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}", model):
            raise ValueError("An explicitly configured model identifier is required.")
        if isinstance(timeout, bool) or not 0 < timeout <= 60:
            raise ValueError("Timeout must be in (0, 60] seconds.")
        self.model = model
        self._host, self._port = url.hostname, url.port or 11434
        self._timeout = timeout
        if keep_alive is not None and not isinstance(keep_alive, (str, int)):
            raise TypeError("keep_alive must be a string, integer or None.")
        self._keep_alive = keep_alive
        self._connection = None
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls):
        return cls(
            os.environ.get("NEXUS_OLLAMA_MODEL", ""),
            base_url=(
                os.environ.get("NEXUS_OLLAMA_URL")
                or os.environ.get("OLLAMA_HOST")
                or "http://127.0.0.1:11434"
            ),
        )

    def close(self):
        with self._lock:
            self._disconnect()

    def _disconnect(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def generate(self, request: ModelRequest) -> ModelResponse:
        if (not isinstance(request, ModelRequest) or not isinstance(request.prompt, str)
                or not isinstance(request.system_instruction, str)):
            raise TypeError("Expected a text ModelRequest.")
        # Direct local-adapter users get the same context semantics as routed
        # users.  Local-only context never needs to be reclassified to be useful
        # to the on-device model.
        request = request.for_local_provider()
        payload = {"model": self.model, "messages": [
            {"role": "system", "content": request.system_instruction},
            {"role": "user", "content": request.prompt}], "stream": False,
            "think": False,
            "options": {"num_ctx": 4096, "num_predict": 512}}
        if self._keep_alive is not None:
            payload["keep_alive"] = self._keep_alive
        if request.json_output:
            payload["format"] = "json"
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(body) > self.MAX_REQUEST_BYTES:
            raise ValueError("Model request exceeds the byte budget.")
        with self._lock:
            try:
                if self._connection is None:
                    self._connection = http.client.HTTPConnection(self._host, self._port, timeout=self._timeout)
                self._connection.request("POST", "/api/chat", body,
                                         {"Content-Type": "application/json", "Accept": "application/json"})
                response = self._connection.getresponse()
                if response.status != 200:
                    raise LocalProviderError("Local model endpoint rejected the request.")
                data = response.read(self.MAX_RESPONSE_BYTES + 1)
                if len(data) > self.MAX_RESPONSE_BYTES:
                    raise LocalProviderError("Local model response exceeds the byte budget.")
                value = json.loads(data)
                if not isinstance(value, dict) or value.get("done") is not True:
                    raise ValueError("Incomplete response")
                message = value.get("message")
                if (not isinstance(message, dict) or message.get("role") != "assistant"
                        or not isinstance(message.get("content"), str)
                        or not message["content"].strip() or message.get("tool_calls")):
                    raise ValueError("Invalid text response")
                return ModelResponse(message["content"], "ollama", self.model)
            except Exception as exc:
                self._disconnect()
                # Provider payloads can contain private text: never echo errors.
                if isinstance(exc, LocalProviderError):
                    raise
                raise LocalProviderError("Local model request failed; no fallback was attempted.") from None
