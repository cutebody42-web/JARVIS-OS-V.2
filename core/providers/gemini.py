"""Gemini adapter for the neutral text-generation contract (not Gemini Live)."""

from collections.abc import Callable

from core.model_provider import ModelRequest, ModelResponse, ModelTier


class GeminiProvider:
    def __init__(
        self,
        *,
        fast_model="gemini-2.5-flash-lite",
        standard_model="gemini-2.5-flash",
        credential_resolver: Callable[[], str | None] | None = None,
    ):
        if credential_resolver is not None and not callable(credential_resolver):
            raise TypeError("credential_resolver must be callable or None")
        self._models = {ModelTier.FAST: fast_model, ModelTier.STANDARD: standard_model}
        self._credential_resolver = credential_resolver

    def generate(self, request: ModelRequest) -> ModelResponse:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be ModelRequest")
        # Defense in depth for direct adapter use. RoutedModelProvider already
        # performs this projection before invoking a provider, but the network
        # adapter must remain safe when used on its own.
        request = request.for_cloud_provider()
        from google import genai
        from memory.config_manager import get_gemini_key

        api_key = (
            self._credential_resolver()
            if self._credential_resolver is not None
            else get_gemini_key()
        )
        if not api_key:
            raise ValueError("Gemini credentials are not configured for the current owner.")
        model = self._models[request.tier]
        config = {"system_instruction": request.system_instruction}
        if request.json_output:
            config["response_mime_type"] = "application/json"
        # Explicit client lifetime and timeout; never configure a global SDK key.
        with genai.Client(api_key=api_key, http_options={"timeout": 30000}) as client:
            response = client.models.generate_content(model=model, contents=request.prompt, config=config)
        if not isinstance(response.text, str) or not response.text.strip():
            raise ValueError("Gemini returned no text.")
        return ModelResponse(response.text.strip(), "gemini", model)
