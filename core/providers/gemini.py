"""Gemini adapter for the neutral text-generation contract (not Gemini Live)."""

from core.model_provider import ModelRequest, ModelResponse, ModelTier


class GeminiProvider:
    def __init__(self, *, fast_model="gemini-2.5-flash-lite", standard_model="gemini-2.5-flash"):
        self._models = {ModelTier.FAST: fast_model, ModelTier.STANDARD: standard_model}

    def generate(self, request: ModelRequest) -> ModelResponse:
        from google import genai
        from memory.config_manager import get_gemini_key

        api_key = get_gemini_key()
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
