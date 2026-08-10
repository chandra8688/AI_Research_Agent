import os
from google import genai
from google.genai import errors

class GeminiProvider:
    def __init__(self):
        from config import settings
        self.api_key = settings.gemini_api_key
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is missing or empty.")

    def generate(self, prompt: str) -> str:
        try:
            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model='gemini-3.5-flash',
                contents=prompt,
            )
            return response.text
        except errors.APIError as e:
            e_str = str(e)
            if getattr(e, 'code', None) in [429, 500, 502, 503, 504] or any(err in e_str for err in ["429", "500", "502", "503", "504", "RESOURCE_EXHAUSTED"]):
                from .errors import RetryableProviderError
                raise RetryableProviderError(f"Gemini API Error: {e_str}")
            from .errors import FatalProviderError
            raise FatalProviderError(f"Gemini API Error: {e_str}")
        except Exception as e:
            e_str = str(e).lower()
            if "timeout" in e_str or "connection" in e_str:
                from .errors import RetryableProviderError
                raise RetryableProviderError(f"Gemini network error: {str(e)}")
            from .errors import FatalProviderError
            raise FatalProviderError(f"Unexpected error during Gemini LLM call: {str(e)}")
