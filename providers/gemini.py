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
            raise RuntimeError(f"Gemini API Error: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during Gemini LLM call: {str(e)}")
