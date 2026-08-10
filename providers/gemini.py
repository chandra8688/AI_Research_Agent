import os
from google import genai
from google.genai import errors

class GeminiProvider:
    def generate(self, prompt: str) -> str:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is missing or empty.")
            
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model='gemini-3.5-flash',
                contents=prompt,
            )
            return response.text
        except errors.APIError as e:
            raise RuntimeError(f"Gemini API Error: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during Gemini LLM call: {str(e)}")
