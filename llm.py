import os
from google import genai
from google.genai import errors

def call_llm(prompt: str) -> str:
    """Calls the Gemini API with the given prompt and returns the text response."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is missing or empty.")
    
    try:
        client = genai.Client(api_key=api_key)
        # Using gemini-3.5-flash as the default development model (2.5 is deprecated)
        response = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=prompt,
        )
        return response.text
    except errors.APIError as e:
        raise RuntimeError(f"Gemini API Error: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during LLM call: {str(e)}")
