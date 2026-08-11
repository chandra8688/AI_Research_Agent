import os
import json
import urllib.request
import urllib.error
from pydantic import BaseModel

class OpenRouterProvider:
    def __init__(self):
        from config import settings
        self.api_key = settings.openrouter_api_key
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is missing or empty.")

    def generate(self, prompt: str) -> str:
        url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Using a reliable free model on OpenRouter
        payload = {
            "model": "google/gemma-2-9b-it:free",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/google/ai-research-agent",
            "X-Title": "AI Research Agent"
        }
        
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers=headers)
        
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content']
                else:
                    raise RuntimeError(f"Malformed OpenRouter response: {result}")
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            if e.code in [429, 500, 502, 503, 504]:
                from .errors import RetryableProviderError
                raise RetryableProviderError(f"OpenRouter API HTTP Error {e.code}: {error_body}")
            from .errors import FatalProviderError
            raise FatalProviderError(f"OpenRouter API HTTP Error {e.code}: {error_body}")
        except urllib.error.URLError as e:
            from .errors import RetryableProviderError
            raise RetryableProviderError(f"OpenRouter network error: {str(e)}")
            from .errors import FatalProviderError
            raise FatalProviderError(f"Unexpected error during OpenRouter LLM call: {str(e)}")

    def generate_structured(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        # Ask the model to output JSON adhering to the schema
        import json
        schema_json = json.dumps(schema.model_json_schema())
        structured_prompt = prompt + f"\n\nYou MUST respond in valid JSON format matching exactly this schema:\n{schema_json}\nReturn ONLY the JSON. Do not include markdown code blocks or any other text."
        
        response_text = self.generate(structured_prompt)
        
        import json
        text = response_text.strip()
        # Clean up response if the model returned markdown code blocks
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        try:
            parsed = json.loads(text)
            return schema(**parsed)
        except json.JSONDecodeError:
            from .errors import FatalProviderError
            raise FatalProviderError(f"OpenRouter returned invalid JSON: {response_text}")
        except Exception as e:
            from .errors import FatalProviderError
            raise FatalProviderError(f"OpenRouter JSON does not match schema: {str(e)}")
