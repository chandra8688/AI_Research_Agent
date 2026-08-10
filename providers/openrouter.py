import os
import json
import urllib.request
import urllib.error

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
            raise RuntimeError(f"OpenRouter API HTTP Error {e.code}: {error_body}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during OpenRouter LLM call: {str(e)}")
