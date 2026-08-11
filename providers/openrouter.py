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
        self.model = settings.llm_model

    def generate(self, prompt: str) -> str:
        url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Using a reliable free model on OpenRouter
        payload = {
            "model": self.model,
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

    def generate_agent_step(self, messages: list[dict], tools: list[dict]) -> "AgentResponse":
        from . import AgentResponse, ToolCall
        import json
        import urllib.request
        import urllib.error
        
        import copy
        
        def lowercase_types(schema):
            if isinstance(schema, dict):
                if "type" in schema and isinstance(schema["type"], str):
                    schema["type"] = schema["type"].lower()
                for v in schema.values():
                    lowercase_types(v)
            elif isinstance(schema, list):
                for item in schema:
                    lowercase_types(item)
                    
        openrouter_tools = []
        for t in tools:
            params = copy.deepcopy(t["parameters"])
            lowercase_types(params)
            
            openrouter_tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": params
                }
            })
            
        or_messages = []
        last_tool_call_id = "unknown_id"
        
        for msg in messages:
            if msg.get("role") == "user":
                if "function_responses" in msg:
                    for resp in msg["function_responses"]:
                        or_messages.append({
                            "role": "tool",
                            "tool_call_id": last_tool_call_id,
                            "name": resp["name"],
                            "content": str(resp["response"].get("result", ""))
                        })
                else:
                    or_messages.append({"role": "user", "content": msg["content"]})
            elif msg.get("role") == "model":
                if "raw_message" in msg:
                    raw = msg["raw_message"]
                    or_messages.append(raw)
                    if "tool_calls" in raw and raw["tool_calls"]:
                        last_tool_call_id = raw["tool_calls"][0]["id"]
                        
        payload = {
            "model": self.model,
            "messages": or_messages,
            "tools": openrouter_tools
        }
        
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(payload).encode('utf-8'),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            method="POST"
        )
        
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                
            choice = result["choices"][0]
            msg_obj = choice["message"]
            
            function_calls = []
            if "tool_calls" in msg_obj and msg_obj["tool_calls"]:
                for tc in msg_obj["tool_calls"]:
                    if tc["type"] == "function":
                        args_str = tc["function"]["arguments"]
                        try:
                            args_dict = json.loads(args_str)
                        except:
                            args_dict = {}
                        function_calls.append(ToolCall(name=tc["function"]["name"], args=args_dict))
                        
            return AgentResponse(
                text=msg_obj.get("content"),
                function_calls=function_calls,
                model_message={"role": "model", "raw_message": msg_obj}
            )
            
        except urllib.error.HTTPError as e:
            if e.code in [429, 500, 502, 503, 504]:
                from .errors import RetryableProviderError
                raise RetryableProviderError(f"OpenRouter API Error: {e.code} - {e.read().decode('utf-8')}")
            from .errors import FatalProviderError
            raise FatalProviderError(f"OpenRouter API Error: {e.code} - {e.read().decode('utf-8')}")
        except urllib.error.URLError as e:
            from .errors import RetryableProviderError
            raise RetryableProviderError(f"OpenRouter network error: {str(e)}")
        except Exception as e:
            from .errors import FatalProviderError
            raise FatalProviderError(f"Unexpected error during OpenRouter tool call: {str(e)}")
