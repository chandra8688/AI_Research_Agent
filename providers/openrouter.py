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
        import json

        schema_obj = schema.model_json_schema()
        schema_json = json.dumps(schema_obj, indent=2)

        # Build field-level guidance so the model clearly understands it must
        # FILL IN values — not reproduce or describe the schema itself.
        required_fields = schema_obj.get("required", [])
        properties = schema_obj.get("properties", {})
        field_lines = []
        for field_name, field_schema in properties.items():
            field_type = field_schema.get("type", "any")
            field_desc = field_schema.get("description", "")
            req_marker = " (required)" if field_name in required_fields else " (optional)"
            desc_part = f" — {field_desc}" if field_desc else ""
            field_lines.append(f'  "{field_name}": <{field_type} value{req_marker}{desc_part}>')
        fields_guidance = "\n".join(field_lines) if field_lines else "  (see schema below)"

        structured_prompt = (
            prompt
            + "\n\n"
            + "=== STRUCTURED OUTPUT INSTRUCTIONS ===\n"
            + "You MUST respond with a single JSON OBJECT that contains actual answer values.\n"
            + "IMPORTANT: Do NOT return the JSON Schema definition.\n"
            + "Do NOT include keys like \"type\", \"properties\", \"required\", \"$defs\", or \"title\" "
            + "unless those are actual field names listed below.\n"
            + "Return ONLY the filled-in JSON instance. No markdown. No explanation outside the JSON.\n\n"
            + "The output JSON object must contain these fields:\n"
            + fields_guidance
            + "\n\n"
            + "Schema (for reference only — describes the SHAPE; do not return it):\n"
            + schema_json
            + "\n\n"
            + "Respond with the filled JSON instance now:"
        )

        response_text = self.generate(structured_prompt)

        text = response_text.strip()
        # Strip Markdown code fences if the model added them
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            from .errors import FatalProviderError
            raise FatalProviderError(f"OpenRouter returned invalid JSON: {response_text}")

        # Explicit guard: reject responses that are schema definitions rather than instances.
        # A schema definition always has "properties" or "$defs" as a top-level key, while
        # a valid instance of any of our schemas never would (they are simple flat objects).
        schema_indicator_keys = {"properties", "$defs", "$schema", "definitions"}
        # Only block if these schema keys appear AND the actual required fields are missing.
        has_schema_keys = bool(schema_indicator_keys.intersection(parsed.keys()))
        has_required_fields = all(f in parsed for f in required_fields) if required_fields else True
        if has_schema_keys and not has_required_fields:
            from .errors import FatalProviderError
            raise FatalProviderError(
                f"OpenRouter returned a JSON Schema definition instead of a populated instance. "
                f"Expected fields: {required_fields}. Got top-level keys: {list(parsed.keys())}"
            )

        try:
            return schema(**parsed)
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
