import os
import json
from pydantic import BaseModel
import groq

class GroqProvider:
    def __init__(self):
        from config import settings
        self.api_key = settings.groq_api_key
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is missing or empty.")
        self.model = settings.groq_model
        self.client = groq.Groq(api_key=self.api_key)

    def _handle_error(self, e: Exception) -> Exception:
        from .errors import RetryableProviderError, FatalProviderError, ProviderError
        
        if isinstance(e, groq.RateLimitError):
            return RetryableProviderError(f"Groq Rate Limit Error: {str(e)}")
        elif isinstance(e, groq.APIConnectionError):
            return RetryableProviderError(f"Groq Connection Error: {str(e)}")
        elif isinstance(e, groq.InternalServerError):
            return RetryableProviderError(f"Groq Server Error: {str(e)}")
        elif isinstance(e, groq.APIStatusError):
            if e.status_code in [429, 500, 502, 503, 504]:
                return RetryableProviderError(f"Groq API HTTP Error {e.status_code}: {str(e)}")
            return FatalProviderError(f"Groq API HTTP Error {e.status_code}: {str(e)}")
        elif isinstance(e, groq.APIError):
            return ProviderError(f"Groq API Error: {str(e)}")
        else:
            return FatalProviderError(f"Unexpected error during Groq LLM call: {str(e)}")

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            raise self._handle_error(e)

    def generate_structured(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        schema_obj = schema.model_json_schema()
        schema_json = json.dumps(schema_obj, indent=2)

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

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": structured_prompt}
                ],
                response_format={"type": "json_object"}
            )
            text = response.choices[0].message.content.strip()

            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                from .errors import FatalProviderError
                raise FatalProviderError(f"Groq returned invalid JSON: {text}")

            schema_indicator_keys = {"properties", "$defs", "$schema", "definitions"}
            has_schema_keys = bool(schema_indicator_keys.intersection(parsed.keys()))
            has_required_fields = all(f in parsed for f in required_fields) if required_fields else True
            if has_schema_keys and not has_required_fields:
                from .errors import FatalProviderError
                raise FatalProviderError(
                    f"Groq returned a JSON Schema definition instead of a populated instance. "
                    f"Expected fields: {required_fields}. Got top-level keys: {list(parsed.keys())}"
                )

            return schema(**parsed)
            
        except Exception as e:
            if type(e).__name__ in ["FatalProviderError", "RetryableProviderError", "ProviderError"]:
                raise e
            from .errors import FatalProviderError
            if "validation error" in str(e).lower() or isinstance(e, ValueError):
                raise FatalProviderError(f"Groq JSON does not match schema: {str(e)}")
            raise self._handle_error(e)

    def generate_agent_step(self, messages: list[dict], tools: list[dict]) -> "AgentResponse":
        from . import AgentResponse, ToolCall
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

        groq_tools = []
        if tools:
            for t in tools:
                params = copy.deepcopy(t["parameters"])
                lowercase_types(params)
                
                groq_tools.append({
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": params
                    }
                })

        groq_messages = []
        last_tool_call_id = "unknown_id"

        for msg in messages:
            if msg.get("role") == "user":
                if "function_responses" in msg:
                    for resp in msg["function_responses"]:
                        groq_messages.append({
                            "role": "tool",
                            "tool_call_id": last_tool_call_id,
                            "name": resp["name"],
                            "content": str(resp["response"].get("result", ""))
                        })
                else:
                    groq_messages.append({"role": "user", "content": msg["content"]})
            elif msg.get("role") == "model":
                if "raw_message" in msg:
                    raw = msg["raw_message"]
                    if "role" not in raw:
                        raw["role"] = "assistant"
                    groq_messages.append(raw)
                    if "tool_calls" in raw and raw["tool_calls"]:
                        last_tool_call_id = raw["tool_calls"][0]["id"]

        try:
            kwargs = {
                "model": self.model,
                "messages": groq_messages,
            }
            if groq_tools:
                kwargs["tools"] = groq_tools

            response = self.client.chat.completions.create(**kwargs)

            choice = response.choices[0]
            msg_obj = choice.message

            function_calls = []
            if msg_obj.tool_calls:
                for tc in msg_obj.tool_calls:
                    if tc.type == "function":
                        args_str = tc.function.arguments
                        try:
                            args_dict = json.loads(args_str)
                        except:
                            args_dict = {}
                        function_calls.append(ToolCall(name=tc.function.name, args=args_dict))

            raw_message = {
                "role": "assistant",
                "content": msg_obj.content,
            }
            if msg_obj.tool_calls:
                raw_message["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in msg_obj.tool_calls
                ]

            return AgentResponse(
                text=msg_obj.content,
                function_calls=function_calls,
                model_message={"role": "model", "raw_message": raw_message}
            )

        except Exception as e:
            raise self._handle_error(e)
