import os
from google import genai
from google.genai import errors
from google.genai import types
from pydantic import BaseModel

class GeminiProvider:
    def __init__(self):
        from config import settings
        self.api_key = settings.gemini_api_key
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is missing or empty.")
        self.model = "gemini-3.5-flash"

    def generate(self, prompt: str) -> str:
        try:
            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model=self.model,
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

    def generate_structured(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        try:
            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
            return response.parsed
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
            raise FatalProviderError(f"Unexpected error during structured Gemini LLM call: {str(e)}")

    def generate_agent_step(self, messages: list[dict], tools: list[dict]) -> "AgentResponse":
        from . import AgentResponse, ToolCall
        from typing import Any
        
        gemini_tools = []
        for t in tools:
            gemini_tools.append(types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=t["parameters"]
            ))
        tool_config = types.Tool(function_declarations=gemini_tools)
        config = types.GenerateContentConfig(tools=[tool_config])
        
        contents = []
        for msg in messages:
            if msg.get("role") == "user":
                if "function_responses" in msg:
                    parts = []
                    for resp in msg["function_responses"]:
                        parts.append(types.Part.from_function_response(
                            name=resp["name"],
                            response=resp["response"]
                        ))
                    contents.append(types.Content(role="user", parts=parts))
                else:
                    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=msg["content"])]))
            elif msg.get("role") == "model":
                if "raw_message" in msg:
                    contents.append(msg["raw_message"])
                    
        try:
            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
            
            function_calls = []
            if response.function_calls:
                for fc in response.function_calls:
                    function_calls.append(ToolCall(name=fc.name, args=dict(fc.args)))
                    
            text = response.text if hasattr(response, 'text') else None
            
            raw_message = None
            if response.candidates:
                raw_message = response.candidates[0].content
                
            return AgentResponse(
                text=text,
                function_calls=function_calls,
                model_message={"role": "model", "raw_message": raw_message}
            )
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
            raise FatalProviderError(f"Unexpected error during Gemini tool call: {str(e)}")
