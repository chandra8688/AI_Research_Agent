import os
from typing import Protocol

from pydantic import BaseModel

from dataclasses import dataclass
from typing import Any

@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]

@dataclass
class AgentResponse:
    text: str | None
    function_calls: list[ToolCall]
    model_message: dict[str, Any]

class LLMProvider(Protocol):
    def generate(self, prompt: str) -> str:
        ...
    def generate_structured(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        ...
    def generate_agent_step(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AgentResponse:
        ...

def get_provider(name: str | None = None) -> LLMProvider:
    from config import settings
    
    if not name:
        name = settings.llm_primary_provider or settings.llm_provider
        
    provider_name = name.lower().strip()
    
    if provider_name == "gemini":
        from .gemini import GeminiProvider
        return GeminiProvider()
    elif provider_name == "openrouter":
        from .openrouter import OpenRouterProvider
        return OpenRouterProvider()
    elif provider_name == "groq":
        from .groq import GroqProvider
        return GroqProvider()
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: '{provider_name}'")
