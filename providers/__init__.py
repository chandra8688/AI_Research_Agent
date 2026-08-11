import os
from typing import Protocol

from pydantic import BaseModel

class LLMProvider(Protocol):
    def generate(self, prompt: str) -> str:
        ...
    def generate_structured(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
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
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: '{provider_name}'")
