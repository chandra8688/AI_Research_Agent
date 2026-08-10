import os
from typing import Protocol

class LLMProvider(Protocol):
    def generate(self, prompt: str) -> str:
        ...

def get_provider() -> LLMProvider:
    from config import settings
    
    provider_name = settings.llm_provider.lower().strip()
    
    if provider_name == "gemini":
        from .gemini import GeminiProvider
        return GeminiProvider()
    elif provider_name == "openrouter":
        from .openrouter import OpenRouterProvider
        return OpenRouterProvider()
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: '{provider_name}'")
