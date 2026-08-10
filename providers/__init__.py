import os
from typing import Protocol

class LLMProvider(Protocol):
    def generate(self, prompt: str) -> str:
        ...

def get_provider() -> LLMProvider:
    provider_name = os.getenv("LLM_PROVIDER", "gemini").lower().strip()
    
    if provider_name == "gemini":
        from .gemini import GeminiProvider
        return GeminiProvider()
    elif provider_name == "openrouter":
        from .openrouter import OpenRouterProvider
        return OpenRouterProvider()
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: '{provider_name}'")
