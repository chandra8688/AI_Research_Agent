import os
from pydantic import BaseModel
import tools

def call_llm(prompt: str) -> str:
    """Calls the configured LLM API (Gemini or OpenRouter) with the given prompt and returns the text response."""
    from config import settings
    from providers import get_provider
    from providers.errors import RetryableProviderError, FatalProviderError
    
    primary_name = settings.llm_primary_provider or settings.llm_provider
    print(f"[LLM] Primary provider: {primary_name}")
    primary_provider = get_provider(primary_name)
    
    try:
        return primary_provider.generate(prompt)
    except RetryableProviderError as e:
        if not settings.llm_fallback_enabled:
            raise
            
        print(f"[LLM] Primary provider failed: {str(e)}")
        print(f"[LLM] Falling back to: {settings.llm_fallback_provider}")
        
        try:
            fallback_provider = get_provider(settings.llm_fallback_provider)
            response = fallback_provider.generate(prompt)
            print("[LLM] Fallback provider succeeded")
            return response
        except Exception as fallback_e:
            raise RuntimeError(f"Both providers failed.\nPrimary error: {str(e)}\nFallback error: {str(fallback_e)}")
    except FatalProviderError:
        # Non-retryable failure, surface immediately without fallback
        raise

def call_llm_structured(prompt: str, schema: type[BaseModel]) -> BaseModel:
    """Calls the configured LLM API (Gemini or OpenRouter) and returns a structured response parsed into the given Pydantic schema."""
    from config import settings
    from providers import get_provider
    from providers.errors import RetryableProviderError, FatalProviderError
    
    primary_name = settings.llm_primary_provider or settings.llm_provider
    print(f"[LLM] Primary provider (structured): {primary_name}")
    primary_provider = get_provider(primary_name)
    
    try:
        return primary_provider.generate_structured(prompt, schema)
    except RetryableProviderError as e:
        if not settings.llm_fallback_enabled:
            raise RuntimeError(str(e))
            
        print(f"[LLM] Primary provider structured failed: {str(e)}")
        print(f"[LLM] Falling back to (structured): {settings.llm_fallback_provider}")
        
        try:
            fallback_provider = get_provider(settings.llm_fallback_provider)
            response = fallback_provider.generate_structured(prompt, schema)
            print("[LLM] Fallback provider structured succeeded")
            return response
        except Exception as fallback_e:
            raise RuntimeError(f"Both providers failed.\nPrimary error: {str(e)}\nFallback error: {str(fallback_e)}")
    except FatalProviderError as e:
        # Non-retryable failure, surface immediately without fallback
        raise RuntimeError(str(e))
    except Exception as e:
        raise RuntimeError(str(e))

