import logging

from pydantic import BaseModel

logger = logging.getLogger(__name__)

def call_llm(prompt: str) -> str:
    """Calls the configured LLM API (Gemini or OpenRouter) with the given prompt and returns the text response."""
    from config import settings
    from providers import get_provider
    from providers.errors import RetryableProviderError, FatalProviderError
    
    primary_name = settings.llm_primary_provider or settings.llm_provider
    logger.debug("Primary provider: %s", primary_name)
    primary_provider = get_provider(primary_name)
    
    try:
        return primary_provider.generate(prompt)
    except RetryableProviderError as e:
        if not settings.llm_fallback_enabled:
            raise
            
        logger.warning("Primary provider failed: %s", e)
        logger.warning("Falling back to: %s", settings.llm_fallback_provider)
        
        try:
            fallback_provider = get_provider(settings.llm_fallback_provider)
            response = fallback_provider.generate(prompt)
            logger.info("Fallback provider succeeded")
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
    logger.debug("Primary provider (structured): %s", primary_name)
    primary_provider = get_provider(primary_name)
    
    try:
        return primary_provider.generate_structured(prompt, schema)
    except RetryableProviderError as e:
        if not settings.llm_fallback_enabled:
            raise RuntimeError(str(e))
            
        logger.warning("Primary provider (structured) failed: %s", e)
        logger.warning("Falling back to (structured): %s", settings.llm_fallback_provider)
        
        try:
            fallback_provider = get_provider(settings.llm_fallback_provider)
            response = fallback_provider.generate_structured(prompt, schema)
            logger.info("Fallback provider (structured) succeeded")
            return response
        except Exception as fallback_e:
            raise RuntimeError(f"Both providers failed.\nPrimary error: {str(e)}\nFallback error: {str(fallback_e)}")
    except FatalProviderError as e:
        # Non-retryable failure, surface immediately without fallback
        raise RuntimeError(str(e))
    except Exception as e:
        raise RuntimeError(str(e))

