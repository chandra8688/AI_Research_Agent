class ProviderError(RuntimeError):
    """Base class for errors raised by LLM providers."""
    pass

class RetryableProviderError(ProviderError):
    """Raised when a provider experiences a transient failure (e.g., 429, 50x, timeout) and can be retried or fallen back."""
    pass

class FatalProviderError(ProviderError):
    """Raised when a provider experiences a non-retryable failure (e.g., 401 Unauthorized, 400 Bad Request)."""
    pass
