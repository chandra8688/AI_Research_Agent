class RetrievalError(RuntimeError):
    """Base class for errors raised by vector store backends."""
    pass

class RetryableRetrievalError(RetrievalError):
    """Raised when a vector backend experiences a transient failure (e.g., connection error, timeout)."""
    pass

class FatalRetrievalError(RetrievalError):
    """Raised when a vector backend experiences a non-retryable failure (e.g., invalid configuration)."""
    pass
