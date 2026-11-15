from fastapi import APIRouter, HTTPException, status
from pydantic import ValidationError
import logging

from config import settings
from api.models import ChatRequest, ChatResponse, HealthResponse, DeleteSessionResponse, ReadyResponse, ConfigResponse
from memory import AgentSession, create_session
from agent import execute_agent

router = APIRouter()

logger = logging.getLogger(__name__)

# In-memory session registry
sessions: dict[str, AgentSession] = {}

@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok")

@router.get("/ready", response_model=ReadyResponse)
def readiness_check():
    provider = (settings.llm_primary_provider or settings.llm_provider).lower().strip()
    vector_db = settings.vector_db.lower().strip()

    # Check Provider
    required_keys = {
        "gemini": ("GEMINI_API_KEY", settings.gemini_api_key),
        "openrouter": ("OPENROUTER_API_KEY", settings.openrouter_api_key),
        "groq": ("GROQ_API_KEY", settings.groq_api_key),
    }
    if provider not in required_keys:
        raise HTTPException(status_code=503, detail=f"Unsupported LLM provider configured: '{provider}'.")
    key_name, key_value = required_keys[provider]
    if not key_value:
        raise HTTPException(status_code=503, detail=f"{provider} provider selected but no {key_name} found.")

    # Check Vector DB
    if vector_db == "pinecone" and not settings.pinecone_api_key:
        raise HTTPException(status_code=503, detail="Pinecone vector DB selected but no PINECONE_API_KEY found.")
        
    return ReadyResponse(status="ready", llm_provider=provider, vector_db=vector_db)

@router.get("/config", response_model=ConfigResponse)
def get_config():
    return ConfigResponse(
        environment=settings.environment,
        llm_provider=settings.llm_provider,
        vector_db=settings.vector_db,
        max_agent_iterations=settings.max_agent_iterations,
        max_reflection_attempts=settings.max_reflection_attempts,
        max_message_length=settings.max_message_length
    )

def _collect_sources(state) -> list[dict]:
    """Deduplicated list of every evidence source (local chunks AND web pages) used."""
    sources = []
    seen = set()
    for ev in state.multi_source_evidence:
        if ev.source_type == "local":
            chunk = str(ev.metadata.get("chunk_index", ""))
            key = ("local", ev.source, chunk)
            entry = {"type": "local", "source": ev.source, "chunk_index": chunk}
        else:
            url = ev.metadata.get("url", "")
            key = ("web", url or ev.source)
            entry = {"type": "web", "source": ev.metadata.get("title") or ev.source, "url": url}
        if key not in seen:
            seen.add(key)
            sources.append(entry)
    return sources


@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty or whitespace.")
        
    if len(message) > settings.max_message_length:
        raise HTTPException(status_code=400, detail=f"Message exceeds maximum allowed length of {settings.max_message_length} characters.")

    # Session handling
    if request.session_id:
        session = sessions.get(request.session_id)
        if not session:
            logger.warning(f"Session ID {request.session_id} not found.")
            raise HTTPException(status_code=404, detail="Session ID not found.")
    else:
        session = create_session()
        sessions[session.session_id] = session
        logger.info(f"Created new session: {session.session_id}")

    try:
        # Note: execute_agent returns (final_answer, state)
        final_answer, state = execute_agent(message, session=session)
        
        sources = _collect_sources(state)

        # Extract trace
        trace_list = []
        for t in state.trace:
            trace_list.append({
                "event_type": t.event_type,
                "iteration": t.iteration,
                "details": t.details
            })
        
        return ChatResponse(
            session_id=session.session_id,
            answer=final_answer,
            iterations=state.iteration,
            tool_calls=len(state.tool_calls),
            sources=sources if sources else None,
            trace=trace_list
        )
    except ValueError as e:
        logger.warning(f"Validation error during agent execution: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        from providers.errors import RetryableProviderError, FatalProviderError
        if isinstance(e, RetryableProviderError) or type(e).__name__ == "RetryableProviderError":
            logger.warning(f"Retryable provider error: {e}")
            if "429" in str(e):
                raise HTTPException(status_code=429, detail="The AI provider is currently overwhelmed or out of quota. Please try again later.")
            raise HTTPException(status_code=503, detail="The AI provider experienced a transient network issue. Please retry your request.")
            
        if isinstance(e, FatalProviderError) or type(e).__name__ == "FatalProviderError":
            # Upstream (LLM provider) failure, e.g. bad request or invalid API key.
            logger.error(f"Fatal provider error: {e}")
            raise HTTPException(status_code=502, detail="The AI provider encountered a fatal error processing the request.")
            
        if isinstance(e, RuntimeError):
            logger.error(f"Runtime error during agent execution: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
        # Catch unexpected errors without exposing internals
        logger.exception("Unexpected error during agent execution.")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during execution.")

@router.delete("/sessions/{session_id}", response_model=DeleteSessionResponse)
def delete_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
        return DeleteSessionResponse(status="deleted")
    raise HTTPException(status_code=404, detail="Session ID not found.")
