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
    provider = settings.llm_provider.lower().strip()
    vector_db = settings.vector_db.lower().strip()
    
    # Check Provider
    if provider == "gemini" and not settings.gemini_api_key:
        raise HTTPException(status_code=503, detail="Gemini provider selected but no GEMINI_API_KEY found.")
    elif provider == "openrouter" and not settings.openrouter_api_key:
        raise HTTPException(status_code=503, detail="OpenRouter provider selected but no OPENROUTER_API_KEY found.")
        
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
        
        # Extract sources from retrieved evidence
        import re
        sources = []
        seen = set()
        for ev in state.retrieved_evidence:
            matches = re.findall(r"Source:\s*(.*?)\s*\(Chunk\s*([^)]+)\)", ev)
            for src, chunk in matches:
                key = f"{src}_{chunk}"
                if key not in seen:
                    seen.add(key)
                    sources.append({"source": src.strip(), "chunk_index": chunk.strip()})
                    
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
            logger.error(f"Fatal provider error: {e}")
            if "400" in str(e):
                raise HTTPException(status_code=502, detail="The AI provider encountered a fatal error processing the request.")
            raise HTTPException(status_code=500, detail="The AI provider encountered a fatal error processing the request.")
            
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
