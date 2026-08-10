from fastapi import APIRouter, HTTPException, status
from pydantic import ValidationError

from api.models import ChatRequest, ChatResponse, HealthResponse, DeleteSessionResponse
from memory import AgentSession, create_session
from agent import execute_agent

router = APIRouter()

# In-memory session registry
sessions: dict[str, AgentSession] = {}

@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok")

@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty or whitespace.")

    # Session handling
    if request.session_id:
        session = sessions.get(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session ID not found.")
    else:
        session = create_session()
        sessions[session.session_id] = session

    try:
        # Note: execute_agent returns (final_answer, state)
        final_answer, state = execute_agent(message, session=session)
        
        return ChatResponse(
            session_id=session.session_id,
            answer=final_answer,
            iterations=state.iteration,
            tool_calls=len(state.tool_calls)
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # Safe controlled runtime errors from agent
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # Catch unexpected errors without exposing internals
        raise HTTPException(status_code=500, detail="An unexpected error occurred during execution.")

@router.delete("/sessions/{session_id}", response_model=DeleteSessionResponse)
def delete_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
        return DeleteSessionResponse(status="deleted")
    raise HTTPException(status_code=404, detail="Session ID not found.")
