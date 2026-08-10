from pydantic import BaseModel, Field

from config import settings

class ChatRequest(BaseModel):
    message: str = Field(..., max_length=settings.max_message_length, description="The message from the user")
    session_id: str | None = Field(default=None, description="The session ID if continuing a conversation")

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    iterations: int
    tool_calls: int
    sources: list[dict] | None = None
    trace: list[dict] | None = None

class HealthResponse(BaseModel):
    status: str

class DeleteSessionResponse(BaseModel):
    status: str

class ReadyResponse(BaseModel):
    status: str
    llm_provider: str
    vector_db: str

class ConfigResponse(BaseModel):
    environment: str
    llm_provider: str
    vector_db: str
    max_agent_iterations: int
    max_reflection_attempts: int
    max_message_length: int
