from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    message: str = Field(..., max_length=10000, description="The message from the user")
    session_id: str | None = Field(default=None, description="The session ID if continuing a conversation")

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    iterations: int
    tool_calls: int

class HealthResponse(BaseModel):
    status: str

class DeleteSessionResponse(BaseModel):
    status: str
