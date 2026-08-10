from dataclasses import dataclass, field
import uuid

@dataclass
class ConversationMessage:
    role: str
    content: str

class ConversationMemory:
    def __init__(self):
        self.messages: list[ConversationMessage] = []
        
    def add_user_message(self, content: str):
        self.messages.append(ConversationMessage(role="user", content=content))
        
    def add_assistant_message(self, content: str):
        self.messages.append(ConversationMessage(role="model", content=content))
        
    def get_messages(self) -> list[ConversationMessage]:
        return self.messages
        
    def clear(self):
        self.messages.clear()
        
    def __len__(self) -> int:
        return len(self.messages)

@dataclass
class AgentSession:
    session_id: str
    memory: ConversationMemory = field(default_factory=ConversationMemory)

def create_session() -> AgentSession:
    return AgentSession(session_id=str(uuid.uuid4()))
