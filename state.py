from dataclasses import dataclass, field
from typing import Any # using Any or type string to avoid circular imports if needed

@dataclass
class AgentState:
    query: str
    contents: list = field(default_factory=list)
    iteration: int = 0
    tool_calls: list = field(default_factory=list)
    tool_results: list = field(default_factory=list)
    retrieved_evidence: list = field(default_factory=list)
    final_answer: str | None = None
    reflection_result: Any = None
    reflection_attempts: int = 0
