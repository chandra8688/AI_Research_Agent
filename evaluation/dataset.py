from dataclasses import dataclass, field
from typing import Optional

@dataclass
class EvaluationCase:
    name: str
    query: str
    expected_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    expected_sources: list[str] = field(default_factory=list)
    require_non_empty_answer: bool = True
    expected_error: Optional[str] = None

EVALUATION_CASES = [
    EvaluationCase(
        name="calculator",
        query="What is 42 multiplied by 7?",
        expected_tools=["calculate_product"],
        forbidden_tools=["search_local_knowledge"],
    ),
    EvaluationCase(
        name="local_rag",
        query="What does the local documentation say about RAG?",
        expected_tools=["search_local_knowledge"],
        expected_sources=["rag_overview.txt"],
    ),
    EvaluationCase(
        name="fine_tuning_rag",
        query="According to the local documentation, how does fine-tuning differ from RAG?",
        expected_tools=["search_local_knowledge"],
        expected_sources=["fine_tuning_overview.txt", "rag_overview.txt"],
    ),
    EvaluationCase(
        name="general_knowledge",
        query="What is the capital of France?",
        forbidden_tools=["search_local_knowledge"],
    ),
    EvaluationCase(
        name="web_search",
        query="Find recent information about retrieval augmented generation.",
        expected_tools=["search_web"],
    ),
    EvaluationCase(
        name="empty_input",
        query="",
        expected_error="ValueError",
    ),
    EvaluationCase(
        name="whitespace_input",
        query="    ",
        expected_error="ValueError",
    ),
    EvaluationCase(
        name="unknown_topic",
        query="What does the local documentation say about quantum computing?",
        expected_tools=["search_local_knowledge"],
    ),
]
