import os
from google import genai
from google.genai import errors
from google.genai import types
from tools import calculate_product, search_web, search_local_knowledge

# ---------------------------------------------------------------------------
# Tool Registry: maps function name → local Python callable
# ---------------------------------------------------------------------------
TOOL_REGISTRY = {
    "calculate_product": calculate_product,
    "search_web": search_web,
    "search_local_knowledge": search_local_knowledge,
}

# ---------------------------------------------------------------------------
# Function Declarations: tells Gemini what tools are available and their shape
# ---------------------------------------------------------------------------
FUNCTION_DECLARATIONS = [
    types.FunctionDeclaration(
        name="calculate_product",
        description="Calculates the product of two integers. Use this whenever multiplication is needed.",
        parameters={
            "type": "OBJECT",
            "properties": {
                "a": {"type": "INTEGER", "description": "The first integer"},
                "b": {"type": "INTEGER", "description": "The second integer"},
            },
            "required": ["a", "b"],
        },
    ),
    types.FunctionDeclaration(
        name="search_web",
        description=(
            "Searches the web for information on a given query using DuckDuckGo. "
            "Returns a list of results with title, URL, and a short snippet. "
            "Use this when you need current or specific external information to answer a question."
        ),
        parameters={
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING", "description": "The search query string"},
                "max_results": {
                    "type": "INTEGER",
                    "description": "Maximum number of results to return (1-10, default 3)",
                },
            },
            "required": ["query"],
        },
    ),
    types.FunctionDeclaration(
        name="search_local_knowledge",
        description=(
            "Searches the local document knowledge base for internal information on a given query. "
            "Use this ONLY when the user asks about local, internal, or provided documents, or topics "
            "where you need context from the internal knowledge base to answer properly."
        ),
        parameters={
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING", "description": "The search query string"},
            },
            "required": ["query"],
        },
    ),
]

MODEL = "gemini-3.5-flash"

from state import AgentState
from memory import AgentSession, create_session
from config import settings
from graph import execute_agent_graph

def execute_agent(prompt: str, max_iterations: int | None = None, session: AgentSession | None = None) -> tuple[str, AgentState]:
    """
    Runs a ReAct-style agent loop using LangGraph orchestration and returns both the final answer and the execution state.
    """
    return execute_agent_graph(prompt, max_iterations, session)

def run_agent(prompt: str, max_iterations: int | None = None, session: AgentSession | None = None) -> str:
    """
    Runs a ReAct-style agent loop.
    Returns only the final string answer to preserve existing behavior.
    """
    answer, _ = execute_agent(prompt, max_iterations, session)
    return answer
