from dataclasses import dataclass, field
import re

def is_simple_query(query: str) -> bool:
    """Conservatively classifies a query as simple factual to use the fast LLM path."""
    if not query:
        return False
        
    query_lower = query.lower().strip()
    
    # Too long? Probably complex.
    if len(query_lower) > 100:
        return False
        
    # Keywords that imply research, recency, comparison, or evidence
    complex_keywords = [
        "latest", "current", "compare", "versus", "vs", "developments",
        "2024", "2025", "2026", "evidence", "sources", "citations",
        "research", "find", "search", "differences", "market", "ecosystem"
    ]
    if any(kw in query_lower for kw in complex_keywords):
        return False
        
    # Must start with simple trivia question patterns
    simple_starts = ["what is", "who is", "who invented", "where is", "what's", "who's", "when did"]
    if any(query_lower.startswith(start) for start in simple_starts):
        return True
        
    # Very conservative default
    return False

@dataclass
class ResearchPlan:
    original_query: str
    intent: str
    requires_web: bool
    requires_local_knowledge: bool
    requires_calculation: bool
    requires_multi_source_research: bool
    steps: list[str] = field(default_factory=list)

def create_research_plan(query: str) -> ResearchPlan:
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")
    
    query_lower = query.lower()
    
    # Check flags
    # We use basic math operators and keywords to detect calculation
    calc_pattern = re.compile(r'(\d+\s*[\+\-\*\/]\s*\d+|calculate|multiply|add|subtract|divide|\*|\+|\-|/)')
    is_calc = bool(calc_pattern.search(query_lower))
    
    # Use deterministic keywords
    local_keywords = ["according to my", "local documentation", "my files", "local documents", "local knowledge", "my document"]
    is_local = any(kw in query_lower for kw in local_keywords)
    
    web_keywords = ["latest", "current", "today", "recent", "news", "web"]
    is_web = any(kw in query_lower for kw in web_keywords)
    
    requires_multi = is_local and is_web
    
    intent = "general_knowledge"
    if requires_multi:
        intent = "comparative_research"
        # Overwrite is_calc since multi-source research typically isn't a calc query, 
        # or we just let it be. We'll set is_calc=False for clarity if it's comparative.
        is_calc = False
    elif is_local:
        intent = "local_research"
        is_calc = False
    elif is_web:
        intent = "web_research"
        is_calc = False
    elif is_calc:
        intent = "calculation"
        
    # Build steps
    steps = []
    if intent == "calculation":
        steps.append("Calculate the result.")
        steps.append("Return the answer.")
    elif intent == "comparative_research":
        steps.append("Search the web for current information.")
        steps.append("Search local knowledge for relevant information.")
        steps.append("Compare evidence from both sources.")
        steps.append("Generate a synthesized answer with source attribution.")
    elif intent == "local_research":
        steps.append("Search local knowledge for information.")
        steps.append("Generate a grounded answer.")
    elif intent == "web_research":
        steps.append("Search the web for information.")
        steps.append("Generate an answer based on search results.")
    else:
        steps.append("Answer the question directly.")
        
    return ResearchPlan(
        original_query=query,
        intent=intent,
        requires_web=is_web,
        requires_local_knowledge=is_local,
        requires_calculation=is_calc,
        requires_multi_source_research=requires_multi,
        steps=steps
    )
