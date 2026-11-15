from dataclasses import dataclass, field
import re

def _contains_any_word(text: str, phrases: list[str]) -> bool:
    """True if any phrase occurs in text as whole word(s), not as a substring of a word."""
    return any(re.search(r"\b" + re.escape(p) + r"\b", text) for p in phrases)


# Arithmetic expressions ("42 * 17", "6x7") or explicit calculation verbs. Bare operator
# characters are NOT enough: hyphens and slashes appear in ordinary text and URLs.
_CALC_PATTERN = re.compile(
    r"\d+(?:\.\d+)?\s*[+\-*/x×÷]\s*\d+"
    r"|\b(?:calculate|compute|multiply|multiplied|times|product of|divide|divided|subtract|sum of)\b"
)


def is_simple_query(query: str) -> bool:
    """Conservatively classifies a query as simple factual to use the fast LLM path."""
    if not query:
        return False
        
    query_lower = query.lower().strip()
    
    # Too long? Probably complex.
    if len(query_lower) > 100:
        return False
        
    # Keywords that imply research, recency, comparison, or evidence
    # Matched as whole words so e.g. "vs" does not fire inside "canvas".
    complex_keywords = [
        "latest", "current", "currently", "now", "today", "recent", "news", "compare",
        "versus", "vs", "developments", "evidence", "sources", "citations", "cite",
        "research", "find", "search", "differences", "market", "ecosystem",
        "local", "documentation", "documents", "my files", "price", "ceo", "president",
    ]
    if _contains_any_word(query_lower, complex_keywords) or re.search(r"\b(19|20)\d{2}\b", query_lower):
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
    
    # Deterministic keyword checks (whole-word matching avoids hits like "currency" -> "current")
    is_calc = bool(_CALC_PATTERN.search(query_lower))

    local_keywords = ["according to my", "local documentation", "my files", "local documents",
                      "local knowledge", "my document", "my documents", "knowledge base"]
    is_local = _contains_any_word(query_lower, local_keywords)

    web_keywords = ["latest", "current", "currently", "today", "recent", "recently", "news", "web", "online", "this week", "this year"]
    is_web = _contains_any_word(query_lower, web_keywords)
    
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
