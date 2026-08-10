from pydantic import BaseModel
from llm import call_llm_structured

class ReflectionResult(BaseModel):
    sufficient: bool
    reason: str

def build_reflection_prompt(query: str, evidence: list) -> str:
    """Builds the prompt for evidence reflection without calling the LLM."""
    evidence_text = "\n\n".join([str(e) for e in evidence])
    
    from langchain_integration import get_reflection_prompt_template
    prompt_template = get_reflection_prompt_template()
    prompt_value = prompt_template.invoke({"query": query, "evidence": evidence_text})
    return prompt_value.to_string()

def evaluate_evidence(query: str, evidence: list) -> ReflectionResult:
    """
    Evaluates whether the retrieved evidence is sufficient to answer the query.
    Returns a structured ReflectionResult.
    """
    if not evidence:
        return ReflectionResult(sufficient=False, reason="No evidence was retrieved.")
        
    prompt = build_reflection_prompt(query, evidence)
    return call_llm_structured(prompt, ReflectionResult)
