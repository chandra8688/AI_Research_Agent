from pydantic import BaseModel
from llm import call_llm_structured

class ReflectionResult(BaseModel):
    sufficient: bool
    reason: str

def build_reflection_prompt(query: str, evidence: list) -> str:
    """Builds the prompt for evidence reflection without calling the LLM."""
    evidence_text = "\n\n".join([str(e) for e in evidence])
    
    prompt = f"""You are a strict evaluator for a research agent.
Your task is to determine whether the provided evidence is sufficient to accurately and fully answer the original query.

RULES:
1. Evaluate ONLY the supplied evidence.
2. Do NOT use outside knowledge to answer the query. 
3. Return `sufficient=True` ONLY when the evidence directly and fully supports the requested answer.
4. Return `sufficient=False` if the evidence is unrelated, incomplete, or does not contain the answer.
5. Provide a brief `reason` explaining your decision based on what the evidence actually contains.

ORIGINAL QUERY:
{query}

SUPPLIED EVIDENCE:
{evidence_text}
"""
    return prompt

def evaluate_evidence(query: str, evidence: list) -> ReflectionResult:
    """
    Evaluates whether the retrieved evidence is sufficient to answer the query.
    Returns a structured ReflectionResult.
    """
    if not evidence:
        return ReflectionResult(sufficient=False, reason="No evidence was retrieved.")
        
    prompt = build_reflection_prompt(query, evidence)
    return call_llm_structured(prompt, ReflectionResult)
