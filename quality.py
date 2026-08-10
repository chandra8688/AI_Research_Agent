import re
from dataclasses import dataclass, field
from typing import Set
from research import EvidenceItem

@dataclass
class ClaimAssessment:
    claim: str
    supported: bool
    confidence: float
    supporting_sources: list[str]
    conflicting_sources: list[str]
    reason: str

@dataclass
class ResearchQualityReport:
    assessments: list[ClaimAssessment]
    overall_grounding_score: float
    unsupported_claims: list[str]
    conflicts_detected: list[str]

def extract_claims(answer: str) -> list[str]:
    """Lightweight deterministic extraction of claims."""
    # Split by common sentence terminators.
    raw_sentences = re.split(r'(?<=[.!?])\s+', answer)
    claims = []
    
    ignore_prefixes = (
        "here is", "based on", "the available", "sources disagree", 
        "i found", "in summary", "according to", "these results",
        "this could not be verified"
    )
    
    for sentence in raw_sentences:
        s = sentence.strip()
        if len(s) < 15:
            continue
        if s.startswith('#') or s.endswith(':'):
            continue
            
        lower_s = s.lower()
        if any(lower_s.startswith(p) for p in ignore_prefixes) and len(s) < 50:
            continue
            
        # Ignore citation blocks e.g. [LOCAL: file] to clean up the claim
        s_clean = re.sub(r'\[.*?\]', '', s).strip()
        # Remove trailing punctuation
        s_clean = re.sub(r'[.!?]+$', '', s_clean).strip()
        if len(s_clean) > 15:
            claims.append(s_clean)
            
    return claims

def get_meaningful_tokens(text: str) -> Set[str]:
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = text.split()
    stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "is", "are", "was", "were", "it", "this", "that", "by", "as", "from"}
    return {t for t in tokens if t not in stop_words and len(t) > 2}

def extract_numbers(text: str) -> Set[str]:
    return set(re.findall(r'\d+(?:\.\d+)?%?', text))

def detect_conflict(claim: str, evidence_text: str) -> bool:
    claim_lower = claim.lower()
    evidence_lower = evidence_text.lower()
    
    claim_nums = extract_numbers(claim_lower)
    ev_nums = extract_numbers(evidence_lower)
    
    if claim_nums and ev_nums and claim_nums.isdisjoint(ev_nums):
        c_tok = get_meaningful_tokens(claim_lower)
        e_tok = get_meaningful_tokens(evidence_lower)
        if len(c_tok.intersection(e_tok)) >= 2:
            return True

    negations = ["does not", "is not", "cannot", "will not"]
    for neg in negations:
        if neg in claim_lower and neg not in evidence_lower:
            affirm = claim_lower.replace(neg, "").strip()
            c_tok = get_meaningful_tokens(affirm)
            e_tok = get_meaningful_tokens(evidence_lower)
            if len(c_tok) > 0 and len(c_tok.intersection(e_tok)) / len(c_tok) >= 0.4:
                return True
        elif neg in evidence_lower and neg not in claim_lower:
            c_tok = get_meaningful_tokens(claim_lower)
            e_tok = get_meaningful_tokens(evidence_lower.replace(neg, ""))
            if len(c_tok) > 0 and len(c_tok.intersection(e_tok)) / len(c_tok) >= 0.4:
                return True
                
    return False

def assess_claims(claims: list[str], evidence: list[EvidenceItem]) -> ResearchQualityReport:
    assessments = []
    all_conflicts = []
    
    if not evidence:
        for claim in claims:
            assessments.append(ClaimAssessment(
                claim=claim, supported=False, confidence=0.0,
                supporting_sources=[], conflicting_sources=[], reason="No evidence provided."
            ))
        return ResearchQualityReport(
            assessments=assessments,
            overall_grounding_score=0.0,
            unsupported_claims=claims,
            conflicts_detected=[]
        )

    for claim in claims:
        claim_tokens = get_meaningful_tokens(claim)
        best_overlap = 0.0
        supporting = []
        conflicting = []
        
        for item in evidence:
            ev_tokens = get_meaningful_tokens(item.content)
            if not claim_tokens:
                overlap = 0.0
            else:
                intersection = claim_tokens.intersection(ev_tokens)
                overlap = len(intersection) / len(claim_tokens)
            
            if detect_conflict(claim, item.content):
                conflicting.append(item.source)
                all_conflicts.append(f"Conflict found regarding: '{claim}' in {item.source}")
            elif overlap >= 0.35: # Generous threshold for token overlap
                supporting.append(item.source)
                if overlap > best_overlap:
                    best_overlap = overlap

        if conflicting:
            supported = len(supporting) > 0
            reason = "Conflicting evidence detected." if not supported else "Partial support but has conflicts."
            assessments.append(ClaimAssessment(
                claim=claim, supported=supported, confidence=best_overlap,
                supporting_sources=supporting, conflicting_sources=conflicting, reason=reason
            ))
        else:
            supported = best_overlap >= 0.35
            reason = "Supported by evidence." if supported else "Insufficient overlap with evidence."
            assessments.append(ClaimAssessment(
                claim=claim, supported=supported, confidence=best_overlap,
                supporting_sources=supporting, conflicting_sources=[], reason=reason
            ))
            
    unsupported = [a.claim for a in assessments if not a.supported]
    score = (len(claims) - len(unsupported)) / len(claims) if claims else 1.0
    
    # Check global evidence conflicts
    # Example: one evidence says 80%, another says 65% for the same topic.
    # To keep it lightweight, we can just look for contradictory claims we already mapped
    # But for AI-220, explicit detection is enough.
    
    return ResearchQualityReport(
        assessments=assessments,
        overall_grounding_score=score,
        unsupported_claims=unsupported,
        conflicts_detected=list(set(all_conflicts))
    )

def validate_citations(answer: str, evidence: list[EvidenceItem]) -> list[str]:
    citations = re.findall(r'\[(?:LOCAL|WEB):\s*(.*?)\]', answer)
    valid_sources = {item.source for item in evidence}
    invalid = []
    
    for cit in citations:
        cit_clean = cit.strip()
        if not any(cit_clean in v or v in cit_clean for v in valid_sources):
            invalid.append(cit_clean)
            
    return invalid
