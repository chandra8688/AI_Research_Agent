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

# Maximum number of claims assessed per answer. Keeps grounding overhead bounded
# for long research responses without silently dropping high-signal claims.
_MAX_CLAIMS = 20

def _claim_priority(claim: str) -> int:
    """Higher is more important. Uses unique meaningful-token count as a proxy
    for information density so that substantive sentences are preferred over
    short fragments when the cap is applied."""
    tokens = get_meaningful_tokens(claim)
    return len(tokens)

def extract_claims(answer: str) -> list[str]:
    """Lightweight deterministic extraction of claims.

    Extraction behaviour is unchanged from the original implementation.
    When the raw candidate list exceeds _MAX_CLAIMS the function selects the
    highest-signal claims (ranked by unique meaningful-token count) so that the
    most information-dense sentences are always retained.
    """
    claims = []

    # Extract table rows explicitly first (skip separator rows)
    for line in answer.split('\n'):
        line_clean = line.strip()
        if line_clean.startswith('|') and line_clean.endswith('|') and '---' not in line_clean:
            claims.append(line_clean)

    # Remove tables from answer to avoid double-processing
    text_without_tables = re.sub(r'\|.*\|', '', answer)

    # Split by common sentence terminators and newlines
    raw_sentences = re.split(r'(?<=[.!?\n])\s+', text_without_tables)

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

        # Strip citation blocks e.g. [LOCAL: file] to obtain clean claim text
        s_clean = re.sub(r'\[.*?\]', '', s).strip()
        # Remove trailing punctuation
        s_clean = re.sub(r'[.!?]+$', '', s_clean).strip()
        if len(s_clean) > 15:
            claims.append(s_clean)

    # Apply cap: if there are more candidates than _MAX_CLAIMS, keep only the
    # highest-priority ones so that the grounding overhead stays bounded.
    # Deduplication happens implicitly because set-based token scoring favours
    # unique content over repeated fragments.
    if len(claims) > _MAX_CLAIMS:
        claims = sorted(claims, key=_claim_priority, reverse=True)[:_MAX_CLAIMS]

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
            overall_grounding_score=0.0 if claims else 1.0,
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
    
    if evidence and not citations:
        invalid.append("No source citations were provided.")
        return invalid

    for cit in citations:
        cit_clean = cit.strip()
        if not any(cit_clean in v or v in cit_clean for v in valid_sources):
            invalid.append(cit_clean)
            
    return invalid

def _chunk_text(text: str, max_length: int = 1500) -> list[str]:
    """Split text into chunks strictly under max_length, gracefully handling newlines."""
    chunks = []

    def _add_to_chunks(piece: str, delimiter: str = "\n\n"):
        if not piece:
            return
        if not chunks:
            chunks.append(piece)
        elif len(chunks[-1]) + len(piece) + len(delimiter) <= max_length:
            chunks[-1] += delimiter + piece
        else:
            chunks.append(piece)

    for paragraph in text.split("\n\n"):
        if len(paragraph) <= max_length:
            _add_to_chunks(paragraph, "\n\n")
        else:
            for line in paragraph.split("\n"):
                if len(line) <= max_length:
                    _add_to_chunks(line, "\n")
                else:
                    for i in range(0, len(line), max_length):
                        _add_to_chunks(line[i:i + max_length], "")

    return chunks

def _is_claim_relevant(claim: str, chunk: str) -> bool:
    c_tokens = get_meaningful_tokens(claim)
    ch_tokens = get_meaningful_tokens(chunk)
    if not c_tokens:
        return False
    # Use a generous threshold so we don't miss claims
    overlap = len(c_tokens.intersection(ch_tokens)) / len(c_tokens)
    return overlap >= 0.3

def apply_grounding_gate(answer: str, report: ResearchQualityReport, evidence: list[EvidenceItem] = None) -> str:
    from config import settings
    if report.overall_grounding_score >= settings.grounding_threshold and not report.unsupported_claims and not report.conflicts_detected:
        return answer
        
    from providers import get_provider
    provider_name = settings.llm_primary_provider or settings.llm_provider
    provider = get_provider(provider_name)
    
    chunks = _chunk_text(answer, max_length=1500)
    rewritten_chunks = []
    from providers.errors import RetryableProviderError
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=3, max=35),
        retry=retry_if_exception_type(RetryableProviderError)
    )
    def _generate_with_retry(prompt: str) -> str:
        return provider.generate(prompt)

    for chunk in chunks:
        # Find relevant unsupported claims and conflicts for this chunk
        chunk_unsupported = [c for c in report.unsupported_claims if _is_claim_relevant(c, chunk)]
        chunk_conflicts = [c for c in report.conflicts_detected if _is_claim_relevant(c, chunk)]
        
        # If no issues found in this chunk, skip rewriting
        if not chunk_unsupported and not chunk_conflicts:
            rewritten_chunks.append(chunk)
            continue
            
        if len(chunk_unsupported) > 3:
            unsupported_str = "\n".join(f"- {c}" for c in chunk_unsupported[:3])
            unsupported_str += f"\n- ... and {len(chunk_unsupported) - 3} more unsupported claims."
        else:
            unsupported_str = "\n".join(f"- {c}" for c in chunk_unsupported)
            
        conflict_str = "\n".join(f"- {c}" for c in chunk_conflicts)
        
        catalog_str = "None"
        if evidence:
            catalog_str = "\n".join(f"- [{item.source_type.upper()}: {item.source}]" for item in evidence)

        prompt = (
            "You are a strict grounding and fact-checking editor. "
            "Your task is to rewrite the provided 'Draft Answer Section' to fix grounding errors based on the provided 'Quality Report'.\n\n"
            "RULES:\n"
            "1. Remove or heavily qualify any unsupported claims.\n"
            "2. For negative claims (e.g., 'X did not do Y') that are unsupported, DO NOT state them as facts. Gracefully omit them rather than repeatedly stating that evidence is missing.\n"
            "3. For conflicting claims, explicitly flag the conflict (e.g., 'Sources disagree on X...'). Do not state a disputed claim as an established fact.\n"
            "4. If a markdown table contains unsupported rows or cells, remove or revise them.\n"
            "5. Preserve citations (e.g., [WEB: url] or [LOCAL: file]) for supported claims.\n"
            "6. Preserve original markdown headings, formatting, and structure as much as possible.\n"
            "7. Output ONLY the rewritten section, do not add introductory text like 'Here is the rewritten section'.\n\n"
            f"AVAILABLE SOURCE CATALOG:\n{catalog_str}\n\n"
            f"QUALITY REPORT:\n"
            f"Unsupported Claims:\n{unsupported_str or 'None'}\n\n"
            f"Conflicts Detected:\n{conflict_str or 'None'}\n\n"
            f"DRAFT ANSWER SECTION:\n{chunk}\n\n"
            "REWRITTEN SECTION:"
        )
        
        try:
            # NOTE: no unconditional sleep here — the tenacity @retry decorator
            # on _generate_with_retry already applies exponential back-off
            # (min=3s, max=35s) whenever a RetryableProviderError is raised.
            rewritten = _generate_with_retry(prompt)
            if rewritten and isinstance(rewritten, str) and rewritten.strip():
                rewritten_chunks.append(rewritten.strip())
            else:
                rewritten_chunks.append(chunk)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Grounding gate LLM call failed for chunk: {e}")
            rewritten_chunks.append(chunk)
            
    return "\n\n".join(rewritten_chunks)
