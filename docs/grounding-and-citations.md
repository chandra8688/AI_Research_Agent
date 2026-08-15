# Grounding and Citations

This document describes the evidence grounding and citation verification system in AI Research Agent V1.0. All logic described here lives in `quality.py`.

---

## Why Grounding Exists

LLMs generate plausible-sounding text that may go beyond what the retrieved evidence actually supports. Without verification, the agent could confidently state facts that were never in any retrieved source.

The grounding system addresses this by:
1. Extracting every factual claim from the LLM''s answer.
2. Checking each claim against the retrieved evidence.
3. Validating every citation tag against the actual source identifiers.
4. Rewriting problematic sections when grounding quality falls below a threshold.

The goal is not to prevent the agent from being uncertain — it is to prevent it from being confidently wrong.

---

## Step 1: Claim Extraction — `extract_claims(answer)`

**Input**: The LLM''s complete text answer.
**Output**: A list of claim strings (capped at 20).

### Extraction logic

1. **Table rows** are extracted first (`|...|` lines, separator rows excluded).
2. **Remaining text** has table markup removed, then is split on sentence terminators (`.`, `!`, `?`, `\n`).
3. Sentences are filtered:
   - Shorter than 15 characters are dropped.
   - Lines starting with `#` (headings) or ending with `:` are dropped.
   - Short sentences (< 50 chars) starting with boilerplate phrases (`"here is"`, `"based on"`, `"according to"`, etc.) are dropped.
4. Citation tags (`[LOCAL: ...]`, `[WEB: ...]`) are stripped from each sentence before it is stored as a claim.

### The 20-claim cap

If more than 20 candidate claims are extracted, the list is sorted by **information density** (unique meaningful token count, after stop-word removal) and the top 20 are retained. This prevents the grounding system from making dozens of LLM calls on very long answers, while ensuring the most information-dense claims are always checked.

```python
_MAX_CLAIMS = 20

def _claim_priority(claim: str) -> int:
    return len(get_meaningful_tokens(claim))

if len(claims) > _MAX_CLAIMS:
    claims = sorted(claims, key=_claim_priority, reverse=True)[:_MAX_CLAIMS]
```

---

## Step 2: Claim Assessment — `assess_claims(claims, evidence)`

**Input**: List of claim strings, list of `EvidenceItem` objects.
**Output**: `ResearchQualityReport` containing per-claim assessments and an overall grounding score.

### Token-overlap method

For each (claim, evidence_item) pair:
1. Meaningful tokens are extracted from both (lowercased, punctuation removed, stop words excluded).
2. `overlap = len(claim_tokens ∩ evidence_tokens) / len(claim_tokens)`
3. If `overlap >= 0.35`, the evidence item is listed as a supporting source.

```python
intersection = claim_tokens.intersection(ev_tokens)
overlap = len(intersection) / len(claim_tokens)
if overlap >= 0.35:
    supporting.append(item.source)
```

A claim is marked `supported=True` if any evidence item meets the overlap threshold and no conflicting evidence is detected.

### Conflict detection — `detect_conflict(claim, evidence_text)`

Two heuristics are applied:

1. **Numerical contradiction**: If the claim contains numbers (`\d+(?:\.\d+)?%?`) and the evidence also contains numbers, but their sets are completely disjoint — **and** the claim and evidence share at least 2 meaningful tokens — a conflict is flagged.

2. **Negation mismatch**: If the claim contains a negation (`"does not"`, `"is not"`, `"cannot"`, `"will not"`) that is not present in the evidence (or vice versa), and at least 40% of the claim''s tokens overlap with the evidence, a conflict is flagged.

> **Known limitation**: Both mechanisms operate on surface-level token patterns. Semantic contradictions that use different vocabulary are not detected.

### Grounding score

```python
score = (len(claims) - len(unsupported_claims)) / len(claims)
```

Ranges from 0.0 (all claims unsupported) to 1.0 (all claims supported).

---

## Step 3: Citation Validation — `validate_citations(answer, evidence)`

**Input**: The LLM''s answer text, list of `EvidenceItem` objects.
**Output**: List of invalid citation strings.

### Rules

1. If evidence was retrieved but **no citations at all** are present in the answer, `"No source citations were provided."` is returned as an invalid citation.
2. Every `[LOCAL: ...]` and `[WEB: ...]` tag is extracted from the answer.
3. For each citation, a substring match is performed against the set of actual `EvidenceItem.source` strings. The check is bidirectional: the citation must be a substring of a source string, **or** the source string must be a substring of the citation.
4. Any citation that fails this check is added to the invalid list.

```python
citations = re.findall(r'\[(?:LOCAL|WEB):\s*(.*?)\]', answer)
valid_sources = {item.source for item in evidence}
invalid = []
for cit in citations:
    cit_clean = cit.strip()
    if not any(cit_clean in v or v in cit_clean for v in valid_sources):
        invalid.append(cit_clean)
```

This is a **strict check** — invented source names, modified URLs, or fabricated titles will fail.

---

## Citation Formats

Two citation formats are enforced:

```
[WEB: Page Title (https://example.com/article)]
[LOCAL: filename.txt]
```

The `source` field of a web `EvidenceItem` is formatted as `"Page Title (URL)"` by `parse_web_evidence()`. The `source` field of a local `EvidenceItem` is the filename string from `load_documents()`.

The LLM is instructed to use these exact formats. The citation validator checks that the content inside the brackets is a recognisable substring of an actual source identifier.

---

## Step 4: Grounding Gate — `apply_grounding_gate(answer, report, evidence)`

The grounding gate is invoked from `graph.py:quality_check` when any of the following are true:
- `report.overall_grounding_score < settings.grounding_threshold` (default 0.70)
- `report.unsupported_claims` is non-empty
- `report.conflicts_detected` is non-empty

### Chunking

The answer is first split into chunks of at most **1 500 characters** by `_chunk_text()`.

The chunker applies a three-level split strategy to guarantee `len(chunk) <= max_length`:
1. Split on `\n\n` (paragraph boundaries).
2. If a paragraph exceeds `max_length`, split further on `\n`.
3. If an individual line still exceeds `max_length`, hard-split by character count.

This strict limit prevents Groq and other providers from receiving oversized grounding requests that would trigger HTTP 413 (Payload Too Large) errors.

### Per-chunk rewriting

For each chunk:
1. `_is_claim_relevant(claim, chunk)` identifies which unsupported claims and conflicts apply to this chunk (token overlap >= 30%).
2. If no issues apply to this chunk, it is passed through **unmodified**.
3. If issues exist, the LLM is called with a fact-checking prompt containing:
   - The available source catalog
   - The list of unsupported claims
   - The list of detected conflicts
   - The draft chunk text

The fact-checking prompt instructs the LLM to:
1. Remove or heavily qualify unsupported claims.
2. For unsupported negative claims (`"X did not Y"`), gracefully omit them rather than inserting hedging language.
3. Flag genuine conflicts (`"Sources disagree on X..."`).
4. Preserve all citation tags for supported claims.
5. Preserve headings and markdown structure.

### Retry logic

The grounding gate LLM call uses `tenacity` exponential back-off (min=3s, max=35s, up to 5 attempts) to handle transient provider rate limits without introducing unconditional sleeps.

### Fallback

If the LLM call fails after all retries, the original unmodified chunk is used. The pipeline does not abort.

---

## Quality Check Flow in `graph.py`

```python
# graph.py:quality_check (simplified)

claims = extract_claims(final_text)                   # <= 20 claims
quality_report = assess_claims(claims, evidence)      # token-overlap grounding
invalid_citations = validate_citations(final_text, evidence)

# First attempt: give the LLM a chance to self-correct
if not refinement_attempted and (unsupported_claims or invalid_citations):
    agent_state.refinement_attempted = True
    # Inject grounding warning into conversation and route back to agent_decide
    return {"next_action": "agent_decide"}

# Second attempt or clean answer: apply grounding gate if needed
if score < threshold or unsupported or conflicts:
    final_text = apply_grounding_gate(final_text, report, evidence)

agent_state.final_answer = final_text
```

---

## Summary of Thresholds and Limits

| Parameter | Value | Source |
|---|---|---|
| Maximum claims assessed | 20 | `quality.py:_MAX_CLAIMS` |
| Token overlap threshold (support) | 35% | `quality.py:assess_claims` |
| Token overlap threshold (conflict negation) | 40% | `quality.py:detect_conflict` |
| Token overlap threshold (chunk relevance) | 30% | `quality.py:_is_claim_relevant` |
| Grounding score threshold | 0.70 | `config.py:grounding_threshold` |
| Max grounding chunk size | 1500 chars | `quality.py:_chunk_text` |
| Max grounding LLM retries | 5 | `quality.py:apply_grounding_gate` |
| Retry back-off | 3–35s exponential | tenacity |
