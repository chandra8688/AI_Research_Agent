# Troubleshooting

This document records real issues that were encountered and fixed during V1.0 development, with precise references to the code changes that resolved them.

---

## Issue 1 — Groq 413 Payload Too Large

**Symptom**: Production Render logs showed:

```
model openai/gpt-oss-120b grounding request 9,511 tokens
Groq TPM limit: 8,000 tokens
-> HTTP 413 Payload Too Large
-> HTTP 429 on retry
```

**Root cause**: `_chunk_text()` in `quality.py` split the answer only on `\n\n` (double newlines). LLM-generated research answers commonly use single newlines for formatting within a paragraph. A 10 000-character section formatted with only `\n` separators was not split at all and was passed as a single grounding prompt, exceeding Groq''s 8 000-token per-message limit.

**Fix**: `quality.py:_chunk_text()` was rewritten with a three-level recursive fallback:
1. Split on `\n\n` (paragraph boundaries).
2. If a paragraph exceeds `max_length`, split further on `\n` (line boundaries).
3. If an individual line still exceeds `max_length`, hard-split by character count.

```python
def _chunk_text(text: str, max_length: int = 1500) -> list[str]:
    chunks = []

    def _add_to_chunks(piece: str, delimiter: str = "\n\n"):
        if not piece: return
        if not chunks: chunks.append(piece)
        elif len(chunks[-1]) + len(piece) + len(delimiter) <= max_length:
            chunks[-1] += delimiter + piece
        else: chunks.append(piece)

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
```

The delimiter length is included in the chunk size check (`len(chunks[-1]) + len(piece) + len(delimiter) <= max_length`), guaranteeing that no returned chunk ever exceeds `max_length`.

**Commit**: `912444e perf: harden grounding chunk limits`

**Verification**: `test_quality.py:test_19_chunk_text_robustness` asserts that chunks from single-newline text and from 2000-character lines are all strictly `<= 1500` characters.

---

## Issue 2 — Stale Chat Session (HTTP 404)

**Symptom**: After a Render deployment restart, the frontend submitted:
```
POST /chat HTTP/1.1 404 Not Found
WARNING: api.routes - Session ID <uuid> not found.
```

**Root cause**: Sessions are stored in a plain Python `dict` in `api/routes.py` — process memory only. When Render restarts the container (e.g., after a new deploy), the session registry is cleared. The browser''s `localStorage` retains the session UUID from the previous deployment, and the next chat submission sends a stale ID that no longer exists.

**Fix**: `frontend/app.js` was updated with a one-time 404 retry. The exact behavior (lines 378–407) is:

1. **Detect 404**: The 404 branch is entered only when `res.status === 404` **and** a `session_id` was included in the payload — meaning the stale session is confirmed as the cause.
2. **Clear `currentSessionId`**: The in-memory `currentSessionId` variable is set to `null`.
3. **Clear the stored session ID on the active conversation**: The matching conversation object in the `conversations` array (keyed by `currentConversationId`) has its `sessionId` field set to `null`, and `saveConversations()` persists this to `localStorage`.
4. **Retry exactly once**: A fresh `POST /chat` request is sent with only `{ message: text }` — no `session_id` — so the backend creates a new session.
5. **On retry success**: The new `session_id` from the response is stored in `currentSessionId`, and `ensureConversationExists()` persists it to `localStorage`. The response renders normally.
6. **On retry failure**: A generic error banner is shown. No further automatic retry is attempted.

```javascript
// app.js lines 378–407 (exact behavior)
} else if (res.status === 404 && payload.session_id) {
    currentSessionId = null;
    const staleConv = conversations.find(c => c.id === currentConversationId);
    if (staleConv) { staleConv.sessionId = null; saveConversations(); }

    const retryPayload = { message: text }; // no session_id — fresh session
    const retryRes = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(retryPayload)
    });

    if (retryRes.ok) {
        const data = await retryRes.json();
        if (data.session_id) {
            currentSessionId = data.session_id;
            ensureConversationExists(text); // persist new session_id
        }
        renderAgentResponseToDOM(data, loadingBox.bubble);
        pushMessageToHistory('agent', data);
    } else {
        // Retry also failed — surface a generic error, do not retry again.
        errorBanner.textContent = 'The agent encountered an error. Please try again.';
        errorBanner.className = 'error-banner';
        loadingBox.msgDiv.remove();
    }
}
```

**Commit**: `01305b3 fix: recover stale chat sessions`

---

## Issue 3 — Grounding Latency (60+ Claim Assessments Per Iteration)

**Symptom**: Production execution traces showed 60+ `claim_assessment` events per research iteration, with total query times of several minutes.

**Root cause 1 — Too many claims**: `extract_claims()` extracted every sentence from the LLM''s answer without a limit. A long research answer with 60–80 sentences generated 60–80 individual claim assessment events per `quality_check` run.

**Root cause 2 — Unconditional sleep**: `apply_grounding_gate()` contained an unconditional `time.sleep(2)` before each LLM call, adding at least 2 seconds per chunk regardless of whether rate limiting was occurring.

**Fix 1 — Claim cap**: A `_MAX_CLAIMS = 20` constant was added. When more than 20 candidate claims are extracted, they are ranked by unique meaningful token count and only the top 20 are retained.

```python
_MAX_CLAIMS = 20

if len(claims) > _MAX_CLAIMS:
    claims = sorted(claims, key=_claim_priority, reverse=True)[:_MAX_CLAIMS]
```

**Fix 2 — Sleep removal**: The unconditional `time.sleep(2)` was removed. Rate-limit back-off is handled exclusively by the `tenacity` `@retry` decorator with exponential back-off (min=3s, max=35s), which only sleeps when a `RetryableProviderError` is actually raised.

**Commit**: `dd63b9e perf: optimize grounding claim assessment`

---

## Issue 4 — Excessive Research Hedging

**Symptom**: Production answers frequently contained repetitive language:
- `"The retrieved evidence did not establish X''s activity regarding Y."`
- `"The evidence does not confirm..."`
- `"Sources could not be found for..."`

This occurred even when evidence existed for the main topic, with boilerplate hedging for minor missing details.

**Root cause**: Two prompt instructions were explicitly commanding this behaviour:

1. In `graph.py:force_synthesis`:
   ```
   "If evidence on a particular entity is missing, explicitly state that it could not be found."
   "If evidence for a specific entity is absent, acknowledge the gap explicitly."
   ```

2. In `quality.py:apply_grounding_gate` (Rule 2):
   ```
   "Rewrite them to state that the retrieved evidence did not establish or mention this
   (e.g., ''The retrieved evidence did not establish X''s activity regarding Y.'').
   Absence of evidence is not evidence of absence."
   ```

The first instructions caused the synthesis LLM to proactively insert gap-acknowledgement text. The second caused the grounding gate to replace every removed negative claim with a verbose "evidence did not establish" sentence, which itself could then be extracted as a new claim and create further hedging cycles.

**Fix**: Both instructions were tightened:

- `graph.py:force_synthesis`: The two "explicitly state missing" instructions were removed entirely. The synthesis prompt simply instructs the LLM not to invent unsupported claims.
- `quality.py:apply_grounding_gate` Rule 2: Changed from "rewrite to state evidence did not establish" to "gracefully omit rather than repeatedly stating that evidence is missing".

Strict citation requirements, grounding logic, conflict detection, and evidence-based uncertainty are preserved. The change only reduces the generation of repetitive formulaic hedging phrases.

**Commit**: `557ab50 refine: reduce unnecessary research hedging`

---

## Common Operational Issues

### Server starts but /ready returns 503

The selected LLM provider does not have an API key set in the environment. Check:
```bash
curl http://localhost:8000/ready
# {"detail": "Gemini provider selected but no GEMINI_API_KEY found."}
```

Set the correct API key or change `LLM_PROVIDER` in your `.env`.

### POST /chat returns 404 immediately

The `session_id` in the request is stale (server was restarted). The frontend handles this automatically. If calling the API manually, omit the `session_id` to create a fresh session.

### Research queries time out or return 429

Provider rate limits have been hit. This is common on free tiers during extended multi-step research. Options:
- Wait and retry (the tenacity back-off in the grounding gate handles transient 429s automatically).
- Use a paid tier or a different provider.
- Reduce `MAX_AGENT_ITERATIONS` or `MAX_REFLECTION_ATTEMPTS` in the environment.

### ChromaDB collection is empty after deployment

If `docs/` contained no `.txt` files when the Docker image was built, the RAG knowledge base is empty. The agent will use web search only. To populate the knowledge base, add `.txt` documents to `docs/` and rebuild the Docker image.

### HuggingFace model fails to download during build

The embedding model download requires internet access during `docker build`. If the build environment is offline, pre-download the model and set `HF_HOME` to point to the cached directory.
