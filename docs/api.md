# API Reference

The AI Research Agent exposes a REST API via FastAPI. All routes are defined in `api/routes.py` and mounted at the root path by `api_server.py`.

Base URL (local): `http://127.0.0.1:8000`

---

## Endpoints

### GET /health

Liveness check. Always returns 200 if the process is running.

**Response:**
```json
{ "status": "ok" }
```

**curl:**
```bash
curl http://127.0.0.1:8000/health
```

---

### GET /ready

Readiness check. Validates that the configured LLM provider has a key set and, if Pinecone is selected, that a Pinecone API key is present.

**Response (200):**
```json
{
  "status": "ready",
  "llm_provider": "gemini",
  "vector_db": "chroma"
}
```

**Error responses:**
| Status | Condition |
|---|---|
| 503 | Gemini selected but `GEMINI_API_KEY` missing |
| 503 | OpenRouter selected but `OPENROUTER_API_KEY` missing |
| 503 | Pinecone vector DB selected but `PINECONE_API_KEY` missing |

**curl:**
```bash
curl http://127.0.0.1:8000/ready
```

---

### GET /config

Returns public runtime configuration. Does not expose API keys or secrets.

**Response (200):**
```json
{
  "environment": "development",
  "llm_provider": "gemini",
  "vector_db": "chroma",
  "max_agent_iterations": 5,
  "max_reflection_attempts": 2,
  "max_message_length": 10000
}
```

**curl:**
```bash
curl http://127.0.0.1:8000/config
```

---

### POST /chat

Submit a research query and receive a grounded answer.

**Request body:**
```json
{
  "message": "What are the latest AI agent developments in 2026?",
  "session_id": "optional-uuid-string"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `message` | string | Yes | The user query. Max length: `max_message_length` (default 10 000) |
| `session_id` | string | No | UUID from a previous response to continue a conversation |

**Behaviour:**
- If `session_id` is omitted, a new session is created.
- If `session_id` is provided and found, the existing conversation memory is used.
- If `session_id` is provided but not found (e.g., after a server restart), HTTP 404 is returned.

**Response (200):**
```json
{
  "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "answer": "According to retrieved sources, ...",
  "iterations": 2,
  "tool_calls": 3,
  "sources": [
    { "source": "rag_overview.txt", "chunk_index": "0" }
  ],
  "trace": [
    {
      "event_type": "graph_start",
      "iteration": 0,
      "details": {}
    },
    {
      "event_type": "claim_extraction",
      "iteration": 2,
      "details": { "count": 12 }
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `session_id` | string | UUID to include in follow-up requests |
| `answer` | string | The final grounded answer |
| `iterations` | int | Number of agent loop iterations used |
| `tool_calls` | int | Total number of tool calls made |
| `sources` | array or null | Local knowledge sources cited (from `retrieved_evidence`; does not include web sources) |
| `trace` | array | Full execution trace with event types and details |

**Note on `sources`**: The `sources` field in the response is populated from `state.retrieved_evidence`, which contains only the raw local knowledge search results. Web sources appear inside the `answer` text as `[WEB: ...]` citations. This is a known inconsistency in the V1.0 response schema.

**Error responses:**

| Status | Condition |
|---|---|
| 400 | Empty message |
| 400 | Message exceeds `max_message_length` |
| 404 | `session_id` provided but session not found |
| 429 | Provider rate limit exceeded (HTTP 429 from upstream) |
| 500 | Runtime error during agent execution |
| 502 | Fatal provider error (bad request to upstream) |
| 503 | Provider transient network failure |

**curl:**
```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the latest developments in AI agents in 2026?"}'
```

Follow-up in the same session:
```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Summarise that in one sentence.", "session_id": "3fa85f64-..."}'
```

---

### DELETE /sessions/{session_id}

Deletes an in-memory session.

**Response (200):**
```json
{ "status": "deleted" }
```

**Error responses:**
| Status | Condition |
|---|---|
| 404 | Session not found |

**curl:**
```bash
curl -X DELETE http://127.0.0.1:8000/sessions/3fa85f64-5717-4562-b3fc-2c963f66afa6
```

---

## Trace Event Types

The `trace` array in the chat response contains events emitted by `AgentState.add_trace()` throughout the pipeline:

| Event type | When emitted |
|---|---|
| `graph_start` | Beginning of graph execution |
| `classified_as_simple_query` | Query is routed to the fast path |
| `research_plan` | After planning is complete |
| `agent_start` | First agent decision iteration |
| `iteration_start` | Each new agent iteration |
| `tool_call` | When a tool is dispatched |
| `tool_result` | When a tool returns a result |
| `research_source_selected` | When `search_web` or `search_local_knowledge` is called |
| `evidence_collected` | After evidence items are accumulated |
| `evidence_synthesis` | Before reflection runs |
| `reflection` | Reflection result (sufficient/insufficient + reason) |
| `reflection_limit_reached` | When max_reflection_attempts is exhausted |
| `force_synthesis` | When force_synthesis node runs |
| `claim_extraction` | Number of claims extracted |
| `claim_assessment` | Per-claim support/conflict result |
| `grounding_check` | Summary of grounding score and unsupported claims |
| `citation_validation` | Summary of citation check results |
| `grounding_gate_triggered` | When the grounding gate is invoked |
| `final_answer` | Answer is committed to session memory |
| `graph_end` | End of graph execution |
| `agent_error` | Any error during execution |
| `fast_llm_path` | Fast path used |

---

## FastAPI Interactive Docs

The auto-generated OpenAPI documentation is available at:
```
http://127.0.0.1:8000/docs
```
