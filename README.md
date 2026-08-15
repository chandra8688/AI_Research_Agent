# AI Research Agent

> An autonomous, grounded research system built on LangGraph that retrieves, synthesises, and validates information from the web and a local knowledge base — with strict citation enforcement throughout.

---

## Overview

AI Research Agent is a production-deployed Python application that answers research questions through a multi-step evidence pipeline rather than relying on an LLM''s parametric memory. For every research query it:

1. **Plans** a retrieval strategy based on query classification.
2. **Retrieves** evidence from DuckDuckGo web search and/or a local vector knowledge base.
3. **Reflects** on whether the gathered evidence is sufficient to answer the question.
4. **Synthesises** an answer grounded exclusively in retrieved evidence.
5. **Validates** every factual claim and citation before the answer reaches the user.

Simple factual questions bypass the full pipeline through a fast path, returning answers directly from the LLM without unnecessary tool calls.

---

## Why This Project Exists

Modern LLMs hallucinate. They confidently state things that are factually wrong, and they rarely distinguish between what they know and what they are inventing. This project explores a pragmatic mitigation: force the model to produce citations from explicitly retrieved sources and then automatically verify those citations before the answer is returned.

The implementation deliberately avoids black-box RAG frameworks in favour of a fully traceable, testable pipeline where every grounding decision is observable.

---

## Key Capabilities

| Capability | Details |
|---|---|
| Autonomous research loop | LangGraph ReAct graph with configurable iteration and reflection limits |
| Web retrieval | DuckDuckGo (`ddgs`) — no API key required |
| Local RAG | ChromaDB vector store with `all-MiniLM-L6-v2` sentence embeddings |
| Multi-source fusion | Optional Chroma + Pinecone retrieval fusion with Reciprocal Rank Fusion |
| Evidence reflection | LLM-based sufficiency evaluation before synthesis |
| Claim grounding | Token-overlap claim assessment against retrieved evidence |
| Citation validation | Strict substring match of `[WEB: ...]` / `[LOCAL: ...]` tags |
| Grounding gate | Automatic answer rewriting when grounding score falls below threshold |
| Simple query fast path | Lightweight direct LLM answer for trivial queries |
| Provider abstraction | Gemini, OpenRouter, and Groq — configurable via environment variables |
| Persistent vector store | Chroma persisted to disk; optional Pinecone cloud fallback |
| Session memory | UUID-keyed in-memory conversation sessions |
| Production deployment | Containerised via Docker, deployed on Render |
| Full test suite | 125 offline unit tests covering all major pipeline components |

---

## Architecture Overview

```
User Browser
     |
     v
FastAPI (api_server.py)
     |  Session lookup / creation
     |  Request validation
     v
LangGraph Graph (graph.py)
     |
     +-- validate          -> input sanitisation, session init
     +-- plan_research     -> intent classification (planning.py)
     |
     +-[simple query]-------> fast_llm_path -> quality_check -> END
     |
     +-- agent_decide      -> LLM chooses next action
     |       |
     |       +-- execute_tools --> search_web (DuckDuckGo)
     |       |                     search_local_knowledge (ChromaDB)
     |       |                     calculate_product
     |       |
     |       +-- collect_evidence -> accumulates EvidenceItems
     |
     +-- reflection        -> evaluate_evidence (reflection.py)
     |       |
     |       +-[insufficient + attempts remain]--> agent_decide (more research)
     |       +-[limit reached]-------------------> force_synthesis
     |
     +-- agent_decide (synthesis) -> LLM generates final answer from evidence
     |
     +-- quality_check (quality.py)
             |
             +-- extract_claims  (<= 20 claims, priority-ranked)
             +-- assess_claims   (token-overlap grounding)
             +-- validate_citations
             +-[grounding issues]--> grounding gate (LLM rewrite)
             +-- final_answer -> response
```

---

## Agent Workflow

1. **Validation** (`graph.py:validate`): Sanitises the prompt, attaches or creates a conversation session, and classifies the query as simple or complex.

2. **Research Planning** (`planning.py:create_research_plan`): Uses keyword matching to determine whether the query requires web search, local knowledge, calculation, or multi-source research. Returns a `ResearchPlan` with intent classification and ordered steps.

3. **Agent Decision Loop** (`graph.py:agent_decide`): Calls the configured LLM provider with the current conversation context and available tool declarations. The LLM either calls a tool or signals that it is ready to produce a final answer.

4. **Tool Execution** (`graph.py:execute_tools`): Dispatches the LLM''s chosen tool call (`search_web`, `search_local_knowledge`, or `calculate_product`) and collects results.

5. **Evidence Collection** (`graph.py:collect_evidence`): Parses raw tool results into structured `EvidenceItem` objects and appends them to the agent state.

6. **Reflection** (`reflection.py:evaluate_evidence`): Calls the LLM with a strict evaluator prompt to judge whether evidence is sufficient (`ReflectionResult.sufficient`). If insufficient and attempts remain, the agent is sent back to search with a refined query. If the attempt limit (`max_reflection_attempts`, default 2) is reached, the pipeline routes to `force_synthesis`.

7. **Synthesis**: The LLM generates a final answer citing retrieved sources using `[WEB: title (URL)]` or `[LOCAL: filename]` tags.

8. **Quality Check** (`quality.py`): Claims are extracted, assessed against evidence, and citations are validated. Grounding issues trigger a targeted LLM rewrite (the grounding gate).

---

## RAG Workflow

```
docs/ directory (.txt files)
        |
        v
load_documents()       <- rag/loader.py  -- reads .txt files
        |
        v
chunk_documents()      <- rag/chunker.py -- 500-char sliding window, 100-char overlap
        |
        v
embed_chunks()         <- rag/embedder.py -- all-MiniLM-L6-v2 (384-dim, CPU, local)
        |
        v
ChromaDB               <- persisted to .chroma_db/
        |
        v (at query time)
embed query -> vector search -> top-3 chunks -> EvidenceItem(source_type="local")
```

Local evidence is ingested once during Docker build (`scripts/build_rag.py`) or on first startup.
Web evidence enters the pipeline through DuckDuckGo snippets and is parsed into `EvidenceItem(source_type="web")` objects.

---

## Grounding and Citation System

The grounding system verifies factual honesty before any answer is returned:

1. **Claim Extraction**: Sentences are parsed from the LLM''s answer, with table rows handled separately. Short or boilerplate sentences are filtered. A maximum of 20 claims are selected by information density (unique meaningful token count) to bound processing overhead.

2. **Claim Assessment**: Each claim is compared against every `EvidenceItem` using set-based token overlap. A claim is marked `supported` if overlap >= 35% with at least one evidence item. Numerical contradictions and negation patterns are checked to detect conflicts.

3. **Citation Validation**: Every `[WEB: ...]` and `[LOCAL: ...]` tag in the answer is checked for a substring match against the actual retrieved source identifiers. Invalid or invented citations are flagged.

4. **Grounding Gate**: If the grounding score falls below the threshold (default 0.70), unsupported claims exist, or conflicts are detected, the answer is chunked (<= 1500 characters/chunk) and each problematic chunk is rewritten by the LLM using a strict fact-checking prompt. Chunks without issues are passed through unmodified.

**Citation formats:**

```
[WEB: Page Title (https://example.com/path)]
[LOCAL: document_filename.txt]
```

> **Limitation**: Claim assessment uses token overlap, not semantic similarity. A claim can technically satisfy the threshold by sharing keywords with evidence that has a different meaning. This is a known limitation of the V1.0 implementation.

---

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| API framework | FastAPI + Uvicorn |
| Orchestration | LangGraph (`StateGraph`) |
| LLM providers | Google Gemini, OpenRouter, Groq |
| Embeddings | `sentence-transformers` / `all-MiniLM-L6-v2` (local, CPU) |
| Primary vector DB | ChromaDB (persisted to disk) |
| Optional vector DB | Pinecone (cloud, configurable fallback) |
| Web search | DuckDuckGo Search (`ddgs`) |
| Prompt templates | LangChain Core (`ChatPromptTemplate`) |
| Settings | `pydantic-settings` |
| Frontend | Vanilla HTML / CSS / JavaScript |
| Containerisation | Docker |
| Deployment | Render (web service) |

---

## Project Structure

```
AI_Research_Agent/
+-- api/
|   +-- models.py          # Pydantic request/response schemas
|   +-- routes.py          # FastAPI route handlers, session management
+-- docs/                  # Technical documentation
+-- frontend/
|   +-- index.html         # Single-page application shell
|   +-- style.css          # Application styling
|   +-- app.js             # Chat UI logic, session management, API calls
+-- providers/
|   +-- __init__.py        # get_provider() factory, AgentResponse dataclass
|   +-- gemini.py          # Google Gemini provider
|   +-- groq.py            # Groq provider
|   +-- openrouter.py      # OpenRouter provider
|   +-- errors.py          # RetryableProviderError, FatalProviderError
+-- rag/
|   +-- loader.py          # .txt document loader -> Document objects
|   +-- chunker.py         # Sliding-window character chunker (500 chars, 100 overlap)
|   +-- embedder.py        # SentenceTransformer embedder (all-MiniLM-L6-v2, 384-dim)
|   +-- chroma_store.py    # ChromaDB vector store implementation
|   +-- pinecone_store.py  # Pinecone vector store implementation
|   +-- store.py           # VectorStore Protocol + FallbackVectorStore + FusionVectorStore
|   +-- reranker.py        # Reciprocal Rank Fusion for multi-backend results
|   +-- pipeline.py        # Knowledge base initialisation pipeline
|   +-- generator.py       # RAG generation helper
+-- scripts/
|   +-- build_rag.py       # Offline knowledge base build script (Docker build phase)
+-- agent.py               # execute_agent() entry point, TOOL_REGISTRY, FUNCTION_DECLARATIONS
+-- api_server.py          # FastAPI application factory, CORS, static file serving
+-- config.py              # pydantic-settings Settings class
+-- graph.py               # Complete LangGraph StateGraph definition
+-- langchain_integration.py  # LangChain adapter: prompt templates, retriever adapter
+-- llm.py                 # call_llm_structured() for structured JSON outputs
+-- memory.py              # ConversationMemory, AgentSession, create_session()
+-- planning.py            # is_simple_query(), create_research_plan(), ResearchPlan
+-- quality.py             # extract_claims, assess_claims, validate_citations, grounding gate
+-- reflection.py          # evaluate_evidence(), ReflectionResult
+-- research.py            # EvidenceItem, parse_local_evidence(), parse_web_evidence()
+-- state.py               # AgentState, TraceEvent, format_trace()
+-- tools.py               # search_web(), search_local_knowledge(), calculate_product()
+-- Dockerfile             # Build with RAG pre-initialisation; starts Uvicorn
+-- docker-compose.yml     # Local container orchestration
+-- requirements.txt       # Python dependencies
+-- .env.example           # Environment variable template
```

---

## Local Installation

**Requirements**: Python 3.12, pip

```bash
# 1. Create and activate a virtual environment
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate

# 2. Install CPU-only PyTorch (required before other dependencies)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. Install remaining dependencies
pip install -r requirements.txt

# 4. Copy and fill in environment variables
cp .env.example .env
# Edit .env with your API keys
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | If using Gemini | — | Google AI Studio API key |
| `OPENROUTER_API_KEY` | If using OpenRouter | — | OpenRouter API key |
| `GROQ_API_KEY` | If using Groq | — | Groq API key |
| `LLM_PROVIDER` | No | `gemini` | Active provider: `gemini`, `openrouter`, or `groq` |
| `LLM_MODEL` | No | provider default | Override model string (e.g., `openai/gpt-oss-20b:free`) |
| `VECTOR_DB` | No | `chroma` | Vector backend: `chroma` or `pinecone` |
| `PINECONE_API_KEY` | If using Pinecone | — | Pinecone API key |
| `PINECONE_INDEX_NAME` | If using Pinecone | — | Pinecone index name |

> **Never commit `.env` to version control.** It is listed in `.gitignore`.

---

## Running Locally

```bash
# Start the server
uvicorn api_server:app --host 127.0.0.1 --port 8000 --reload

# Windows via venv
.\venv\Scripts\uvicorn.exe api_server:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in your browser.
FastAPI interactive docs: `http://127.0.0.1:8000/docs`

---

## Docker

```bash
# Build (also pre-initialises the RAG knowledge base)
docker build -t ai-research-agent .

# Run
docker run -p 8000:8000 \
  -e GEMINI_API_KEY=your_key \
  -e LLM_PROVIDER=gemini \
  ai-research-agent

# Via docker-compose (reads from .env)
docker-compose up --build
```

---

## Deployment (Render)

The application is deployed to Render as a Docker-based web service. Render reads the `Dockerfile` and injects environment variables configured in the Render dashboard. See [docs/deployment.md](docs/deployment.md) for details.

---

## API Overview

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/ready` | Readiness check — validates API keys |
| `GET` | `/config` | Public configuration (no secrets) |
| `POST` | `/chat` | Submit a research query |
| `DELETE` | `/sessions/{id}` | Delete an in-memory session |

Full documentation: [docs/api.md](docs/api.md)

---

## Testing

```bash
python -m unittest discover
```

125 offline unit tests — no live API calls, no network, no database required.

| Test file | Coverage area |
|---|---|
| `test_agent_providers.py` | Provider abstraction |
| `test_api_integration.py` | FastAPI routes |
| `test_config.py` | Settings |
| `test_graph.py` | LangGraph nodes and routing |
| `test_langchain.py` | LangChain adapter |
| `test_llm_fallback.py` | LLM provider fallback |
| `test_planning.py` | Query classification |
| `test_quality.py` | Claim extraction, grounding, chunking |
| `test_research.py` | Evidence parsing |
| `test_retrieval_fallback.py` | Vector store fallback |
| `test_retrieval_fusion.py` | RRF fusion |
| `test_structured_llm.py` | Structured LLM output |
| `test_validate_citations.py` | Citation validation |

---

## Example Research Queries

Research queries (full pipeline):
```
What are the latest major developments in AI agents in 2026?
Compare the EV battery targets of Toyota and Samsung SDI.
What is the current state of open-source LLMs?
```

Simple queries (fast path, no research):
```
What is the capital of France?
Who invented the telephone?
```

---

## Known Limitations

- **Token-overlap grounding**: Claim assessment uses keyword overlap, not semantic similarity. Claims sharing keywords with semantically different evidence may pass incorrectly.
- **Web source quality**: DuckDuckGo results are accepted without domain authority filtering. SEO blogs and primary sources are treated equally.
- **In-memory sessions**: Sessions are lost on server restart. The frontend recovers automatically via a 404-retry mechanism.
- **Synchronous execution**: All operations are blocking. Long research queries hold the handler until complete.
- **Free-tier rate limits**: Extended tool loops may hit provider rate limits (429) on free tiers.
- **No semantic deduplication**: Repeated similar evidence items from multiple search iterations are not deduplicated.

---

## Future Improvements

- Semantic (embedding-based) claim-source alignment to replace token overlap
- Domain authority scoring in web retrieval
- Asynchronous request handling
- Persistent session storage (Redis)
- Streaming responses to the frontend
- Automated grounding metrics and evaluation harness

---

## Project Status

**V1.0 — Feature Complete and Frozen**

| Commit | Description |
|---|---|
| `557ab50` | refine: reduce unnecessary research hedging |
| `912444e` | perf: harden grounding chunk limits |
| `01305b3` | fix: recover stale chat sessions |
| `dd63b9e` | perf: optimize grounding claim assessment |
| `8fe53a1` | fix: enforce grounded research citations |
| `1f8d273` | feat: containerize deployment with build-time RAG |

Production deployment live on Render. All 125 tests pass.

---

## Author

**Chandra Shekar**
[GitHub: chandra8688](https://github.com/chandra8688)
