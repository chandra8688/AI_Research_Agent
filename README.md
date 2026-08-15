# AI Research Agent V1.0

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_Workflow-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![Render](https://img.shields.io/badge/Render-Deployed-000000?logo=render&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-125_Passing-success)

> **A production-oriented autonomous research agent that combines web research, local knowledge retrieval, multi-step planning, reflection, and rigorous citation validation.**

## Why this project?
Modern LLMs hallucinate. They confidently state things that are factually wrong, and rarely distinguish between what they know and what they are inventing. This project explores a pragmatic mitigation: **force the model to produce citations from explicitly retrieved sources, and automatically verify those citations before the answer is returned.**

This implementation deliberately avoids black-box RAG frameworks in favour of a fully traceable, testable pipeline where every grounding decision is observable.

---

## What happens when a user asks a question?

The agent executes a dynamic, multi-step pipeline. Simple questions take a fast path, while complex research triggers the full autonomous loop:

```text
User Query
   ↓
Research Planning
   ↓
Tool Selection (LLM chooses actions)
   ↓
Web / Local Retrieval (DuckDuckGo / ChromaDB)
   ↓
Evidence Collection
   ↓
Reflection (Is this enough to answer?)
   ↓
Evidence Synthesis
   ↓
Claim Assessment (Fact-checking vs Evidence)
   ↓
Grounding Check (Rewriting unsupported claims)
   ↓
Citation Validation
   ↓
Final Answer
```

---

## Technical Highlights

This project demonstrates several advanced AI engineering patterns:
- **[LangGraph-based Orchestration](docs/architecture.md)**: A stateful `StateGraph` for the autonomous research loop, replacing brittle prompt-chaining.
- **[Multi-Step Research Loop](docs/architecture.md)**: The agent can iteratively search, collect evidence, and reflect multiple times before synthesising.
- **[Dual-Channel Retrieval](docs/rag-pipeline.md)**: Fuses live web search (`ddgs`) with a local RAG vector store.
- **[Reflection & Sufficiency](docs/architecture.md)**: An LLM-as-a-judge node evaluates if the gathered evidence actually answers the prompt.
- **[Grounding & Citation Validation](docs/grounding-and-citations.md)**: Extracts factual claims, assesses them against retrieved evidence via token-overlap, and strictly validates all citation tags (e.g., `[WEB: ...]`).
- **[Provider Abstraction](docs/architecture.md)**: Seamlessly switch between Gemini, OpenRouter, and Groq via environment variables.
- **[Session Memory](docs/api.md)**: UUID-keyed conversation tracking.
- **[Containerised Deployment](docs/deployment.md)**: Dockerized setup with build-time RAG ingestion, deployed on Render.
- **[Comprehensive Testing](#testing)**: 125 offline unit tests covering routing, fallback mechanisms, and quality checks.

---

## Demo

*Add screenshots/demo GIF here when available.*

---

## Documentation Hub

Detailed documentation is available in the `docs/` directory:

- 🏗️ **[Architecture](docs/architecture.md)** — System diagram, LangGraph nodes, and core flow.
- 🔍 **[RAG Pipeline](docs/rag-pipeline.md)** — Ingestion, chunking strategy, and multi-backend vector retrieval.
- 🛡️ **[Grounding & Citations](docs/grounding-and-citations.md)** — Claim extraction, conflict detection, and the grounding gate.
- 🔌 **[API Reference](docs/api.md)** — FastAPI endpoints, request/response schemas, and trace payloads.
- 🚀 **[Deployment](docs/deployment.md)** — Docker setup, build-time ingestion, and Render configuration.
- 🔧 **[Troubleshooting](docs/troubleshooting.md)** — Post-mortems for real issues fixed during V1.0 development.

---

## Tech Stack

**AI / Agent**
- **LangGraph**: Stateful agent orchestration
- **LangChain Core**: Prompt templating

**RAG / Retrieval**
- **ChromaDB**: Primary local vector store
- **Pinecone**: Optional cloud fallback vector store
- **Sentence-Transformers**: `all-MiniLM-L6-v2` local CPU embeddings
- **DuckDuckGo Search**: Live web retrieval (`ddgs`)

**Backend**
- **Python 3.12**
- **FastAPI + Uvicorn**: REST API and static file serving
- **Pydantic**: Configuration and schema validation

**Frontend**
- **HTML/CSS/JS**: Vanilla, zero-build-step frontend

**Deployment**
- **Docker**: Containerization with baked-in knowledge base
- **Render**: Web service hosting

**Testing**
- **Unittest / HTTPX**: 125 offline tests

---

## Honest Limitations

This is a V1.0 release. It is not an AGI, nor is it 100% hallucination-free. The system has known constraints:
- **Token-overlap grounding**: Claim assessment relies on keyword overlap. Claims sharing keywords with semantically different evidence may pass incorrectly (semantic verification is planned).
- **Web source quality**: DuckDuckGo results are accepted without domain authority filtering.
- **In-memory sessions**: Sessions are lost on server restart (the frontend handles recovery gracefully).
- **Synchronous execution**: Long research queries hold the HTTP handler until complete.
- **Provider rate limits**: Extended tool loops may hit 429 limits on free-tier LLM providers.

---

## Installation & Running Locally

**Requirements**: Python 3.12, pip

```bash
# 1. Virtual environment setup
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
# source venv/bin/activate   # macOS / Linux

# 2. Install CPU-only PyTorch (required before other dependencies)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. Install remaining dependencies
pip install -r requirements.txt

# 4. Environment variables
cp .env.example .env
# Edit .env with your API keys (e.g., GEMINI_API_KEY)

# 5. Run the server
uvicorn api_server:app --host 127.0.0.1 --port 8000 --reload
```
Access the application at `http://127.0.0.1:8000`.

---

## Docker

The Docker build automatically ingests the local knowledge base (`docs/*.txt`).

```bash
docker build -t ai-research-agent .

docker run -p 8000:8000 \
  -e GEMINI_API_KEY=your_key \
  -e LLM_PROVIDER=gemini \
  ai-research-agent
```

---

## Testing

The project is heavily tested to ensure pipeline reliability without requiring network access.

```bash
python -m unittest discover
```
*(Expect 125 tests to pass in ~2-4 seconds)*

---

## Project Status
**V1.0 — Feature Complete and Frozen**

Production deployment is live on Render. The core architecture is stable and thoroughly tested.

## Author
**Chandra Shekar**<br>
[GitHub: chandra8688](https://github.com/chandra8688)
