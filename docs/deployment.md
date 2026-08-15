# Deployment

This document covers Docker containerisation and Render deployment for AI Research Agent V1.0.

---

## Docker

### Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# System dependencies for native extensions (chromadb, sentence-transformers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch (must be installed before requirements.txt)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-build the RAG knowledge base at image build time
ENV HF_HOME=/app/.hf_cache
RUN python scripts/build_rag.py

EXPOSE 8000

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build-time RAG Initialisation

The `RUN python scripts/build_rag.py` step runs during the Docker image build, not at container startup. This:

- Downloads the `all-MiniLM-L6-v2` embedding model from HuggingFace (~80 MB) and caches it at `/app/.hf_cache/`.
- Loads `.txt` documents from the `docs/` directory.
- Chunks, embeds, and stores them in ChromaDB at `.chroma_db/`.
- Bakes the populated `.chroma_db/` and `.hf_cache/` directories directly into the image.

**Result**: At container startup there is no embedding model download and no knowledge base initialisation delay. The vector store is immediately queryable.

> If the `docs/` directory contains no `.txt` files, `build_rag.py` completes silently with an empty knowledge base. The agent will fall back to web-only research.

### docker-compose.yml

```yaml
version: '3.8'

services:
  ai-agent:
    build: .
    container_name: ai-research-agent
    ports:
      - "8000:8000"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-}
      - PINECONE_API_KEY=${PINECONE_API_KEY:-}
      - LLM_PROVIDER=${LLM_PROVIDER:-gemini}
      - VECTOR_DB=${VECTOR_DB:-chroma}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
```

Environment variables are read from the host `.env` file.

### Build and run locally

```bash
# Build image (~5 minutes on first build due to PyTorch and model download)
docker build -t ai-research-agent .

# Run with environment variables
docker run -p 8000:8000 \
  -e GEMINI_API_KEY=your_key \
  -e LLM_PROVIDER=gemini \
  ai-research-agent

# Run via docker-compose (reads .env automatically)
docker-compose up --build

# Verify the container is healthy
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

### Image size

The image is large due to CPU PyTorch, sentence-transformers, and the pre-built ChromaDB knowledge base. Exact size depends on the host platform and cached layers. Expect a multi-hundred-megabyte image.

---

## Render Deployment

The application is deployed on [Render](https://render.com) as a **Docker-based web service**.

### Configuration

There is no `render.yaml` file in the repository. Deployment is configured through the Render web dashboard:

| Setting | Value |
|---|---|
| Environment | Docker |
| Dockerfile path | `./Dockerfile` |
| Port | 8000 |
| Health check path | `/health` |

### Environment Variables

Set the following in the Render dashboard (Environment → Environment Variables):

| Variable | Required |
|---|---|
| `GEMINI_API_KEY` | If using Gemini |
| `OPENROUTER_API_KEY` | If using OpenRouter |
| `GROQ_API_KEY` | If using Groq |
| `LLM_PROVIDER` | Yes (`gemini`, `openrouter`, or `groq`) |
| `VECTOR_DB` | No (defaults to `chroma`) |
| `PINECONE_API_KEY` | If using Pinecone |
| `PINECONE_INDEX_NAME` | If using Pinecone |

### Deployment process

1. Push commits to `origin/master`.
2. Render detects the push and rebuilds the Docker image.
3. The build runs `scripts/build_rag.py` during the image build step.
4. On successful build, Render deploys the new container and routes traffic.
5. The previous container continues serving traffic until the new one passes health checks.

### Production considerations

- **Session memory**: Sessions are stored in process memory. A Render deployment restart (e.g., due to a new deploy or Render''s instance recycling) clears all active sessions. The frontend handles this with a one-time 404 retry.
- **RAM**: The embedding model and ChromaDB collection are held in process memory. On hosting plans with limited RAM, complex multi-step research workloads may approach available memory limits. Monitor usage if running on a constrained plan.
- **Response time**: Research queries are synchronous and may take 30–90 seconds depending on the number of tool calls, reflection iterations, and grounding gate invocations. This is expected behaviour.
- **Rate limits**: On provider free tiers, extended research workflows may encounter HTTP 429 responses. The grounding gate uses tenacity retry logic (up to 5 attempts, exponential back-off 3–35s) to handle this.
- **No persistent storage**: ChromaDB data is baked into the Docker image. Adding new documents requires rebuilding the image.

---

## Application Entrypoint

The server is started by:
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

`api_server.py` loads environment variables via `python-dotenv` and registers the FastAPI router and static file handlers. The startup event logs the configured provider and vector DB.

There is no production WSGI/ASGI worker configuration (e.g., Gunicorn) in V1.0 — Uvicorn runs as a single-process server.
