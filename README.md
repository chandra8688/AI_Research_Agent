# AI Research Agent

The AI Research Agent is an advanced, autonomous research system designed to retrieve, synthesize, and validate information across multiple sources. Built on LangGraph orchestration, it conducts multi-step deep research and uses provider abstraction to maintain uptime.

## What It Does
The system takes a user query, plans a research approach, and uses autonomous agents to search the web and local documents. It compiles evidence, critically reflects on whether it has sufficient information to answer the question, and enforces strict citation grounding. It also features a "fast path" to quickly answer simple factual queries without unnecessary deep research.

## Core Capabilities
- **LangGraph orchestration**: Deterministic cyclic state management for agentic workflows.
- **Web research/search**: Live web search integration.
- **RAG**: Retrieval-Augmented Generation across multiple sources.
- **Chroma vector database**: Local embeddings storage.
- **Structured LLM outputs**: Enforces specific JSON formats for agent decision making.
- **Reflection/evidence evaluation**: Critically assesses the retrieved evidence against the user query.
- **Forced synthesis after reflection limit**: Ensures the agent provides a final answer when max reflection attempts are reached.
- **Provider abstraction**: Seamless fallback between configured LLM providers.
- **Simple-query fast path**: Bypasses deep research for trivial factual questions.
- **FastAPI backend**: Fast, asynchronous web framework exposing REST endpoints.
- **Vanilla JS frontend**: Simple, reliable web user interface.

## Architecture Overview
The high-level request flow operates as follows:

```text
User
   ↓
FastAPI
   ↓
Query planning/classification
   ↓
LangGraph
   ↓
Agent/tool loop
   ↓
Web search / local knowledge
   ↓
Reflection
   ↓
Force synthesis when reflection limit is reached
   ↓
Quality check
   ↓
Final answer
   ↓
Frontend
```

## LLM Provider Configuration
The application uses environment variables for provider configuration.

- `LLM_PROVIDER`: Set the active primary provider (e.g., `openrouter` or `gemini`).
- `LLM_MODEL`: Set the specific model string (e.g., `nvidia/nemotron-3-super-120b-a12b:free`).
- `OPENROUTER_API_KEY`: Your API key for OpenRouter.
- `GEMINI_API_KEY`: Your API key for Google Gemini.

> [!IMPORTANT]
> The `.env` file is strictly local and must never be committed to source control. Use `.env.example` as a template for required keys.

## Installation
The current supported deployment is native Windows Python execution (Docker is currently deferred).

1. Create a virtual environment:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install CPU-only PyTorch (if required by your hardware/setup):
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

3. Install dependencies:
```powershell
pip install -r requirements.txt
```

## Running the Application
Start the FastAPI server:
```powershell
.\venv\Scripts\uvicorn.exe api_server:app --host 127.0.0.1 --port 8000
```
Access the web interface by navigating to `http://127.0.0.1:8000` in your browser.

## API Endpoints
- `GET /health`: Returns basic health status.
- `GET /ready`: Returns readiness status and provider availability.
- `GET /config`: Returns public application configuration (no secrets).
- `POST /chat`: Main chat endpoint for research queries (expects JSON payload with `message`).

## Testing
Run the complete offline test suite:
```powershell
.\venv\Scripts\python.exe -m unittest discover
```
Currently validated offline test count: **109/109 tests passing**.

## Error Handling
The application gracefully maps downstream provider exceptions to the client:
- **429 (Too Many Requests)**: Handled when the AI provider is temporarily rate-limited.
- **502 (Bad Gateway)**: Handled when the AI provider returns an upstream error.
- **503 (Service Unavailable)**: Handled when the research service provider is unreachable.
- **Network failure**: Handled gracefully by the frontend if the backend disconnects.

## Known Limitations
- **OpenRouter free-model rate limits**: The free tier imposes strict limits that can interrupt multi-step workflows.
- **Deep research time**: Complex multi-source deep research can take several minutes to complete.
- **Incomplete evaluation**: A planned six-query research-quality evaluation was INCOMPLETE because free-model daily limits blocked tests 2–6.
- **Live validation**: We successfully ran at least one successful end-to-end research workflow using OpenRouter models.
- **Deferred deployment**: Docker containerization is currently deferred.
- **Source quality constraints**: The final output quality heavily depends on the search providers and the specific web results available at execution time.
- **Simple-query limitations**: The simple-query fast path intentionally avoids unnecessary research, meaning it will not provide citations for factual trivia.

## Validation Status

**VERIFIED:**
- 109 offline tests passing
- FastAPI health/ready/config endpoints
- OpenRouter provider integration
- OpenRouter tool calling
- structured outputs
- reflection
- reflection attempt limit
- force synthesis
- frontend error handling
- at least one successful end-to-end research workflow

**NOT FULLY VERIFIED:**
- complete six-query research evaluation
- production-scale reliability
- Docker deployment

## Project Structure
```text
project/
├── api/             # FastAPI routers and HTTP models
├── docs/            # Architecture, setup, and evaluation documentation
├── frontend/        # Vanilla HTML/CSS/JS user interface
├── providers/       # LLM provider abstraction layer (Gemini, OpenRouter)
├── rag/             # Retrieval-augmented generation and vector tools
├── graph.py         # LangGraph orchestration state machine
├── planning.py      # Query classification and research planning
├── quality.py       # Evidence grounding and citation verification
├── reflection.py    # Autonomous evidence evaluation
└── state.py         # Agent execution state definitions
```

## Security
- The `.env` file is explicitly ignored in `.gitignore`.
- API keys are never committed to the repository.
- The `/config` API endpoint does not expose sensitive credentials.
- Upstream provider errors are intercepted and sanitized before reaching the frontend to prevent leaking internal stack traces or API keys.
