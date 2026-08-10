# Setup Guide

This guide explains how to configure and run the AI Research Agent locally.

## 1. Clone Repository

Download the repository to your local machine:
```bash
git clone <repository_url>
cd AI_Research_Agent
```

## 2. Create Virtual Environment

Ensure you are using Python 3.10 or greater. Create a dedicated virtual environment:
```bash
python -m venv .venv
```

## 3. Activate Environment

Activate the environment before proceeding:
- **Windows:**
  ```powershell
  .\.venv\Scripts\activate
  ```
- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```

## 4. Install Requirements

Install all necessary dependencies. (This step installs FastAPI, LangGraph, LangChain Core, Chroma, Pinecone, and SentenceTransformers).
```bash
pip install -r requirements.txt
```

## 5. Configure Environment Variables

Create a `.env` file in the root directory. Configure your providers according to your requirements. 

**DO NOT COMMIT YOUR `.env` FILE TO SOURCE CONTROL.**

```env
# Primary LLM (Required)
GEMINI_API_KEY=<your-key>

# Fallback LLM (Optional: Enables failover if Gemini is rate limited)
OPENROUTER_API_KEY=<your-key>

# Fallback Vector DB (Optional: Enables cloud RAG backend)
PINECONE_API_KEY=<your-key>
```

## 6. Initialize Local Knowledge Base

By default, the RAG system reads from local directories to build its index.
1. Place any internal `.txt` files into the `knowledge/` directory (or directory specified by your configuration).
2. The agent's Chroma database will automatically index these files upon the first query using the local `all-MiniLM-L6-v2` embedding model.

## 7. Run Offline Tests

Validate the architecture via the offline test suite before running the API:
```bash
python -m unittest discover -p "test_*.py"
```
*Note: All tests execute entirely offline via mocks and do not consume your API quotas.*

## 8. Start FastAPI

Launch the backend REST API:
```bash
python api_server.py
```
Or via uvicorn directly:
```bash
uvicorn api_server:app --host 127.0.0.1 --port 8000
```

## 9. Open Frontend

The FastAPI application natively serves the static web UI. Navigate to:
[http://127.0.0.1:8000/](http://127.0.0.1:8000/)

You can now chat directly with the AI Research Agent.
