FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# curl is used by the docker-compose healthcheck; build-essential for native wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch (~200MB instead of the ~2GB CUDA build), then the rest
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Build the RAG knowledge base (and download the embedding model) at build time
ENV HF_HOME=/app/.hf_cache
RUN python scripts/build_rag.py

# The embedding model is baked into the image: never call the Hugging Face Hub at runtime
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 8000

# Render (and most PaaS hosts) inject $PORT; default to 8000 for local runs
CMD ["sh", "-c", "uvicorn api_server:app --host 0.0.0.0 --port ${PORT:-8000}"]
