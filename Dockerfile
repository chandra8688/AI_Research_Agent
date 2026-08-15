FROM python:3.12-slim

WORKDIR /app

# Install system dependencies if required by native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Prepare the RAG knowledge base during the build phase
ENV HF_HOME=/app/.hf_cache
RUN python scripts/build_rag.py

# Expose FastAPI port
EXPOSE 8000

# Run in production mode (no auto-reload)
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
