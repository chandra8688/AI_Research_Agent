from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from api.routes import router
from rag.pipeline import initialize_knowledge_base

from config import settings
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s:\t%(name)s - %(message)s")
logger = logging.getLogger(__name__)

# Load env variables for Gemini/Pinecone before running the server
load_dotenv()

app = FastAPI(
    title="AI Research Agent API",
    description="HTTP API layer for the AI Research Agent",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.on_event("startup")
def startup_event():
    # Initialize the local RAG knowledge base safely
    logger.info(f"Starting {settings.app_name} in {settings.environment} mode.")
    logger.info(f"LLM Provider: {settings.llm_provider} | Vector DB: {settings.vector_db}")
    try:
        initialize_knowledge_base()
    except Exception as e:
        logger.warning(f"Failed to initialize local knowledge base during startup: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=True)
