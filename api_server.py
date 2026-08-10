from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from api.routes import router
from rag.pipeline import initialize_knowledge_base

# Load env variables for Gemini/Pinecone before running the server
load_dotenv()

app = FastAPI(
    title="AI Research Agent API",
    description="HTTP API layer for the AI Research Agent",
    version="1.0.0"
)

# CORS configuration for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.on_event("startup")
def startup_event():
    # Initialize the local RAG knowledge base safely
    try:
        initialize_knowledge_base()
    except Exception as e:
        print(f"Warning: Failed to initialize local knowledge base during startup: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=True)
