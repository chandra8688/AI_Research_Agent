from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "AI Research Agent"
    environment: str = "development"
    
    # LLM Provider Configuration
    llm_provider: str = "gemini"
    llm_primary_provider: str | None = None
    llm_fallback_provider: str = "openrouter"
    llm_fallback_enabled: bool = False
    
    gemini_api_key: str | None = None
    openrouter_api_key: str | None = None
    
    # Vector DB Configuration
    vector_db: str = "chroma"
    chroma_persist_directory: str = ".chroma_db"
    pinecone_api_key: str | None = None
    pinecone_index_name: str | None = None
    
    # Agent/API Constraints
    max_message_length: int = 10000
    max_agent_iterations: int = 5
    max_reflection_attempts: int = 2
    
    # CORS
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173"
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
