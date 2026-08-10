import os
from typing import Protocol
import numpy as np

from rag.loader import Document

class VectorStore(Protocol):
    def add_documents(self, chunks: list[Document], embeddings: list[np.ndarray]) -> None:
        ...

    def count(self) -> int:
        ...

    def search(self, query_embedding: np.ndarray, k: int = 3) -> list[Document]:
        ...

def get_vector_store() -> VectorStore:
    """
    Factory function to retrieve the configured VectorStore backend.
    Checks the VECTOR_DB environment variable (default: chroma).
    """
    from config import settings
    
    db_type = settings.vector_db.lower().strip()
    
    if db_type == "chroma":
        from rag.chroma_store import ChromaStore
        return ChromaStore(persist_directory=settings.chroma_persist_directory, collection_name="rag_collection")
    elif db_type == "pinecone":
        from rag.pinecone_store import PineconeStore
        return PineconeStore()
    else:
        raise ValueError(f"Unsupported VECTOR_DB: '{db_type}'. Must be 'chroma' or 'pinecone'.")
