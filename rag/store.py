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

def _get_backend(name: str) -> VectorStore:
    db_type = name.lower().strip()
    if db_type == "chroma":
        from config import settings
        from rag.chroma_store import ChromaStore
        return ChromaStore(persist_directory=settings.chroma_persist_directory, collection_name="rag_collection")
    elif db_type == "pinecone":
        from rag.pinecone_store import PineconeStore
        return PineconeStore()
    else:
        from rag.errors import FatalRetrievalError
        raise FatalRetrievalError(f"Unsupported VECTOR_DB: '{db_type}'. Must be 'chroma' or 'pinecone'.")


class FallbackVectorStore(VectorStore):
    def __init__(self):
        from config import settings
        self.primary_name = settings.vector_db.lower().strip()
        self.fallback_name = settings.vector_db_fallback.lower().strip()
        self.fallback_enabled = settings.vector_db_fallback_enabled
        self._primary = None
        self._fallback = None

    def _get_primary(self) -> VectorStore:
        if self._primary is None:
            self._primary = _get_backend(self.primary_name)
        return self._primary

    def _get_fallback(self) -> VectorStore:
        if self._fallback is None:
            self._fallback = _get_backend(self.fallback_name)
        return self._fallback

    def add_documents(self, chunks: list[Document], embeddings: list[np.ndarray]) -> None:
        self._get_primary().add_documents(chunks, embeddings)

    def count(self) -> int:
        return self._get_primary().count()

    def search(self, query_embedding: np.ndarray, k: int = 3) -> list[Document]:
        from rag.errors import RetryableRetrievalError, FatalRetrievalError
        
        try:
            primary = self._get_primary()
            results = primary.search(query_embedding, k)
            print(f"[VECTOR] Backend: {self.primary_name}")
            print(f"[VECTOR] Retrieved {len(results)} chunks")
            return results
        except RetryableRetrievalError as e:
            if not self.fallback_enabled:
                raise
                
            print(f"[VECTOR] Primary backend: {self.primary_name}")
            print(f"[VECTOR] Primary backend failed: {str(e)}")
            print(f"[VECTOR] Falling back to: {self.fallback_name}")
            
            try:
                fallback = self._get_fallback()
                results = fallback.search(query_embedding, k)
                print(f"[VECTOR] Retrieved {len(results)} chunks from fallback backend")
                return results
            except Exception as fallback_e:
                raise RuntimeError(f"Both backends failed.\nPrimary error: {str(e)}\nFallback error: {str(fallback_e)}")
        except FatalRetrievalError:
            raise


def get_vector_store() -> VectorStore:
    """
    Factory function to retrieve the configured VectorStore backend,
    now wrapped with fallback routing capabilities.
    """
    return FallbackVectorStore()
