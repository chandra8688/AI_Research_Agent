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


class FusionVectorStore(VectorStore):
    def __init__(self):
        from config import settings
        self.chroma_name = "chroma"
        self.pinecone_name = "pinecone"
        self.top_k = settings.retrieval_fusion_top_k
        self.final_k = settings.retrieval_final_k
        self._chroma = None
        self._pinecone = None

    def _get_chroma(self) -> VectorStore:
        if self._chroma is None:
            self._chroma = _get_backend(self.chroma_name)
        return self._chroma

    def _get_pinecone(self) -> VectorStore:
        if self._pinecone is None:
            self._pinecone = _get_backend(self.pinecone_name)
        return self._pinecone

    def add_documents(self, chunks: list[Document], embeddings: list[np.ndarray]) -> None:
        # Fusion mode normally only reads, but we can write to both or just chroma.
        # Let's write to Chroma by default to preserve local consistency.
        self._get_chroma().add_documents(chunks, embeddings)

    def count(self) -> int:
        return self._get_chroma().count()

    def search(self, query_embedding: np.ndarray, k: int = 3) -> list[Document]:
        from rag.errors import RetryableRetrievalError, FatalRetrievalError
        from rag.reranker import fuse_results
        
        chroma_results = []
        pinecone_results = []
        
        print(f"[RETRIEVAL] Mode: fusion")
        
        # 1. Fetch from Chroma
        try:
            chroma_store = self._get_chroma()
            # Fetch fusion top_k initially
            chroma_results = chroma_store.search(query_embedding, self.top_k)
            print(f"[RETRIEVAL] Chroma results: {len(chroma_results)}")
        except Exception as e:
            print(f"[RETRIEVAL] Chroma fusion error: {e}")
            
        # 2. Fetch from Pinecone
        try:
            pinecone_store = self._get_pinecone()
            pinecone_results = pinecone_store.search(query_embedding, self.top_k)
            print(f"[RETRIEVAL] Pinecone results: {len(pinecone_results)}")
        except Exception as e:
            print(f"[RETRIEVAL] Pinecone fusion error: {e}")
            
        if not chroma_results and not pinecone_results:
            # Check if both failed or simply returned nothing
            return []
            
        # 3. Fuse
        # Use requested 'k' from caller if it's less than final_k setting, or just use the config setting
        # Actually caller k is used as final_k
        k = min(k, self.final_k)
        
        fused = fuse_results(chroma_results, pinecone_results, k)
        print(f"[RETRIEVAL] Unique results: {len(set((d.metadata.get('source'), d.metadata.get('chunk_index')) for d in fused))}")
        print(f"[RETRIEVAL] Final results: {len(fused)}")
        
        return fused


def get_vector_store() -> VectorStore:
    """
    Factory function to retrieve the configured VectorStore backend,
    now wrapped with fallback or fusion capabilities.
    """
    from config import settings
    if getattr(settings, 'retrieval_fusion_enabled', False):
        return FusionVectorStore()
    return FallbackVectorStore()
