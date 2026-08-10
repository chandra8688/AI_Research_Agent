import os
import numpy as np
from pinecone import Pinecone
from rag.loader import Document

class PineconeStore:
    def __init__(self):
        """
        Initializes the Pinecone client and connects to the specified index.
        Requires PINECONE_API_KEY and PINECONE_INDEX_NAME environment variables.
        """
        from config import settings
        
        api_key = settings.pinecone_api_key
        index_name = settings.pinecone_index_name
        
        if not api_key:
            raise ValueError("pinecone_api_key configuration is missing or empty.")
        if not index_name:
            raise ValueError("pinecone_index_name configuration is missing or empty.")
            
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)

    def add_documents(self, chunks: list[Document], embeddings: list[np.ndarray]) -> None:
        """
        Stores chunk texts, embeddings, and metadata into the Pinecone index.
        """
        if not chunks or not embeddings:
            return
            
        if len(chunks) != len(embeddings):
            raise ValueError(f"Length mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")

        vectors = []
        for chunk, emb in zip(chunks, embeddings):
            source = chunk.metadata.get("source", "unknown")
            chunk_idx = chunk.metadata.get("chunk_index", 0)
            chunk_id = f"{source}_chunk_{chunk_idx}"
            
            # Pinecone metadata must be a flat dictionary of strings/numbers/booleans/lists of strings
            meta = chunk.metadata.copy()
            # Ensure text content is stored in metadata so we can retrieve it
            meta["content"] = chunk.content
            
            vectors.append({
                "id": chunk_id,
                "values": emb.tolist(),
                "metadata": meta
            })

        # Pinecone upserts in batches. We can upsert them all if the batch is reasonable.
        # For huge lists, we would batch them, but here we expect modest numbers or handle it simply.
        self.index.upsert(vectors=vectors)

    def count(self) -> int:
        """Returns the total number of items in the collection."""
        stats = self.index.describe_index_stats()
        return stats.total_vector_count or 0

    def search(self, query_embedding: np.ndarray, k: int = 3) -> list[Document]:
        """
        Performs a similarity search using the provided query embedding.
        """
        if k <= 0:
            return []
            
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=k,
            include_metadata=True
        )
        
        returned_chunks = []
        for match in results.get("matches", []):
            meta = match.get("metadata", {})
            score = match.get("score", 0.0)
            
            # Extract content from metadata and create a fresh metadata dict for the Document
            content = meta.pop("content", "")
            
            new_meta = meta.copy()
            new_meta["distance"] = score # We use 'score' in Pinecone, acting as distance/similarity
            
            returned_chunks.append(Document(
                content=content,
                metadata=new_meta
            ))
            
        return returned_chunks
