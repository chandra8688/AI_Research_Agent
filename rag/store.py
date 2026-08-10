import os
import chromadb
import numpy as np
from rag.loader import Document

class VectorStore:
    def __init__(self, persist_directory: str = ".chroma_db", collection_name: str = "rag_collection"):
        """
        Initializes a persistent local ChromaDB client and gets/creates the collection.
        
        Args:
            persist_directory: Local folder to store the database files.
            collection_name: Name of the collection to use within ChromaDB.
        """
        # Ensure the directory exists (or ChromaDB will create it)
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # get_or_create_collection prevents duplicate errors on reload
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, chunks: list[Document], embeddings: list[np.ndarray]) -> None:
        """
        Stores chunk texts, embeddings, and metadata into the Chroma collection.
        
        Args:
            chunks: List of Document objects (the text chunks).
            embeddings: List of numpy arrays (the vectors).
        """
        if not chunks or not embeddings:
            return
            
        if len(chunks) != len(embeddings):
            raise ValueError(f"Length mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")

        ids = []
        documents = []
        metadatas = []
        embeddings_list = []

        for chunk, emb in zip(chunks, embeddings):
            # Generate a unique ID for each chunk based on source and index
            source = chunk.metadata.get("source", "unknown")
            chunk_idx = chunk.metadata.get("chunk_index", 0)
            chunk_id = f"{source}_chunk_{chunk_idx}"
            
            ids.append(chunk_id)
            documents.append(chunk.content)
            metadatas.append(chunk.metadata)
            embeddings_list.append(emb.tolist()) # chromadb expects python lists, not numpy arrays

        # Upsert adds new items or updates existing items with the same ID
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings_list
        )

    def count(self) -> int:
        """Returns the total number of items in the collection."""
        return self.collection.count()

    def get_item(self, item_id: str) -> dict:
        """Retrieves a single item from the collection by ID."""
        return self.collection.get(ids=[item_id], include=["documents", "metadatas", "embeddings"])
