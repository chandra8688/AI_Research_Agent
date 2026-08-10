import os
import chromadb
import numpy as np
from rag.loader import Document

class ChromaStore:
    def __init__(self, persist_directory: str = ".chroma_db", collection_name: str = "rag_collection"):
        """
        Initializes a persistent local ChromaDB client and gets/creates the collection.
        
        Args:
            persist_directory: Local folder to store the database files.
            collection_name: Name of the collection to use within ChromaDB.
        """
        # Ensure the directory exists (or ChromaDB will create it)
        self.persist_directory = persist_directory
        try:
            self.client = chromadb.PersistentClient(path=persist_directory)
            # get_or_create_collection prevents duplicate errors on reload
            self.collection = self.client.get_or_create_collection(name=collection_name)
        except Exception as e:
            from rag.errors import RetryableRetrievalError
            raise RetryableRetrievalError(f"ChromaDB initialization failed: {str(e)}")

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

    def search(self, query_embedding: np.ndarray, k: int = 3) -> list[Document]:
        """
        Performs a similarity search using the provided query embedding.
        
        Args:
            query_embedding: A numpy array representing the embedded query.
            k: The maximum number of results to return.
            
        Returns:
            A list of Document objects with distance scores included in their metadata.
        """
        if k <= 0:
            return []
            
        total_docs = self.count()
        if total_docs == 0:
            return []
            
        # Ensure k doesn't exceed the total documents in the collection
        n_results = min(k, total_docs)
        
        # ChromaDB query expects a list of lists for embeddings
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
        except ValueError as e:
            from rag.errors import FatalRetrievalError
            raise FatalRetrievalError(f"ChromaDB query value error: {str(e)}")
        except Exception as e:
            from rag.errors import RetryableRetrievalError
            raise RetryableRetrievalError(f"ChromaDB search failed: {str(e)}")
        
        # Results are returned as lists of lists (one list per query)
        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []
        
        returned_chunks = []
        for doc, meta, dist in zip(docs, metas, distances):
            # Create a new metadata dict to avoid mutating the stored one
            # and inject the distance metric
            new_meta = meta.copy() if meta else {}
            new_meta["distance"] = dist
            
            # Reconstruct the Document object
            returned_chunks.append(Document(
                content=doc,
                metadata=new_meta
            ))
            
        return returned_chunks
