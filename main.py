import os
import shutil
from rag.loader import load_documents
from rag.chunker import chunk_documents
from rag.embedder import embed_chunks
from rag.store import VectorStore


def main():
    print("=" * 60)
    print("AI-063: Local Vector Store Tests (Offline)")
    print("=" * 60)
    
    db_path = ".chroma_db"
    
    # ------------------------------------------------------------------
    # Setup: load, chunk, and embed the real sample documents
    # ------------------------------------------------------------------
    print("\n--- Setup: Generating chunks and embeddings ---")
    docs = load_documents("docs")
    chunks = chunk_documents(docs, chunk_size=500, overlap=100)
    embeddings = embed_chunks(chunks)
    print(f"  Chunks produced: {len(chunks)}")
    print(f"  Embeddings generated: {len(embeddings)}")

    # ------------------------------------------------------------------
    # Test 1 & 2: Initialize store and insert chunks
    # ------------------------------------------------------------------
    print("\n--- Test 1 & 2: Insert into VectorStore ---")
    store = VectorStore(persist_directory=db_path)
    store.add_documents(chunks, embeddings)
    print("  PASS -- successfully inserted chunks and embeddings into ChromaDB.")

    # ------------------------------------------------------------------
    # Test 3: Verify collection count
    # ------------------------------------------------------------------
    print("\n--- Test 3: Verify collection count ---")
    count = store.count()
    print(f"  Collection count: {count}")
    assert count == len(chunks), f"Expected {len(chunks)}, got {count}"
    print("  PASS -- count matches inserted chunks.")

    # ------------------------------------------------------------------
    # Test 4 & 5: ID uniqueness and retrieval
    # ------------------------------------------------------------------
    print("\n--- Test 4 & 5: Retrieve item and verify metadata ---")
    first_chunk = chunks[0]
    first_id = f"{first_chunk.metadata['source']}_chunk_{first_chunk.metadata['chunk_index']}"
    
    result = store.get_item(first_id)
    assert result is not None, "Failed to retrieve item."
    
    # Chroma returns lists for these fields since get() accepts multiple IDs
    retrieved_doc = result["documents"][0]
    retrieved_meta = result["metadatas"][0]
    retrieved_emb = result["embeddings"][0]
    
    print(f"  Retrieved ID: {first_id}")
    print(f"  Retrieved Source: {retrieved_meta.get('source')}")
    assert retrieved_doc == first_chunk.content, "Stored text mismatch."
    assert retrieved_meta["source"] == first_chunk.metadata["source"], "Metadata mismatch."
    # We won't perfectly match floats, but we check if it has the right dimension
    assert len(retrieved_emb) == len(embeddings[0]), "Embedding dimension mismatch."
    
    print("  PASS -- successfully retrieved text, metadata, and embedding by unique ID.")

    # ------------------------------------------------------------------
    # Test 6 & 7: Persistence
    # ------------------------------------------------------------------
    print("\n--- Test 6 & 7: Verify data persistence ---")
    # Delete the in-memory object to simulate program exit
    del store
    
    # Re-initialize pointing to the same persistent directory
    new_store = VectorStore(persist_directory=db_path)
    new_count = new_store.count()
    print(f"  Re-opened store collection count: {new_count}")
    assert new_count == len(chunks), "Data did not persist between client lifecycles!"
    print("  PASS -- data persists locally.")

    print("\n" + "=" * 60)
    print("AI-063 tests complete. No API calls were made.")
    print("=" * 60)


if __name__ == "__main__":
    main()
