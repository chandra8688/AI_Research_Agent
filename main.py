import os
import shutil
import numpy as np
from rag.loader import load_documents
from rag.chunker import chunk_documents
from rag.embedder import embed_chunks
from rag.store import VectorStore


def main():
    print("=" * 60)
    print("AI-064: Vector Similarity Retrieval Tests (Offline)")
    print("=" * 60)
    
    db_path = ".chroma_db"
    
    # Clean up previous tests to ensure a fresh test environment
    if os.path.exists(db_path):
        try:
            shutil.rmtree(db_path)
        except Exception:
            pass # might fail on windows if locked, but chroma will handle it
    
    # ------------------------------------------------------------------
    # Setup: load, chunk, embed, and store the real sample documents
    # ------------------------------------------------------------------
    print("\n--- Setup: Building Pipeline ---")
    docs = load_documents("docs")
    chunks = chunk_documents(docs, chunk_size=500, overlap=100)
    embeddings = embed_chunks(chunks)
    
    store = VectorStore(persist_directory=db_path)
    store.add_documents(chunks, embeddings)
    total_stored = store.count()
    print(f"  Chunks stored: {total_stored}")

    # ------------------------------------------------------------------
    # Test 1 & 2: Query and verify top result
    # ------------------------------------------------------------------
    print("\n--- Test 1 & 2: Query for RAG ---")
    query_text = "What is RAG?"
    print(f"  Query: '{query_text}'")
    
    # Note: we use our existing embedder to get the query vector
    # We create a dummy Document object just to use our chunk embedder pipeline,
    # or we can pass the raw string if the embedder was modified, but our embedder 
    # takes a list of Document objects. Wait, looking at embed_chunks, it iterates 
    # over chunks.content. Let's make a dummy chunk.
    from rag.loader import Document
    query_chunk = Document(content=query_text, metadata={"source": "query"})
    query_embedding = embed_chunks([query_chunk])[0]
    
    results_k3 = store.search(query_embedding, k=3)
    
    print(f"  Returned {len(results_k3)} results.")
    assert len(results_k3) == min(3, total_stored), "Result count mismatch for k=3"
    
    top_result = results_k3[0]
    print(f"  Top result source: {top_result.metadata['source']}")
    print(f"  Top result distance: {top_result.metadata['distance']:.4f}")
    assert "RAG" in top_result.content or "rag" in top_result.metadata["source"].lower(), "Top result doesn't seem relevant to RAG."
    assert "distance" in top_result.metadata, "Distance score missing from metadata."
    assert "source" in top_result.metadata, "Source metadata missing."
    assert "chunk_index" in top_result.metadata, "Chunk index metadata missing."
    print("  PASS -- Top-k results are relevant and contain expected metadata.")

    # ------------------------------------------------------------------
    # Test 3 & 4: Test k=1 and result counts
    # ------------------------------------------------------------------
    print("\n--- Test 3 & 4: Test k=1 and boundaries ---")
    results_k1 = store.search(query_embedding, k=1)
    print(f"  Returned {len(results_k1)} results for k=1.")
    assert len(results_k1) == 1, "Expected exactly 1 result for k=1."
    
    results_k100 = store.search(query_embedding, k=100)
    print(f"  Returned {len(results_k100)} results for k=100 (total docs={total_stored}).")
    assert len(results_k100) == total_stored, "Result count should not exceed total stored chunks."
    print("  PASS -- k boundaries respected.")

    # ------------------------------------------------------------------
    # Test 5: Empty query handling
    # ------------------------------------------------------------------
    print("\n--- Test 5: Empty query ---")
    empty_chunk = Document(content="", metadata={})
    empty_embedding = embed_chunks([empty_chunk])
    
    # Our embedder returns an empty list for empty content
    if not empty_embedding:
        print("  Query embedding was empty (as expected for empty text).")
        # In a real app we'd skip search, but let's test if we pass a zero vector
        dummy_zero = np.zeros(384, dtype=np.float32)
        empty_results = store.search(dummy_zero, k=3)
        print(f"  Zero-vector search returned {len(empty_results)} results.")
        assert len(empty_results) > 0, "Zero-vector should still return something due to ANN."
    else:
        empty_results = store.search(empty_embedding[0], k=3)
        print(f"  Empty text search returned {len(empty_results)} results.")
    
    print("  PASS -- Empty/zero query handled gracefully.")


    print("\n" + "=" * 60)
    print("AI-064 tests complete. No API calls were made.")
    print("=" * 60)


if __name__ == "__main__":
    main()
