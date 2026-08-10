import numpy as np
from rag.loader import Document, load_documents
from rag.chunker import chunk_documents
from rag.embedder import embed_chunks, EMBEDDING_DIM


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Computes cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    print("=" * 60)
    print("AI-062: Embeddings Tests (Offline, CPU)")
    print("=" * 60)
    print("Note: First run will download all-MiniLM-L6-v2 (~80MB).\n")

    # ------------------------------------------------------------------
    # Setup: load and chunk the real sample documents
    # ------------------------------------------------------------------
    docs = load_documents("docs")
    chunks = chunk_documents(docs, chunk_size=500, overlap=100)
    print(f"Documents loaded : {len(docs)}")
    print(f"Chunks produced  : {len(chunks)}\n")

    # ------------------------------------------------------------------
    # Test 1: Generate embeddings
    # ------------------------------------------------------------------
    print("--- Test 1: Generate embeddings ---")
    embeddings = embed_chunks(chunks)
    print(f"  Embeddings generated: {len(embeddings)}")
    for i, emb in enumerate(embeddings):
        src = chunks[i].metadata["source"]
        idx = chunks[i].metadata["chunk_index"]
        print(f"  [{src}] chunk {idx}  shape={emb.shape}  dtype={emb.dtype}")

    # ------------------------------------------------------------------
    # Test 2: Count matches chunk count
    # ------------------------------------------------------------------
    print("\n--- Test 2: Count == chunk count ---")
    assert len(embeddings) == len(chunks), (
        f"Expected {len(chunks)} embeddings, got {len(embeddings)}"
    )
    print(f"  PASS -- {len(embeddings)} embeddings == {len(chunks)} chunks.")

    # ------------------------------------------------------------------
    # Test 3: Dimensionality is (384,) for all embeddings
    # ------------------------------------------------------------------
    print("\n--- Test 3: Dimensionality == (384,) ---")
    for i, emb in enumerate(embeddings):
        assert emb.shape == (EMBEDDING_DIM,), (
            f"Chunk {i}: expected shape ({EMBEDDING_DIM},), got {emb.shape}"
        )
    print(f"  PASS -- all {len(embeddings)} embeddings have shape ({EMBEDDING_DIM},).")

    # ------------------------------------------------------------------
    # Test 4: Identical text produces near-identical embedding
    # ------------------------------------------------------------------
    print("\n--- Test 4: Identical text -> cosine similarity >= 0.9999 ---")
    same_text = "Retrieval-Augmented Generation is a technique for grounding LLMs."
    doc_a = Document(content=same_text, metadata={"source": "test_a.txt"})
    doc_b = Document(content=same_text, metadata={"source": "test_b.txt"})
    emb_a, emb_b = embed_chunks([doc_a, doc_b])
    sim = cosine_similarity(emb_a, emb_b)
    print(f"  Cosine similarity of identical text: {sim:.6f}")
    assert sim >= 0.9999, f"Expected >= 0.9999, got {sim:.6f}"
    print("  PASS -- identical text produces near-identical embeddings.")

    # ------------------------------------------------------------------
    # Test 5: Empty input returns []
    # ------------------------------------------------------------------
    print("\n--- Test 5: Empty input returns [] ---")
    result = embed_chunks([])
    assert result == [], f"Expected [], got {result}"
    print("  PASS -- empty input returns empty list.")

    print("\n" + "=" * 60)
    print("AI-062 tests complete. No API calls were made.")
    print("=" * 60)


if __name__ == "__main__":
    main()
