import warnings
from rag.loader import Document, load_documents
from rag.chunker import chunk_document, chunk_documents


def main():
    print("=" * 60)
    print("AI-061: Document Chunking Tests (Offline)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Test 1: Happy path — chunk the real sample docs
    # ------------------------------------------------------------------
    print("\n--- Test 1: Chunk real sample documents ---")
    docs = load_documents("docs")
    chunks = chunk_documents(docs, chunk_size=500, overlap=100)

    for chunk in chunks:
        src   = chunk.metadata["source"]
        idx   = chunk.metadata["chunk_index"]
        total = chunk.metadata["chunk_count"]
        chars = len(chunk.content)
        print(f"  [{src}] chunk {idx+1}/{total}  |  {chars} chars")

    print(f"\n  Total chunks produced: {len(chunks)}")

    # ------------------------------------------------------------------
    # Test 2: Metadata verification
    # ------------------------------------------------------------------
    print("\n--- Test 2: Metadata verification ---")
    for chunk in chunks:
        assert "source" in chunk.metadata,      "Missing 'source' key"
        assert "chunk_index" in chunk.metadata,  "Missing 'chunk_index' key"
        assert "chunk_count" in chunk.metadata,  "Missing 'chunk_count' key"
        assert chunk.metadata["chunk_index"] >= 0
        assert chunk.metadata["chunk_count"] >= 1
    print("  PASS — all chunks have correct metadata keys and values.")

    # ------------------------------------------------------------------
    # Test 3: Overlap verification between adjacent chunks
    # ------------------------------------------------------------------
    print("\n--- Test 3: Overlap between adjacent chunks ---")
    # Grab the chunks for the first document only
    first_source = docs[0].metadata["source"]
    first_doc_chunks = [c for c in chunks if c.metadata["source"] == first_source]

    if len(first_doc_chunks) >= 2:
        tail = first_doc_chunks[0].content[-100:]
        head = first_doc_chunks[1].content[:100]
        overlap_found = any(word in head for word in tail.split() if len(word) > 4)
        print(f"  Tail of chunk 0 (last 100 chars): {tail!r}")
        print(f"  Head of chunk 1 (first 100 chars): {head!r}")
        print(f"  Overlap detected: {overlap_found}")
        assert overlap_found, "Expected word overlap between adjacent chunks"
        print("  PASS — adjacent chunks share overlapping content.")
    else:
        print(f"  SKIP — '{first_source}' produced only 1 chunk (document shorter than chunk_size).")

    # ------------------------------------------------------------------
    # Test 4: Short document produces exactly one chunk
    # ------------------------------------------------------------------
    print("\n--- Test 4: Short document -> exactly one chunk ---")
    short_doc = Document(content="Short text.", metadata={"source": "test_short.txt"})
    short_chunks = chunk_document(short_doc, chunk_size=500, overlap=100)
    print(f"  Chunks produced: {len(short_chunks)}")
    assert len(short_chunks) == 1, f"Expected 1 chunk, got {len(short_chunks)}"
    assert short_chunks[0].metadata["chunk_count"] == 1
    assert short_chunks[0].metadata["chunk_index"] == 0
    print("  PASS — short document produces exactly one chunk with correct metadata.")

    # ------------------------------------------------------------------
    # Test 5: Invalid argument handling
    # ------------------------------------------------------------------
    print("\n--- Test 5: Invalid argument handling ---")
    dummy = Document(content="Some content.", metadata={"source": "dummy.txt"})

    try:
        chunk_document(dummy, chunk_size=0, overlap=0)
        print("  FAIL — should have raised ValueError for chunk_size=0")
    except ValueError as e:
        print(f"  PASS — chunk_size=0: {e}")

    try:
        chunk_document(dummy, chunk_size=100, overlap=100)
        print("  FAIL — should have raised ValueError for overlap >= chunk_size")
    except ValueError as e:
        print(f"  PASS — overlap >= chunk_size: {e}")

    print("\n" + "=" * 60)
    print("AI-061 tests complete. No API calls were made.")
    print("=" * 60)


if __name__ == "__main__":
    main()
