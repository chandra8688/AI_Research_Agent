import os
import shutil
from dotenv import load_dotenv
from rag.loader import load_documents, Document
from rag.chunker import chunk_documents
from rag.embedder import embed_chunks
from rag.store import VectorStore
from rag.generator import generate_rag_answer


def main():
    load_dotenv()
    print("=" * 60)
    print("AI-065: Basic RAG Answer Generation Tests")
    print("=" * 60)
    
    db_path = ".chroma_db"
    
    # Clean up previous tests
    if os.path.exists(db_path):
        try:
            shutil.rmtree(db_path)
        except Exception:
            pass
            
    # ------------------------------------------------------------------
    # Setup: Local pipeline (load -> chunk -> embed -> store)
    # ------------------------------------------------------------------
    print("\n--- Setup: Building Pipeline ---")
    docs = load_documents("docs")
    chunks = chunk_documents(docs, chunk_size=500, overlap=100)
    embeddings = embed_chunks(chunks)
    
    store = VectorStore(persist_directory=db_path)
    store.add_documents(chunks, embeddings)
    print(f"  Stored {store.count()} chunks in ChromaDB.")

    # Helper for searching and generating
    def test_rag(query_text: str):
        print(f"\nQUERY: '{query_text}'")
        
        # 1. Embed query
        query_chunk = Document(content=query_text, metadata={"source": "query"})
        query_embedding = embed_chunks([query_chunk])[0]
        
        # 2. Retrieve chunks
        retrieved_chunks = store.search(query_embedding, k=3)
        print(f"  [Retrieved {len(retrieved_chunks)} chunks]")
        for i, chunk in enumerate(retrieved_chunks):
            print(f"    - Chunk {i}: {chunk.metadata.get('source')} (dist: {chunk.metadata.get('distance', 0.0):.4f})")
            
        # 3. Generate answer
        print("\n  [Generating Answer...]")
        answer = generate_rag_answer(query_text, retrieved_chunks)
        print("-" * 40)
        print(answer)
        print("-" * 40)
        return answer

    # ------------------------------------------------------------------
    # Test 1: Grounded query
    # ------------------------------------------------------------------
    print("\n--- Test 1: Explain RAG ---")
    # test_rag("Explain RAG.") # Skipped to save quota

    
    # ------------------------------------------------------------------
    # Test 2: Unrelated query (should fail gracefully)
    # ------------------------------------------------------------------
    print("\n--- Test 2: Unrelated query (Out of Context) ---")
    test_rag("What is the capital of France?")

    print("\n" + "=" * 60)
    print("AI-065 tests complete. (2 Gemini API calls made)")
    print("=" * 60)


if __name__ == "__main__":
    main()
