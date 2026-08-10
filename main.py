import os
import shutil
from dotenv import load_dotenv

from rag.loader import load_documents
from rag.chunker import chunk_documents
from rag.embedder import embed_chunks
from rag.store import VectorStore
from tools import search_local_knowledge
from agent import run_agent

def main():
    load_dotenv()
    print("=" * 60)
    print("AI-066: Agentic RAG Integration Tests")
    print("=" * 60)

    db_path = ".chroma_db"

    # ------------------------------------------------------------------
    # Setup: Ensure VectorStore is populated
    # ------------------------------------------------------------------
    print("\n--- Setup: Building Pipeline ---")
    if os.path.exists(db_path):
        try:
            shutil.rmtree(db_path)
        except Exception:
            pass

    docs = load_documents("docs")
    chunks = chunk_documents(docs, chunk_size=500, overlap=100)
    embeddings = embed_chunks(chunks)

    store = VectorStore(persist_directory=db_path)
    store.add_documents(chunks, embeddings)
    print(f"  Stored {store.count()} chunks in ChromaDB.")


    # ------------------------------------------------------------------
    # Test 1: Direct tool test (OFFLINE)
    # ------------------------------------------------------------------
    print("\n--- Test 1: Direct offline tool test ---")
    offline_res = search_local_knowledge("What is RAG?")
    print("-" * 40)
    print(offline_res)
    print("-" * 40)
    assert "Error:" not in offline_res, "Local tool failed unexpectedly."
    assert "[Evidence" in offline_res, "Formatted evidence missing."

    # ------------------------------------------------------------------
    # Test 4: Empty query handling (OFFLINE)
    # ------------------------------------------------------------------
    print("\n--- Test 4: Direct offline empty query test ---")
    empty_res = search_local_knowledge("")
    print(f"  Result: {empty_res}")
    assert "Error" in empty_res, "Empty query did not return an error string."

    # ------------------------------------------------------------------
    # Test 3: Calculator regression (AGENT)
    # ------------------------------------------------------------------
    print("\n--- Test 3: Agent calculator regression ---")
    print("  Prompt: 'What is 42 multiplied by 7?'")
    try:
        # Note: using a small max_iterations to protect quota if it gets confused
        calc_answer = run_agent("What is 42 multiplied by 7?", max_iterations=3)
        print("\n  Final Answer:", calc_answer)
    except Exception as e:
        print(f"\n  Agent failed: {e}")

    # ------------------------------------------------------------------
    # Test 2: Local knowledge RAG query (AGENT)
    # ------------------------------------------------------------------
    print("\n--- Test 2: Agent local-knowledge RAG test ---")
    print("  Prompt: 'What does our local documentation say about RAG versus fine-tuning?'")
    try:
        rag_answer = run_agent("What does our local documentation say about RAG versus fine-tuning?", max_iterations=3)
        print("\n  Final Answer:", rag_answer)
    except Exception as e:
        print(f"\n  Agent failed: {e}")

    print("\n" + "=" * 60)
    print("AI-066 tests complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
