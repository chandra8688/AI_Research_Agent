import os

from rag.loader import load_documents
from rag.chunker import chunk_documents
from rag.embedder import embed_chunks
from rag.store import VectorStore, get_vector_store

def initialize_knowledge_base(docs_dir: str = "docs") -> VectorStore:
    """
    Initializes the local RAG knowledge base if it does not exist.
    Otherwise, returns a VectorStore instance pointing to the existing database.
    """
    store = get_vector_store()
    
    # If the collection already has chunks, we assume it's initialized.
    # Note: store.count() will return > 0 if data exists.
    try:
        count = store.count()
    except Exception:
        count = 0
        
    if count > 0:
        return store
        
    print("Initializing local knowledge base...")
    if not os.path.exists(docs_dir):
        print(f"  Warning: '{docs_dir}' directory not found. No documents loaded.")
        return store

    docs = load_documents(docs_dir)
    if not docs:
        print(f"  Warning: No documents found in '{docs_dir}'.")
        return store
        
    print(f"  Chunking {len(docs)} document(s)...")
    chunks = chunk_documents(docs, chunk_size=500, overlap=100)
    
    print(f"  Embedding {len(chunks)} chunk(s) locally. This may take a moment...")
    embeddings = embed_chunks(chunks)
    
    print(f"  Storing {len(chunks)} chunk(s) into ChromaDB...")
    store.add_documents(chunks, embeddings)
    print("Local knowledge base initialization complete.")
    
    return store
