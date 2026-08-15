import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("Starting RAG knowledge base build process...")
    
    try:
        # Import after environment setup in case other modules depend on it
        from rag.pipeline import initialize_knowledge_base
        
        # 2. Initialize the existing RAG pipeline
        store = initialize_knowledge_base()
        
        # 3. Verify that the vector store contains documents
        count = store.count()
        if count == 0:
            print("Error: Knowledge base initialization completed, but vector store contains 0 documents.")
            sys.exit(1)
            
        # 5. Exit successfully when the knowledge base is ready
        print(f"Success: Knowledge base prepared. Vector store contains {count} documents.")
        sys.exit(0)
        
    except Exception as e:
        # 4. Fail with a non-zero exit code if construction fails
        print(f"Error during RAG build process: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
