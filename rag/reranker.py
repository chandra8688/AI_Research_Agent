from rag.loader import Document

def fuse_results(chroma_docs: list[Document], pinecone_docs: list[Document], final_k: int) -> list[Document]:
    """
    Fuses results from Chroma and Pinecone using Reciprocal Rank Fusion (RRF)
    and removes duplicates based on source and chunk_index.
    """
    # RRF formula: RRF_score = sum(1 / (k + rank)) where k = 60
    K = 60
    
    scored_chunks = {} # key: (source, chunk_index), value: Document
    
    # Process Chroma
    for rank, doc in enumerate(chroma_docs, start=1):
        source = doc.metadata.get("source", "unknown")
        chunk_idx = doc.metadata.get("chunk_index", 0)
        key = (source, chunk_idx)
        
        rrf_score = 1.0 / (K + rank)
        
        new_meta = doc.metadata.copy()
        new_meta["backend"] = "chroma"
        if "distance" in new_meta:
            new_meta["raw_score"] = new_meta["distance"]
        new_meta["normalized_score"] = rrf_score
        
        fused_doc = Document(content=doc.content, metadata=new_meta)
        scored_chunks[key] = fused_doc

    # Process Pinecone
    for rank, doc in enumerate(pinecone_docs, start=1):
        source = doc.metadata.get("source", "unknown")
        chunk_idx = doc.metadata.get("chunk_index", 0)
        key = (source, chunk_idx)
        
        rrf_score = 1.0 / (K + rank)
        
        if key in scored_chunks:
            # Duplicate found, combine scores
            existing_doc = scored_chunks[key]
            existing_doc.metadata["normalized_score"] += rrf_score
            existing_doc.metadata["backend"] = "chroma+pinecone"
            if "distance" in doc.metadata:
                existing_doc.metadata["pinecone_raw_score"] = doc.metadata["distance"]
        else:
            new_meta = doc.metadata.copy()
            new_meta["backend"] = "pinecone"
            if "distance" in new_meta:
                new_meta["raw_score"] = new_meta["distance"]
            new_meta["normalized_score"] = rrf_score
            
            fused_doc = Document(content=doc.content, metadata=new_meta)
            scored_chunks[key] = fused_doc
            
    # Sort by normalized score descending
    sorted_docs = sorted(scored_chunks.values(), key=lambda d: d.metadata["normalized_score"], reverse=True)
    
    # Return final Top-K
    return sorted_docs[:final_k]
