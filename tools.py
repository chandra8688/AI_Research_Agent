def calculate_product(a: int, b: int) -> int:
    """Calculates the product of two integers."""
    return a * b


def search_web(query: str, max_results: int = 3) -> str:
    """
    Searches the web using DuckDuckGo and returns formatted results.

    Each result includes: title, URL, and a short snippet.
    Returns an error message string on failure so the LLM can handle it gracefully.
    """
    from ddgs import DDGS

    if not query or not query.strip():
        return "Error: search query cannot be empty."

    try:
        max_results = int(max_results)
        if max_results < 1:
            max_results = 1
        elif max_results > 10:
            max_results = 10
    except (TypeError, ValueError):
        max_results = 3

    try:
        results = list(DDGS().text(query.strip(), max_results=max_results))
    except Exception as e:
        return f"Error: search failed — {str(e)}"

    if not results:
        return "No results found for the given query."

    formatted = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", "No title")
        url = r.get("href", "No URL")
        snippet = r.get("body", "No snippet available.")
        formatted.append(f"[Result {i}]\nTitle: {title}\nURL: {url}\nSnippet: {snippet}")

    return "\n\n".join(formatted)


def search_local_knowledge(query: str) -> str:
    """
    Searches the local knowledge base (VectorStore) for relevant chunks based on the query.
    Returns formatted evidence including source metadata.
    """
    if not query or not query.strip():
        return "Error: search query cannot be empty."

    try:
        from rag.loader import Document
        from rag.embedder import embed_chunks
        from rag.store import get_vector_store
        
        # We need a quick dummy chunk to embed the query since embed_chunks expects a list of Documents
        query_chunk = Document(content=query.strip(), metadata={})
        query_embedding = embed_chunks([query_chunk])
        
        if not query_embedding:
            return "Error: could not generate embedding for query."
            
        store = get_vector_store()
        if store.count() == 0:
            return "Error: local vector database is empty. No documents available to search."
            
        results = store.search(query_embedding[0], k=3)
        if not results:
            return "No relevant local documents found for the given query."
            
        formatted = []
        for i, chunk in enumerate(results, start=1):
            source = chunk.metadata.get("source", "unknown")
            chunk_idx = chunk.metadata.get("chunk_index", "unknown")
            dist = chunk.metadata.get("distance", "unknown")
            
            # Optionally format distance to a neat float if it is one
            dist_str = f"{dist:.4f}" if isinstance(dist, float) else str(dist)
            
            formatted.append(
                f"[Evidence {i}]\nSource: {source} (Chunk {chunk_idx})\nDistance: {dist_str}\nText: {chunk.content}"
            )
            
        return "\n\n".join(formatted)

    except Exception as e:
        return f"Error: local knowledge search failed — {str(e)}"
