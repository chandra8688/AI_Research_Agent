from rag.loader import Document
from llm import call_llm

def generate_rag_answer(query: str, chunks: list[Document]) -> str:
    """
    Generates an answer using the LLM based *only* on the retrieved chunks.
    
    Args:
        query: The user's question.
        chunks: The list of retrieved Document objects from the VectorStore.
        
    Returns:
        The generated answer string.
    """
    
    context_parts = []
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        chunk_idx = chunk.metadata.get("chunk_index", 0)
        context_parts.append(f"--- Document [{source}] (Chunk {chunk_idx}) ---\n{chunk.content}\n")
        
    context_str = "\n".join(context_parts)
    
    prompt = f"""You are a helpful and precise assistant. 
You will be provided with a set of retrieved document chunks as context, followed by a user query.

Your task is to answer the query using ONLY the provided context.
- If the answer is present in the context, provide a detailed and clear explanation.
- Where practical, cite the source of your information using the bracketed document source name (e.g., "[rag_overview.txt]").
- If the context does NOT contain enough information to answer the query, you MUST state that the information is not available in the provided context. Do NOT invent or infer an answer from outside knowledge.

CONTEXT:
{context_str}

USER QUERY:
{query}
"""

    return call_llm(prompt)
