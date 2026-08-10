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
    
    from langchain_integration import get_rag_prompt_template
    prompt_template = get_rag_prompt_template()
    prompt_value = prompt_template.invoke({"context": context_str, "query": query})
    prompt = prompt_value.to_string()

    return call_llm(prompt)
