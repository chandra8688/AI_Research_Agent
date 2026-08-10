from typing import Any, List
from langchain_core.documents import Document as LCDocument
from langchain_core.prompts import ChatPromptTemplate
from rag.loader import Document

# -----------------------------------------------------------------------------
# 1. Document Abstraction
# -----------------------------------------------------------------------------
def to_langchain_document(doc: Document) -> LCDocument:
    """
    Converts our custom Document to a LangChain Core Document safely,
    preserving all metadata.
    """
    return LCDocument(
        page_content=doc.content,
        metadata=doc.metadata.copy()
    )

# -----------------------------------------------------------------------------
# 2. Prompt Templates
# -----------------------------------------------------------------------------
def get_rag_prompt_template() -> ChatPromptTemplate:
    """
    Creates a ChatPromptTemplate for RAG generation.
    """
    system_msg = (
        "You are a helpful and precise assistant. "
        "You will be provided with a set of retrieved document chunks as context, followed by a user query.\n\n"
        "Your task is to answer the query using ONLY the provided context.\n"
        "- If the answer is present in the context, provide a detailed and clear explanation.\n"
        "- Where practical, cite the source of your information using the bracketed document source name (e.g., \"[rag_overview.txt]\").\n"
        "- If the context does NOT contain enough information to answer the query, you MUST state that the information is not available in the provided context. Do NOT invent or infer an answer from outside knowledge."
    )
    
    user_msg = "CONTEXT:\n{context}\n\nUSER QUERY:\n{query}"
    
    return ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("user", user_msg)
    ])

def get_reflection_prompt_template() -> ChatPromptTemplate:
    """
    Creates a ChatPromptTemplate for evidence reflection.
    """
    system_msg = (
        "You are a strict evaluator for a research agent.\n"
        "Your task is to determine whether the provided evidence is sufficient to accurately and fully answer the original query.\n\n"
        "RULES:\n"
        "1. Evaluate ONLY the supplied evidence.\n"
        "2. Do NOT use outside knowledge to answer the query. \n"
        "3. Return `sufficient=True` ONLY when the evidence directly and fully supports the requested answer.\n"
        "4. Return `sufficient=False` if the evidence is unrelated, incomplete, or does not contain the answer.\n"
        "5. Provide a brief `reason` explaining your decision based on what the evidence actually contains."
    )
    
    user_msg = "ORIGINAL QUERY:\n{query}\n\nSUPPLIED EVIDENCE:\n{evidence}"
    
    return ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("user", user_msg)
    ])

# -----------------------------------------------------------------------------
# 3. Retriever Adapter
# -----------------------------------------------------------------------------
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from rag.store import get_vector_store
from rag.embedder import embed_chunks
from rag.loader import Document as CustomDocument

class LangChainRetrieverAdapter(BaseRetriever):
    """
    Adapter that exposes our existing VectorStore hierarchy (Chroma, Pinecone,
    Fusion, Fallback) as a LangChain BaseRetriever.
    """
    k: int = 3
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[LCDocument]:
        # 1. Embed query using our existing pipeline
        # Create a dummy Document to embed
        dummy_chunk = CustomDocument(content=query)
        emb = embed_chunks([dummy_chunk])[0]
        
        # 2. Invoke our existing vector store
        store = get_vector_store()
        results = store.search(emb, k=self.k)
        
        # 3. Convert results to LangChain Documents
        return [to_langchain_document(doc) for doc in results]
