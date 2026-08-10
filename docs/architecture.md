# System Architecture

The AI Research Agent relies on a strict, layered architecture designed to cleanly separate HTTP concerns, orchestration, intelligence, external systems, and observability.

## 1. System Overview
The system is constructed with a FastAPI backend serving as the entrypoint for users (via web UI or REST APIs), which offloads user queries to a LangGraph-orchestrated ReAct loop. The agent retrieves knowledge through a unified multi-backend RAG pipeline and leverages fallback mechanisms for both retrievers and LLM providers.

## 2. API Layer
Built with FastAPI. It handles synchronous and asynchronous client requests, initializes session variables, and manages user state contexts before passing processing down to the LangGraph execution block. 

## 3. LangGraph Orchestration
The primary driver of the agent's logic. It replaces manual python loops with a strictly validated, cyclic Directed Acyclic Graph (DAG) using `StateGraph`. The graph natively manages loop cycles (like research and tool usage), handles dynamic error catching, and tracks iteration limits.

## 4. Agent State
A robust Pydantic/dataclass structured `AgentState` object tracks every execution. It stores the conversation memory, list of tool calls, multi-source evidence, LLM reflection attempts, research plans, and extensive metric tracing outputs.

## 5. Planning
Executed immediately when a user query enters the graph. The planning module utilizes the LLM to structurally classify the query's intent (e.g., does this need a calculator? web search? multi-source synthesis?) and prepares a deterministic strategy before tools are indiscriminately chosen.

## 6. Tools
The agent uses an extensible Tool Registry pattern (`tools.py`) to bind Python functions directly to LLM `FunctionDeclarations`. Available tools currently include a math calculator, DuckDuckGo web search, and a localized document retriever.

## 7. RAG Layer
Retrieval-Augmented Generation processes document loaders, embedding vectorization (via SentenceTransformers `all-MiniLM-L6-v2`), and querying. The layer relies on retrieving localized knowledge to ground the LLM's final generated answer.

## 8. Vector Store Abstraction
Vector databases are decoupled via a Python `Protocol` interface. The implementation can dynamically swap or utilize different vector stores under a single, unified method signature (`search`, `add_documents`).

## 9. Chroma
The default, localized vector database running in flat-file/in-memory mode for rapid, zero-dependency environment setups and immediate knowledge grounding.

## 10. Pinecone
The integrated cloud vector database. If Chroma is inaccessible or requires a scaled enterprise dataset, Pinecone serves as the backend database option.

## 11. Retrieval Fallback
A safety wrapper around the vector abstraction layer. If Chroma (the primary backend) triggers a `RetryableRetrievalError` (e.g., connection timeout or database lock), this layer dynamically switches the query to the Pinecone backend seamlessly.

## 12. Multi-Retriever Fusion
Instead of choosing one database or another, this module deliberately queries BOTH Chroma and Pinecone simultaneously to assemble the broadest spectrum of potential evidence.

## 13. Reciprocal Rank Fusion
RRF is used to safely combine documents returned by Multi-Retriever Fusion. Because distance scores in Chroma and similarity scores in Pinecone are mathematically incompatible, RRF relies entirely on the document's rank position to normalize and merge the strongest evidence.

## 14. Evidence Reflection
Once evidence is collected, the agent halts generation and executes a separate LLM evaluation phase. It rigorously examines the assembled text and answers a binary question: "Is this context actually sufficient to answer the prompt?" If insufficient, the agent refines its query and loops back to retrieve more documents.

## 15. Research Synthesis
When multiple knowledge sources are engaged (e.g., Local Knowledge + Web Knowledge), the synthesis module dynamically parses, merges, and tags the varying source formats so the LLM understands its origin.

## 16. Quality / Claim Grounding
Before any text is sent to the user, the agent extracts individual factual assertions (claims) from its own generated response. It then cross-validates each claim exclusively against the retrieved evidence. If unsupported claims exist, the agent forcibly scrubs or refines them.

## 17. Conversation Memory
FastAPI requests are mapped to unique UUID sessions. The conversation module stores trailing LLM iterations and limits context windows so follow-up inquiries correctly recall previous context.

## 18. LLM Provider Abstraction
Interaction with foundational models is stripped of provider-specific nuances. The `Provider` abstract class forces a consistent standard for initializing prompt contexts, generation, and error surface bubbling.

## 19. Gemini/OpenRouter Fallback
To ensure production availability, the system natively implements provider fault tolerance. If the primary LLM (Gemini) encounters rate limits (`429`) or server downtime (`503`), the agent automatically pivots to OpenRouter models to fulfill the request.

## 20. LangChain Core Integration
Selectively utilizes `langchain-core` for standardized document conversions and `ChatPromptTemplate` implementations. By isolating these within an adapter design (`langchain_integration.py`), the codebase gains standard interface features without fully coupling the architecture to LangChain.

## 21. Observability
Every major component appends dictionary-style traces into the `AgentState`. This creates a granular audit log tracking LLM performance times, retrieved documents, grounding scores, tool calls, and API failures.

## 22. Frontend
A static web interface dynamically loaded via the FastAPI server. It exposes a simple chat window parsing the agent's final text and displaying traced source citations, highlighting the agent's multi-step decision pipeline.
