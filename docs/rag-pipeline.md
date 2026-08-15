# RAG Pipeline

This document describes the Retrieval-Augmented Generation (RAG) pipeline in AI Research Agent V1.0.

The pipeline has two distinct phases: **ingestion** (build-time or first-startup) and **retrieval** (per-query at runtime).

---

## Local vs Web Evidence

The agent uses two independent retrieval channels:

| Channel | Source | Parser | `source_type` |
|---|---|---|---|
| Local knowledge | ChromaDB vector store | `parse_local_evidence()` | `"local"` |
| Web search | DuckDuckGo (`ddgs`) | `parse_web_evidence()` | `"web"` |

Both channels produce `EvidenceItem` objects (`research.py`) and are accumulated together in `agent_state.multi_source_evidence`. The grounding and citation systems treat them identically except for citation format.

---

## Ingestion Pipeline

```
docs/ directory
    (contains .txt files — e.g., rag_overview.txt, evaluation.md converted to .txt)
         |
         v
rag/loader.py  load_documents(directory)
    - Reads all .txt files sorted alphabetically for determinism
    - Each file -> Document(content=<full text>, metadata={"source": filename})
    - Returns [] if directory missing or no .txt files found
         |
         v
rag/chunker.py  chunk_documents(docs, chunk_size=500, overlap=100)
    - Sliding-window character chunker
    - Chunk 0: chars [0:500]
    - Chunk 1: chars [400:900]   (overlap of 100 chars)
    - Short documents produce exactly one chunk
    - Each chunk carries metadata: source, chunk_index, chunk_count
         |
         v
rag/embedder.py  embed_chunks(chunks)
    - Model: sentence-transformers/all-MiniLM-L6-v2
    - Embedding dimension: 384
    - Runs on CPU — no GPU required, no API key required
    - Model is downloaded on first use (~80 MB) to HF_HOME (.hf_cache/)
    - Encodes all chunk texts in a single batch call
    - Returns list of numpy arrays, one per chunk
         |
         v
rag/chroma_store.py  ChromaStore.add_documents(chunks, embeddings)
    - Persisted to .chroma_db/ directory
    - Collection name: "rag_collection"
    - Documents are stored with content and metadata
```

### When does ingestion happen?

- **Docker build**: `scripts/build_rag.py` runs `initialize_knowledge_base()` during the Docker build step. The populated `.chroma_db/` is baked into the image.
- **Local development**: `rag/pipeline.py:initialize_knowledge_base()` checks if the Chroma collection has any documents. If empty, it ingests the `docs/` directory automatically on first startup.
- **Subsequent starts**: If the collection already has documents (`store.count() > 0`), ingestion is skipped.

---

## Retrieval Pipeline

### Local Knowledge Retrieval

```
User query string
         |
         v
tools.py  search_local_knowledge(query)
    - Creates a dummy Document from the query string
    - Calls embed_chunks([query_doc]) -> query embedding (384-dim numpy array)
    - Checks store.count() -- returns error if empty
         |
         v
rag/store.py  FallbackVectorStore.search(query_embedding, k=3)
    - Calls ChromaStore.search() as primary
    - If ChromaStore raises RetryableRetrievalError, falls back to PineconeStore
         |
         v
rag/chroma_store.py  ChromaStore.search(query_embedding, k=3)
    - Queries the ChromaDB collection for the k nearest neighbours
    - Returns list of Document objects with metadata (source, chunk_index, distance)
         |
         v
Formatted as:
    "[Evidence 1]\nSource: filename.txt (Chunk 0)\nDistance: 0.1234\nText: ..."
         |
         v
research.py  parse_local_evidence(text)
    -> list[EvidenceItem(source_type="local")]
```

### Web Retrieval

```
User query string
         |
         v
tools.py  search_web(query, max_results=3)
    - Calls DDGS().text(query, max_results=max_results)
    - max_results is clamped to [1, 10]
    - Returns on empty results or search failure with an error string
         |
         v
Formatted as:
    "[Result 1]\nTitle: Page Title\nURL: https://...\nSnippet: ..."
         |
         v
research.py  parse_web_evidence(text)
    -> list[EvidenceItem(source_type="web", source="Page Title (https://...)")]
```

---

## Vector Store Abstraction

`rag/store.py` defines a `VectorStore` Protocol with three methods:
- `add_documents(chunks, embeddings)`
- `count() -> int`
- `search(query_embedding, k) -> list[Document]`

Three implementations exist:

### `FallbackVectorStore` (default)
Wraps a primary and fallback backend. If the primary backend raises `RetryableRetrievalError`, it transparently switches to the fallback. Fatal errors are propagated immediately without fallback.

Default: primary = ChromaDB, fallback = Pinecone (when `VECTOR_DB_FALLBACK_ENABLED=True`).

### `FusionVectorStore` (optional)
Enabled when `RETRIEVAL_FUSION_ENABLED=True`. Queries both ChromaDB and Pinecone simultaneously and merges results using Reciprocal Rank Fusion (`rag/reranker.py`).

**Why RRF?** ChromaDB uses L2 distance (lower = better). Pinecone uses cosine similarity (higher = better). The raw scores are mathematically incompatible. RRF ignores raw scores entirely and merges by rank position — a document ranked #1 by Chroma and #2 by Pinecone gets a higher combined score than a document ranked #5 by either.

### `ChromaStore` (`rag/chroma_store.py`)
Default implementation. Uses the `chromadb` library with a persistent directory (`.chroma_db/`).

### `PineconeStore` (`rag/pinecone_store.py`)
Cloud vector store. Requires `PINECONE_API_KEY` and `PINECONE_INDEX_NAME`.

---

## Evidence Formatting

After retrieval, evidence items are formatted for injection into LLM context via `research.py:format_combined_evidence()`:

```
[LOCAL: filename.txt]
<chunk text>

[WEB: Page Title (https://example.com)]
<snippet text>
```

This formatted string is inserted into the LLM''s conversation context during the synthesis step.

---

## Limitations

- Only `.txt` files are ingested from the `docs/` directory. Markdown, PDF, or other formats are not supported in V1.0.
- The DuckDuckGo web search accepts whatever results the search engine returns — there is no domain authority ranking.
- Web evidence is limited to snippets (typically 200–400 characters). Full page content is not fetched.
- The local knowledge base is static — adding new documents requires rebuilding or reinitialising the vector store.
