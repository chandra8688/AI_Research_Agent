# RAG Pipeline

The Retrieval-Augmented Generation (RAG) pipeline operates independently of the agent orchestration, functioning as a standalone modular knowledge ingestion and extraction engine.

## Execution Sequence

1. **Document Loading:** Raw text data is extracted via the `loader.py`. 
2. **Chunking:** The continuous text is spliced into smaller, context-preserving components.
3. **Embedding:** The application uses the local SentenceTransformers model `all-MiniLM-L6-v2` (which runs without external API limits) to generate 384-dimensional dense vectors.
4. **Vector Storage:** These embeddings are ingested into either Chroma (local) or Pinecone (cloud).
5. **Query Embedding:** During agent querying, the target query string is similarly embedded through `all-MiniLM-L6-v2`.
6. **Retrieval:** The vector stores fetch the top-k matches closest in the high-dimensional space.
7. **Fallback:** If the primary store fails (e.g., Chroma crashes), the pipeline immediately bounces the identical query to the fallback secondary store.
8. **Multi-Retriever Fusion:** If the agent is instructed to perform deep research, it intentionally triggers both vector backends simultaneously to aggregate results.
9. **Reciprocal Rank Fusion (RRF):** The results from Fusion are deduplicated and merged using their mathematical positional rank.
10. **Evidence Generation:** Final matched chunks are structurally transformed into explicit `EvidenceItem` models.
11. **Generation:** The RAG generator parses evidence into a controlled generation prompt.
12. **Reflection:** The LLM strictly evaluates the generated evidence for relevance.
13. **Grounding:** Claim validation ensures generated text didn't hallucinate outside of the supplied RRF-merged documents.

## Document Metadata

To maintain provenance throughout the pipeline, the `Document` and `EvidenceItem` structure tightly preserves metadata:
- **`source`**: The origin filename or URL.
- **`chunk_index`**: The exact structural position of the text in the original document.
- **`distance` / `raw_score`**: The raw algorithm score retrieved directly from the vector store.
- **`backend`**: Identifies whether the document came from "chroma" or "pinecone".
- **`normalized_score`**: A post-processed confidence score (e.g., an RRF adjusted decimal) utilized for general sorting.

## Chroma Distance vs. Pinecone Similarity

A massive complexity in dual-backend RAG systems is varying mathematical measurement approaches:
- **Chroma** generally relies on `L2 distance` (Euclidean distance) where a **LOWER** score indicates a stronger match (closer).
- **Pinecone** generally relies on `Cosine Similarity` where a **HIGHER** score (closer to 1.0) indicates a stronger match.

## Why Reciprocal Rank Fusion (RRF)?

Because Chroma distance algorithms and Pinecone similarity algorithms are mathematically incompatible, comparing a Chroma score of `0.2` directly against a Pinecone score of `0.8` is impossible. 

Instead of attempting fragile mathematical normalizations across algorithms, **Reciprocal Rank Fusion** ignores the raw score entirely. It instead looks at the document's placement in the result list (e.g., Rank 1, Rank 2). The formula calculates a combined score based purely on these positional ranks across all returning vector databases, ensuring the strongest overall documents bubble to the top seamlessly.
