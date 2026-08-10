from rag.loader import Document


def chunk_document(
    doc: Document,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[Document]:
    """
    Splits a single Document into overlapping fixed-size character chunks.

    Strategy: sliding window.
        - Chunk 0: chars [0 : chunk_size]
        - Chunk 1: chars [chunk_size - overlap : 2*chunk_size - overlap]
        - ...and so on until the end of content.

    A document shorter than chunk_size produces exactly one chunk.

    Args:
        doc:        The source Document to split.
        chunk_size: Maximum number of characters per chunk (must be >= 1).
        overlap:    Number of characters shared between adjacent chunks
                    (must be < chunk_size).

    Returns:
        A list of Document objects, each representing one chunk.

    Raises:
        ValueError: If chunk_size < 1 or overlap >= chunk_size.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )

    content = doc.content
    step = chunk_size - overlap

    # Collect all start positions
    starts = list(range(0, max(len(content), 1), step))

    chunks: list[Document] = []
    for idx, start in enumerate(starts):
        chunk_text = content[start : start + chunk_size]
        if not chunk_text:
            break
        chunks.append(
            Document(
                content=chunk_text,
                metadata={
                    **doc.metadata,
                    "chunk_index": idx,
                    "chunk_count": None,  # filled in below
                },
            )
        )

    # Back-fill chunk_count now that we know the total
    total = len(chunks)
    for chunk in chunks:
        chunk.metadata["chunk_count"] = total

    return chunks


def chunk_documents(
    docs: list[Document],
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[Document]:
    """
    Applies chunk_document to a list of Documents and returns a flat list of chunks.
    """
    all_chunks: list[Document] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, chunk_size=chunk_size, overlap=overlap))
    return all_chunks
