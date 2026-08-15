import os
import numpy as np

# Safely default HF_HOME to a local directory so the downloaded model
# cache is stored inside the project folder, ensuring it is preserved
# in the deployment slug (for both build phase and runtime).
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.path.join(os.getcwd(), ".hf_cache")

from sentence_transformers import SentenceTransformer
from rag.loader import Document

# Model is downloaded once on first use (~80MB) and cached locally.
# CPU-only, no API key required.
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def embed_chunks(chunks: list[Document]) -> list[np.ndarray]:
    """
    Generates a dense embedding vector for each Document chunk.

    Uses sentence-transformers `all-MiniLM-L6-v2`, which runs locally on CPU.
    All chunk texts are encoded in a single batch call for efficiency.

    Args:
        chunks: A list of Document objects (typically produced by chunk_documents).

    Returns:
        A list of numpy arrays, one per chunk, each of shape (384,).
        Returns an empty list if the input is empty.
    """
    if not chunks:
        return []

    model = SentenceTransformer(MODEL_NAME)

    texts = [chunk.content for chunk in chunks]

    # encode() returns a 2-D numpy array of shape (n_chunks, 384)
    embeddings_matrix = model.encode(texts, convert_to_numpy=True)

    # Split into a list of 1-D arrays, one per chunk
    return [embeddings_matrix[i] for i in range(len(chunks))]
