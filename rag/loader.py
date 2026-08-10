import os
import warnings
from dataclasses import dataclass, field


@dataclass
class Document:
    """A single document unit: raw text content plus metadata."""
    content: str
    metadata: dict = field(default_factory=dict)


def load_documents(directory: str) -> list[Document]:
    """
    Loads all .txt files from the given directory.

    Returns a list of Document objects sorted by filename for determinism.
    Each Document contains:
        content  - the full text of the file
        metadata - {"source": filename}

    Handles gracefully:
        - Directory does not exist  → returns []  (with a warning)
        - No .txt files found       → returns []  (with a warning)
        - Unreadable file           → skips file  (with a warning)
    """
    if not os.path.isdir(directory):
        warnings.warn(f"load_documents: directory not found: '{directory}'")
        return []

    txt_files = sorted(
        f for f in os.listdir(directory) if f.endswith(".txt")
    )

    if not txt_files:
        warnings.warn(f"load_documents: no .txt files found in '{directory}'")
        return []

    documents: list[Document] = []
    for filename in txt_files:
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                content = fh.read()
            documents.append(
                Document(content=content, metadata={"source": filename})
            )
        except (OSError, UnicodeDecodeError) as e:
            warnings.warn(f"load_documents: could not read '{filename}': {e}")

    return documents
