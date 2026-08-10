from dataclasses import dataclass, field
from typing import Any
import re

@dataclass
class EvidenceItem:
    content: str
    source: str
    source_type: str
    metadata: dict = field(default_factory=dict)
    relevance: float | None = None

def parse_local_evidence(text: str) -> list[EvidenceItem]:
    items = []
    chunks = text.split("[Evidence ")
    for chunk in chunks[1:]:
        match = re.search(r"Source:\s*(.*?)\s*\(Chunk\s*([^)]+)\)\s*Distance:\s*([^\n]+)\s*Text:\s*(.*)", chunk, re.DOTALL)
        if match:
            source = match.group(1).strip()
            chunk_idx = match.group(2).strip()
            dist = match.group(3).strip()
            content = match.group(4).strip()
            items.append(EvidenceItem(
                content=content,
                source=source,
                source_type="local",
                metadata={"chunk_index": chunk_idx, "distance": dist}
            ))
    return items

def parse_web_evidence(text: str) -> list[EvidenceItem]:
    items = []
    chunks = text.split("[Result ")
    for chunk in chunks[1:]:
        match = re.search(r"Title:\s*([^\n]+)\s*URL:\s*([^\n]+)\s*Snippet:\s*(.*)", chunk, re.DOTALL)
        if match:
            title = match.group(1).strip()
            url = match.group(2).strip()
            snippet = match.group(3).strip()
            items.append(EvidenceItem(
                content=snippet,
                source=f"{title} ({url})",
                source_type="web",
                metadata={"title": title, "url": url}
            ))
    return items

def combine_evidence(local_text: str = "", web_text: str = "") -> list[EvidenceItem]:
    local_items = parse_local_evidence(local_text) if local_text else []
    web_items = parse_web_evidence(web_text) if web_text else []
    return local_items + web_items

def format_combined_evidence(items: list[EvidenceItem]) -> str:
    if not items:
        return "No evidence collected."
    formatted = []
    for item in items:
        if item.source_type == "local":
            formatted.append(f"[LOCAL: {item.source}]\n{item.content}")
        else:
            formatted.append(f"[WEB: {item.source}]\n{item.content}")
    return "\n\n".join(formatted)
