from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    source: str
    text: str
    embedding: tuple[float, ...] = ()


@dataclass(frozen=True)
class SearchResult:
    chunk: DocumentChunk
    score: float


@dataclass(frozen=True)
class ChatAnswer:
    text: str
    sources: tuple[str, ...]

