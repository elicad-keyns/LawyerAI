from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from src.domain.entities import DocumentChunk, SearchResult


class EmbeddingPort(ABC):
    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class LanguageModelPort(ABC):
    @abstractmethod
    def answer(self, question: str, context: str) -> str: ...


class VectorStorePort(ABC):
    @abstractmethod
    def replace(self, chunks: Iterable[DocumentChunk]) -> int: ...

    @abstractmethod
    def search(self, vector: Sequence[float], limit: int) -> list[SearchResult]: ...

    @abstractmethod
    def count(self) -> int: ...


class DocumentReaderPort(ABC):
    @abstractmethod
    def read(self, path: str) -> str: ...

