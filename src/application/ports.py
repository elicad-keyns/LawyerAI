from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from typing import Optional
from src.domain.entities import DocumentChunk, SearchResult


class EmbeddingPort(ABC):
    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class LanguageModelPort(ABC):
    @abstractmethod
    def answer(self, question: str, context: str) -> str: ...

    def stream_answer(self, question: str, context: str, on_event: Optional[Callable] = None):
        """Потоковый режим; адаптеры без streaming используют один фрагмент."""
        yield self.answer(question, context)


class QueryRewriterPort(ABC):
    @abstractmethod
    def rewrite(self, question: str) -> str: ...


class VectorStorePort(ABC):
    @abstractmethod
    def replace(self, chunks: Iterable[DocumentChunk]) -> int: ...

    @abstractmethod
    def search(self, vector: Sequence[float], limit: int, query: str = "") -> list[SearchResult]: ...

    @abstractmethod
    def count(self) -> int: ...


class ArticleRepositoryPort(ABC):
    @abstractmethod
    def find_article(self, number: str): ...


class DocumentReaderPort(ABC):
    @abstractmethod
    def read(self, path: str) -> str: ...
