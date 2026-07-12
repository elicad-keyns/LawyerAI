from pathlib import Path
import re
from time import perf_counter
from src.application.ports import ArticleRepositoryPort, DocumentReaderPort, EmbeddingPort, LanguageModelPort, VectorStorePort
from src.domain.entities import ChatAnswer, DocumentChunk


class AskLegalQuestion:
    def __init__(self, embedder: EmbeddingPort, store: VectorStorePort, llm: LanguageModelPort, top_k: int = 5, articles: ArticleRepositoryPort = None):
        self._embedder, self._store, self._llm, self._top_k = embedder, store, llm, top_k
        self._articles = articles

    def execute(self, question: str) -> ChatAnswer:
        stream, sources = self.execute_stream(question)
        return ChatAnswer("".join(stream), sources)

    def execute_stream(self, question: str, on_event=None):
        emit = on_event or (lambda *_: None)
        question = question.strip()
        if not question:
            raise ValueError("Вопрос не может быть пустым")
        emit("rag.question.validated", {"characters": len(question)})
        article_numbers = re.findall(r"(?i)стат(?:ья|ьи|ью|ье|ьей|ьёй|ей)\s*[№N]?\s*(\d+(?:\.\d+)?)", question)
        if self._articles and len(set(article_numbers)) == 1:
            number = article_numbers[0]
            emit("article.lookup.started", {"number": number})
            article = self._articles.find_article(number)
            if article:
                source, text = article
                formatted = self._format_article(text)
                emit("article.lookup.completed", {"number": number, "characters": len(formatted), "mode": "verbatim_without_llm"})
                return iter([formatted]), (source,)
            emit("article.lookup.missed", {"number": number})
        search_query = question
        started = perf_counter()
        emit("rag.embedding.started", {"model_input_characters": len(search_query)})
        # paraphrase-multilingual-MiniLM кодирует обычный текст без E5-префиксов
        vector = self._embedder.embed([search_query])[0]
        emit("rag.embedding.completed", {"duration_ms": round((perf_counter() - started) * 1000), "dimensions": len(vector)})
        started = perf_counter()
        emit("rag.search.started", {"top_k": self._top_k, "indexed_chunks": self._store.count()})
        results = self._store.search(vector, self._top_k, search_query)
        emit("rag.search.completed", {
            "duration_ms": round((perf_counter() - started) * 1000),
            "results": [{"source": r.chunk.source, "score": round(r.score, 4), "characters": len(r.chunk.text)} for r in results],
        })
        if not results:
            return iter(["База ТК РФ пока не проиндексирована. Добавьте документы в папку documents."]), ()
        context = "\n\n".join(f"[{i + 1}] {r.chunk.source}\n{r.chunk.text}" for i, r in enumerate(results))
        sources = tuple(dict.fromkeys(r.chunk.source for r in results))
        emit("rag.context.prepared", {"characters": len(context), "sources": len(sources)})
        return self._llm.stream_answer(question, context, emit), sources

    @staticmethod
    def _format_article(text: str) -> str:
        text = re.sub(r"\s+(\d+)\)\s+", r"\n\1) ", text).strip()
        return text



class IndexDocuments:
    def __init__(self, reader: DocumentReaderPort, embedder: EmbeddingPort, store: VectorStorePort, chunk_size: int = 900, overlap: int = 140):
        self._reader, self._embedder, self._store = reader, embedder, store
        self._chunk_size, self._overlap = chunk_size, overlap

    def execute(self, paths: list[str]) -> int:
        raw: list[tuple[str, str]] = []
        for path in paths:
            text = self._reader.read(path)
            for n, part in enumerate(self._split(text)):
                raw.append((f"{Path(path).name}, фрагмент {n + 1}", part))
        if not raw:
            return self._store.replace([])
        vectors = self._embedder.embed([text for _, text in raw])
        chunks = [DocumentChunk(str(i), source, text, tuple(vector)) for i, ((source, text), vector) in enumerate(zip(raw, vectors))]
        return self._store.replace(chunks)

    def _split(self, text: str) -> list[str]:
        text = " ".join(text.split())
        parts, start = [], 0
        while start < len(text):
            end = min(start + self._chunk_size, len(text))
            if end < len(text):
                boundary = text.rfind(". ", start + self._chunk_size // 2, end)
                if boundary > start:
                    end = boundary + 1
            parts.append(text[start:end].strip())
            start = max(end - self._overlap, start + 1)
        return [p for p in parts if len(p) > 80]
