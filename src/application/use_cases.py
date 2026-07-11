from pathlib import Path
from src.application.ports import DocumentReaderPort, EmbeddingPort, LanguageModelPort, VectorStorePort
from src.domain.entities import ChatAnswer, DocumentChunk


class AskLegalQuestion:
    def __init__(self, embedder: EmbeddingPort, store: VectorStorePort, llm: LanguageModelPort, top_k: int = 5):
        self._embedder, self._store, self._llm, self._top_k = embedder, store, llm, top_k

    def execute(self, question: str) -> ChatAnswer:
        stream, sources = self.execute_stream(question)
        return ChatAnswer("".join(stream), sources)

    def execute_stream(self, question: str):
        question = question.strip()
        if not question:
            raise ValueError("Вопрос не может быть пустым")
        search_query = question
        normalized = question.lower()
        if "уволиться" in normalized or "собственн" in normalized:
            search_query += " Расторжение трудового договора по инициативе работника статья 80 увольнение по собственному желанию"
        vector = self._embedder.embed([f"query: {search_query}"])[0]
        results = self._store.search(vector, self._top_k, question)
        if not results:
            return iter(["База ТК РФ пока не проиндексирована. Добавьте документы в папку documents."]), ()
        context = "\n\n".join(f"[{i + 1}] {r.chunk.source}\n{r.chunk.text}" for i, r in enumerate(results))
        sources = tuple(dict.fromkeys(r.chunk.source for r in results))
        return self._llm.stream_answer(question, context), sources


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
        vectors = self._embedder.embed([f"passage: {text}" for _, text in raw])
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
