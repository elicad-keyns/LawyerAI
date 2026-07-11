import json
import math
import re
from pathlib import Path
from threading import Lock
from collections.abc import Iterable, Sequence
from src.application.ports import DocumentReaderPort, EmbeddingPort, LanguageModelPort, VectorStorePort
from src.domain.entities import DocumentChunk, SearchResult


class FastEmbedAdapter(EmbeddingPort):
    def __init__(self, model_name: str, cache_dir: str | None = None):
        self._model_name, self._cache_dir, self._model = model_name, cache_dir, None
        self._lock = Lock()

    def _get(self):
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding
                self._model = TextEmbedding(model_name=self._model_name, cache_dir=self._cache_dir)
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [v.tolist() for v in self._get().embed(list(texts), batch_size=16)]


class LlamaCppAdapter(LanguageModelPort):
    def __init__(self, model_path: str, repo: str, filename: str, threads: int, context: int, max_tokens: int, temperature: float):
        self._path, self._repo, self._filename = Path(model_path), repo, filename
        self._threads, self._context, self._max_tokens, self._temperature = threads, context, max_tokens, temperature
        self._model, self._lock = None, Lock()

    def _get(self):
        with self._lock:
            if self._model is None:
                if not self._path.exists():
                    from huggingface_hub import hf_hub_download
                    downloaded = hf_hub_download(self._repo, self._filename, local_dir=self._path.parent)
                    Path(downloaded).replace(self._path)
                from llama_cpp import Llama
                self._model = Llama(model_path=str(self._path), n_ctx=self._context, n_threads=self._threads, n_batch=128, verbose=False)
        return self._model

    def answer(self, question: str, context: str) -> str:
        result = self._get()(self._prepare_prompt(question, context), **self._generation_options())
        return self._remove_repetitions(result["choices"][0]["text"])

    def stream_answer(self, question: str, context: str):
        model = self._get()
        result = model(self._prepare_prompt(question, context), stream=True, **self._generation_options())
        for part in result:
            token = part["choices"][0].get("text", "")
            if token:
                yield token

    def _prepare_prompt(self, question: str, context: str) -> str:
        prefix = ("Ты юридический помощник по трудовому праву РФ. Отвечай только по приведённым фрагментам. "
                  "Если данных недостаточно, честно скажи об этом. Пиши кратко, по-русски, указывай номера статей, если они есть. "
                  "Не выдумывай нормы. Не добавляй предупреждений, оговорок и дисклеймеров. Не повторяй предложения и абзацы. "
                  "Сначала дай прямой ответ, затем при необходимости перечисли основания кратким списком.\n\nФРАГМЕНТЫ:\n")
        suffix = f"\n\nВОПРОС: {question}\nОТВЕТ:"
        model = self._get()

        # RAG-контекст ограничивается реальными токенами конкретной модели,
        # чтобы prompt + ответ никогда не превышали n_ctx.
        fixed_tokens = len(model.tokenize((prefix + suffix).encode("utf-8"), add_bos=True))
        context_budget = self._context - self._max_tokens - fixed_tokens - 16
        if context_budget < 64:
            raise ValueError("LLM_CONTEXT слишком мал для вопроса и MAX_TOKENS")
        context_tokens = model.tokenize(context.encode("utf-8"), add_bos=False)[:context_budget]
        safe_context = model.detokenize(context_tokens).decode("utf-8", errors="ignore")
        return prefix + safe_context + suffix

    def _generation_options(self) -> dict:
        return dict(
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            top_p=0.9,
            repeat_penalty=1.18,
            stop=["ВОПРОС:", "ФРАГМЕНТЫ:", "Предупреждение:", "Disclaimer:"],
        )

    @staticmethod
    def _remove_repetitions(text: str) -> str:
        """Удаляет повторные абзацы и типовые дисклеймеры слабых моделей."""
        text = re.split(r"(?i)предупреждение\s*:|ответ не заменяет консультацию|disclaimer\s*:", text, maxsplit=1)[0]
        paragraphs = re.split(r"\n\s*\n", text.strip())
        unique: list[str] = []
        seen: set[str] = set()
        for paragraph in paragraphs:
            normalized = re.sub(r"\W+", " ", paragraph.lower()).strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique.append(paragraph.strip())
        return "\n\n".join(unique).strip()


class JsonVectorStore(VectorStorePort):
    def __init__(self, path: str):
        self._path, self._chunks, self._lock = Path(path), [], Lock()
        self._load()

    def _load(self):
        if self._path.exists():
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._chunks = [DocumentChunk(x["id"], x["source"], x["text"], tuple(x["embedding"])) for x in data]

    def replace(self, chunks: Iterable[DocumentChunk]) -> int:
        with self._lock:
            self._chunks = list(chunks)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = [{"id": c.id, "source": c.source, "text": c.text, "embedding": c.embedding} for c in self._chunks]
            self._path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            return len(self._chunks)

    def search(self, vector: Sequence[float], limit: int, query: str = "") -> list[SearchResult]:
        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            norm = math.sqrt(sum(x*x for x in a) * sum(y*y for y in b))
            return dot / norm if norm else 0.0

        # Номера статей — идентификаторы, а не семантика. Эмбеддинги могут
        # считать статьи 77 и 87 похожими, поэтому точный номер получает
        # приоритет над векторной близостью.
        article_match = re.search(r"(?i)стат(?:ья|ьи|ью|ье|ей)\s*[№N]?\s*(\d+(?:\.\d+)?)", query)
        article = article_match.group(1) if article_match else ""
        article_pattern = re.compile(rf"(?i)стат(?:ья|ьи|ью|ье|ей)\s*[№N]?\s*{re.escape(article)}(?!\d|\.\d)") if article else None
        query_words = {w for w in re.findall(r"[а-яёa-z]{4,}", query.lower()) if w not in {"какой", "какая", "какие", "статья"}}

        def hybrid_score(chunk: DocumentChunk) -> float:
            score = cosine(vector, chunk.embedding)
            normalized_text = chunk.text.lower().replace("ё", "е")
            if article_pattern and article_pattern.search(chunk.text):
                score += 3.0
            if query_words:
                matches = sum(word.replace("ё", "е") in normalized_text for word in query_words)
                score += 0.25 * matches / len(query_words)
            return score

        ranked = sorted((SearchResult(c, hybrid_score(c)) for c in self._chunks), key=lambda r: r.score, reverse=True)
        return ranked[:limit]

    def count(self) -> int:
        return len(self._chunks)


class LocalDocumentReader(DocumentReaderPort):
    def read(self, path: str) -> str:
        file = Path(path)
        if file.suffix.lower() == ".pdf":
            from pypdf import PdfReader
            return "\n".join(page.extract_text() or "" for page in PdfReader(file).pages)
        if file.suffix.lower() in {".txt", ".md"}:
            return file.read_text(encoding="utf-8")
        raise ValueError(f"Неподдерживаемый формат: {file.suffix}. Используйте PDF, TXT или MD")
