import json
import math
import re
from time import perf_counter
from pathlib import Path
from threading import Lock
from collections.abc import Iterable, Sequence
from typing import Optional
from src.application.ports import DocumentReaderPort, EmbeddingPort, LanguageModelPort, VectorStorePort
from src.domain.entities import DocumentChunk, SearchResult


class FastEmbedAdapter(EmbeddingPort):
    def __init__(self, model_name: str, cache_dir: Optional[str] = None):
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
    def __init__(self, model_path: str, repo: str, filename: str, threads: int, context: int, max_tokens: int, temperature: float, batch: int = 512):
        self._path, self._repo, self._filename = Path(model_path), repo, filename
        self._threads, self._context, self._max_tokens, self._temperature, self._batch = threads, context, max_tokens, temperature, batch
        self._model, self._lock = None, Lock()

    def _get(self):
        with self._lock:
            if self._model is None:
                if not self._path.exists():
                    from huggingface_hub import hf_hub_download
                    downloaded = hf_hub_download(self._repo, self._filename, local_dir=self._path.parent)
                    Path(downloaded).replace(self._path)
                from llama_cpp import Llama
                self._model = Llama(
                    model_path=str(self._path),
                    n_ctx=self._context,
                    n_threads=self._threads,
                    n_threads_batch=self._threads,
                    n_batch=self._batch,
                    use_mmap=True,
                    use_mlock=False,
                    verbose=False,
                )
        return self._model

    def answer(self, question: str, context: str) -> str:
        result = self._get()(self._prepare_prompt(question, context), **self._generation_options())
        return self._remove_repetitions(result["choices"][0]["text"])

    def stream_answer(self, question: str, context: str, on_event=None):
        emit = on_event or (lambda *_: None)
        started = perf_counter()
        emit("llm.loading.started", {"model": self._path.name, "threads": self._threads, "batch": self._batch, "context_window": self._context})
        model = self._get()
        emit("llm.loading.completed", {"duration_ms": round((perf_counter() - started) * 1000)})
        prompt = self._prepare_prompt(question, context, emit)
        options = self._generation_options()
        emit("llm.generation.started", {"max_tokens": self._max_tokens, "temperature": self._temperature, "repeat_penalty": options["repeat_penalty"]})
        result = model(prompt, stream=True, **options)
        tokens = 0
        for part in result:
            token = part["choices"][0].get("text", "")
            if token:
                # Маленькие Qwen иногда переключаются на китайский в середине
                # русского ответа. Завершаем поток до появления мусорного текста.
                if re.search(r"[\u3400-\u9fff]", token):
                    emit("llm.generation.stopped", {"reason": "unexpected_cjk_token", "tokens": tokens})
                    break
                tokens += 1
                if tokens == 1 or tokens % 10 == 0:
                    emit("llm.generation.progress", {"tokens": tokens})
                yield token
        emit("llm.generation.completed", {"tokens": tokens})

    def _prepare_prompt(self, question: str, context: str, on_event=None) -> str:
        system = ("Ты помощник по Трудовому кодексу РФ. Отвечай строго по предоставленному контексту на русском языке. "
                  "Не придумывай факты. Не пиши слова «ОТВЕТ», «ПРОБЛЕМА», предупреждения или дисклеймеры. "
                  "Не повторяй текст и не используй китайские иероглифы. Если спрашивают конкретную статью, используй прежде всего фрагмент, начинающийся с её заголовка, "
                  "и не подменяй содержание случайными ссылками на эту статью. Дай прямой краткий ответ простым языком.")
        prefix = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\nКонтекст из ТК РФ:\n"
        suffix = f"\n\nВопрос пользователя: {question}<|im_end|>\n<|im_start|>assistant\n"
        model = self._get()

        # RAG-контекст ограничивается реальными токенами конкретной модели,
        # чтобы prompt + ответ никогда не превышали n_ctx.
        fixed_tokens = len(model.tokenize((prefix + suffix).encode("utf-8"), add_bos=True))
        context_budget = self._context - self._max_tokens - fixed_tokens - 16
        if context_budget < 64:
            raise ValueError("LLM_CONTEXT слишком мал для вопроса и MAX_TOKENS")
        context_tokens = model.tokenize(context.encode("utf-8"), add_bos=False)[:context_budget]
        safe_context = model.detokenize(context_tokens).decode("utf-8", errors="ignore")
        if on_event:
            on_event("llm.prompt.prepared", {"fixed_tokens": fixed_tokens, "context_tokens": len(context_tokens), "context_budget": context_budget, "context_truncated": len(context_tokens) >= context_budget})
        return prefix + safe_context + suffix

    def _generation_options(self) -> dict:
        return dict(
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            top_p=0.9,
            repeat_penalty=1.18,
            stop=["<|im_end|>", "<|im_start|>", "ПРОБЛЕМА:", "ОТВЕТ:", "Предупреждение:", "Disclaimer:"],
        )

    @staticmethod
    def _remove_repetitions(text: str) -> str:
        """Удаляет повторные абзацы и типовые дисклеймеры слабых моделей."""
        text = re.split(r"(?i)предупреждение\s*:|ответ не заменяет консультацию|disclaimer\s*:|проблема\s*:|ответ\s*:", text, maxsplit=1)[0]
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
        article_match = re.search(r"(?i)стат(?:ья|ьи|ью|ье|ьей|ьёй|ей)\s*[№N]?\s*(\d+(?:\.\d+)?)", query)
        article = article_match.group(1) if article_match else ""
        article_pattern = re.compile(rf"(?i)стат(?:ья|ьи|ью|ье|ей)\s*[№N]?\s*{re.escape(article)}(?!\d|\.\d)") if article else None
        # В PDF заголовок часто идёт сразу после названия главы без точки:
        # «Глава 13... Статья 77». Заглавная «Статья» отличает заголовок от
        # обычных ссылок вида «согласно статье 77».
        article_heading_pattern = re.compile(rf"(?:^|\s)Статья\s+{re.escape(article)}\s*[.\s]", re.MULTILINE) if article else None
        query_words = {w for w in re.findall(r"[а-яёa-z]{4,}", query.lower()) if w not in {"какой", "какая", "какие", "статья"}}

        def hybrid_score(chunk: DocumentChunk) -> float:
            score = cosine(vector, chunk.embedding)
            normalized_text = chunk.text.lower().replace("ё", "е")
            if article_heading_pattern and article_heading_pattern.search(chunk.text):
                score += 10.0
            elif article_pattern and article_pattern.search(chunk.text):
                score += 1.5
            if query_words:
                matches = sum(word.replace("ё", "е") in normalized_text for word in query_words)
                score += 0.25 * matches / len(query_words)
            return score

        ranked = sorted((SearchResult(c, hybrid_score(c)) for c in self._chunks), key=lambda r: r.score, reverse=True)

        # Для запроса конкретной статьи возвращаем её начало и следующие
        # последовательные фрагменты. Так длинные статьи не обрываются после
        # первого чанка и не заменяются случайными перекрёстными ссылками.
        if article_heading_pattern:
            heading_index = next((i for i, chunk in enumerate(self._chunks) if article_heading_pattern.search(chunk.text)), None)
            if heading_index is not None:
                section = self._chunks[heading_index:heading_index + limit]
                return [SearchResult(chunk, 20.0 - offset) for offset, chunk in enumerate(section)]
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
