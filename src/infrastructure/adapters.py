import json
import math
from pathlib import Path
from threading import Lock
from collections.abc import Iterable, Sequence
from src.application.ports import DocumentReaderPort, EmbeddingPort, LanguageModelPort, VectorStorePort
from src.domain.entities import DocumentChunk, SearchResult


class FastEmbedAdapter(EmbeddingPort):
    def __init__(self, model_name: str):
        self._model_name, self._model = model_name, None
        self._lock = Lock()

    def _get(self):
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding
                self._model = TextEmbedding(model_name=self._model_name)
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [v.tolist() for v in self._get().embed(list(texts), batch_size=16)]


class LlamaCppAdapter(LanguageModelPort):
    def __init__(self, model_path: str, repo: str, filename: str, threads: int, context: int, max_tokens: int):
        self._path, self._repo, self._filename = Path(model_path), repo, filename
        self._threads, self._context, self._max_tokens = threads, context, max_tokens
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
        prompt = ("Ты юридический помощник по трудовому праву РФ. Отвечай только по приведённым фрагментам. "
                  "Если данных недостаточно, честно скажи об этом. Пиши кратко, по-русски, указывай номера статей, если они есть. "
                  "Не выдумывай нормы. Добавь предупреждение, что ответ не заменяет консультацию юриста.\n\n"
                  f"ФРАГМЕНТЫ:\n{context}\n\nВОПРОС: {question}\nОТВЕТ:")
        result = self._get()(prompt, max_tokens=self._max_tokens, temperature=0.15, top_p=0.9, stop=["ВОПРОС:", "ФРАГМЕНТЫ:"])
        return result["choices"][0]["text"].strip()


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

    def search(self, vector: Sequence[float], limit: int) -> list[SearchResult]:
        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            norm = math.sqrt(sum(x*x for x in a) * sum(y*y for y in b))
            return dot / norm if norm else 0.0
        ranked = sorted((SearchResult(c, cosine(vector, c.embedding)) for c in self._chunks), key=lambda r: r.score, reverse=True)
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

