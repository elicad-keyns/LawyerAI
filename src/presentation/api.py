import asyncio
import json
import logging
import queue
import shutil
import threading
import time
import uuid
from pathlib import Path
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from src.application.use_cases import AskLegalQuestion, IndexDocuments
from src.config import Settings
from src.infrastructure.adapters import FastEmbedAdapter, JsonVectorStore, LlamaCppAdapter, LocalDocumentReader

settings = Settings()
embedder = FastEmbedAdapter(settings.embedding_model, settings.embedding_cache_dir)
runtime_index = Path(settings.data_dir) / "index" / "chunks.json"
bundled_index = Path("artifacts/index/chunks.json")
if bundled_index.exists():
    runtime_index.parent.mkdir(parents=True, exist_ok=True)
    # Репозиторий содержит канонический предварительно собранный индекс.
    # При каждом деплое обновляем Volume, чтобы он не продолжал использовать
    # старые embedding-векторы после изменения модели или индексации.
    if not runtime_index.exists() or runtime_index.read_bytes() != bundled_index.read_bytes():
        shutil.copyfile(bundled_index, runtime_index)
store = JsonVectorStore(str(runtime_index))
llm = LlamaCppAdapter(
    settings.model_path,
    settings.model_repo,
    settings.model_file,
    settings.threads,
    settings.context,
    settings.max_tokens,
    settings.temperature,
    settings.batch,
)
ask = AskLegalQuestion(embedder, store, llm, settings.top_k, store, llm, llm)
indexer = IndexDocuments(LocalDocumentReader(), embedder, store)
app = FastAPI(title="ПравоТруд", docs_url=None, redoc_url=None)
logger = logging.getLogger("pravotrud")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


def authorize(x_access_key: str = Header(default="")):
    if not settings.access_key:
        raise HTTPException(503, "На сервере не задан ACCESS_KEY")
    if x_access_key != settings.access_key:
        raise HTTPException(401, "Неверный ключ доступа")


@app.get("/api/health")
def health():
    return {"status": "ok", "chunks": store.count()}


@app.post("/api/auth", dependencies=[Depends(authorize)])
def auth():
    return {"ok": True}


@app.post("/api/chat", dependencies=[Depends(authorize)])
async def chat(body: ChatRequest):
    request_id = uuid.uuid4().hex[:8]

    def events():
        output: queue.Queue = queue.Queue()
        request_started = time.perf_counter()

        def trace(name, details=None):
            output.put(("log", {"event": name, "details": details or {}, "elapsed_ms": round((time.perf_counter() - request_started) * 1000), "request_id": request_id}))

        def produce():
            try:
                trace("request.accepted", {"characters": len(body.message)})
                token_stream, sources = ask.execute_stream(body.message, trace)
                for token in token_stream:
                    output.put(("token", token))
                output.put(("sources", sources))
                trace("request.completed", {"duration_ms": round((time.perf_counter() - request_started) * 1000), "sources": len(sources)})
            except Exception as error:
                logger.exception("Local LLM stream failed")
                trace("request.failed", {"error_type": type(error).__name__, "message": str(error) or "неизвестная ошибка"})
                output.put(("error", f"Сбой локальной модели ({type(error).__name__}): {str(error) or 'неизвестная ошибка'}"))
            finally:
                output.put(("done", None))

        threading.Thread(target=produce, daemon=True).start()
        while True:
            try:
                kind, payload = output.get(timeout=10)
            except queue.Empty:
                yield json.dumps({"type": "heartbeat", "data": {"elapsed_ms": round((time.perf_counter() - request_started) * 1000), "request_id": request_id}}) + "\n"
                continue
            if kind == "done":
                break
            yield json.dumps({"type": kind, "data": payload}, ensure_ascii=False) + "\n"

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/index", dependencies=[Depends(authorize)])
async def rebuild_index():
    paths = [str(p) for p in Path("documents").glob("**/*") if p.suffix.lower() in {".pdf", ".txt", ".md"}]
    if not paths:
        raise HTTPException(400, "Папка documents не содержит PDF, TXT или MD")
    count = await asyncio.to_thread(indexer.execute, paths)
    return {"chunks": count, "files": len(paths)}


web = Path(__file__).parent / "web"
app.mount("/assets", StaticFiles(directory=web), name="assets")


@app.get("/")
def home():
    return FileResponse(web / "index.html")
