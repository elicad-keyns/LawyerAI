import asyncio
import logging
import shutil
from pathlib import Path
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from src.application.use_cases import AskLegalQuestion, IndexDocuments
from src.config import Settings
from src.infrastructure.adapters import FastEmbedAdapter, JsonVectorStore, LlamaCppAdapter, LocalDocumentReader

settings = Settings()
embedder = FastEmbedAdapter(settings.embedding_model)
runtime_index = Path(settings.data_dir) / "index" / "chunks.json"
bundled_index = Path("artifacts/index/chunks.json")
if not runtime_index.exists() and bundled_index.exists():
    runtime_index.parent.mkdir(parents=True, exist_ok=True)
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
)
ask = AskLegalQuestion(embedder, store, llm, settings.top_k)
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
    try:
        answer = await asyncio.to_thread(ask.execute, body.message)
        return {"answer": answer.text, "sources": answer.sources}
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    except Exception as error:
        logger.exception("Local LLM request failed")
        message = str(error).strip() or "неизвестная ошибка"
        raise HTTPException(500, f"Сбой локальной модели ({type(error).__name__}): {message}") from error


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
