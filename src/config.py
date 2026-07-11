import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    access_key: str = os.getenv("ACCESS_KEY", "")
    data_dir: str = os.getenv("DATA_DIR", "data")
    model_repo: str = os.getenv("MODEL_REPO", "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
    model_file: str = os.getenv("MODEL_FILE", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    model_path: str = os.getenv("MODEL_PATH", "data/models/tinyllama.gguf")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    threads: int = int(os.getenv("LLM_THREADS", "4"))
    context: int = int(os.getenv("LLM_CONTEXT", "2048"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "400"))
    top_k: int = int(os.getenv("TOP_K", "5"))

