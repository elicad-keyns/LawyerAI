import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    access_key: str = os.getenv("ACCESS_KEY", "")
    data_dir: str = os.getenv("DATA_DIR", "data")
    model_repo: str = os.getenv("MODEL_REPO", "Qwen/Qwen2.5-1.5B-Instruct-GGUF")
    model_file: str = os.getenv("MODEL_FILE", "qwen2.5-1.5b-instruct-q4_k_m.gguf")
    model_path: str = os.getenv("MODEL_PATH", "data/models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embedding_cache_dir: str = os.getenv("EMBEDDING_CACHE_DIR", "data/models/fastembed")
    threads: int = int(os.getenv("LLM_THREADS", "8"))
    context: int = int(os.getenv("LLM_CONTEXT", "1536"))
    batch: int = int(os.getenv("LLM_BATCH", "512"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "220"))
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.15"))
    top_k: int = int(os.getenv("TOP_K", "5"))

    def __post_init__(self) -> None:
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("LLM_TEMPERATURE должна быть в диапазоне от 0.0 до 2.0")
