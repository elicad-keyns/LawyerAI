FROM python:3.11-slim-bookworm AS builder
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake git curl ca-certificates libopenblas-dev && rm -rf /var/lib/apt/lists/*
ENV CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" CMAKE_BUILD_PARALLEL_LEVEL=8
WORKDIR /app
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# Модель встраивается в образ на этапе сборки. Railway не нужно скачивать её
# при первом пользовательском запросе. URL можно заменить через Docker build arg.
ARG MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf?download=true"
RUN mkdir -p /bundled-model \
    && curl --fail --location --retry 5 --retry-delay 5 \
       --output /bundled-model/model.gguf "$MODEL_URL" \
    && test -s /bundled-model/model.gguf

# Эмбеддинг-модель также кэшируется внутри образа, иначе первый поисковый
# запрос всё равно потребовал бы доступ Railway к внешней сети.
RUN pip install --no-cache-dir /wheels/* \
    && python -c "from fastembed import TextEmbedding; list(TextEmbedding(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', cache_dir='/bundled-fastembed').embed(['проверка']))"

FROM python:3.11-slim-bookworm
WORKDIR /app
COPY --from=builder /wheels /wheels
COPY --from=builder /bundled-model/model.gguf /app/models/model.gguf
COPY --from=builder /bundled-fastembed /app/models/fastembed
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libopenblas0-pthread && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir /wheels/* && rm -rf /wheels
COPY . .
RUN mkdir -p /data/models /data/index /app/documents
ENV PORT=8000 PYTHONUNBUFFERED=1 DATA_DIR=/data MODEL_PATH=/app/models/model.gguf EMBEDDING_CACHE_DIR=/app/models/fastembed
EXPOSE 8000
CMD ["sh", "-c", "uvicorn src.presentation.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
