FROM python:3.11-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake && rm -rf /var/lib/apt/lists/*
ENV CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_BLAS=OFF" CMAKE_BUILD_PARALLEL_LEVEL=4
WORKDIR /app
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels
COPY . .
RUN mkdir -p /data/models /data/index /app/documents
ENV PORT=8000 PYTHONUNBUFFERED=1 DATA_DIR=/data
EXPOSE 8000
CMD ["sh", "-c", "uvicorn src.presentation.api:app --host 0.0.0.0 --port ${PORT:-8000}"]

