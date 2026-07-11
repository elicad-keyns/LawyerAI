# ПравоТруд

Минималистичный защищённый чат с полностью локальным RAG по ТК РФ. Генерация работает через TinyLlama 1.1B Q4 (`llama.cpp`, CPU), эмбеддинги — через мультиязычный MiniLM в ONNX. После первой загрузки запросы не уходят во внешние AI API.

## Архитектура

Зависимости направлены внутрь: `presentation → application ← infrastructure`, а `application → domain`. Use cases зависят только от абстрактных портов, конкретные LLM, embedder, хранилище и reader подключаются в composition root `src/presentation/api.py`.

## Локальный запуск

```bash
cp .env.example .env
docker build -t pravotrud .
docker run --env-file .env -p 8000:8000 -v pravotrud-data:/data pravotrud
```

Откройте `http://localhost:8000`. При первом индексировании скачивается embedder, при первом вопросе — GGUF-модель, поэтому первый запрос заметно дольше.

## Загрузка ТК РФ

1. Добавьте актуальный PDF/TXT/MD в `documents/`.
2. Запустите сервис.
3. Перестройте индекс:

```bash
curl -X POST http://localhost:8000/api/index -H "X-Access-Key: change-me"
```

## Предварительная индексация перед деплоем

Чтобы Railway получил уже готовый индекс, положите ТК РФ в `documents/` и выполните локально:

```bash
python -m pip install fastembed pypdf
python -m scripts.index_documents
```

Команда создаст `artifacts/index/chunks.json`. Добавьте этот файл в Git вместе с проектом. При первом запуске контейнер автоматически скопирует готовый индекс в Railway Volume `/data`. Эмбеддинг-модель при обработке вопросов должна совпадать с моделью, использованной при индексации; поэтому не меняйте `EMBEDDING_MODEL` после сборки индекса.

Проверить готовый артефакт можно так:

```bash
ls -lh artifacts/index/chunks.json
```

## Railway

Создайте сервис из репозитория, добавьте Volume с mount path `/data` и переменную `ACCESS_KEY` со случайным длинным значением. Остальные переменные описаны в `.env.example`. Для 4 CPU / 4 GB оставьте `LLM_THREADS=4`, `LLM_CONTEXT=2048`; ожидаемая RAM после прогрева около 2–3 GB. Модель очень компактна, поэтому ответы нужно воспринимать только как навигацию по найденным фрагментам.

Важно: статический ключ защищает от случайного публичного использования, но не является полноценной системой пользователей. Сайт работает только по HTTPS Railway; ключ хранится в localStorage браузера.
