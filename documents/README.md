# Документы для RAG

Положите сюда актуальный Трудовой кодекс РФ в формате PDF, TXT или MD, затем вызовите защищённый endpoint:

```bash
curl -X POST https://YOUR-APP.up.railway.app/api/index -H "X-Access-Key: YOUR_KEY"
```

Для Railway документ лучше добавить в репозиторий до сборки. Индекс и модели сохраняются в `/data`; подключите Railway Volume к этому пути.

Если не хотите хранить исходный текст кодекса в деплое, соберите индекс заранее командой `python -m scripts.index_documents`, закоммитьте `artifacts/index/chunks.json`, а сам документ удалите из `documents/`. Контейнер автоматически подхватит готовый индекс.
