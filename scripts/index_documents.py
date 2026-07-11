"""Собирает переносимый RAG-индекс до Docker/Railway-деплоя."""
import argparse
from pathlib import Path

from src.application.use_cases import IndexDocuments
from src.config import Settings
from src.infrastructure.adapters import FastEmbedAdapter, JsonVectorStore, LocalDocumentReader


def main() -> None:
    parser = argparse.ArgumentParser(description="Индексация документов ТК РФ")
    parser.add_argument("--documents", default="documents", help="Папка с PDF, TXT или MD")
    parser.add_argument("--output", default="artifacts/index/chunks.json", help="Файл готового индекса")
    args = parser.parse_args()

    paths = [
        str(path)
        for path in Path(args.documents).glob("**/*")
        if path.suffix.lower() in {".pdf", ".txt", ".md"} and path.name.lower() != "readme.md"
    ]
    if not paths:
        raise SystemExit(f"В {args.documents} не найдено документов PDF, TXT или MD")

    settings = Settings()
    use_case = IndexDocuments(
        LocalDocumentReader(),
        FastEmbedAdapter(settings.embedding_model),
        JsonVectorStore(args.output),
    )
    count = use_case.execute(paths)
    size_mb = Path(args.output).stat().st_size / 1024 / 1024
    print(f"Готово: {len(paths)} файлов, {count} фрагментов, {size_mb:.1f} МБ")


if __name__ == "__main__":
    main()

