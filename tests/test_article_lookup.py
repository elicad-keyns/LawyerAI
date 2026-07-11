from src.domain.entities import DocumentChunk
from src.infrastructure.adapters import JsonVectorStore


def test_full_article_is_reconstructed_from_overlapping_chunks(tmp_path):
    store = JsonVectorStore(str(tmp_path / "index.json"))
    store.replace([
        DocumentChunk("1", "tk.pdf, фрагмент 1", "Глава 1. Статья 1. Цели и задачи. Целями являются гарантии и", ()),
        DocumentChunk("2", "tk.pdf, фрагмент 2", "гарантии и защита прав работников. Статья 2. Основные принципы.", ()),
    ])
    source, text = store.find_article("1")
    assert source == "tk.pdf"
    assert text == "Статья 1. Цели и задачи. Целями являются гарантии и защита прав работников."
    assert "Статья 2" not in text
