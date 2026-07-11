from src.domain.entities import DocumentChunk
from src.infrastructure.adapters import JsonVectorStore


def test_exact_article_number_beats_semantic_similarity(tmp_path):
    store = JsonVectorStore(str(tmp_path / "index.json"))
    store.replace([
        DocumentChunk("wrong", "tk.pdf", "Статья 87. Хранение персональных данных", (1.0, 0.0)),
        DocumentChunk("right", "tk.pdf", "Статья 77. Общие основания прекращения трудового договора", (0.0, 1.0)),
    ])
    results = store.search((1.0, 0.0), 2, "Статья 77 ТК РФ")
    assert results[0].chunk.id == "right"
