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


def test_article_heading_beats_cross_reference(tmp_path):
    store = JsonVectorStore(str(tmp_path / "index.json"))
    store.replace([
        DocumentChunk("reference", "tk.pdf", "Увольнение осуществляется по основаниям статьи 77 настоящего Кодекса.", (1.0, 0.0)),
        DocumentChunk("heading", "tk.pdf", "Статья 77. Общие основания прекращения трудового договора", (0.0, 1.0)),
    ])
    assert store.search((1.0, 0.0), 2, "Что означает статья 77?")[0].chunk.id == "heading"


def test_instrumental_case_and_article_continuation(tmp_path):
    store = JsonVectorStore(str(tmp_path / "index.json"))
    store.replace([
        DocumentChunk("wrong", "tk.pdf", "Порядок освидетельствования согласно статье 1 закона.", (1.0, 0.0)),
        DocumentChunk("start", "tk.pdf", "Статья 1. Цели и задачи трудового законодательства", (0.0, 1.0)),
        DocumentChunk("continuation", "tk.pdf", "Целями трудового законодательства являются установление государственных гарантий.", (0.0, 1.0)),
    ])
    results = store.search((1.0, 0.0), 2, "Что установлено статьёй 1 ТК РФ?")
    assert [result.chunk.id for result in results] == ["start", "continuation"]


def test_bm25_is_generic_and_not_topic_specific(tmp_path):
    store = JsonVectorStore(str(tmp_path / "index.json"))
    store.replace([
        DocumentChunk("semantic", "tk.pdf", "Общие положения трудового законодательства", (1.0, 0.0)),
        DocumentChunk("lexical", "tk.pdf", "Ежегодный оплачиваемый отпуск предоставляется работникам", (0.7, 0.3)),
    ])
    results = store.search((1.0, 0.0), 2, "ежегодный оплачиваемый отпуск")
    assert results[0].chunk.id == "lexical"
