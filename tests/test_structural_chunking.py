from src.application.use_cases import IndexDocuments
from tests.test_use_cases import Embed, Reader, Store


def test_long_article_repeats_heading_in_continuation():
    indexer = IndexDocuments(Reader(), Embed(), Store(), chunk_size=180, overlap=40)
    text = "Преамбула. Статья 80. Расторжение договора. " + "Работник предупреждает работодателя. " * 20 + " Статья 81. Инициатива работодателя. Основания расторжения договора перечисляются в настоящей статье."
    chunks = indexer._split(text)
    article_80 = [chunk for chunk in chunks if chunk.startswith("Статья 80.")]
    assert len(article_80) > 1
    assert all("Расторжение договора" in chunk[:100] for chunk in article_80)
    assert any(chunk.startswith("Статья 81.") for chunk in chunks)
    assert len(chunks) < 20
