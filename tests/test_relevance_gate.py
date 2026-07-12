from src.application.use_cases import AskLegalQuestion
from tests.test_use_cases import Embed, Llm, Store
from src.domain.entities import DocumentChunk, SearchResult


class LowScoreStore(Store):
    chunks = [DocumentChunk("1", "tk.pdf", "Нерелевантный текст", (1.0, 0.0))]
    def search(self, vector, limit, query="", exact_article=""):
        return [SearchResult(self.chunks[0], 0.3)]


def test_out_of_domain_question_does_not_reach_llm():
    answer = AskLegalQuestion(Embed(), LowScoreStore(), Llm(), min_relevance_score=0.62).execute("Вопрос вне базы")
    assert "не найдено достаточно релевантных норм" in answer.text
