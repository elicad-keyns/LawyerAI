from src.application.use_cases import AskLegalQuestion
from tests.test_use_cases import Embed, Llm, Store
from src.domain.entities import DocumentChunk, SearchResult


class CandidateStore(Store):
    def __init__(self):
        self.chunks = [DocumentChunk(str(i), "tk.pdf", f"Статья {i}. Текст", (1.0, 0.0)) for i in range(12)]
    def search(self, vector, limit, query="", exact_article=""):
        return [SearchResult(chunk, 1.0 - i / 100) for i, chunk in enumerate(self.chunks[:limit])]


class Reranker:
    def rerank(self, question, candidates, limit):
        return ["7", "3", "1"]


def test_llm_reranker_selects_from_broad_candidates():
    answer = AskLegalQuestion(Embed(), CandidateStore(), Llm(), top_k=3, reranker=Reranker()).execute("Вопрос")
    assert answer.sources
