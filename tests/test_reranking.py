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


def test_candidate_diversity_limits_repeated_article_chunks():
    chunks = [DocumentChunk(str(i), "tk.pdf", "Статья 349.1. Продолжение", (1.0, 0.0)) for i in range(6)]
    chunks += [DocumentChunk("81", "tk.pdf", "Статья 81. Общая норма", (1.0, 0.0))]
    store = CandidateStore()
    store.chunks = chunks
    class CaptureReranker:
        seen = []
        def rerank(self, question, candidates, limit):
            self.seen = candidates
            return [chunk.id for chunk in candidates[:limit]]
    reranker = CaptureReranker()
    AskLegalQuestion(Embed(), store, Llm(), top_k=3, reranker=reranker).execute("Вопрос")
    assert sum(chunk.text.startswith("Статья 349.1") for chunk in reranker.seen) <= 2
    assert any(chunk.text.startswith("Статья 81") for chunk in reranker.seen)
