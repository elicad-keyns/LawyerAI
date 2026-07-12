from src.application.ports import DocumentReaderPort, EmbeddingPort, LanguageModelPort, VectorStorePort
from src.application.use_cases import AskLegalQuestion, IndexDocuments
from src.domain.entities import DocumentChunk, SearchResult


class Embed(EmbeddingPort):
    inputs=[]
    def embed(self, texts): self.inputs.extend(texts); return [[1.0, 0.0] for _ in texts]
class Llm(LanguageModelPort):
    def answer(self, question, context): return f"Ответ: {question} | {context}"
class Store(VectorStorePort):
    chunks=[]
    def replace(self, chunks): self.chunks=list(chunks); return len(self.chunks)
    def search(self, vector, limit, query="", exact_article=""): return [SearchResult(self.chunks[0], 1.0)] if self.chunks else []
    def count(self): return len(self.chunks)
class Reader(DocumentReaderPort):
    def read(self, path): return "Статья 1. " + "Трудовое право регулирует отношения. " * 50


def test_rag_flow():
    store=Store()
    assert IndexDocuments(Reader(), Embed(), store).execute(["tk.txt"]) > 0
    answer=AskLegalQuestion(Embed(), store, Llm()).execute("Что регулирует кодекс?")
    assert "Что регулирует" in answer.text
    assert answer.sources


def test_empty_index_message():
    answer=AskLegalQuestion(Embed(), Store(), Llm()).execute("Вопрос")
    assert "не проиндексирована" in answer.text


def test_embedding_text_has_no_e5_prefix_for_minilm():
    embedder=Embed()
    AskLegalQuestion(embedder, Store(), Llm()).execute("Как уйти с работы?")
    assert embedder.inputs == ["Как уйти с работы?"]
