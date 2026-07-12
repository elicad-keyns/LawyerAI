from src.application.use_cases import AskLegalQuestion
from tests.test_use_cases import Embed, Llm, Store


class Rewriter:
    def rewrite(self, question):
        return "расторжение трудового договора по инициативе работника"


def test_rewritten_query_is_embedded_without_fixed_dictionary():
    embedder = Embed()
    use_case = AskLegalQuestion(embedder, Store(), Llm(), rewriter=Rewriter())
    use_case.execute("Как уйти с работы?")
    assert embedder.inputs == ["расторжение трудового договора по инициативе работника"]
