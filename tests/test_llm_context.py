class FakeModel:
    def __init__(self): self.prompt = ""
    def tokenize(self, value, add_bos=False): return list(value)
    def detokenize(self, tokens): return bytes(tokens)
    def __call__(self, prompt, **kwargs): self.prompt = prompt; return {"choices": [{"text": "ответ"}]}


def test_llm_adapter_limits_rag_context():
    from src.infrastructure.adapters import LlamaCppAdapter
    adapter = LlamaCppAdapter("x", "x", "x", 1, 1200, 100, 0.15)
    fake = FakeModel()
    adapter._model = fake
    assert adapter.answer("вопрос", "контекст" * 500) == "ответ"
    assert len(fake.prompt.encode()) <= 1100


def test_llm_adapter_removes_repeated_disclaimer():
    from src.infrastructure.adapters import LlamaCppAdapter
    text = "Применяется статья 80.\n\nПрименяется статья 80.\n\nПредупреждение: Ответ не заменяет консультацию."
    assert LlamaCppAdapter._remove_repetitions(text) == "Применяется статья 80."


def test_qwen_chatml_prompt_is_used():
    from src.infrastructure.adapters import LlamaCppAdapter
    adapter = LlamaCppAdapter("x", "x", "x", 1, 1600, 100, 0.1)
    adapter._model = FakeModel()
    prompt = adapter._prepare_prompt("Что в статье 77?", "Статья 77. Основания прекращения договора")
    assert prompt.startswith("<|im_start|>system")
    assert prompt.endswith("<|im_start|>assistant\n")
