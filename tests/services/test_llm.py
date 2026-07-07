from app.services import llm


def test_llm_disabled_without_provider(monkeypatch):
    monkeypatch.delenv("RAG_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("RAG_LLM_MODEL", raising=False)
    monkeypatch.delenv("RAG_LLM_API_KEY", raising=False)
    monkeypatch.delenv("RAG_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert llm.is_llm_enabled() is False


def test_openai_compatible_llm_generates_answer(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)

            class Message:
                content = "这是基于资料的回答。[source 1]"

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.chat = FakeChat()

    monkeypatch.setattr(llm, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("RAG_LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("RAG_LLM_MODEL", "test-chat-model")
    monkeypatch.setenv("RAG_LLM_API_KEY", "test-key")
    monkeypatch.setenv("RAG_LLM_BASE_URL", "https://example.test/v1")

    answer = llm.generate_grounded_answer("prompt text")

    assert answer == "这是基于资料的回答。[source 1]"
    assert captured["client_kwargs"]["api_key"] == "test-key"
    assert captured["client_kwargs"]["base_url"] == "https://example.test/v1"
    assert captured["model"] == "test-chat-model"
    assert captured["messages"][1]["content"] == "prompt text"
