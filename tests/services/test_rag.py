from app.models import RetrievalSearchResult
from app.services.rag import build_rag_prompt, build_rag_response, build_rag_sources


def _retrieval_result(content: str = "abcdef ghijkl mnopqr") -> RetrievalSearchResult:
    return RetrievalSearchResult(
        rank=1,
        content=content,
        score=0.12,
        score_type="pgvector_distance_lower_is_better",
        file_id="file-123",
        source_file="guide.md",
        chunk_index=3,
        page=7,
        metadata={
            "file_id": "file-123",
            "source_file": "guide.md",
            "chunk_id": "chunk-123",
            "chunk_index": 3,
            "digest": "digest-123",
        },
    )


def test_build_rag_sources_truncates_context_and_keeps_traceability():
    sources = build_rag_sources([_retrieval_result()], max_context_chars=10)

    assert len(sources) == 1
    assert sources[0].content == "abcdef ghi"
    assert sources[0].chunk_id == "chunk-123"
    assert sources[0].source_file == "guide.md"
    assert sources[0].chunk_index == 3
    assert sources[0].score == 0.12
    assert sources[0].score_type == "pgvector_distance_lower_is_better"


def test_build_rag_prompt_contains_grounding_rules_and_source_labels():
    sources = build_rag_sources([_retrieval_result()], max_context_chars=100)
    prompt = build_rag_prompt("What is the guide about?", sources)

    assert "只能使用给定资料回答" in prompt
    assert "如果资料不足" in prompt
    assert "[source 1]" in prompt
    assert "file=guide.md" in prompt
    assert "chunk_index=3" in prompt
    assert "What is the guide about?" in prompt


def test_build_rag_response_refuses_when_retrieval_is_empty():
    response = build_rag_response(
        query="No answer question",
        file_id="file-123",
        retrieval_results=[],
        max_context_chars=100,
    )

    assert response["refusal"] is True
    assert response["answer_strategy"] == "no_context_refusal"
    assert response["used_context_count"] == 0
    assert response["sources"] == []
    assert "资料不足" in response["answer"]


def test_build_rag_response_uses_all_returned_sources():
    second = _retrieval_result(content="second source content")
    second.rank = 2
    second.metadata["chunk_id"] = "chunk-456"

    response = build_rag_response(
        query="What is the guide about?",
        file_id="file-123",
        retrieval_results=[_retrieval_result(), second],
        max_context_chars=1000,
    )

    assert response["refusal"] is False
    assert response["answer_strategy"] == "extractive_context_v1"
    assert response["used_context_count"] == 2
    assert "[source 1]" in response["answer"]
    assert "[source 2]" in response["answer"]
    assert "second source content" in response["answer"]


def test_build_rag_response_can_use_llm_grounded_strategy(monkeypatch):
    monkeypatch.setattr(
        "app.services.rag.generate_grounded_answer",
        lambda prompt: "LLM grounded answer [source 1]",
    )

    response = build_rag_response(
        query="What is the guide about?",
        file_id="file-123",
        retrieval_results=[_retrieval_result()],
        max_context_chars=1000,
        use_llm=True,
    )

    assert response["refusal"] is False
    assert response["answer_strategy"] == "llm_grounded_context_v1"
    assert response["answer"] == "LLM grounded answer [source 1]"


def test_build_rag_response_falls_back_when_llm_fails(monkeypatch):
    def fail(_prompt):
        raise RuntimeError("llm unavailable")

    monkeypatch.setattr("app.services.rag.generate_grounded_answer", fail)

    response = build_rag_response(
        query="What is the guide about?",
        file_id="file-123",
        retrieval_results=[_retrieval_result()],
        max_context_chars=1000,
        use_llm=True,
    )

    assert response["refusal"] is False
    assert response["answer_strategy"] == "extractive_context_v1"
    assert "[source 1]" in response["answer"]


def test_build_rag_response_marks_llm_refusal(monkeypatch):
    monkeypatch.setattr(
        "app.services.rag.generate_grounded_answer",
        lambda prompt: "资料不足，无法基于已上传文档回答该问题。",
    )

    response = build_rag_response(
        query="What is the guide about?",
        file_id="file-123",
        retrieval_results=[_retrieval_result()],
        max_context_chars=1000,
        use_llm=True,
    )

    assert response["refusal"] is True
    assert response["answer_strategy"] == "low_confidence_refusal"


def test_build_rag_response_marks_english_llm_refusal(monkeypatch):
    monkeypatch.setattr(
        "app.services.rag.generate_grounded_answer",
        lambda prompt: (
            "The exact API key cannot be determined from the materials given. "
            "[source 1]"
        ),
    )

    response = build_rag_response(
        query="What is the guide about?",
        file_id="file-123",
        retrieval_results=[_retrieval_result()],
        max_context_chars=1000,
        use_llm=True,
    )

    assert response["refusal"] is True
    assert response["answer_strategy"] == "low_confidence_refusal"


def test_build_rag_sources_warns_and_falls_back_for_legacy_missing_chunk_id(caplog):
    result = _retrieval_result()
    result.metadata.pop("chunk_id")

    sources = build_rag_sources([result], max_context_chars=100)

    assert sources[0].chunk_id == "digest-123"
    assert "Missing chunk_id" in caplog.text
