import hashlib
import pytest
from pydantic import ValidationError

from app.models import (
    DocumentModel,
    QueryRequestBody,
    RagChatRequest,
    RagGlobalChatRequest,
)


def test_generate_digest():
    content = "Hello, World!"
    model = DocumentModel(page_content=content)
    expected_digest = hashlib.md5(content.encode()).hexdigest()
    assert model.generate_digest() == expected_digest


def test_document_model_metadata_uses_independent_default_dicts():
    first = DocumentModel(page_content="one")
    second = DocumentModel(page_content="two")

    first.metadata["source"] = "guide.md"

    assert second.metadata == {}


@pytest.mark.parametrize("k", [0, 21])
def test_query_request_rejects_out_of_range_k(k):
    with pytest.raises(ValidationError):
        QueryRequestBody(query="hello", file_id="file-123", k=k)


def test_rag_chat_request_limits_context_size():
    with pytest.raises(ValidationError):
        RagChatRequest(query="hello", file_id="file-123", max_context_chars=199)

    with pytest.raises(ValidationError):
        RagChatRequest(query="hello", file_id="file-123", max_context_chars=8001)


def test_rag_global_chat_request_limits_context_size():
    with pytest.raises(ValidationError):
        RagGlobalChatRequest(query="hello", max_context_chars=199)

    with pytest.raises(ValidationError):
        RagGlobalChatRequest(query="hello", max_context_chars=8001)


def test_resume_business_tables_are_declared():
    from app.db_models import Base

    assert {"documents", "document_chunks", "chat_messages"} <= set(
        Base.metadata.tables
    )


def test_document_chunk_table_has_traceability_columns():
    from app.db_models import Base

    columns = Base.metadata.tables["document_chunks"].columns

    for column_name in [
        "document_id",
        "chunk_index",
        "content",
        "char_length",
        "source_file",
        "page",
        "embedding_id",
        "metadata",
    ]:
        assert column_name in columns


def test_chat_messages_table_can_store_sources():
    from app.db_models import Base

    columns = Base.metadata.tables["chat_messages"].columns

    assert "conversation_id" in columns
    assert "role" in columns
    assert "content" in columns
    assert "sources" in columns
