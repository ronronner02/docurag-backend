from langchain_core.documents import Document

from app.routes.document_routes import _prepare_documents_sync


def test_prepare_documents_adds_traceable_chunk_metadata(monkeypatch):
    monkeypatch.setattr("app.routes.document_routes.CHUNK_SIZE", 12)
    monkeypatch.setattr("app.routes.document_routes.CHUNK_OVERLAP", 0)

    source = Document(
        page_content="alpha beta gamma delta",
        metadata={"source": "/tmp/uploads/guide.md", "page": 3},
    )

    chunks = _prepare_documents_sync(
        data=[source],
        file_id="file-123",
        user_id="user-456",
        clean_content=False,
    )

    assert len(chunks) >= 2

    first = chunks[0]
    assert first.metadata["file_id"] == "file-123"
    assert first.metadata["user_id"] == "user-456"
    assert first.metadata["chunk_id"]
    assert first.metadata["chunk_index"] == 0
    assert first.metadata["char_length"] == len(first.page_content)
    assert first.metadata["source_file"] == "guide.md"
    assert first.metadata["page"] == 3
    assert first.metadata["digest"]

    for index, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_id"]
        assert chunk.metadata["chunk_index"] == index
        assert chunk.metadata["char_length"] == len(chunk.page_content)

    chunk_ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))
