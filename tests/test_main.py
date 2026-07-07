import os
import jwt
import datetime
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor

from main import app
from app.routes import document_routes

client = TestClient(app)


@pytest.fixture
def auth_headers():
    jwt_secret = "testsecret"
    os.environ["JWT_SECRET"] = jwt_secret
    payload = {
        "id": "testuser",
        "exp": datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(hours=1),
    }
    token = jwt.encode(payload, jwt_secret, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(autouse=True)
def override_vector_store(monkeypatch):
    from app.config import vector_store
    from app.services.vector_store.async_pg_vector import AsyncPgVector
    from app.routes import document_routes

    # Clear the LRU cache and patch the cached function to return dummy embeddings
    document_routes.get_cached_query_embedding.cache_clear()
    monkeypatch.setattr("app.services.rag.is_llm_enabled", lambda: False)

    def dummy_get_cached_query_embedding(query):
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(
        document_routes, "get_cached_query_embedding", dummy_get_cached_query_embedding
    )

    # Initialize thread pool for tests since TestClient doesn't run lifespan
    if not hasattr(app.state, "thread_pool") or app.state.thread_pool is None:
        app.state.thread_pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="test-worker"
        )

    # Override get_all_ids as an async function - patch at CLASS level to bypass run_in_executor
    async def dummy_get_all_ids(self, executor=None):
        return ["testid1", "testid2"]

    monkeypatch.setattr(AsyncPgVector, "get_all_ids", dummy_get_all_ids)

    # Override get_filtered_ids as an async function.
    async def dummy_get_filtered_ids(self, ids, executor=None):
        dummy_ids = ["testid1", "testid2"]
        return [id for id in dummy_ids if id in ids]

    monkeypatch.setattr(AsyncPgVector, "get_filtered_ids", dummy_get_filtered_ids)

    async def dummy_get_file_ids(self, ids, executor=None):
        dummy_ids = ["testid1", "testid2"]
        return [id for id in dummy_ids if id in ids]

    monkeypatch.setattr(AsyncPgVector, "get_file_ids", dummy_get_file_ids)

    async def dummy_get_ids_by_file_ids(self, ids, executor=None):
        return [f"{file_id}_chunk_1" for file_id in ids]

    monkeypatch.setattr(
        AsyncPgVector, "get_ids_by_file_ids", dummy_get_ids_by_file_ids
    )

    # Override get_documents_by_ids as an async function.
    async def dummy_get_documents_by_ids(self, ids, executor=None):
        return [
            Document(page_content="Test content", metadata={"file_id": id})
            for id in ids
        ]

    monkeypatch.setattr(
        AsyncPgVector, "get_documents_by_ids", dummy_get_documents_by_ids
    )

    # Override embedding_function with a dummy that doesn't call OpenAI
    class DummyEmbedding:
        def embed_query(self, query):
            return [0.1, 0.2, 0.3]

    vector_store.embedding_function = DummyEmbedding()

    # Override similarity search to return a tuple (Document, score).
    def _extract_file_id(filter):
        file_filter = filter.get("file_id", "testid1") if filter else "testid1"
        if isinstance(file_filter, dict):
            if "$like" in file_filter:
                pattern = str(file_filter["$like"])
                prefix = pattern[:-1] if pattern.endswith("%") else pattern
                return f"{prefix}sample"
            return file_filter.get("$eq") or "testid1"
        return file_filter

    def dummy_similarity_search_with_score_by_vector(self, embedding, k, filter):
        doc = Document(
            page_content="Queried content",
            metadata={
                "file_id": _extract_file_id(filter),
                "user_id": "testuser",
            },
        )
        return [(doc, 0.9)]

    async def dummy_asimilarity_search_with_score_by_vector(
        self, embedding, k, filter=None, executor=None
    ):
        doc = Document(
            page_content="Queried content",
            metadata={
                "file_id": _extract_file_id(filter),
                "user_id": "testuser",
                "source_file": "guide.md",
                "chunk_id": "chunk-testid1-1",
                "chunk_index": 2,
                "page": 5,
            },
        )
        return [(doc, 0.9)]

    monkeypatch.setattr(
        AsyncPgVector,
        "similarity_search_with_score_by_vector",
        dummy_similarity_search_with_score_by_vector,
    )
    monkeypatch.setattr(
        AsyncPgVector,
        "asimilarity_search_with_score_by_vector",
        dummy_asimilarity_search_with_score_by_vector,
    )

    # Override document addition functions.
    def dummy_add_documents(self, docs, ids):
        return ids

    async def dummy_aadd_documents(self, docs, ids=None, executor=None):
        return ids

    monkeypatch.setattr(AsyncPgVector, "add_documents", dummy_add_documents)
    monkeypatch.setattr(AsyncPgVector, "aadd_documents", dummy_aadd_documents)

    # Override delete function.
    async def dummy_delete(self, ids=None, collection_only=False, executor=None):
        return None

    monkeypatch.setattr(AsyncPgVector, "delete", dummy_delete)

    async def dummy_delete_by_file_ids(self, file_ids, executor=None):
        return None

    monkeypatch.setattr(AsyncPgVector, "delete_by_file_ids", dummy_delete_by_file_ids)


def test_get_all_ids(auth_headers):
    response = client.get("/ids", headers=auth_headers)
    assert response.status_code == 200
    json_data = response.json()
    assert isinstance(json_data, list)
    assert "testid1" in json_data


def test_get_documents_by_ids(auth_headers):
    response = client.get(
        "/documents", params={"ids": ["testid1"]}, headers=auth_headers
    )
    assert response.status_code == 200
    json_data = response.json()
    assert isinstance(json_data, list)
    assert json_data[0]["page_content"] == "Test content"
    assert json_data[0]["metadata"]["file_id"] == "testid1"


def test_delete_documents(auth_headers):
    response = client.request(
        "DELETE", "/documents", json=["testid1"], headers=auth_headers
    )
    assert response.status_code == 200
    json_data = response.json()
    assert "Documents for" in json_data["message"]


def test_query_embeddings_by_file_id(auth_headers):
    data = {
        "query": "Test query",
        "file_id": "testid1",
        "k": 4,
        "entity_id": "testuser",
    }
    response = client.post("/query", json=data, headers=auth_headers)
    assert response.status_code == 200
    json_data = response.json()
    assert isinstance(json_data, list)
    if json_data:
        doc = json_data[0][0]
        assert doc["page_content"] == "Queried content"


def test_retrieval_search_returns_structured_results(auth_headers):
    data = {
        "query": "Test query",
        "file_id": "testid1",
        "k": 4,
        "entity_id": "testuser",
    }
    response = client.post("/retrieval/search", json=data, headers=auth_headers)
    assert response.status_code == 200

    json_data = response.json()
    assert json_data["query"] == "Test query"
    assert json_data["file_id"] == "testid1"
    assert json_data["top_k"] == 4
    assert json_data["result_count"] == 1

    result = json_data["results"][0]
    assert result["rank"] == 1
    assert result["content"] == "Queried content"
    assert result["score"] == 0.9
    assert result["score_type"] == "pgvector_distance_lower_is_better"
    assert result["file_id"] == "testid1"
    assert result["source_file"] == "guide.md"
    assert result["chunk_index"] == 2
    assert result["page"] == 5


def test_global_retrieval_search_returns_structured_results(auth_headers):
    data = {
        "query": "Test query",
        "k": 4,
        "entity_id": "testuser",
    }
    response = client.post("/retrieval/search_global", json=data, headers=auth_headers)
    assert response.status_code == 200

    json_data = response.json()
    assert json_data["query"] == "Test query"
    assert json_data["top_k"] == 4
    assert json_data["candidate_mode"] == "global_vector_search"
    assert json_data["result_count"] == 1

    result = json_data["results"][0]
    assert result["rank"] == 1
    assert result["content"] == "Queried content"
    assert result["score"] == 0.9
    assert result["file_id"] == "testid1"
    assert result["source_file"] == "guide.md"


def test_global_retrieval_search_can_scope_by_file_id_prefix(auth_headers):
    data = {
        "query": "Test query",
        "k": 4,
        "entity_id": "testuser",
        "file_id_prefix": "hotpotqa-day11-",
    }
    response = client.post("/retrieval/search_global", json=data, headers=auth_headers)
    assert response.status_code == 200

    json_data = response.json()
    assert json_data["candidate_mode"] == "global_vector_search_prefix"
    assert json_data["file_id_prefix"] == "hotpotqa-day11-"

    result = json_data["results"][0]
    assert result["file_id"].startswith("hotpotqa-day11-")


def test_rag_chat_returns_answer_and_sources(auth_headers):
    data = {
        "query": "Test query",
        "file_id": "testid1",
        "k": 4,
        "entity_id": "testuser",
    }
    response = client.post("/rag/chat", json=data, headers=auth_headers)
    assert response.status_code == 200

    json_data = response.json()
    assert json_data["query"] == "Test query"
    assert json_data["file_id"] == "testid1"
    assert json_data["refusal"] is False
    assert json_data["answer_strategy"] == "extractive_context_v1"
    assert json_data["used_context_count"] == 1
    assert "Queried content" in json_data["answer"]

    source = json_data["sources"][0]
    assert source["rank"] == 1
    assert source["source_file"] == "guide.md"
    assert source["chunk_id"] == "chunk-testid1-1"
    assert source["chunk_index"] == 2
    assert source["page"] == 5
    assert source["score"] == 0.9
    assert source["score_type"] == "pgvector_distance_lower_is_better"


def test_rag_chat_can_use_llm_grounded_strategy(auth_headers, monkeypatch):
    monkeypatch.setattr(
        "app.services.rag.generate_grounded_answer",
        lambda prompt: "LLM grounded API answer [source 1]",
    )

    data = {
        "query": "Test query",
        "file_id": "testid1",
        "k": 4,
        "entity_id": "testuser",
        "use_llm": True,
    }
    response = client.post("/rag/chat", json=data, headers=auth_headers)
    assert response.status_code == 200

    json_data = response.json()
    assert json_data["refusal"] is False
    assert json_data["answer_strategy"] == "llm_grounded_context_v1"
    assert json_data["answer"] == "LLM grounded API answer [source 1]"
    assert json_data["used_context_count"] == 1


def test_rag_chat_refuses_when_no_context(auth_headers, monkeypatch):
    from app.services.vector_store.async_pg_vector import AsyncPgVector

    async def no_results(self, embedding, k, filter=None, executor=None):
        return []

    monkeypatch.setattr(
        AsyncPgVector,
        "asimilarity_search_with_score_by_vector",
        no_results,
    )

    data = {
        "query": "No answer question",
        "file_id": "testid1",
        "k": 4,
        "entity_id": "testuser",
    }
    response = client.post("/rag/chat", json=data, headers=auth_headers)
    assert response.status_code == 200

    json_data = response.json()
    assert json_data["refusal"] is True
    assert json_data["answer_strategy"] == "no_context_refusal"
    assert json_data["used_context_count"] == 0
    assert json_data["sources"] == []
    assert "资料不足" in json_data["answer"]


def test_rag_chat_global_returns_answer_and_sources(auth_headers):
    data = {
        "query": "Test global query",
        "k": 4,
        "entity_id": "testuser",
        "use_llm": False,
    }
    response = client.post("/rag/chat_global", json=data, headers=auth_headers)
    assert response.status_code == 200

    json_data = response.json()
    assert json_data["query"] == "Test global query"
    assert json_data["candidate_mode"] == "global_vector_search"
    assert json_data["file_id_prefix"] is None
    assert json_data["refusal"] is False
    assert json_data["answer_strategy"] == "extractive_context_v1"
    assert json_data["used_context_count"] == 1
    assert "Queried content" in json_data["answer"]

    source = json_data["sources"][0]
    assert source["source_file"] == "guide.md"
    assert source["metadata"]["file_id"] == "testid1"


def test_rag_chat_global_can_scope_by_file_id_prefix(auth_headers, monkeypatch):
    monkeypatch.setattr(
        "app.services.rag.generate_grounded_answer",
        lambda prompt: "Global LLM answer [source 1]",
    )

    data = {
        "query": "Test global prefix query",
        "k": 4,
        "entity_id": "testuser",
        "file_id_prefix": "hotpotqa-day11-",
        "use_llm": True,
    }
    response = client.post("/rag/chat_global", json=data, headers=auth_headers)
    assert response.status_code == 200

    json_data = response.json()
    assert json_data["candidate_mode"] == "global_vector_search_prefix"
    assert json_data["file_id_prefix"] == "hotpotqa-day11-"
    assert json_data["answer_strategy"] == "llm_grounded_context_v1"
    assert json_data["answer"] == "Global LLM answer [source 1]"
    assert json_data["sources"][0]["metadata"]["file_id"].startswith("hotpotqa-day11-")


def test_rag_chat_global_refuses_when_no_context(auth_headers, monkeypatch):
    from app.services.vector_store.async_pg_vector import AsyncPgVector

    async def no_results(self, embedding, k, filter=None, executor=None):
        return []

    monkeypatch.setattr(
        AsyncPgVector,
        "asimilarity_search_with_score_by_vector",
        no_results,
    )

    data = {
        "query": "No answer global question",
        "k": 4,
        "entity_id": "testuser",
        "use_llm": True,
    }
    response = client.post("/rag/chat_global", json=data, headers=auth_headers)
    assert response.status_code == 200

    json_data = response.json()
    assert json_data["refusal"] is True
    assert json_data["answer_strategy"] == "no_context_refusal"
    assert json_data["used_context_count"] == 0
    assert json_data["sources"] == []
    assert "资料不足" in json_data["answer"]


def test_embed_local_file(tmp_path, auth_headers, monkeypatch):
    # Monkeypatch RAG_UPLOAD_DIR so the file is within the allowed directory.
    monkeypatch.setattr(document_routes, "RAG_UPLOAD_DIR", str(tmp_path))

    # Create a temporary file inside the patched upload dir.
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a test document.")

    data = {
        "filepath": "test.txt",
        "filename": "test.txt",
        "file_content_type": "text/plain",
        "file_id": "testid1",
    }
    response = client.post("/local/embed", json=data, headers=auth_headers)
    assert response.status_code == 200, f"Response: {response.text}"
    json_data = response.json()
    assert json_data["status"] is True
    assert json_data["file_id"] == "testid1"


def test_embed_file(tmp_path, auth_headers):
    file_content = "This is a test file for the embed endpoint."
    test_file = tmp_path / "test_embed.txt"
    test_file.write_text(file_content)
    with test_file.open("rb") as f:
        response = client.post(
            "/embed",
            data={"file_id": "testid1", "entity_id": "testuser"},
            files={"file": ("test_embed.txt", f, "text/plain")},
            headers=auth_headers,
        )
    assert response.status_code == 200, f"Response: {response.text}"
    json_data = response.json()
    assert json_data["status"] is True
    assert json_data["file_id"] == "testid1"


def test_load_document_context(auth_headers):
    response = client.get("/documents/testid1/context", headers=auth_headers)
    assert response.status_code == 200, f"Response: {response.text}"
    content = response.text
    assert "testid1" in content or "Test content" in content


def test_embed_file_upload(tmp_path, auth_headers, monkeypatch):
    file_content = "Test content for embed upload."
    test_file = tmp_path / "upload_test.txt"
    test_file.write_text(file_content)

    with test_file.open("rb") as f:
        response = client.post(
            "/embed-upload",
            data={"file_id": "testid1", "entity_id": "testuser"},
            files={"uploaded_file": ("upload_test.txt", f, "text/plain")},
            headers=auth_headers,
        )
    assert response.status_code == 200, f"Response: {response.text}"
    json_data = response.json()
    assert json_data["status"] is True
    assert json_data["file_id"] == "testid1"


def test_query_multiple(auth_headers):
    data = {
        "query": "Test query multiple",
        "file_ids": ["testid1", "testid2"],
        "k": 4,
    }
    response = client.post("/query_multiple", json=data, headers=auth_headers)
    assert response.status_code == 200, f"Response: {response.text}"
    json_data = response.json()
    assert isinstance(json_data, list)
    if json_data:
        doc = json_data[0][0]
        assert doc["page_content"] == "Queried content"


def test_extract_text_from_file(tmp_path, auth_headers):
    """Test the /text endpoint for text extraction without embeddings."""
    file_content = "This is a test file for text extraction.\nIt has multiple lines.\nAnd should be extracted properly."
    test_file = tmp_path / "test_text_extraction.txt"
    test_file.write_text(file_content)

    with test_file.open("rb") as f:
        response = client.post(
            "/text",
            data={"file_id": "test_text_123", "entity_id": "testuser"},
            files={"file": ("test_text_extraction.txt", f, "text/plain")},
            headers=auth_headers,
        )

    assert response.status_code == 200, f"Response: {response.text}"
    json_data = response.json()

    # Check response structure
    assert "text" in json_data
    assert "file_id" in json_data
    assert "filename" in json_data
    assert "known_type" in json_data

    # Check response content
    assert json_data["text"] == file_content
    assert json_data["file_id"] == "test_text_123"
    assert json_data["filename"] == "test_text_extraction.txt"
    assert json_data["known_type"] is True  # text files are known types
