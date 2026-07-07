# tests/conftest.py
import os

# Set environment variables early so config picks up test settings.
os.environ["TESTING"] = "1"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ["RAG_LLM_PROVIDER"] = "disabled"
# Set DB_HOST (and DSN) to dummy values to avoid real connection attempts.
os.environ["DB_HOST"] = "localhost"  # or any dummy value
os.environ["DSN"] = "dummy://"
os.environ.setdefault("RAG_OPENAI_API_KEY", "test-key-for-unit-tests")
os.environ.setdefault("OPENAI_API_KEY", "test-key-for-unit-tests")
os.environ["EMBEDDINGS_PROVIDER"] = "openai"
os.environ["EMBEDDINGS_MODEL"] = "text-embedding-3-small"

# -- Patch the vector store classes to bypass DB connection --

# Do this *before* importing any app modules.
from app.services.vector_store.async_pg_vector import AsyncPgVector
from langchain_community.vectorstores.pgvector import PGVector


def dummy_post_init(self):
    # Skip extension creation
    pass


AsyncPgVector.__post_init__ = dummy_post_init
PGVector.__post_init__ = dummy_post_init

from langchain_core.documents import Document


class DummyVectorStore:
    def get_all_ids(self) -> list[str]:
        return ["testid1", "testid2"]

    def get_filtered_ids(self, ids) -> list[str]:
        dummy_ids = ["testid1", "testid2"]
        return [id for id in dummy_ids if id in ids]

    async def get_documents_by_ids(self, ids: list[str]) -> list[Document]:
        return [
            Document(page_content="Test content", metadata={"file_id": id})
            for id in ids
        ]

    def similarity_search_with_score_by_vector(self, embedding, k: int, filter: dict):
        doc = Document(
            page_content="Queried content",
            metadata={
                "file_id": filter.get("file_id", "testid1"),
                "user_id": "testuser",
            },
        )
        return [(doc, 0.9)]

    def add_documents(self, documents, ids=None, **kwargs):
        return ids

    async def aadd_documents(self, documents, ids=None, **kwargs):
        return ids

    async def delete(self, ids=None, collection_only: bool = False):
        return None

    # Implement the missing as_retriever() method
    def as_retriever(self):
        # Return self or wrap with a dummy retriever if needed.
        return self
