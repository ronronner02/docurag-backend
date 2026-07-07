# app/routes/document_routes.py
import os
import uuid
from pathlib import Path
import hashlib
import traceback
import aiofiles
import aiofiles.os
from shutil import copyfileobj
from typing import List, Iterable, Optional, Union, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor
from fastapi import (
    APIRouter,
    Request,
    UploadFile,
    HTTPException,
    File,
    Form,
    Body,
    Query,
    status,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from functools import lru_cache
import asyncio

if TYPE_CHECKING:
    from app.services.vector_store.async_pg_vector import AsyncPgVector
    from app.services.vector_store.atlas_mongo_vector import AtlasMongoVector
    from langchain_community.vectorstores.pgvector import PGVector as PgVector

from app.config import (
    logger,
    vector_store,
    VECTOR_DB_TYPE,
    VectorDBType,
    RAG_UPLOAD_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MAX_QUEUE_SIZE,
    RAG_DISTANCE_THRESHOLD,
)

# Warn once at import time if the user set a threshold under Atlas, where
# the score direction is inverted (Atlas vectorSearchScore: higher = better)
# and naive `score <= threshold` would keep the *weaker* matches. We scope
# the filter to pgvector only until we grow a first-class "min similarity"
# semantic for Atlas.
#
# Inspect the raw env var here rather than the parsed RAG_DISTANCE_THRESHOLD:
# the parser in app.config deliberately skips the float() cast under Atlas
# (so non-numeric stale values don't break startup), which means the parsed
# value is always None for Atlas — and relying on it would suppress the
# warning we want operators to see.
if (
    VECTOR_DB_TYPE == VectorDBType.ATLAS_MONGO
    and os.getenv("RAG_DISTANCE_THRESHOLD") not in (None, "")
):
    logger.warning(
        "RAG_DISTANCE_THRESHOLD is set but VECTOR_DB_TYPE=atlas-mongo; "
        "Atlas returns similarity scores (higher = better) which would "
        "invert the filter semantics, so the threshold will be ignored."
    )


def _apply_distance_threshold(documents):
    """Drop (doc, score) tuples whose distance exceeds RAG_DISTANCE_THRESHOLD.

    Only applied for pgvector, where similarity_search_with_score_by_vector
    returns a distance (lower = more similar). Skipped for Atlas because its
    score is a similarity (higher = better) and applying the same comparison
    would keep the weakest matches and drop the strongest.
    """
    if RAG_DISTANCE_THRESHOLD is None:
        return documents
    if VECTOR_DB_TYPE == VectorDBType.ATLAS_MONGO:
        return documents
    return [(doc, score) for doc, score in documents if score <= RAG_DISTANCE_THRESHOLD]
from app.constants import ERROR_MESSAGES
from app.models import (
    StoreDocument,
    QueryRequestBody,
    DocumentResponse,
    QueryMultipleBody,
    GlobalRetrievalSearchRequest,
    GlobalRetrievalSearchResponse,
    RetrievalSearchResponse,
    RetrievalSearchResult,
    RagChatRequest,
    RagChatResponse,
    RagGlobalChatRequest,
    RagGlobalChatResponse,
)
from app.services.rag import build_rag_response
from app.services.vector_store.async_pg_vector import AsyncPgVector
from app.utils.document_loader import (
    get_loader,
    clean_text,
    process_documents,
    cleanup_temp_encoding_file,
)
from app.utils.health import is_health_ok

router = APIRouter()


def _retrieval_score_type() -> str:
    if VECTOR_DB_TYPE == VectorDBType.PGVECTOR:
        return "pgvector_distance_lower_is_better"
    if VECTOR_DB_TYPE == VectorDBType.ATLAS_MONGO:
        return "atlas_similarity_higher_is_better"
    return "unknown"


def _format_retrieval_results(documents) -> List[RetrievalSearchResult]:
    score_type = _retrieval_score_type()
    results = []
    for rank, (document, score) in enumerate(documents, start=1):
        metadata = dict(document.metadata or {})
        source_path = metadata.get("source")
        source_file = metadata.get("source_file")
        if source_file is None and source_path:
            source_file = Path(str(source_path)).name
        results.append(
            RetrievalSearchResult(
                rank=rank,
                content=document.page_content,
                score=score,
                score_type=score_type,
                file_id=metadata.get("file_id"),
                source_file=source_file,
                chunk_index=metadata.get("chunk_index"),
                page=metadata.get("page"),
                metadata=metadata,
            )
        )
    return results


def _escape_like_prefix(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _normalize_file_id_prefix(file_id_prefix: Optional[str]) -> Optional[str]:
    if file_id_prefix is None:
        return None
    prefix = file_id_prefix.strip()
    if not prefix:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT("file_id_prefix cannot be blank"),
        )
    return prefix


def _build_file_id_prefix_filter(file_id_prefix: Optional[str]) -> Optional[dict]:
    prefix = _normalize_file_id_prefix(file_id_prefix)
    if prefix is None:
        return None
    return {"file_id": {"$like": f"{_escape_like_prefix(prefix)}%"}}


def calculate_num_batches(total: int, batch_size: int) -> int:
    """Calculate the number of batches needed to process total items."""
    if batch_size <= 0:
        return 1
    return (total + batch_size - 1) // batch_size


def get_user_id(request: Request, entity_id: str = None) -> str:
    """Extract user ID from request or entity_id."""
    if entity_id and validate_file_path(RAG_UPLOAD_DIR, os.path.join(entity_id, "_")) is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT("Invalid entity_id"),
        )

    if not hasattr(request.state, "user"):
        return entity_id if entity_id else "public"

    user_id = request.state.user.get("id")
    if entity_id and entity_id != user_id:
        logger.warning(
            "Ignoring unauthorized entity_id %s for authenticated user %s",
            entity_id,
            user_id,
        )
    return user_id


async def save_upload_file_async(file: UploadFile, temp_file_path: str) -> None:
    """Save uploaded file asynchronously."""
    try:
        async with aiofiles.open(temp_file_path, "wb") as temp_file:
            chunk_size = 64 * 1024  # 64 KB
            while content := await file.read(chunk_size):
                await temp_file.write(content)
    except Exception as e:
        logger.error(
            "Failed to save uploaded file | Path: %s | Error: %s | Traceback: %s",
            temp_file_path,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save the uploaded file. Error: {str(e)}",
        )


def save_upload_file_sync(file: UploadFile, temp_file_path: str) -> None:
    """Save uploaded file synchronously."""
    try:
        with open(temp_file_path, "wb") as temp_file:
            copyfileobj(file.file, temp_file)
    except Exception as e:
        logger.error(
            "Failed to save uploaded file | Path: %s | Error: %s | Traceback: %s",
            temp_file_path,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save the uploaded file. Error: {str(e)}",
        )


def validate_file_path(base_dir: str, file_path: str) -> Optional[str]:
    """Validate that file_path resolves within base_dir. Returns resolved absolute path or None."""
    if not file_path or not file_path.strip():
        return None
    if "\x00" in file_path:
        return None
    try:
        allowed = Path(base_dir).resolve()
        requested = Path(os.path.join(base_dir, file_path)).resolve()
        requested.relative_to(allowed)
        return str(requested)
    except (ValueError, RuntimeError, TypeError, OSError):
        return None


def _make_unique_temp_path(user_id: str, filename: str) -> Optional[str]:
    """Build a unique temp file path under RAG_UPLOAD_DIR/{user_id}/ to prevent
    concurrent upload collisions. Returns a validated absolute path, or None if
    the raw filename would escape RAG_UPLOAD_DIR (path traversal rejection)."""
    # Validate the raw filename to reject traversal attempts
    if validate_file_path(RAG_UPLOAD_DIR, os.path.join(user_id, filename)) is None:
        return None
    # unique_name is stem + "_" + [0-9a-f]{32} + suffix — no path separators,
    # so it cannot escape the directory validated above.
    p = Path(filename)
    unique_name = f"{p.stem}_{uuid.uuid4().hex}{p.suffix}"
    return str(Path(RAG_UPLOAD_DIR, user_id, unique_name).resolve())


async def load_file_content(
    filename: str,
    content_type: str,
    file_path: str,
    executor,
    raw_text: bool = False,
) -> tuple:
    """Load file content using appropriate loader.

    Pass ``raw_text=True`` when the caller wants verbatim file contents (e.g.
    the ``/text`` endpoint) so text-formatted files are not semantically
    parsed.
    """
    loader = None
    try:
        loader, known_type, file_ext = get_loader(
            filename, content_type, file_path, raw_text=raw_text
        )
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(executor, lambda: list(loader.lazy_load()))
        return data, known_type, file_ext
    finally:
        # Clean up temporary UTF-8 file if it was created for encoding conversion
        if loader is not None:
            cleanup_temp_encoding_file(loader)


def extract_text_from_documents(documents: List[Document], file_ext: str) -> str:
    """Extract text content from loaded documents."""
    text_content = ""
    if documents:
        for doc in documents:
            if hasattr(doc, "page_content"):
                # Clean text if it's a PDF
                if file_ext == "pdf":
                    text_content += clean_text(doc.page_content) + "\n"
                else:
                    text_content += doc.page_content + "\n"

    # Remove trailing newline
    return text_content.rstrip("\n")


async def cleanup_temp_file_async(file_path: str) -> None:
    """Clean up temporary file asynchronously."""
    try:
        await aiofiles.os.remove(file_path)
    except Exception as e:
        logger.error(
            "Failed to remove temporary file | Path: %s | Error: %s | Traceback: %s",
            file_path,
            str(e),
            traceback.format_exc(),
        )


@router.get("/ids")
async def get_all_ids(request: Request):
    try:
        if isinstance(vector_store, AsyncPgVector):
            ids = await vector_store.get_all_ids(executor=request.app.state.thread_pool)
        else:
            ids = vector_store.get_all_ids()

        return list(set(ids))
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in get_all_ids | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Failed to get all IDs | Error: %s | Traceback: %s",
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    try:
        if await is_health_ok():
            return {"status": "UP"}
        else:
            logger.error("Health check failed")
            return {"status": "DOWN"}, 503
    except Exception as e:
        logger.error(
            "Error during health check | Error: %s | Traceback: %s",
            str(e),
            traceback.format_exc(),
        )
        return {"status": "DOWN", "error": str(e)}, 503


@router.get("/documents", response_model=list[DocumentResponse])
async def get_documents_by_ids(request: Request, ids: list[str] = Query(...)):
    try:
        if isinstance(vector_store, AsyncPgVector):
            existing_ids = await vector_store.get_filtered_ids(
                ids, executor=request.app.state.thread_pool
            )
            documents = await vector_store.get_documents_by_ids(
                ids, executor=request.app.state.thread_pool
            )
        else:
            existing_ids = vector_store.get_filtered_ids(ids)
            documents = vector_store.get_documents_by_ids(ids)

        # Ensure all requested ids exist
        if not all(id in existing_ids for id in ids):
            raise HTTPException(status_code=404, detail="One or more IDs not found")

        # Ensure documents list is not empty
        if not documents:
            raise HTTPException(
                status_code=404, detail="No documents found for the given IDs"
            )

        return documents
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in get_documents_by_ids | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error getting documents by IDs | IDs: %s | Error: %s | Traceback: %s",
            ids,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents")
async def delete_documents(request: Request, document_ids: List[str] = Body(...)):
    try:
        existing_ids = await _get_existing_file_ids(document_ids, request)

        if not all(id in existing_ids for id in document_ids):
            raise HTTPException(status_code=404, detail="One or more IDs not found")

        await _delete_file_vectors(document_ids, request)

        file_count = len(document_ids)
        return {
            "message": f"Documents for {file_count} file{'s' if file_count > 1 else ''} deleted successfully"
        }
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in delete_documents | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Failed to delete documents | IDs: %s | Error: %s | Traceback: %s",
            document_ids,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


# Cache the embedding function with LRU cache
@lru_cache(maxsize=128)
def get_cached_query_embedding(query: str):
    return vector_store.embedding_function.embed_query(query)


async def _retrieve_authorized_documents(
    body: Union[QueryRequestBody, GlobalRetrievalSearchRequest, RagGlobalChatRequest],
    request: Request,
    metadata_filter: Optional[dict] = None,
    requested_file_id: Optional[str] = None,
):
    if not hasattr(request.state, "user"):
        user_authorized = body.entity_id if body.entity_id else "public"
    else:
        user_authorized = request.state.user.get("id")

    authorized_documents = []

    embedding = get_cached_query_embedding(body.query)

    if isinstance(vector_store, AsyncPgVector):
        documents = await vector_store.asimilarity_search_with_score_by_vector(
            embedding,
            k=body.k,
            filter=metadata_filter,
            executor=request.app.state.thread_pool,
        )
    else:
        documents = vector_store.similarity_search_with_score_by_vector(
            embedding, k=body.k, filter=metadata_filter
        )

    documents = _apply_distance_threshold(documents)

    if not documents:
        return authorized_documents

    for document, score in documents:
        doc_metadata = document.metadata or {}
        doc_user_id = doc_metadata.get("user_id")
        if doc_user_id is None or doc_user_id == user_authorized:
            authorized_documents.append((document, score))
            continue

        logger.warning(
            "Unauthorized chunk filtered out | requested_entity_id=%s | "
            "user=%s | document_user_id=%s | file_id=%s | chunk_id=%s",
            body.entity_id,
            user_authorized,
            doc_user_id,
            requested_file_id,
            doc_metadata.get("chunk_id"),
        )

    return authorized_documents


async def _retrieve_authorized_documents_by_file_id(
    body: QueryRequestBody,
    request: Request,
):
    return await _retrieve_authorized_documents(
        body,
        request,
        metadata_filter={"file_id": {"$eq": body.file_id}},
        requested_file_id=body.file_id,
    )


@router.post("/query")
async def query_embeddings_by_file_id(
    body: QueryRequestBody,
    request: Request,
):
    try:
        return await _retrieve_authorized_documents_by_file_id(body, request)

    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in query_embeddings_by_file_id | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error in query embeddings | File ID: %s | Query: %s | Error: %s | Traceback: %s",
            body.file_id,
            body.query,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrieval/search", response_model=RetrievalSearchResponse)
async def search_retrieval_results(
    body: QueryRequestBody,
    request: Request,
):
    try:
        documents = await _retrieve_authorized_documents_by_file_id(body, request)
        results = _format_retrieval_results(documents)
        return RetrievalSearchResponse(
            query=body.query,
            file_id=body.file_id,
            top_k=body.k,
            result_count=len(results),
            results=results,
        )
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in search_retrieval_results | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error in retrieval search | File ID: %s | Query: %s | Error: %s | Traceback: %s",
            body.file_id,
            body.query,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrieval/search_global", response_model=GlobalRetrievalSearchResponse)
async def search_global_retrieval_results(
    body: GlobalRetrievalSearchRequest,
    request: Request,
):
    try:
        file_id_prefix = _normalize_file_id_prefix(body.file_id_prefix)
        metadata_filter = _build_file_id_prefix_filter(file_id_prefix)
        documents = await _retrieve_authorized_documents(
            body, request, metadata_filter=metadata_filter
        )
        results = _format_retrieval_results(documents)
        return GlobalRetrievalSearchResponse(
            query=body.query,
            top_k=body.k,
            result_count=len(results),
            candidate_mode=(
                "global_vector_search_prefix"
                if file_id_prefix
                else "global_vector_search"
            ),
            file_id_prefix=file_id_prefix,
            results=results,
        )
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in search_global_retrieval_results | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error in global retrieval search | Query: %s | Error: %s | Traceback: %s",
            body.query,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/chat", response_model=RagChatResponse)
async def rag_chat(
    body: RagChatRequest,
    request: Request,
):
    query_body = QueryRequestBody(
        query=body.query,
        file_id=body.file_id,
        k=body.k,
        entity_id=body.entity_id,
    )
    try:
        documents = await _retrieve_authorized_documents_by_file_id(query_body, request)
        retrieval_results = _format_retrieval_results(documents)
        return RagChatResponse(
            **build_rag_response(
                query=body.query,
                file_id=body.file_id,
                retrieval_results=retrieval_results,
                max_context_chars=body.max_context_chars,
                use_llm=body.use_llm,
            )
        )
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in rag_chat | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error in rag chat | File ID: %s | Query: %s | Error: %s | Traceback: %s",
            body.file_id,
            body.query,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/chat_global", response_model=RagGlobalChatResponse)
async def rag_chat_global(
    body: RagGlobalChatRequest,
    request: Request,
):
    try:
        file_id_prefix = _normalize_file_id_prefix(body.file_id_prefix)
        metadata_filter = _build_file_id_prefix_filter(file_id_prefix)
        documents = await _retrieve_authorized_documents(
            body, request, metadata_filter=metadata_filter
        )
        retrieval_results = _format_retrieval_results(documents)
        rag_response = build_rag_response(
            query=body.query,
            file_id=file_id_prefix or "__global__",
            retrieval_results=retrieval_results,
            max_context_chars=body.max_context_chars,
            use_llm=body.use_llm,
        )
        return RagGlobalChatResponse(
            query=rag_response["query"],
            candidate_mode=(
                "global_vector_search_prefix"
                if file_id_prefix
                else "global_vector_search"
            ),
            file_id_prefix=file_id_prefix,
            answer=rag_response["answer"],
            refusal=rag_response["refusal"],
            answer_strategy=rag_response["answer_strategy"],
            used_context_count=rag_response["used_context_count"],
            sources=rag_response["sources"],
        )
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in rag_chat_global | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error in global rag chat | Query: %s | Error: %s | Traceback: %s",
            body.query,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


async def _process_documents_async_pipeline(
    documents: List[Document],
    file_id: str,
    vector_store: "AsyncPgVector",
    executor: "ThreadPoolExecutor",
) -> List[str]:
    """
    Process documents using async producer-consumer pattern for batched embedding and insertion.

    Args:
        documents: List of Document objects to process
        file_id: Unique identifier for the file being processed
        vector_store: AsyncPgVector instance for document storage
        executor: ThreadPoolExecutor for concurrent operations

    Returns:
        List of document IDs that were successfully inserted
    """
    total_chunks = len(documents)
    if total_chunks == 0:
        return []

    # Create queues for producer-consumer pattern
    # embedding_queue is bounded to limit document data held in memory.
    # results_queue is unbounded — it holds only small UUID lists, and the
    # drain loop runs after gather(), so bounding it would deadlock when
    # num_batches > maxsize.
    embedding_queue = asyncio.Queue(maxsize=EMBEDDING_MAX_QUEUE_SIZE)
    results_queue = asyncio.Queue()
    all_ids = []

    num_batches = calculate_num_batches(total_chunks, EMBEDDING_BATCH_SIZE)

    logger.info(
        "Starting async pipeline for file %s: %d chunks with %d batch size",
        file_id,
        total_chunks,
        EMBEDDING_BATCH_SIZE,
    )

    async def batch_producer():
        """Produce document batches and put them in the queue."""
        try:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * EMBEDDING_BATCH_SIZE
                end_idx = min(start_idx + EMBEDDING_BATCH_SIZE, total_chunks)
                batch_documents = documents[start_idx:end_idx]
                batch_ids = _vector_ids_for_documents(
                    batch_documents, file_id, start_idx
                )

                logger.info(
                    "Generating embeddings for batch %d/%d: chunks %d-%d",
                    batch_idx + 1,
                    num_batches,
                    start_idx,
                    end_idx - 1,
                )

                # Put batch in queue for processing
                await embedding_queue.put(
                    (batch_documents, batch_ids, batch_idx + 1, num_batches)
                )
        except Exception as e:
            logger.error("Error in batch producer: %s", e)
            raise
        finally:
            # Always signal end of production
            await embedding_queue.put(None)

    async def embedding_consumer():
        """Consume batches from queue, embed and insert into database."""
        try:
            while True:
                item = await embedding_queue.get()
                if item is None:  # End signal
                    embedding_queue.task_done()
                    break

                batch_documents, batch_ids, batch_num, total_batches = item

                logger.info(
                    "Inserting batch %d/%d into database (%d chunks)",
                    batch_num,
                    total_batches,
                    len(batch_documents),
                )

                try:
                    # Insert batch into database
                    batch_result_ids = await vector_store.aadd_documents(
                        batch_documents, ids=batch_ids, executor=executor
                    )
                    await results_queue.put(batch_result_ids)
                except Exception as e:
                    logger.error(
                        "Error processing batch %d/%d: %s", batch_num, total_batches, e
                    )
                    await results_queue.put(e)  # Put exception object
                finally:
                    embedding_queue.task_done()

        except Exception as e:
            logger.error("Fatal error in embedding consumer: %s", e)
            await results_queue.put(e)
            raise

    producer_task = None
    consumer_task = None

    try:
        # Start producer and consumer concurrently
        producer_task = asyncio.create_task(batch_producer())
        consumer_task = asyncio.create_task(embedding_consumer())

        # Wait for both to complete
        await asyncio.gather(producer_task, consumer_task, return_exceptions=False)

        # Collect results from all batches
        for _ in range(num_batches):
            result = await results_queue.get()
            if isinstance(result, Exception):
                raise result
            all_ids.extend(result)

        logger.info(
            "Async pipeline completed for file %s: %d embeddings created",
            file_id,
            len(all_ids),
        )

        return all_ids

    except Exception as e:
        logger.error("Pipeline failed for file %s: %s", file_id, e)
        if consumer_task is not None or producer_task is not None:
            # if one of the tasks is still running, cancel it
            if consumer_task is not None and not consumer_task.done():
                consumer_task.cancel()
            if producer_task is not None and not producer_task.done():
                producer_task.cancel()

            # Await cancelled tasks to ensure proper cleanup
            if consumer_task is None:
                await asyncio.gather(producer_task, return_exceptions=True)
            elif producer_task is None:
                await asyncio.gather(consumer_task, return_exceptions=True)
            else:
                await asyncio.gather(
                    consumer_task, producer_task, return_exceptions=True
                )

        # Attempt rollback only if we inserted something
        if all_ids:
            try:
                logger.warning("Performing rollback of file %s", file_id)
                await _rollback_inserted_vectors_async(vector_store, all_ids, executor)
                logger.info("Rollback completed for file %s", file_id)
            except Exception as cleanup_error:
                logger.error("Rollback failed for file %s: %s", file_id, cleanup_error)

        # Re-raise the original error
        raise


async def _process_documents_batched_sync(
    documents: List[Document],
    file_id: str,
    vector_store: Union["PgVector", "AtlasMongoVector"],
    executor: "ThreadPoolExecutor",
) -> List[str]:
    """
    Process documents in batches using synchronous vector store operations.

    Args:
        documents: List of Document objects to process
        file_id: Unique identifier for the file being processed
        vector_store: Synchronous vector store instance (ExtendedPgVector or AtlasMongoVector)
        executor: ThreadPoolExecutor for running sync operations

    Returns:
        List of document IDs that were successfully inserted
    """
    total_chunks = len(documents)
    if total_chunks == 0:
        return []

    all_ids = []
    num_batches = calculate_num_batches(total_chunks, EMBEDDING_BATCH_SIZE)

    logger.info(
        "Processing file %s with sync batching: %d batches of %d chunks each",
        file_id,
        num_batches,
        EMBEDDING_BATCH_SIZE,
    )

    loop = asyncio.get_running_loop()

    for batch_idx in range(num_batches):
        start_idx = batch_idx * EMBEDDING_BATCH_SIZE
        end_idx = min(start_idx + EMBEDDING_BATCH_SIZE, total_chunks)
        batch_documents = documents[start_idx:end_idx]
        batch_ids = _vector_ids_for_documents(batch_documents, file_id, start_idx)

        logger.info(
            "Processing batch %d/%d: chunks %d-%d (%d chunks)",
            batch_idx + 1,
            num_batches,
            start_idx,
            end_idx - 1,
            len(batch_documents),
        )

        try:
            # Wrap sync call in executor to avoid blocking the event loop
            batch_result_ids = await loop.run_in_executor(
                executor,
                lambda docs=batch_documents, ids=batch_ids: vector_store.add_documents(
                    docs, ids=ids
                ),
            )
            all_ids.extend(batch_result_ids)

        except Exception as batch_error:
            logger.error("Batch %d failed: %s", batch_idx + 1, batch_error)

            # Rollback entire file from vector store
            if (
                all_ids
            ):  # any batch succeeded (i.e., any chunks for this file were inserted)
                logger.warning("Rolling back file %s due to batch failure", file_id)
                try:
                    await loop.run_in_executor(
                        executor,
                        lambda: _rollback_inserted_vectors_sync(
                            vector_store, all_ids, file_id
                        ),
                    )
                    logger.info("Rollback completed for file %s", file_id)
                except Exception as rollback_error:
                    logger.error(
                        "Rollback failed for file %s: %s", file_id, rollback_error
                    )

            raise batch_error

    return all_ids


def generate_digest(page_content: str) -> str:
    return hashlib.md5(page_content.encode("utf-8", "ignore")).hexdigest()


def _vector_ids_for_documents(
    documents: List[Document], file_id: str, start_index: int = 0
) -> List[str]:
    ids = []
    for offset, doc in enumerate(documents):
        metadata = dict(doc.metadata or {})
        chunk_id = metadata.get("chunk_id")
        if not chunk_id:
            digest = metadata.get("digest") or generate_digest(doc.page_content)
            chunk_id = f"{file_id}_{digest}_{start_index + offset}"
            metadata["chunk_id"] = chunk_id
            doc.metadata = metadata
        ids.append(str(chunk_id))
    return ids


async def _get_existing_file_ids(file_ids: List[str], request: Request) -> List[str]:
    if isinstance(vector_store, AsyncPgVector):
        if hasattr(vector_store, "get_file_ids"):
            return await vector_store.get_file_ids(
                file_ids, executor=request.app.state.thread_pool
            )
        return await vector_store.get_filtered_ids(
            file_ids, executor=request.app.state.thread_pool
        )

    if hasattr(vector_store, "get_file_ids"):
        return vector_store.get_file_ids(file_ids)
    return vector_store.get_filtered_ids(file_ids)


async def _delete_file_vectors(file_ids: List[str], request: Request) -> None:
    if isinstance(vector_store, AsyncPgVector):
        if hasattr(vector_store, "delete_by_file_ids"):
            await vector_store.delete_by_file_ids(
                file_ids, executor=request.app.state.thread_pool
            )
            return
        chunk_ids = await vector_store.get_ids_by_file_ids(
            file_ids, executor=request.app.state.thread_pool
        )
        await vector_store.delete(
            ids=chunk_ids or file_ids, executor=request.app.state.thread_pool
        )
        return

    if hasattr(vector_store, "delete_by_file_ids"):
        vector_store.delete_by_file_ids(file_ids)
        return

    chunk_ids = (
        vector_store.get_ids_by_file_ids(file_ids)
        if hasattr(vector_store, "get_ids_by_file_ids")
        else []
    )
    vector_store.delete(ids=chunk_ids or file_ids)


async def _rollback_inserted_vectors_async(
    vector_store: "AsyncPgVector",
    inserted_ids: List[str],
    executor: "ThreadPoolExecutor",
) -> None:
    if inserted_ids:
        await vector_store.delete(ids=inserted_ids, executor=executor)


def _rollback_inserted_vectors_sync(
    vector_store: Union["PgVector", "AtlasMongoVector"],
    inserted_ids: List[str],
    file_id: str,
) -> None:
    if not inserted_ids:
        return
    if VECTOR_DB_TYPE == VectorDBType.ATLAS_MONGO and hasattr(
        vector_store, "delete_by_file_ids"
    ):
        vector_store.delete_by_file_ids([file_id])
        return
    vector_store.delete(ids=inserted_ids)


def _prepare_documents_sync(
    data: Iterable[Document],
    file_id: str,
    user_id: str,
    clean_content: bool,
) -> List[Document]:
    """
    Synchronous document preparation - runs in executor to avoid blocking event loop.
    Handles text splitting, cleaning, and metadata preparation.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    documents = text_splitter.split_documents(data)

    # If `clean_content` is True, clean the page_content of each document (remove null bytes)
    if clean_content:
        for doc in documents:
            doc.page_content = clean_text(doc.page_content)

    # Preparing documents with traceable chunk metadata for insertion.
    prepared_documents = []
    for chunk_index, doc in enumerate(documents):
        chunk_id = uuid.uuid4().hex
        metadata = dict(doc.metadata or {})
        source_path = metadata.get("source")
        source_file = Path(str(source_path)).name if source_path else None
        metadata.update(
            {
                "file_id": file_id,
                "user_id": user_id,
                "chunk_id": chunk_id,
                "digest": generate_digest(doc.page_content),
                "chunk_index": chunk_index,
                "char_length": len(doc.page_content),
                "source_file": source_file,
            }
        )
        prepared_documents.append(
            Document(page_content=doc.page_content, metadata=metadata)
        )

    return prepared_documents


async def store_data_in_vector_db(
    data: Iterable[Document],
    file_id: str,
    user_id: str = "",
    clean_content: bool = False,
    executor=None,
) -> bool:
    # Run document preparation in executor to avoid blocking the event loop
    loop = asyncio.get_running_loop()
    docs = await loop.run_in_executor(
        executor,
        _prepare_documents_sync,
        data,
        file_id,
        user_id,
        clean_content,
    )

    try:
        if EMBEDDING_BATCH_SIZE <= 0:
            # synchronously embed the file and insert into vector store in one go
            vector_ids = _vector_ids_for_documents(docs, file_id)
            if isinstance(vector_store, AsyncPgVector):
                ids = await vector_store.aadd_documents(
                    docs, ids=vector_ids, executor=executor
                )
            else:
                ids = vector_store.add_documents(docs, ids=vector_ids)
        else:
            # asynchronously embed the file and insert into vector store as it is embedding
            # to lessen memory impact and speed up slightly as the majority of the document
            # is inserted into db by the time it is fully embedded

            if isinstance(vector_store, AsyncPgVector):
                ids = await _process_documents_async_pipeline(
                    docs, file_id, vector_store, executor
                )
            else:
                # Fallback to batched processing for sync vector stores
                ids = await _process_documents_batched_sync(
                    docs, file_id, vector_store, executor
                )

        return {"message": "Documents added successfully", "ids": ids}

    except Exception as e:
        logger.error(
            "Failed to store data in vector DB | File ID: %s | User ID: %s | Error: %s | Traceback: %s",
            file_id,
            user_id,
            str(e),
            traceback.format_exc(),
        )
        return {"message": "An error occurred while adding documents.", "error": str(e)}


@router.post("/local/embed")
async def embed_local_file(
    document: StoreDocument, request: Request, entity_id: str = None
):
    file_path = validate_file_path(RAG_UPLOAD_DIR, document.filepath)

    # Check if the file exists and if it is within the allowed upload directory
    if file_path is None or not os.path.exists(file_path):
        logger.warning("Path validation failed for local embed: %s", document.filepath)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.FILE_NOT_FOUND,
        )

    user_id = get_user_id(request, entity_id)

    loader = None
    try:
        loader, known_type, file_ext = get_loader(
            document.filename, document.file_content_type, file_path
        )
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(
            request.app.state.thread_pool, lambda: list(loader.lazy_load())
        )

        result = await store_data_in_vector_db(
            data,
            document.file_id,
            user_id,
            clean_content=file_ext == "pdf",
            executor=request.app.state.thread_pool,
        )

        if result:
            return {
                "status": True,
                "file_id": document.file_id,
                "filename": document.filename,
                "known_type": known_type,
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=ERROR_MESSAGES.DEFAULT(),
            )
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in embed_local_file | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(e)
        if "No pandoc was found" in str(e):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.PANDOC_NOT_INSTALLED,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.DEFAULT(e),
            )
    finally:
        # Clean up temporary UTF-8 file if it was created for encoding conversion
        if loader is not None:
            cleanup_temp_encoding_file(loader)


@router.post("/embed")
async def embed_file(
    request: Request,
    file_id: str = Form(...),
    file: UploadFile = File(...),
    entity_id: str = Form(None),
):
    response_status = True
    response_message = "File processed successfully."
    known_type = None

    user_id = get_user_id(request, entity_id)
    validated_file_path = _make_unique_temp_path(user_id, file.filename)

    if validated_file_path is None:
        logger.warning("Path validation failed for embed: %s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT("Invalid request"),
        )

    try:
        os.makedirs(os.path.dirname(validated_file_path), exist_ok=True)
        await save_upload_file_async(file, validated_file_path)
        data, known_type, file_ext = await load_file_content(
            file.filename,
            file.content_type,
            validated_file_path,
            request.app.state.thread_pool,
        )

        result = await store_data_in_vector_db(
            data=data,
            file_id=file_id,
            user_id=user_id,
            clean_content=file_ext == "pdf",
            executor=request.app.state.thread_pool,
        )

        if not result:
            response_status = False
            response_message = "Failed to process/store the file data."
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process/store the file data.",
            )
        elif "error" in result:
            response_status = False
            response_message = "Failed to process/store the file data."
            if isinstance(result["error"], str):
                response_message = result["error"]
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="An unspecified error occurred.",
                )
    except HTTPException as http_exc:
        response_status = False
        response_message = f"HTTP Exception: {http_exc.detail}"
        logger.error(
            "HTTP Exception in embed_file | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        response_status = False
        response_message = f"Error during file processing: {str(e)}"
        logger.error(
            "Error during file processing: %s\nTraceback: %s",
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error during file processing: {str(e)}",
        )
    finally:
        await cleanup_temp_file_async(validated_file_path)

    return {
        "status": response_status,
        "message": response_message,
        "file_id": file_id,
        "filename": file.filename,
        "known_type": known_type,
    }


@router.get("/documents/{id}/context")
async def load_document_context(request: Request, id: str):
    ids = [id]
    try:
        if isinstance(vector_store, AsyncPgVector):
            existing_ids = await vector_store.get_filtered_ids(
                ids, executor=request.app.state.thread_pool
            )
            documents = await vector_store.get_documents_by_ids(
                ids, executor=request.app.state.thread_pool
            )
        else:
            existing_ids = vector_store.get_filtered_ids(ids)
            documents = vector_store.get_documents_by_ids(ids)

        # Ensure the requested id exists
        if not all(id in existing_ids for id in ids):
            raise HTTPException(
                status_code=404, detail="The specified file_id was not found"
            )

        # Ensure documents list is not empty
        if not documents:
            raise HTTPException(
                status_code=404, detail="No document found for the given ID"
            )

        return process_documents(documents)
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in load_document_context | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error loading document context | Document ID: %s | Error: %s | Traceback: %s",
            id,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


@router.post("/embed-upload")
async def embed_file_upload(
    request: Request,
    file_id: str = Form(...),
    uploaded_file: UploadFile = File(...),
    entity_id: str = Form(None),
):
    user_id = get_user_id(request, entity_id)

    validated_temp_file_path = _make_unique_temp_path(user_id, uploaded_file.filename)

    if validated_temp_file_path is None:
        logger.warning(
            "Path validation failed for embed-upload: %s", uploaded_file.filename
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT("Invalid request"),
        )

    try:
        os.makedirs(os.path.dirname(validated_temp_file_path), exist_ok=True)
        await save_upload_file_async(uploaded_file, validated_temp_file_path)
        data, known_type, file_ext = await load_file_content(
            uploaded_file.filename,
            uploaded_file.content_type,
            validated_temp_file_path,
            request.app.state.thread_pool,
        )

        result = await store_data_in_vector_db(
            data,
            file_id,
            user_id,
            clean_content=file_ext == "pdf",
            executor=request.app.state.thread_pool,
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process/store the file data.",
            )
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in embed_file_upload | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error during file processing | File: %s | Error: %s | Traceback: %s",
            uploaded_file.filename,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error during file processing: {str(e)}",
        )
    finally:
        await cleanup_temp_file_async(validated_temp_file_path)

    return {
        "status": True,
        "message": "File processed successfully.",
        "file_id": file_id,
        "filename": uploaded_file.filename,
        "known_type": known_type,
    }


@router.post("/query_multiple")
async def query_embeddings_by_file_ids(request: Request, body: QueryMultipleBody):
    try:
        # Get the embedding of the query text
        embedding = get_cached_query_embedding(body.query)

        # Perform similarity search with the query embedding and filter by the file_ids in metadata
        if isinstance(vector_store, AsyncPgVector):
            documents = await vector_store.asimilarity_search_with_score_by_vector(
                embedding,
                k=body.k,
                filter={"file_id": {"$in": body.file_ids}},
                executor=request.app.state.thread_pool,
            )
        else:
            documents = vector_store.similarity_search_with_score_by_vector(
                embedding, k=body.k, filter={"file_id": {"$in": body.file_ids}}
            )

        documents = _apply_distance_threshold(documents)

        # Ensure documents list is not empty
        if not documents:
            raise HTTPException(
                status_code=404, detail="No documents found for the given query"
            )

        return documents
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in query_embeddings_by_file_ids | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error in query multiple embeddings | File IDs: %s | Query: %s | Error: %s | Traceback: %s",
            body.file_ids,
            body.query,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/text")
async def extract_text_from_file(
    request: Request,
    file_id: str = Form(...),
    file: UploadFile = File(...),
    entity_id: str = Form(None),
):
    """
    Extract text content from an uploaded file without creating embeddings.
    Returns the raw text content for text parsing purposes.
    """
    user_id = get_user_id(request, entity_id)
    validated_temp_file_path = _make_unique_temp_path(user_id, file.filename)

    if validated_temp_file_path is None:
        logger.warning("Path validation failed for text extraction: %s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT("Invalid request"),
        )

    try:
        os.makedirs(os.path.dirname(validated_temp_file_path), exist_ok=True)
        await save_upload_file_async(file, validated_temp_file_path)
        data, known_type, file_ext = await load_file_content(
            file.filename,
            file.content_type,
            validated_temp_file_path,
            request.app.state.thread_pool,
            raw_text=True,
        )

        # Extract text content from loaded documents
        text_content = extract_text_from_documents(data, file_ext)

        return {
            "text": text_content,
            "file_id": file_id,
            "filename": file.filename,
            "known_type": known_type,
        }

    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in extract_text_from_file | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error during text extraction | File: %s | Error: %s | Traceback: %s",
            file.filename,
            str(e),
            traceback.format_exc(),
        )
        if "No pandoc was found" in str(e):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.PANDOC_NOT_INSTALLED,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error during text extraction: {str(e)}",
            )
    finally:
        await cleanup_temp_file_async(validated_temp_file_path)
