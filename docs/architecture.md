# Architecture

DocuRAG Backend is organized around a document ingestion path and a retrieval / answer path.

## Ingestion Path

```text
POST /embed or /embed-upload
-> save uploaded file under RAG_UPLOAD_DIR
-> validate path and isolate upload
-> choose document loader
-> parse file into Document objects
-> split into chunks
-> enrich metadata
-> generate embeddings
-> store text, vectors, and metadata in pgvector
```

Important metadata fields:

```text
file_id
user_id
chunk_id
digest
chunk_index
char_length
source_file
page
```

The metadata makes retrieved chunks traceable and enables file-level, user-level, and prefix-level filtering.

## Retrieval Path

Single-document retrieval:

```text
POST /retrieval/search
-> embed query
-> pgvector similarity search
-> metadata filter: file_id
-> permission filter
-> structured RetrievalSearchResponse
```

Global retrieval:

```text
POST /retrieval/search_global
-> embed query
-> optional file_id_prefix filter
-> pgvector similarity search
-> structured GlobalRetrievalSearchResponse
```

Prefix-scoped global retrieval is used when a loaded vector store contains multiple datasets or experiments and an evaluation must be isolated without deleting historical data.

## RAG Answer Path

```text
POST /rag/chat or /rag/chat_global
-> retrieve top-k chunks
-> build RagSource objects
-> build grounded prompt
-> optional OpenAI-compatible LLM call
-> answer + sources + refusal + answer_strategy
```

Answer strategies:

```text
extractive_context_v1
llm_grounded_context_v1
no_context_refusal
low_confidence_refusal
```

If no context is retrieved, the service refuses immediately. If the LLM judges the context insufficient, the response is marked as a low-confidence refusal. If the LLM layer is disabled or fails, the service falls back to an extractive answer based on returned sources.

## Main Modules

| Module | Responsibility |
|---|---|
| `main.py` | FastAPI app creation, middleware, lifespan setup, router registration |
| `app/config.py` | Environment config, embedding provider initialization, vector store initialization |
| `app/routes/document_routes.py` | Upload, text extraction, retrieval, RAG endpoints |
| `app/services/rag.py` | Sources, prompt construction, refusal detection, answer assembly |
| `app/services/llm.py` | Optional OpenAI-compatible LLM client |
| `app/services/vector_store/` | pgvector and Atlas vector store adapters |
| `app/db_models.py` | Business-level SQLAlchemy models for documents, chunks, and chat messages |
| `scripts/` | Smoke tests and HotpotQA / BEIR evaluation utilities |

## Deployment Shape

```text
Client
-> FastAPI container
-> PostgreSQL + pgvector container
-> optional external LLM endpoint
```

The default Docker Compose setup starts both FastAPI and pgvector. Hugging Face model files and uploaded files are mounted into local runtime directories that are ignored by Git.
