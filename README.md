# DocuRAG Backend

DocuRAG Backend is a FastAPI-based retrieval-augmented generation (RAG) service for document ingestion, vector search, source-grounded answers, and retrieval evaluation.

The project focuses on an end-to-end backend pipeline:

```text
document upload
-> document loader
-> chunk splitting and metadata enrichment
-> embedding generation
-> PostgreSQL / pgvector storage
-> scoped or global vector retrieval
-> source-grounded RAG answer generation
-> retrieval and answer-layer evaluation
```

## Features

- FastAPI backend with typed Pydantic request and response models.
- Document ingestion with path validation and upload isolation.
- Chunk metadata for traceability: `file_id`, `user_id`, `chunk_id`, `digest`, `chunk_index`, `source_file`, and `page`.
- Configurable embedding providers, with local Hugging Face embeddings supported by default in `.env.example`.
- PostgreSQL + pgvector vector store with async wrappers and metadata filtering.
- Structured retrieval APIs for single-file and global knowledge-base search.
- Prefix-scoped global retrieval for clean evaluation boundaries.
- RAG chat APIs returning `answer`, `sources`, `refusal`, and `answer_strategy`.
- Optional OpenAI-compatible LLM generation layer for grounded answers with `[source n]` citations.
- Extractive fallback and no-context / low-confidence refusal behavior.
- HotpotQA / BEIR subset evaluation scripts for retrieval and answer-layer grounding metrics.
- Docker Compose setup for local API + pgvector development.

## Architecture

```text
Client
-> FastAPI app
-> /embed or /embed-upload
-> Loader
-> RecursiveCharacterTextSplitter
-> Embedding provider
-> pgvector
-> /retrieval/search or /retrieval/search_global
-> /rag/chat or /rag/chat_global
-> answer + sources + refusal
```

See [docs/architecture.md](docs/architecture.md) for a fuller architecture walkthrough.

## Quick Start

### 1. Configure environment

Copy the example environment file:

```bash
cp .env.example .env
```

The example configuration uses:

```env
VECTOR_DB_TYPE=pgvector
EMBEDDINGS_PROVIDER=huggingface
EMBEDDINGS_MODEL=BAAI/bge-m3
RAG_LLM_PROVIDER=disabled
```

To enable LLM answer generation, set the optional `RAG_LLM_*` variables in your local `.env`. Do not commit real API keys.

### 2. Start with Docker Compose

```bash
docker compose up --build
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

### 3. Run locally

Use Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

If you run the API locally while pgvector runs in Docker, set:

```env
DB_HOST=127.0.0.1
DB_PORT=5433
```

## Core APIs

### Upload and embed a document

```bash
curl -X POST "http://127.0.0.1:8000/embed" \
  -F "file_id=demo-doc-1" \
  -F "file=@examples/demo.txt;type=text/plain"
```

### Search within one document

```bash
curl -X POST "http://127.0.0.1:8000/retrieval/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is this document about?",
    "file_id": "demo-doc-1",
    "k": 5
  }'
```

### Search globally

```bash
curl -X POST "http://127.0.0.1:8000/retrieval/search_global" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What does the knowledge base say?",
    "k": 5
  }'
```

### Search globally within a prefix scope

```bash
curl -X POST "http://127.0.0.1:8000/retrieval/search_global" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What does the evaluation subset say?",
    "k": 5,
    "file_id_prefix": "hotpotqa-day11-"
  }'
```

### RAG chat

```bash
curl -X POST "http://127.0.0.1:8000/rag/chat_global" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Answer using retrieved sources.",
    "k": 5,
    "max_context_chars": 2200,
    "use_llm": true
  }'
```

The response includes:

```text
answer
refusal
answer_strategy
used_context_count
sources[]
```

## Evaluation

This repository includes scripts for HotpotQA / BEIR subset evaluation:

- `scripts/hotpotqa_prepare_subset.py`
- `scripts/hotpotqa_import_subset.py`
- `scripts/evaluate_hotpotqa_retrieval.py`
- `scripts/evaluate_hotpotqa_rag_answers.py`
- `scripts/evaluate_rag_refusals.py`

The reported project metrics are based on a strict subset, not the full HotpotQA / BEIR benchmark:

```text
Dataset: BEIR HotpotQA
Queries: 100
Candidate docs: 5200
Relevant support docs: 200
Random negatives: 5000
Embedding: BAAI/bge-m3
Vector store: PostgreSQL + pgvector
```

Final subset metrics:

```text
Prefix-scoped global retrieval:
Recall@5 = 96.5%
All-Support@5 = 93.0%
Recall@10 = 98.0%
All-Support@10 = 96.0%

100-query answer-layer evaluation:
support-source hit rate = 100%
all-support-in-sources rate = 93.0%
citation marker rate = 100%
LLM grounded answer rate = 92.0%
low-confidence refusal rate = 8.0%
answer_error_count = 0
```

These are subset grounding and retrieval metrics. They are not official full-corpus HotpotQA / BEIR benchmark results and are not official answer EM/F1 scores.

See [docs/evaluation.md](docs/evaluation.md) for commands and metric definitions.

## Tests

Install test dependencies:

```bash
pip install -r test_requirements.txt
```

Run the test suite:

```bash
pytest
```

Run selected tests:

```bash
pytest tests/test_main.py tests/services/test_rag.py tests/test_hotpotqa_eval_scripts.py -q
```

## Security Notes

- Never commit `.env`, API keys, model service credentials, uploaded files, or downloaded datasets.
- `data/` and `uploads/` are runtime directories and are ignored by Git.
- Use `file_id`, `entity_id`, and metadata filters to keep retrieval scoped to authorized documents.
- The LLM layer is optional; embeddings and retrieval can run locally without an LLM API key.
- If an LLM is enabled, only retrieved source snippets are sent to the model provider.

## License

See [LICENSE](LICENSE).
