# API Examples

These examples assume the API is running at `http://127.0.0.1:8000`.

## Health Check

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status": "UP"}
```

## Upload And Embed

```bash
curl -X POST "http://127.0.0.1:8000/embed" \
  -F "file_id=demo-doc-1" \
  -F "file=@examples/demo.txt;type=text/plain"
```

## Single-File Retrieval

```bash
curl -X POST "http://127.0.0.1:8000/retrieval/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is this document about?",
    "file_id": "demo-doc-1",
    "k": 5
  }'
```

## Global Retrieval

```bash
curl -X POST "http://127.0.0.1:8000/retrieval/search_global" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What does the knowledge base say?",
    "k": 5
  }'
```

## Prefix-Scoped Global Retrieval

```bash
curl -X POST "http://127.0.0.1:8000/retrieval/search_global" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What does this evaluation subset contain?",
    "k": 5,
    "file_id_prefix": "hotpotqa-day11-"
  }'
```

## Single-File RAG Chat

```bash
curl -X POST "http://127.0.0.1:8000/rag/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Answer with citations from this document.",
    "file_id": "demo-doc-1",
    "k": 5,
    "max_context_chars": 2200,
    "use_llm": false
  }'
```

## Global RAG Chat

```bash
curl -X POST "http://127.0.0.1:8000/rag/chat_global" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Answer with citations from the knowledge base.",
    "k": 5,
    "max_context_chars": 2200,
    "use_llm": true
  }'
```

The response includes:

```json
{
  "query": "...",
  "answer": "... [source 1]",
  "refusal": false,
  "answer_strategy": "llm_grounded_context_v1",
  "used_context_count": 5,
  "sources": []
}
```

## LLM Configuration

The LLM layer is disabled by default. To enable it, set the following in your local `.env`:

```env
RAG_LLM_PROVIDER=openai_compatible
RAG_LLM_MODEL=<chat-model>
RAG_LLM_API_KEY=<api-key>
RAG_LLM_BASE_URL=<optional-base-url>
RAG_LLM_TEMPERATURE=0
RAG_LLM_MAX_OUTPUT_TOKENS=512
```

Do not commit a real `.env` file.
