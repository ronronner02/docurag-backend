# Deployment

## Docker Compose

Copy the example environment file:

```bash
cp .env.example .env
```

Start the API and pgvector:

```bash
docker compose up --build
```

The API listens on:

```text
http://127.0.0.1:8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Database Only

To run only PostgreSQL + pgvector:

```bash
docker compose -f db-compose.yaml up
```

Then run the API locally with:

```env
DB_HOST=127.0.0.1
DB_PORT=5433
```

## Local Python Runtime

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Hugging Face Embeddings

The default `.env.example` uses:

```env
EMBEDDINGS_PROVIDER=huggingface
EMBEDDINGS_MODEL=BAAI/bge-m3
HF_HOME=./data/huggingface
```

The first run downloads model files into `data/huggingface`, which is ignored by Git.

## Optional LLM Layer

The LLM layer is disabled by default:

```env
RAG_LLM_PROVIDER=disabled
```

To enable OpenAI-compatible chat completion:

```env
RAG_LLM_PROVIDER=openai_compatible
RAG_LLM_MODEL=<chat-model>
RAG_LLM_API_KEY=<api-key>
RAG_LLM_BASE_URL=<optional-base-url>
RAG_LLM_TEMPERATURE=0
RAG_LLM_MAX_OUTPUT_TOKENS=512
```

The LLM only receives retrieved source snippets, not the full vector store.

## Runtime Directories

These directories are generated locally and ignored by Git:

```text
data/
uploads/
.venv/
```

Do not publish downloaded datasets, user uploads, model caches, or real `.env` files.
