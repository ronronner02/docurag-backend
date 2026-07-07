# Evaluation

This project includes retrieval and answer-layer evaluation scripts for HotpotQA / BEIR subset experiments.

## Dataset Boundary

The project metrics are reported on a strict subset:

```text
Dataset: BEIR HotpotQA
Queries: 100
Candidate docs: 5200
Relevant support docs: 200
Random negatives: 5000
Embedding: BAAI/bge-m3
Vector store: PostgreSQL + pgvector
```

This is not a full HotpotQA / BEIR full-corpus benchmark.

## Retrieval Evaluation

The retrieval script can evaluate either restricted candidate mode or loaded-vector-store global mode.

Prefix-scoped global retrieval example:

```bash
python scripts/evaluate_hotpotqa_retrieval.py \
  --subset-dir data/derived/hotpotqa/day11_labeled_100_neg5000_random \
  --base-url http://127.0.0.1:8000 \
  --ks 1,3,5,10 \
  --global-search \
  --global-file-id-prefix hotpotqa-day11- \
  --output data/derived/hotpotqa/day11_labeled_100_neg5000_random/retrieval_report.json
```

Project retrieval metrics:

```text
Recall@5 = 96.5%
All-Support@5 = 93.0%
Recall@10 = 98.0%
All-Support@10 = 96.0%
NDCG@5 = 96.56%
MAP@10 = 95.44%
```

## Answer-Layer Evaluation

The answer-layer script calls `/rag/chat_global` and measures source grounding behavior.

Example:

```bash
python scripts/evaluate_hotpotqa_rag_answers.py \
  --subset-dir data/derived/hotpotqa/day11_labeled_100_neg5000_random \
  --base-url http://127.0.0.1:8000 \
  --k 5 \
  --limit 100 \
  --max-context-chars 2200 \
  --use-llm \
  --global-file-id-prefix hotpotqa-day11- \
  --output data/derived/hotpotqa/day11_labeled_100_neg5000_random/rag_answer_report.json
```

Answer-layer metrics:

```text
support-source hit rate = 100%
all-support-in-sources rate = 93.0%
citation marker rate = 100%
LLM grounded answer rate = 92.0%
low-confidence refusal rate = 8.0%
answer_error_count = 0
average source count = 4.8
```

These metrics evaluate whether returned sources overlap qrels and whether answers include citation markers. They do not compute official answer exact match or token F1.

## Refusal Evaluation

Refusal evaluation checks whether the RAG layer refuses when context is missing or insufficient.

```bash
python scripts/evaluate_rag_refusals.py \
  --base-url http://127.0.0.1:8000 \
  --use-llm \
  --global-file-id-prefix hotpotqa-day11-
```

The refusal smoke test is a safety check, not an official HotpotQA metric.

## Metric Definitions

| Metric | Meaning |
|---|---|
| `Recall@K` | Fraction of qrels support docs retrieved in the Top-K results |
| `All-Support@K` | Whether all support docs for a query appear in Top-K |
| `NDCG@K` | Ranking quality with stronger weight for earlier relevant results |
| `MAP@10` | Mean average precision over Top-10 |
| `support-source hit rate` | Whether returned answer sources include at least one qrels support doc |
| `all-support-in-sources rate` | Whether returned answer sources include all qrels support docs |
| `citation marker rate` | Whether the answer text includes `[source n]` citation markers |
| `low-confidence refusal rate` | Fraction of calls where the model judged retrieved context insufficient |

## Reporting Rule

Correct:

```text
On a BEIR HotpotQA 100-query / 5200-doc strict subset, prefix-scoped global retrieval reached Recall@5=96.5% and All-Support@5=93.0%.
```

Incorrect:

```text
Full HotpotQA / BEIR benchmark Recall@5=96.5%.
```
