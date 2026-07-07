#!/usr/bin/env python3
"""Prepare a small HotpotQA retrieval subset from local BEIR parquet files.

If qrels are provided, the subset is suitable for labeled retrieval evaluation.
If qrels are missing, the script still writes a smoke subset but marks it as
unlabeled, so metrics are not accidentally fabricated.

Prepared subsets are restricted-candidate evaluations, not full HotpotQA/BEIR
benchmarks.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open("r", encoding="utf-8", newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters="\t, ")
        reader = csv.reader(f, dialect)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if row[0].lower() in {"query-id", "query_id", "qid"}:
                continue
            if len(row) >= 4:
                query_id, _, doc_id, score = row[:4]
            elif len(row) >= 3:
                query_id, doc_id, score = row[:3]
            else:
                continue
            try:
                score_int = int(float(score))
            except ValueError:
                continue
            if score_int > 0:
                qrels[str(query_id)][str(doc_id)] = score_int
    return dict(qrels)


def read_parquet_rows(path: Path, columns: list[str], limit: int | None = None):
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    rows = []
    remaining = limit
    for batch in parquet_file.iter_batches(columns=columns, batch_size=10_000):
        batch_rows = batch.to_pylist()
        if remaining is None:
            rows.extend(batch_rows)
            continue
        rows.extend(batch_rows[:remaining])
        remaining -= min(remaining, len(batch_rows))
        if remaining <= 0:
            break
    return rows


def iter_rows(path: Path, columns: list[str] | None = None) -> Iterator[dict]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if columns is not None:
                    row = {column: row.get(column) for column in columns}
                yield row
        return

    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(columns=columns, batch_size=100_000):
        for row in batch.to_pylist():
            yield row


def read_rows(path: Path, columns: list[str], limit: int | None = None) -> list[dict]:
    if path.suffix == ".parquet":
        return read_parquet_rows(path, columns, limit)

    rows = []
    for row in iter_rows(path, columns):
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def read_rows_by_ids(path: Path, ids: set[str]) -> list[dict]:
    rows = []
    remaining = set(ids)
    for row in iter_rows(path, ["_id", "title", "text"]):
        row_id = str(row["_id"])
        if row_id in remaining:
            rows.append(row)
            remaining.remove(row_id)
        if not remaining:
            break
    return rows


def read_rows_by_ids_with_negatives(
    path: Path,
    ids: set[str],
    negative_limit: int,
    negative_strategy: str = "deterministic",
    seed: int = 42,
):
    if negative_strategy == "bm25_hard":
        raise NotImplementedError(
            "bm25_hard negative sampling is not implemented yet. Use random for now."
        )

    relevant_rows = []
    negative_rows = []
    remaining = set(ids)

    if negative_strategy == "random":
        rng = random.Random(seed)
        seen = 0
        reservoir_ids: set[str] = set()
        for row in iter_rows(path, ["_id", "title", "text"]):
            row_id = str(row["_id"])
            if row_id in remaining:
                relevant_rows.append(row)
                remaining.remove(row_id)
                continue
            if row_id in ids or row_id in reservoir_ids:
                continue

            seen += 1
            if len(negative_rows) < negative_limit:
                negative_rows.append(row)
                reservoir_ids.add(row_id)
                continue

            j = rng.randint(0, seen - 1)
            if j < negative_limit:
                old_id = str(negative_rows[j]["_id"])
                reservoir_ids.remove(old_id)
                negative_rows[j] = row
                reservoir_ids.add(row_id)
        return relevant_rows, negative_rows

    for row in iter_rows(path, ["_id", "title", "text"]):
        row_id = str(row["_id"])
        if row_id in remaining:
            relevant_rows.append(row)
            remaining.remove(row_id)
        elif len(negative_rows) < negative_limit and row_id not in ids:
            negative_rows.append(row)
        if not remaining and len(negative_rows) >= negative_limit:
            break
    return relevant_rows, negative_rows


def select_queries(
    all_queries: list[dict],
    qrels: dict[str, dict[str, int]],
    query_limit: int,
    sample_strategy: str,
    seed: int,
) -> list[dict]:
    candidate_queries = [query for query in all_queries if str(query["_id"]) in qrels]
    if sample_strategy == "first":
        return candidate_queries[:query_limit]

    rng = random.Random(seed)
    return rng.sample(candidate_queries, min(query_limit, len(candidate_queries)))


def resolve_data_path(path: Path, kind: str) -> Path:
    if path.exists():
        return path

    fallbacks = {
        "corpus": [
            Path("data/raw/hotpotqa/corpus/corpus-00000-of-00001.parquet"),
            Path("data/raw/hotpotqa_beir/extracted/hotpotqa/corpus.jsonl"),
        ],
        "queries": [
            Path("data/raw/hotpotqa/queries/queries-00000-of-00001.parquet"),
            Path("data/raw/hotpotqa_beir/extracted/hotpotqa/queries.jsonl"),
        ],
    }
    for fallback in fallbacks.get(kind, []):
        if fallback.exists():
            return fallback
    raise FileNotFoundError(f"{kind.capitalize()} file not found: {path}")


def build_file_id(doc_id: str, prefix: str) -> str:
    safe = str(doc_id).replace("/", "_").replace("\\", "_")
    return f"{prefix}{safe}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        default="data/raw/hotpotqa/corpus/corpus-00000-of-00001.parquet",
    )
    parser.add_argument(
        "--queries",
        default="data/raw/hotpotqa/queries/queries-00000-of-00001.parquet",
    )
    parser.add_argument("--qrels", default=None)
    parser.add_argument("--out-dir", default="data/derived/hotpotqa/day09_subset")
    parser.add_argument("--query-limit", type=int, default=20)
    parser.add_argument("--smoke-corpus-limit", type=int, default=50)
    parser.add_argument(
        "--negative-corpus-limit",
        type=int,
        default=0,
        help="When qrels are provided, add this many non-relevant corpus rows as distractors.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-strategy", choices=["first", "random"], default="random")
    parser.add_argument(
        "--negative-strategy",
        choices=["deterministic", "random", "bm25_hard"],
        default="random",
    )
    parser.add_argument("--negative-seed", type=int, default=None)
    parser.add_argument("--file-id-prefix", default="hotpotqa-")
    args = parser.parse_args()

    corpus_path = resolve_data_path(Path(args.corpus), "corpus")
    queries_path = resolve_data_path(Path(args.queries), "queries")
    out_dir = Path(args.out_dir)
    qrels_path = Path(args.qrels) if args.qrels else None
    negative_seed = args.negative_seed if args.negative_seed is not None else args.seed

    has_qrels = qrels_path is not None and qrels_path.exists()
    qrels = load_qrels(qrels_path) if has_qrels else {}

    all_queries = read_rows(queries_path, ["_id", "title", "text"])

    if has_qrels:
        selected_queries = select_queries(
            all_queries, qrels, args.query_limit, args.sample_strategy, args.seed
        )
        relevant_doc_ids = {
            doc_id
            for query in selected_queries
            for doc_id in qrels.get(str(query["_id"]), {})
        }
        if args.negative_corpus_limit > 0:
            relevant_docs, negative_docs = read_rows_by_ids_with_negatives(
                corpus_path,
                relevant_doc_ids,
                args.negative_corpus_limit,
                args.negative_strategy,
                negative_seed,
            )
            selected_docs = relevant_docs + negative_docs
        else:
            relevant_docs = read_rows_by_ids(corpus_path, relevant_doc_ids)
            negative_docs = []
            selected_docs = relevant_docs
        selected_qrels = [
            {
                "query_id": str(query["_id"]),
                "doc_id": str(doc_id),
                "score": score,
                "file_id": build_file_id(str(doc_id), args.file_id_prefix),
            }
            for query in selected_queries
            for doc_id, score in qrels.get(str(query["_id"]), {}).items()
        ]
    else:
        if args.sample_strategy == "first":
            selected_queries = all_queries[: args.query_limit]
        else:
            rng = random.Random(args.seed)
            selected_queries = rng.sample(
                all_queries, min(args.query_limit, len(all_queries))
            )
        selected_docs = read_rows(corpus_path, ["_id", "title", "text"], limit=args.smoke_corpus_limit)
        relevant_docs = []
        negative_docs = []
        selected_qrels = []

    corpus_rows = [
        {
            "doc_id": str(row["_id"]),
            "file_id": build_file_id(str(row["_id"]), args.file_id_prefix),
            "title": row.get("title") or "",
            "text": row.get("text") or "",
        }
        for row in selected_docs
    ]
    query_rows = [
        {
            "query_id": str(row["_id"]),
            "title": row.get("title") or "",
            "text": row.get("text") or "",
        }
        for row in selected_queries
    ]

    corpus_count = write_jsonl(out_dir / "corpus.jsonl", corpus_rows)
    query_count = write_jsonl(out_dir / "queries.jsonl", query_rows)
    qrels_count = write_jsonl(out_dir / "qrels.jsonl", selected_qrels)

    evaluation_type = (
        "strict_subset_eval"
        if has_qrels
        and args.query_limit >= 100
        and args.negative_corpus_limit >= 5000
        else "labeled_smoke_test"
        if has_qrels
        else "unlabeled_smoke_test"
    )
    manifest = {
        "dataset": "BeIR/hotpotqa",
        "subset_dir": str(out_dir),
        "has_qrels": has_qrels,
        "corpus_rows": corpus_count,
        "relevant_corpus_rows": len(relevant_docs),
        "negative_corpus_rows": len(negative_docs),
        "query_rows": query_count,
        "qrels_rows": qrels_count,
        "file_id_prefix": args.file_id_prefix,
        "sample_strategy": args.sample_strategy,
        "negative_strategy": args.negative_strategy,
        "seed": args.seed,
        "negative_seed": negative_seed,
        "is_full_corpus_eval": False,
        "is_benchmark_claim": False,
        "evaluation_type": evaluation_type,
        "notes": (
            "Qrels are available. This subset can be used for labeled retrieval evaluation, "
            "but it is not a full HotpotQA/BEIR benchmark unless full corpus retrieval is used."
            if has_qrels
            else "qrels missing: this is an unlabeled smoke subset, not a Recall@K eval set."
        ),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
