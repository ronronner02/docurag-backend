#!/usr/bin/env python3
"""Legacy Hit@K-style evaluator for a prepared HotpotQA subset.

Use ``evaluate_hotpotqa_retrieval.py`` for separated Hit@K, Recall@K,
All-Support@K, MRR, and BEIR-style IR metrics.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

try:
    from .smoke_embeddings import post_json
except ImportError:
    from smoke_embeddings import post_json


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_recall_at_k(
    retrieved_by_query: dict[str, list[str]],
    relevant_by_query: dict[str, set[str]],
    ks: list[int],
) -> dict[str, float]:
    if not relevant_by_query:
        return {}
    metrics = {}
    total = len(relevant_by_query)
    for k in ks:
        hits = 0
        for query_id, relevant_file_ids in relevant_by_query.items():
            retrieved = retrieved_by_query.get(query_id, [])[:k]
            if relevant_file_ids.intersection(retrieved):
                hits += 1
        metrics[f"recall@{k}"] = hits / total
    return metrics


def parse_query_multiple_response(body) -> list[str]:
    file_ids: list[str] = []
    if not isinstance(body, list):
        return file_ids
    for item in body:
        if not isinstance(item, list) or not item:
            continue
        document = item[0]
        if not isinstance(document, dict):
            continue
        metadata = document.get("metadata") or {}
        file_id = metadata.get("file_id")
        if file_id:
            file_ids.append(str(file_id))
    return file_ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-dir", default="data/derived/hotpotqa/day09_subset")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--ks", default="1,3,5")
    parser.add_argument("--auth-token", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    subset_dir = Path(args.subset_dir)
    queries = read_jsonl(subset_dir / "queries.jsonl")
    corpus = read_jsonl(subset_dir / "corpus.jsonl")
    qrels = read_jsonl(subset_dir / "qrels.jsonl")
    ks = [int(k.strip()) for k in args.ks.split(",") if k.strip()]

    if not qrels:
        report = {
            "ok": False,
            "reason": "qrels.jsonl is missing or empty; cannot compute Recall@K.",
            "subset_dir": str(subset_dir),
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2

    candidate_file_ids = [row["file_id"] for row in corpus]
    max_k = max(ks)
    relevant_by_query: dict[str, set[str]] = defaultdict(set)
    for row in qrels:
        relevant_by_query[row["query_id"]].add(row["file_id"])

    retrieved_by_query: dict[str, list[str]] = {}
    details = []

    for query in queries:
        payload = {
            "query": query["text"],
            "file_ids": candidate_file_ids,
            "k": max_k,
        }
        try:
            status, body = post_json(
                args.base_url,
                "/query_multiple",
                payload,
                args.timeout,
                args.auth_token,
            )
            if status != 200:
                raise RuntimeError(f"HTTP {status}: {body}")
            retrieved = parse_query_multiple_response(body)
        except Exception as exc:
            retrieved = []
            body = {"error": str(exc)}
        retrieved_by_query[query["query_id"]] = retrieved
        details.append(
            {
                "query_id": query["query_id"],
                "query": query["text"],
                "relevant_file_ids": sorted(relevant_by_query.get(query["query_id"], set())),
                "retrieved_file_ids": retrieved,
            }
        )

    metrics = compute_recall_at_k(retrieved_by_query, relevant_by_query, ks)
    report = {
        "ok": True,
        "subset_dir": str(subset_dir),
        "query_count": len(queries),
        "candidate_file_count": len(candidate_file_ids),
        "qrels_query_count": len(relevant_by_query),
        "ks": ks,
        "metrics": metrics,
        "details": details,
    }
    if args.report:
        Path(args.report).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
