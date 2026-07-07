#!/usr/bin/env python3
"""Evaluate HotpotQA retrieval on a prepared subset.

By default this script uses ``/query_multiple`` with a provided candidate
file_id set. With ``--global-search`` it uses ``/retrieval/search_global`` and
does not pass candidate file_ids. Both modes report their boundary explicitly.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .smoke_embeddings import post_json
except ImportError:
    from smoke_embeddings import post_json


RESTRICTED_LIMITATIONS = [
    "This is not a full HotpotQA/BEIR benchmark.",
    "Candidate documents are restricted by file_ids passed to /query_multiple.",
    "Negatives are sampled from a subset, not guaranteed hard negatives.",
    "Metrics should not be reported as full-corpus benchmark performance.",
]

GLOBAL_SEARCH_LIMITATIONS = [
    "This is not a full HotpotQA/BEIR benchmark unless the vector store contains the full corpus.",
    "Global search runs against the currently loaded vector store collection.",
    "The subset manifest is still used only for queries, qrels, and evidence accounting.",
    "Metrics should be reported as loaded-vector-store performance, not official BEIR performance.",
]

PREFIX_GLOBAL_SEARCH_LIMITATIONS = [
    "This is not a full HotpotQA/BEIR benchmark unless the prefix covers the full corpus.",
    "Global search is scoped by file_id_prefix, so other loaded documents are excluded.",
    "The subset manifest is still used only for queries, qrels, and evidence accounting.",
    "Metrics should be reported as prefix-scoped loaded-vector-store performance, not official BEIR performance.",
]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_ks(raw: str) -> list[int]:
    return sorted({int(k.strip()) for k in raw.split(",") if k.strip()})


def candidate_mode_for_args(args: argparse.Namespace) -> str:
    if args.global_search and args.global_file_id_prefix:
        return "global_vector_search_prefix"
    if args.global_search:
        return "global_vector_search"
    return "restricted_file_ids"


def limitations_for_args(args: argparse.Namespace) -> list[str]:
    if args.global_search and args.global_file_id_prefix:
        return PREFIX_GLOBAL_SEARCH_LIMITATIONS
    if args.global_search:
        return GLOBAL_SEARCH_LIMITATIONS
    return RESTRICTED_LIMITATIONS


def build_global_search_payload(
    query_text: str,
    k: int,
    entity_id: str | None = None,
    file_id_prefix: str | None = None,
) -> dict:
    payload = {
        "query": query_text,
        "k": k,
    }
    if entity_id:
        payload["entity_id"] = entity_id
    if file_id_prefix:
        payload["file_id_prefix"] = file_id_prefix
    return payload


def build_relevance_maps(qrels_rows: list[dict]) -> tuple[dict[str, dict[str, int]], dict[str, set[str]]]:
    qrels_by_query: dict[str, dict[str, int]] = defaultdict(dict)
    relevant_by_query: dict[str, set[str]] = defaultdict(set)
    for row in qrels_rows:
        query_id = str(row["query_id"])
        doc_key = str(row.get("file_id") or row.get("doc_id"))
        score = int(row.get("score", 1))
        if score <= 0:
            continue
        qrels_by_query[query_id][doc_key] = score
        relevant_by_query[query_id].add(doc_key)
    return dict(qrels_by_query), dict(relevant_by_query)


def score_for_eval(raw_score: Any, rank: int) -> float:
    try:
        return -float(raw_score)
    except (TypeError, ValueError):
        return -float(rank)


def parse_query_multiple_response(body: Any) -> list[dict]:
    results: list[dict] = []
    seen_file_ids: set[str] = set()
    if not isinstance(body, list):
        return results

    for rank, item in enumerate(body, start=1):
        if not isinstance(item, list) or not item:
            continue
        document = item[0]
        raw_score = item[1] if len(item) > 1 else None
        if not isinstance(document, dict):
            continue
        metadata = document.get("metadata") or {}
        file_id = metadata.get("file_id")
        if not file_id:
            continue
        file_id = str(file_id)
        if file_id in seen_file_ids:
            continue
        seen_file_ids.add(file_id)
        results.append(
            {
                "rank": rank,
                "file_id": file_id,
                "raw_score": raw_score,
                "score_for_eval": score_for_eval(raw_score, rank),
            }
        )
    return results


def parse_retrieval_search_response(body: Any) -> list[dict]:
    results: list[dict] = []
    seen_file_ids: set[str] = set()
    if not isinstance(body, dict):
        return results

    for index, item in enumerate(body.get("results", []), start=1):
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata") or {}
        file_id = item.get("file_id") or metadata.get("file_id")
        if not file_id:
            continue
        file_id = str(file_id)
        if file_id in seen_file_ids:
            continue
        seen_file_ids.add(file_id)
        rank = int(item.get("rank") or index)
        raw_score = item.get("score")
        results.append(
            {
                "rank": rank,
                "file_id": file_id,
                "raw_score": raw_score,
                "score_for_eval": score_for_eval(raw_score, rank),
            }
        )
    return results


def build_run(retrieved_by_query: dict[str, list[dict]]) -> dict[str, dict[str, float]]:
    run: dict[str, dict[str, float]] = {}
    for query_id, results in retrieved_by_query.items():
        run[query_id] = {
            result["file_id"]: float(result["score_for_eval"]) for result in results
        }
    return run


def dcg(relevances: list[int]) -> float:
    return sum((2**rel - 1) / math.log2(rank + 1) for rank, rel in enumerate(relevances, start=1))


def precision_at_k(ranked_doc_ids: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(ranked_doc_ids[:k]) & relevant) / k


def recall_at_k(ranked_doc_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked_doc_ids[:k]) & relevant) / len(relevant)


def ndcg_at_k(
    ranked_doc_ids: list[str], qrels_for_query: dict[str, int], k: int
) -> float:
    if not qrels_for_query:
        return 0.0
    relevances = [qrels_for_query.get(doc_id, 0) for doc_id in ranked_doc_ids[:k]]
    ideal_relevances = sorted(qrels_for_query.values(), reverse=True)[:k]
    ideal = dcg(ideal_relevances)
    if ideal == 0:
        return 0.0
    return dcg(relevances) / ideal


def average_precision_at_k(
    ranked_doc_ids: list[str], relevant: set[str], k: int
) -> float:
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / min(len(relevant), k)


def aggregate(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def compute_project_metrics(
    ranked_by_query: dict[str, list[str]],
    relevant_by_query: dict[str, set[str]],
    ks: list[int],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    query_ids = sorted(relevant_by_query)
    for k in ks:
        hits = []
        recalls = []
        all_supports = []
        reciprocal_ranks = []
        for query_id in query_ids:
            relevant = relevant_by_query[query_id]
            ranked = ranked_by_query.get(query_id, [])
            top_k = set(ranked[:k])
            overlap = top_k & relevant
            hits.append(1.0 if overlap else 0.0)
            recalls.append(len(overlap) / len(relevant) if relevant else 0.0)
            all_supports.append(1.0 if relevant <= top_k else 0.0)
            reciprocal_rank = 0.0
            for rank, doc_id in enumerate(ranked[:k], start=1):
                if doc_id in relevant:
                    reciprocal_rank = 1.0 / rank
                    break
            reciprocal_ranks.append(reciprocal_rank)

        metrics[f"hit@{k}"] = aggregate(hits)
        metrics[f"recall@{k}"] = aggregate(recalls)
        metrics[f"all_support@{k}"] = aggregate(all_supports)
        metrics[f"mrr@{k}"] = aggregate(reciprocal_ranks)
    return metrics


def compute_official_metrics_fallback(
    qrels_by_query: dict[str, dict[str, int]],
    ranked_by_query: dict[str, list[str]],
    ks: list[int],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    query_ids = sorted(qrels_by_query)
    for k in ks:
        metrics[f"ndcg_cut_{k}"] = aggregate(
            [ndcg_at_k(ranked_by_query.get(qid, []), qrels_by_query[qid], k) for qid in query_ids]
        )
        metrics[f"recall_{k}"] = aggregate(
            [
                recall_at_k(ranked_by_query.get(qid, []), set(qrels_by_query[qid]), k)
                for qid in query_ids
            ]
        )
        metrics[f"P_{k}"] = aggregate(
            [
                precision_at_k(ranked_by_query.get(qid, []), set(qrels_by_query[qid]), k)
                for qid in query_ids
            ]
        )
    if 10 in ks:
        metrics["map_cut_10"] = aggregate(
            [
                average_precision_at_k(ranked_by_query.get(qid, []), set(qrels_by_query[qid]), 10)
                for qid in query_ids
            ]
        )
    return metrics


def normalize_pytrec_metric_name(name: str) -> str:
    return name.replace(".", "_")


def compute_official_metrics_pytrec(
    qrels_by_query: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    ks: list[int],
) -> dict[str, float]:
    import pytrec_eval

    metric_names = set()
    for k in ks:
        metric_names.add(f"ndcg_cut.{k}")
        metric_names.add(f"recall.{k}")
        metric_names.add(f"P.{k}")
    if 10 in ks:
        metric_names.add("map_cut.10")

    evaluator = pytrec_eval.RelevanceEvaluator(qrels_by_query, metric_names)
    per_query = evaluator.evaluate(run)
    metrics = {}
    for metric_name in sorted(metric_names):
        metrics[normalize_pytrec_metric_name(metric_name)] = aggregate(
            [query_metrics.get(metric_name, 0.0) for query_metrics in per_query.values()]
        )
    return metrics


def compute_official_metrics(
    qrels_by_query: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    ranked_by_query: dict[str, list[str]],
    ks: list[int],
    use_pytrec_eval: bool,
) -> tuple[dict[str, float], str]:
    if use_pytrec_eval:
        try:
            return compute_official_metrics_pytrec(qrels_by_query, run, ks), "pytrec_eval"
        except ImportError:
            pass
    return compute_official_metrics_fallback(qrels_by_query, ranked_by_query, ks), "python_fallback"


def evaluate(args: argparse.Namespace) -> tuple[dict, int]:
    subset_dir = Path(args.subset_dir)
    queries = read_jsonl(subset_dir / "queries.jsonl")
    corpus = read_jsonl(subset_dir / "corpus.jsonl")
    qrels_rows = read_jsonl(subset_dir / "qrels.jsonl")
    manifest = read_json(subset_dir / "manifest.json")
    ks = parse_ks(args.ks)

    if args.global_file_id_prefix and not args.global_search:
        return (
            {
                "ok": False,
                "reason": "--global-file-id-prefix requires --global-search.",
                "subset_dir": str(subset_dir),
                "candidate_mode": candidate_mode_for_args(args),
                "limitations": limitations_for_args(args),
            },
            2,
        )

    if not qrels_rows:
        return (
            {
                "ok": False,
                "reason": "qrels.jsonl is missing or empty; cannot compute retrieval metrics.",
                "subset_dir": str(subset_dir),
                "candidate_mode": candidate_mode_for_args(args),
                "limitations": limitations_for_args(args),
            },
            2,
        )

    candidate_file_ids = [row["file_id"] for row in corpus]
    max_k = max(ks)
    qrels_by_query, relevant_by_query = build_relevance_maps(qrels_rows)
    retrieved_by_query: dict[str, list[dict]] = {}
    details = []

    for query in queries:
        query_id = str(query["query_id"])
        if args.global_search:
            endpoint = "/retrieval/search_global"
            payload = build_global_search_payload(
                query_text=query["text"],
                k=max_k,
                entity_id=args.entity_id,
                file_id_prefix=args.global_file_id_prefix,
            )
        else:
            endpoint = "/query_multiple"
            payload = {
                "query": query["text"],
                "file_ids": candidate_file_ids,
                "k": max_k,
            }
        try:
            status, body = post_json(
                args.base_url,
                endpoint,
                payload,
                args.timeout,
                args.auth_token,
            )
            if status != 200:
                raise RuntimeError(f"HTTP {status}: {body}")
            if args.global_search:
                retrieved = parse_retrieval_search_response(body)
            else:
                retrieved = parse_query_multiple_response(body)
        except Exception as exc:
            retrieved = []
            body = {"error": str(exc)}

        retrieved_by_query[query_id] = retrieved
        details.append(
            {
                "query_id": query_id,
                "query": query["text"],
                "relevant_file_ids": sorted(relevant_by_query.get(query_id, set())),
                "retrieved_file_ids": [result["file_id"] for result in retrieved],
                "raw_response_error": body if isinstance(body, dict) and "error" in body else None,
            }
        )

    ranked_by_query = {
        query_id: [result["file_id"] for result in results]
        for query_id, results in retrieved_by_query.items()
    }
    run = build_run(retrieved_by_query)
    official_metrics, official_backend = compute_official_metrics(
        qrels_by_query, run, ranked_by_query, ks, args.use_pytrec_eval
    )
    project_metrics = compute_project_metrics(ranked_by_query, relevant_by_query, ks)

    report = {
        "ok": True,
        "evaluation_type": manifest.get("evaluation_type", "labeled_smoke_test"),
        "candidate_mode": candidate_mode_for_args(args),
        "retrieval_endpoint": (
            "/retrieval/search_global" if args.global_search else "/query_multiple"
        ),
        "global_file_id_prefix": (
            args.global_file_id_prefix if args.global_search else None
        ),
        "query_count": len(queries),
        "candidate_file_count": None if args.global_search else len(candidate_file_ids),
        "subset_candidate_file_count": len(candidate_file_ids),
        "qrels_query_count": len(relevant_by_query),
        "ks": ks,
        "official_ir_metrics_backend": official_backend,
        "official_ir_metrics": official_metrics,
        "project_metrics": project_metrics,
        "limitations": limitations_for_args(args),
        "details": details,
    }
    return report, 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-dir", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--ks", default="1,3,5,10")
    parser.add_argument("--output", default=None)
    parser.add_argument("--use-pytrec-eval", action="store_true")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--auth-token", default=None)
    parser.add_argument("--global-search", action="store_true")
    parser.add_argument("--global-file-id-prefix", default=None)
    parser.add_argument("--entity-id", default=None)
    args = parser.parse_args()

    report, exit_code = evaluate(args)
    if args.output:
        Path(args.output).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
