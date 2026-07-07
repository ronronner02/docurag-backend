#!/usr/bin/env python3
"""Evaluate global RAG answer grounding on a prepared HotpotQA subset.

This script calls ``/rag/chat_global`` and scores answer-level behavior:
whether returned sources overlap qrels, whether all support docs are present,
whether the answer includes source markers, and whether the model refused.

It is intentionally separate from retrieval metrics. Retrieval Recall@K scores
the ranker; this script scores the answer layer built on top of retrieval.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .smoke_embeddings import post_json
except ImportError:
    from smoke_embeddings import post_json


GLOBAL_RAG_LIMITATIONS = [
    "This is not a full HotpotQA/BEIR benchmark unless the vector store contains the full corpus.",
    "The script evaluates answer-layer behavior from /rag/chat_global, not official answer exact match or F1.",
    "Support hit metrics use returned source file_ids against qrels; they do not judge natural-language correctness.",
    "Use a small --limit for low-cost LLM smoke tests; use --limit 100 only when cost is acceptable.",
]

PREFIX_GLOBAL_RAG_LIMITATIONS = [
    "This is not a full HotpotQA/BEIR benchmark unless the prefix covers the full corpus.",
    "Global RAG is scoped by file_id_prefix, so other loaded documents are excluded.",
    "The script evaluates answer-layer behavior from /rag/chat_global, not official answer exact match or F1.",
    "Support hit metrics use returned source file_ids against qrels; they do not judge natural-language correctness.",
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


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def build_relevance_map(qrels_rows: list[dict]) -> dict[str, set[str]]:
    relevant_by_query: dict[str, set[str]] = defaultdict(set)
    for row in qrels_rows:
        score = int(row.get("score", 1))
        if score <= 0:
            continue
        query_id = str(row["query_id"])
        file_id = str(row.get("file_id") or row.get("doc_id"))
        relevant_by_query[query_id].add(file_id)
    return dict(relevant_by_query)


def candidate_mode(file_id_prefix: str | None) -> str:
    if file_id_prefix:
        return "global_rag_prefix"
    return "global_rag"


def limitations(file_id_prefix: str | None) -> list[str]:
    if file_id_prefix:
        return PREFIX_GLOBAL_RAG_LIMITATIONS
    return GLOBAL_RAG_LIMITATIONS


def build_global_rag_payload(
    query_text: str,
    k: int,
    max_context_chars: int,
    use_llm: bool,
    entity_id: str | None = None,
    file_id_prefix: str | None = None,
) -> dict:
    payload = {
        "query": query_text,
        "k": k,
        "max_context_chars": max_context_chars,
        "use_llm": use_llm,
    }
    if entity_id:
        payload["entity_id"] = entity_id
    if file_id_prefix:
        payload["file_id_prefix"] = file_id_prefix
    return payload


def answer_has_source_marker(answer: str) -> bool:
    return re.search(r"\[source\s+\d+\]", answer, flags=re.IGNORECASE) is not None


def parse_rag_chat_global_response(body: Any) -> dict:
    if not isinstance(body, dict):
        return {
            "answer": "",
            "answer_strategy": "invalid_response",
            "refusal": True,
            "source_file_ids": [],
            "source_count": 0,
            "answer_has_citation": False,
        }

    answer = str(body.get("answer") or "")
    source_file_ids = []
    seen_file_ids = set()
    for source in body.get("sources", []):
        if not isinstance(source, dict):
            continue
        metadata = source.get("metadata") or {}
        file_id = metadata.get("file_id")
        if not file_id:
            continue
        file_id = str(file_id)
        if file_id in seen_file_ids:
            continue
        seen_file_ids.add(file_id)
        source_file_ids.append(file_id)

    return {
        "answer": answer,
        "answer_strategy": str(body.get("answer_strategy") or ""),
        "refusal": bool(body.get("refusal")),
        "source_file_ids": source_file_ids,
        "source_count": len(body.get("sources", []) or []),
        "answer_has_citation": answer_has_source_marker(answer),
    }


def aggregate(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def compute_rag_answer_metrics(details: list[dict]) -> dict[str, float]:
    query_count = len(details)
    support_hits = []
    all_supports = []
    citation_markers = []
    refusals = []
    llm_grounded = []
    low_confidence_refusals = []
    no_context_refusals = []
    extractive_answers = []
    source_counts = []

    for detail in details:
        relevant = set(detail.get("relevant_file_ids", []))
        source_file_ids = set(detail.get("source_file_ids", []))
        strategy = detail.get("answer_strategy")

        support_hits.append(1.0 if relevant & source_file_ids else 0.0)
        all_supports.append(1.0 if relevant and relevant <= source_file_ids else 0.0)
        citation_markers.append(1.0 if detail.get("answer_has_citation") else 0.0)
        refusals.append(1.0 if detail.get("refusal") else 0.0)
        llm_grounded.append(1.0 if strategy == "llm_grounded_context_v1" else 0.0)
        low_confidence_refusals.append(1.0 if strategy == "low_confidence_refusal" else 0.0)
        no_context_refusals.append(1.0 if strategy == "no_context_refusal" else 0.0)
        extractive_answers.append(1.0 if strategy == "extractive_context_v1" else 0.0)
        source_counts.append(float(detail.get("source_count") or 0))

    return {
        "query_count": float(query_count),
        "retrieved_support_hit_rate": aggregate(support_hits),
        "all_support_in_sources_rate": aggregate(all_supports),
        "citation_marker_rate": aggregate(citation_markers),
        "refusal_rate": aggregate(refusals),
        "llm_grounded_strategy_rate": aggregate(llm_grounded),
        "low_confidence_refusal_rate": aggregate(low_confidence_refusals),
        "no_context_refusal_rate": aggregate(no_context_refusals),
        "extractive_strategy_rate": aggregate(extractive_answers),
        "average_source_count": aggregate(source_counts),
    }


def evaluate(args: argparse.Namespace) -> tuple[dict, int]:
    subset_dir = Path(args.subset_dir)
    queries = read_jsonl(subset_dir / "queries.jsonl")
    qrels_rows = read_jsonl(subset_dir / "qrels.jsonl")
    manifest = read_json(subset_dir / "manifest.json")

    if not qrels_rows:
        return (
            {
                "ok": False,
                "reason": "qrels.jsonl is missing or empty; cannot compute source grounding metrics.",
                "subset_dir": str(subset_dir),
                "candidate_mode": candidate_mode(args.global_file_id_prefix),
                "limitations": limitations(args.global_file_id_prefix),
            },
            2,
        )

    if args.limit is not None and args.limit >= 0:
        queries = queries[: args.limit]

    relevant_by_query = build_relevance_map(qrels_rows)
    details = []

    for query in queries:
        query_id = str(query["query_id"])
        payload = build_global_rag_payload(
            query_text=query["text"],
            k=args.k,
            max_context_chars=args.max_context_chars,
            use_llm=args.use_llm,
            entity_id=args.entity_id,
            file_id_prefix=args.global_file_id_prefix,
        )
        try:
            status, body = post_json(
                args.base_url,
                "/rag/chat_global",
                payload,
                args.timeout,
                args.auth_token,
            )
            if status != 200:
                raise RuntimeError(f"HTTP {status}: {body}")
            parsed = parse_rag_chat_global_response(body)
            error = None
        except Exception as exc:
            parsed = parse_rag_chat_global_response({})
            error = str(exc)

        relevant = sorted(relevant_by_query.get(query_id, set()))
        source_file_ids = parsed["source_file_ids"]
        detail = {
            "query_id": query_id,
            "query": query["text"],
            "relevant_file_ids": relevant,
            "source_file_ids": source_file_ids,
            "support_hit": bool(set(relevant) & set(source_file_ids)),
            "all_support_in_sources": bool(set(relevant) and set(relevant) <= set(source_file_ids)),
            "answer_has_citation": parsed["answer_has_citation"],
            "refusal": parsed["refusal"],
            "answer_strategy": parsed["answer_strategy"],
            "source_count": parsed["source_count"],
            "answer_preview": parsed["answer"][:300],
            "raw_response_error": error,
        }
        details.append(detail)

    report = {
        "ok": True,
        "evaluation_type": manifest.get("evaluation_type", "labeled_smoke_test"),
        "candidate_mode": candidate_mode(args.global_file_id_prefix),
        "rag_endpoint": "/rag/chat_global",
        "global_file_id_prefix": args.global_file_id_prefix,
        "query_count": len(queries),
        "qrels_query_count": len(relevant_by_query),
        "k": args.k,
        "max_context_chars": args.max_context_chars,
        "use_llm": args.use_llm,
        "metrics": compute_rag_answer_metrics(details),
        "limitations": limitations(args.global_file_id_prefix),
        "details": details,
    }
    return report, 0


def main() -> int:
    configure_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-dir", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-context-chars", type=int, default=2000)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--global-file-id-prefix", default=None)
    parser.add_argument("--entity-id", default=None)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--auth-token", default=None)
    parser.add_argument("--output", default=None)
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
