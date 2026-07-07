#!/usr/bin/env python3
"""Import a prepared HotpotQA subset into the running DocuRAG API."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

try:
    from .smoke_embeddings import post_multipart_embed
except ImportError:
    from smoke_embeddings import post_multipart_embed


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def index_results_by_file_id(report: dict) -> dict[str, dict]:
    results = {}
    for result in report.get("results", []):
        file_id = result.get("file_id")
        if file_id:
            results[str(file_id)] = result
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-dir", default="data/derived/hotpotqa/day09_subset")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--auth-token", default=None)
    parser.add_argument("--entity-id", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    subset_dir = Path(args.subset_dir)
    corpus_path = subset_dir / "corpus.jsonl"
    rows = read_jsonl(corpus_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    report_path = Path(args.report) if args.report else None
    previous_report = read_json(report_path) if args.resume and report_path else {}
    results_by_file_id = index_results_by_file_id(previous_report)
    skipped_existing = 0
    imported_this_run = 0
    failed_this_run = 0
    attempted_this_run = 0

    report = {
        "base_url": args.base_url,
        "subset_dir": str(subset_dir),
        "attempted": len(rows),
        "attempted_this_run": 0,
        "skipped_existing": 0,
        "imported": 0,
        "imported_this_run": 0,
        "failed": 0,
        "failed_this_run": 0,
        "results": [],
    }

    for index, row in enumerate(rows, start=1):
        file_id = str(row["file_id"])
        previous_result = results_by_file_id.get(file_id)
        if args.resume and previous_result and previous_result.get("ok") is True:
            skipped_existing += 1
            if args.progress_every > 0 and index % args.progress_every == 0:
                print(
                    f"progress {index}/{len(rows)} skipped_existing={skipped_existing}",
                    file=sys.stderr,
                )
            continue

        attempted_this_run += 1
        title = row.get("title") or ""
        text = row.get("text") or ""
        content = f"{title}\n\n{text}".strip()
        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            temp_path = Path(f.name)

        try:
            status, body = post_multipart_embed(
                args.base_url,
                file_id,
                temp_path,
                args.timeout,
                args.auth_token,
                args.entity_id,
            )
            ok = status == 200 and isinstance(body, dict) and body.get("status") is True
        except Exception as exc:
            status = None
            body = {"error": str(exc)}
            ok = False
        finally:
            temp_path.unlink(missing_ok=True)

        if ok:
            imported_this_run += 1
        else:
            failed_this_run += 1
        results_by_file_id[file_id] = {
            "doc_id": row["doc_id"],
            "file_id": file_id,
            "ok": ok,
            "status": status,
            "body": body,
        }

        if report_path:
            current_results = [
                results_by_file_id[str(item["file_id"])]
                for item in rows
                if str(item["file_id"]) in results_by_file_id
            ]
            checkpoint = {
                **report,
                "attempted_this_run": attempted_this_run,
                "skipped_existing": skipped_existing,
                "imported_this_run": imported_this_run,
                "failed_this_run": failed_this_run,
                "imported": sum(1 for item in current_results if item.get("ok") is True),
                "failed": sum(
                    1 for item in current_results if item.get("ok") is not True
                ),
                "results": current_results,
            }
            write_json(report_path, checkpoint)

        if args.progress_every > 0 and index % args.progress_every == 0:
            print(
                "progress "
                f"{index}/{len(rows)} imported_this_run={imported_this_run} "
                f"failed_this_run={failed_this_run} skipped_existing={skipped_existing}",
                file=sys.stderr,
            )
        if args.fail_fast and not ok:
            break

    final_results = [
        results_by_file_id[str(row["file_id"])]
        for row in rows
        if str(row["file_id"]) in results_by_file_id
    ]
    report["attempted_this_run"] = attempted_this_run
    report["skipped_existing"] = skipped_existing
    report["imported_this_run"] = imported_this_run
    report["failed_this_run"] = failed_this_run
    report["imported"] = sum(1 for item in final_results if item.get("ok") is True)
    report["failed"] = sum(1 for item in final_results if item.get("ok") is not True)
    report["results"] = final_results

    if report_path:
        write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
