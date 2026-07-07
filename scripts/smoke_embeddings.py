#!/usr/bin/env python3
"""Run a tiny end-to-end smoke test against a running DocuRAG API.

The script intentionally talks to HTTP endpoints instead of importing app code.
That verifies the same path a real client uses:

health -> embed -> retrieval/search -> rag/chat
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib import error, request


DEFAULT_TEXT = (
    "DocuRAG is a FastAPI retrieval augmented generation backend. "
    "It stores document chunks in pgvector using configurable embeddings, "
    "then returns grounded answers with source metadata."
)


def _url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + path


def _headers(auth_token: str | None = None, content_type: str | None = None) -> dict:
    headers = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def _read_response(resp) -> tuple[int, dict | list | str]:
    raw = resp.read().decode("utf-8", errors="replace")
    try:
        return resp.status, json.loads(raw)
    except json.JSONDecodeError:
        return resp.status, raw


def _open(req: request.Request, timeout: int) -> tuple[int, dict | list | str]:
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            return _read_response(resp)
    except error.HTTPError as exc:
        status, body = _read_response(exc)
        raise RuntimeError(f"HTTP {status}: {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Request failed: {exc}") from exc


def get_json(base_url: str, path: str, timeout: int) -> tuple[int, dict | list | str]:
    req = request.Request(_url(base_url, path), method="GET")
    return _open(req, timeout)


def post_json(
    base_url: str,
    path: str,
    payload: dict,
    timeout: int,
    auth_token: str | None,
) -> tuple[int, dict | list | str]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        _url(base_url, path),
        data=body,
        method="POST",
        headers=_headers(auth_token, "application/json"),
    )
    return _open(req, timeout)


def post_multipart_embed(
    base_url: str,
    file_id: str,
    file_path: Path,
    timeout: int,
    auth_token: str | None,
    entity_id: str | None,
) -> tuple[int, dict | list | str]:
    boundary = f"----docurag-smoke-{uuid.uuid4().hex}"
    parts: list[bytes] = []

    def add_field(name: str, value: str) -> None:
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        parts.append(value.encode("utf-8"))
        parts.append(b"\r\n")

    add_field("file_id", file_id)
    if entity_id:
        add_field("entity_id", entity_id)

    parts.append(f"--{boundary}\r\n".encode())
    parts.append(
        (
            'Content-Disposition: form-data; name="file"; '
            f'filename="{file_path.name}"\r\n'
            "Content-Type: text/plain\r\n\r\n"
        ).encode()
    )
    parts.append(file_path.read_bytes())
    parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())

    body = b"".join(parts)
    req = request.Request(
        _url(base_url, "/embed"),
        data=body,
        method="POST",
        headers=_headers(auth_token, f"multipart/form-data; boundary={boundary}"),
    )
    return _open(req, timeout)


def assert_status(name: str, status: int, expected: int = 200) -> None:
    if status != expected:
        raise AssertionError(f"{name} returned {status}, expected {expected}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--file-id", default=f"day8-embedding-smoke-{int(time.time())}")
    parser.add_argument("--query", default="What is DocuRAG?")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--max-context-chars", type=int, default=800)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--auth-token", default=None)
    parser.add_argument("--entity-id", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    report: dict = {
        "base_url": args.base_url,
        "file_id": args.file_id,
        "query": args.query,
        "steps": [],
    }

    try:
        status, health = get_json(args.base_url, "/health", args.timeout)
        assert_status("health", status)
        report["steps"].append({"name": "health", "status": status, "body": health})

        with NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(DEFAULT_TEXT)
            temp_path = Path(f.name)

        try:
            status, embed = post_multipart_embed(
                args.base_url,
                args.file_id,
                temp_path,
                args.timeout,
                args.auth_token,
                args.entity_id,
            )
        finally:
            temp_path.unlink(missing_ok=True)

        assert_status("embed", status)
        report["steps"].append({"name": "embed", "status": status, "body": embed})

        retrieval_payload = {
            "query": args.query,
            "file_id": args.file_id,
            "k": args.k,
        }
        if args.entity_id:
            retrieval_payload["entity_id"] = args.entity_id

        status, retrieval = post_json(
            args.base_url,
            "/retrieval/search",
            retrieval_payload,
            args.timeout,
            args.auth_token,
        )
        assert_status("retrieval/search", status)
        if not isinstance(retrieval, dict) or retrieval.get("result_count", 0) < 1:
            raise AssertionError(f"retrieval/search returned no results: {retrieval}")
        report["steps"].append(
            {"name": "retrieval/search", "status": status, "body": retrieval}
        )

        rag_payload = {
            "query": args.query,
            "file_id": args.file_id,
            "k": args.k,
            "max_context_chars": args.max_context_chars,
        }
        if args.entity_id:
            rag_payload["entity_id"] = args.entity_id

        status, rag = post_json(
            args.base_url,
            "/rag/chat",
            rag_payload,
            args.timeout,
            args.auth_token,
        )
        assert_status("rag/chat", status)
        if not isinstance(rag, dict) or rag.get("used_context_count", 0) < 1:
            raise AssertionError(f"rag/chat returned no context: {rag}")
        report["steps"].append({"name": "rag/chat", "status": status, "body": rag})

        report["ok"] = True
    except Exception as exc:
        report["ok"] = False
        report["error"] = str(exc)

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
