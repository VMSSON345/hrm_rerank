#!/usr/bin/env python3
"""Submit document ingestion jobs and wait for terminal status."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request


def request_json(
    url: str, *, payload: dict[str, object] | None = None
) -> dict[str, object]:
    data = None
    method = "GET"
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        method = "POST"
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url, data=data, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=30.0) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    if not isinstance(result, dict):
        raise RuntimeError(f"Expected JSON object from {url}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("documents", nargs="+")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--skip-mineru-health-check",
        action="store_true",
        help="Allow native-only ingestion when the configured MinerU API is unavailable",
    )
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.timeout <= 0:
        parser.error("poll interval and timeout must be positive")

    base_url = args.base_url.rstrip("/")
    if not args.skip_mineru_health_check:
        health = request_json(base_url + "/health")
        mineru = health.get("mineru")
        mineru_status = (
            str(mineru.get("status", "unknown"))
            if isinstance(mineru, dict)
            else "unknown"
        )
        if mineru_status != "healthy":
            raise RuntimeError(
                "MinerU is not healthy; start scripts/run_mineru_api.sh before "
                f"submitting scanned PDFs. Service health: {mineru}"
            )

    jobs: list[tuple[str, str]] = []
    for document in args.documents:
        result = request_json(
            base_url + "/v1/ingest",
            payload={"document": document, "rebuild": args.rebuild},
        )
        job_id = str(result.get("job_id", ""))
        if not job_id:
            raise RuntimeError(f"Ingest response has no job_id: {result}")
        jobs.append((document, job_id))
        print(f"submitted document={document} job={job_id}", flush=True)

    deadline = time.monotonic() + args.timeout
    pending = dict(jobs)
    last_states: dict[str, tuple[str, object, object]] = {}
    failures: list[str] = []
    while pending:
        if time.monotonic() >= deadline:
            waiting = ", ".join(f"{doc}:{job}" for doc, job in pending.items())
            raise RuntimeError(f"Timed out waiting for ingestion: {waiting}")
        for document, job_id in list(pending.items()):
            result = request_json(base_url + f"/v1/jobs/{job_id}")
            status = str(result.get("status", "unknown"))
            completed = result.get("documents_completed", 0)
            total = result.get("documents_total", 0)
            state = (status, completed, total)
            if last_states.get(job_id) != state:
                print(
                    f"job={job_id} status={status} documents={completed}/{total}",
                    flush=True,
                )
                last_states[job_id] = state
            if status == "ready":
                del pending[document]
            elif status == "failed":
                failures.append(
                    f"{document}: {result.get('error')}"
                )
                del pending[document]
        if pending:
            time.sleep(args.poll_seconds)
    if failures:
        raise RuntimeError(
            "Ingestion failed:\n- " + "\n- ".join(failures)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
