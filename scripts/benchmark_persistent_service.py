from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request


def post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300.0) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure repeated answers against an already-running HRM service"
    )
    parser.add_argument("document")
    parser.add_argument("question")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")

    endpoint = args.base_url.rstrip("/") + "/v1/answer"
    measurements: list[dict[str, object]] = []
    for run in range(1, args.repeats + 1):
        started_at = time.perf_counter()
        result = post_json(
            endpoint,
            {"document": args.document, "question": args.question},
        )
        wall_seconds = time.perf_counter() - started_at
        measurements.append(
            {
                "run": run,
                "wall_seconds": wall_seconds,
                "service_answer_seconds": result.get("service_answer_seconds"),
                "status": result.get("status"),
                "answer": result.get("answer"),
                "unit": result.get("unit"),
                "stage_timings": result.get("stage_timings"),
                "page_index": result.get("page_index"),
            }
        )
    wall = [float(item["wall_seconds"]) for item in measurements]
    report = {
        "endpoint": endpoint,
        "document": args.document,
        "question": args.question,
        "repeats": args.repeats,
        "wall_seconds_mean": statistics.mean(wall),
        "wall_seconds_min": min(wall),
        "wall_seconds_max": max(wall),
        "measurements": measurements,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
