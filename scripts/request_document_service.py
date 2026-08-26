#!/usr/bin/env python3
"""Submit one document question to the persistent HRM service."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("document")
    parser.add_argument("question")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--top-k-chunks", type=int, default=12)
    parser.add_argument("--neighbor-chunks", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    payload = {
        "document": args.document,
        "question": args.question,
        "top_k_chunks": args.top_k_chunks,
        "neighbor_chunks": args.neighbor_chunks,
    }
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/v1/answer",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code}: {detail}", file=sys.stderr)
        return 1
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Document service request failed: {exc}", file=sys.stderr)
        return 1

    rendered = result.get("rendered")
    if not isinstance(rendered, str):
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
