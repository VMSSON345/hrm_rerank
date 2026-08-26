#!/usr/bin/env python3
"""Render one hash-locked development-only PDF page for visual QA."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

import fitz


PROTECTED = re.compile(r"(?:test|holdout|final|blind|validation)", re.IGNORECASE)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--page", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.source, args.output):
        if PROTECTED.search(str(path.resolve()).replace("\\", "/")):
            raise ValueError(f"Protected path is not allowed: {path}")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {args.output}")
    if _sha256(args.source) != args.expected_source_sha256:
        raise ValueError("Source PDF SHA-256 mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open(args.source)
    try:
        pixmap = document[args.page - 1].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        pixmap.save(args.output)
    finally:
        document.close()
    print(json.dumps({"output": str(args.output.resolve()), "render_sha256": _sha256(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
