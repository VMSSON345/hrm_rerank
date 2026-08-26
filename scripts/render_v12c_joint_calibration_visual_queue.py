#!/usr/bin/env python3
"""Render a frozen V12-C joint visual-QA queue with source and render hashes."""

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


def _refuse(path: Path) -> None:
    if PROTECTED.search(str(path.resolve()).replace("\\", "/")):
        raise ValueError(f"Protected path is not allowed: {path}")


def render(queue_path: Path, render_dir: Path) -> dict[str, object]:
    _refuse(queue_path)
    _refuse(render_dir)
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    if queue.get("classification") != "development_only_joint_calibration_pending_visual_qa":
        raise ValueError("Queue is not development-only joint calibration")
    render_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for row in queue["records"]:
        source = Path(str(row["source_path"]))
        _refuse(source)
        if _sha256(source) != row["source_sha256"]:
            raise ValueError(f"Source hash mismatch: {source}")
        page = int(row["page"])
        prefix = "pos" if row["queue_stratum"] == "positive_candidate" else "neg"
        output = render_dir / f"{prefix}__{source.stem}__p{page}.png"
        document = fitz.open(source)
        try:
            pixmap = document[page - 1].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            pixmap.save(output)
        finally:
            document.close()
        records.append({
            "source_name": row["source_name"],
            "company_key": row["company_key"],
            "page": page,
            "queue_stratum": row["queue_stratum"],
            "proof_or_failure_type": row["proof_or_failure_type"],
            "render_path": str(output.resolve()),
            "render_sha256": _sha256(output),
        })
    return {
        "protocol": "v12c-joint-calibration-render-queue-v1",
        "classification": "development_only_joint_calibration_pending_visual_qa",
        "queue_path": str(queue_path.resolve()),
        "queue_sha256": _sha256(queue_path),
        "pages_rendered": len(records),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {args.output}")
    result = render(args.queue, args.render_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pages_rendered": result["pages_rendered"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
