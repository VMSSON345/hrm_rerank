#!/usr/bin/env python3
"""Render the exact PDF pages referenced by a SUBTRACT review cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import fitz


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--layout-audit", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.report}")
    layout = json.loads(args.layout_audit.read_text(encoding="utf-8"))
    documents = layout.get("documents")
    if not isinstance(documents, dict):
        raise ValueError("Layout audit document map is missing")
    records = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rendered: list[dict[str, object]] = []
    args.render_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        operands = record["program"]["operands"]
        candidate = record["candidates"][operands[0]]
        source_name = str(candidate["source_name"])
        page_number = int(candidate["page"])
        source_info = documents.get(source_name)
        if not isinstance(source_info, dict):
            raise ValueError(f"Source is absent from layout audit: {source_name}")
        source = Path(str(source_info["source_path"]))
        expected_hash = str(record["supervision"]["source_sha256"])
        if _sha256(source) != expected_hash:
            raise ValueError(f"Source hash mismatch: {source_name}")
        output = args.render_dir / f"{source.stem}__p{page_number}.png"
        document = fitz.open(source)
        try:
            page = document[page_number - 1]
            pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
            pixmap.save(output)
        finally:
            document.close()
        rendered.append(
            {
                "example_id": record["example_id"],
                "source_name": source_name,
                "source_sha256": expected_hash,
                "page": page_number,
                "render": str(output.resolve()),
            }
        )
    report = {
        "schema": "subtract-development-cohort-render-v1",
        "classification": "development_only_visual_review",
        "blind_test_used": False,
        "input": str(args.input.resolve()),
        "input_sha256": _sha256(args.input),
        "layout_audit": str(args.layout_audit.resolve()),
        "layout_audit_sha256": _sha256(args.layout_audit),
        "pages_rendered": len(rendered),
        "records": rendered,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"pages_rendered": len(rendered)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
