#!/usr/bin/env python3
"""Render development-only V12-C negative/ambiguity discovery pages."""

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


def render(discovery_path: Path, render_dir: Path) -> dict[str, object]:
    _refuse(discovery_path)
    _refuse(render_dir)
    discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
    if discovery.get("classification") != "development_only_negative_control_discovery":
        raise ValueError("Discovery is not an approved development negative-control discovery")
    records = discovery.get("records")
    if not isinstance(records, list):
        raise ValueError("Discovery records are missing")
    render_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[dict[str, object]] = []
    for record in records:
        source_path = Path(str(record["source_path"]))
        _refuse(source_path)
        if _sha256(source_path) != record["source_sha256"]:
            raise ValueError(f"Source hash mismatch: {source_path}")
        page_number = int(record["page"])
        output = render_dir / f"{source_path.stem}__p{page_number}.png"
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite render: {output}")
        document = fitz.open(source_path)
        try:
            pixmap = document[page_number - 1].get_pixmap(
                matrix=fitz.Matrix(2.0, 2.0), alpha=False
            )
            pixmap.save(output)
        finally:
            document.close()
        rendered.append({
            "source_name": record["source_name"],
            "source_sha256": record["source_sha256"],
            "company_key": record["company_key"],
            "page": page_number,
            "candidate_type": record["candidate_type"],
            "render_path": str(output.resolve()),
            "render_sha256": _sha256(output),
        })
    return {
        "protocol": "v12c-negative-ambiguity-render-queue-v1",
        "classification": "development_only_negative_control_pending_visual_qa",
        "protected_evaluation_sources_used": False,
        "model_invoked": False,
        "threshold_changed": False,
        "discovery_path": str(discovery_path.resolve()),
        "discovery_sha256": _sha256(discovery_path),
        "pages_rendered": len(rendered),
        "records": rendered,
        "ready_for_threshold_change": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {args.output}")
    report = render(args.discovery, args.render_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pages_rendered": report["pages_rendered"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
