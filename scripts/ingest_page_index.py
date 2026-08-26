#!/usr/bin/env python3
"""Build the runtime page index without loading the HRM/Torch stack."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrm_finance_qa.document_page_index import DocumentPageIndex, PageRecord
from hrm_finance_qa.document_sources import OCRConfig, discover_documents


def require_healthy_mineru(base_url: str) -> None:
    endpoint = base_url.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(endpoint, timeout=5.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.HTTPError) as exc:
        raise RuntimeError(f"MinerU API is unavailable at {endpoint}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("status") != "healthy":
        raise RuntimeError(f"MinerU API is not healthy: {payload}")


def render_page_records_markdown(records: Sequence[PageRecord]) -> str:
    if not records:
        return ""
    first = records[0]
    lines = [
        "---",
        f"source_name: {json.dumps(first.source_name, ensure_ascii=False)}",
        f"document_fingerprint: {first.document_fingerprint}",
        "generated_from_page_index: true",
        "---",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"## Page {record.page_number}",
                "",
                (
                    "<!-- "
                    f"extraction_method={record.extraction_method}; "
                    f"native_usable={str(record.native_usable).lower()}; "
                    f"native_reason={record.native_reason}; "
                    f"ocr_state={record.ocr_state}"
                    " -->"
                ),
                "",
            ]
        )
        if record.searchable_text.strip():
            lines.extend([record.searchable_text.strip(), ""])
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build persistent page indexes through a running MinerU API"
    )
    parser.add_argument("documents", nargs="+", type=Path)
    parser.add_argument(
        "--page-index-path",
        type=Path,
        default=ROOT / "cache" / "document_page_index.sqlite3",
    )
    parser.add_argument(
        "--mineru-cache-dir",
        type=Path,
        default=ROOT / "cache" / "mineru",
    )
    parser.add_argument("--mineru-api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--mineru-api-timeout", type=float, default=7200.0)
    parser.add_argument("--mineru-api-page-batch-size", type=int, default=4)
    parser.add_argument(
        "--mineru-backend",
        choices=("hybrid-engine", "pipeline"),
        default="hybrid-engine",
    )
    parser.add_argument(
        "--mineru-effort", choices=("medium", "high"), default="medium"
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        help="Write one page-provenanced Markdown file per indexed PDF",
    )
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    require_healthy_mineru(args.mineru_api_url)
    index = DocumentPageIndex(args.page_index_path)
    ocr_config = OCRConfig(
        enabled=True,
        mineru_cache_dir=args.mineru_cache_dir,
        mineru_api_url=args.mineru_api_url,
        mineru_api_timeout_seconds=args.mineru_api_timeout,
        mineru_api_page_batch_size=args.mineru_api_page_batch_size,
        mineru_backend=args.mineru_backend,
        mineru_effort=args.mineru_effort,
    )
    for requested in args.documents:
        for document in discover_documents(requested):
            if document.suffix.lower() != ".pdf":
                continue
            records, stats = index.load_or_build(
                document, ocr_config=ocr_config, rebuild=args.rebuild
            )
            export_path = None
            if args.export_dir is not None:
                export_dir = args.export_dir.expanduser().resolve()
                export_dir.mkdir(parents=True, exist_ok=True)
                export_path = export_dir / f"{document.stem}.md"
                export_path.write_text(
                    render_page_records_markdown(records),
                    encoding="utf-8",
                )
            print(
                json.dumps(
                    {
                        "document": str(document),
                        "stats": asdict(stats),
                        "markdown_export": (
                            str(export_path) if export_path is not None else None
                        ),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
