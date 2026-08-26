#!/usr/bin/env python3
"""Materialize MinerU Markdown only for selected development review pages."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections import Counter
from hashlib import sha256
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrm_finance_qa.document_sources import OCRConfig, _load_or_generate_mineru_pages


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_mineru(url: str) -> dict[str, object]:
    endpoint = url.rstrip("/") + "/health"
    with urllib.request.urlopen(endpoint, timeout=5.0) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("status") != "healthy":
        raise RuntimeError(f"MinerU API is not healthy: {payload}")
    return payload


def _contiguous_batches(pages: list[int], batch_size: int) -> list[list[int]]:
    batches: list[list[int]] = []
    current: list[int] = []
    for page in sorted(pages):
        if current and (page != current[-1] + 1 or len(current) >= batch_size):
            batches.append(current)
            current = []
        current.append(page)
    if current:
        batches.append(current)
    return batches


def _selected_pages(
    item: dict[str, object], *, required_reason: str | None = None
) -> list[int]:
    total_pages = item.get("total_pages")
    if not isinstance(total_pages, int) or total_pages < 1:
        raise ValueError("Manifest document has invalid total_pages")
    selected: set[int] = set()
    pages = item.get("pages", [])
    if not isinstance(pages, list):
        raise ValueError("Manifest pages must be a list")
    for page in pages:
        if not isinstance(page, dict):
            raise ValueError("Manifest page must be an object")
        if page.get("selection") not in {"core", "neighbor", "discovery"}:
            continue
        reasons = page.get("reasons", [])
        if required_reason is not None and (
            not isinstance(reasons, list) or required_reason not in reasons
        ):
            continue
        page_number = page.get("page")
        if not isinstance(page_number, int) or not 1 <= page_number <= total_pages:
            raise ValueError("Selected page is outside the document")
        selected.add(page_number)
    return sorted(selected)


def _write_markdown(path: Path, source: Path, pages: dict[int, list[str]]) -> None:
    lines = [
        "---",
        f"source_name: {json.dumps(source.name, ensure_ascii=False)}",
        "extraction_method: mineru",
        "selection: development_financial_review",
        "---",
        "",
    ]
    for page, page_lines in sorted(pages.items()):
        lines.extend([f"## Page {page}", "", *page_lines, ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="OCR manifest review pages through a persistent MinerU API"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--mineru-api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--mineru-cache-dir", type=Path, default=ROOT / "cache" / "mineru")
    parser.add_argument("--page-batch-size", type=int, default=4)
    parser.add_argument(
        "--required-reason",
        help="OCR only pages carrying this structure-selection reason",
    )
    parser.add_argument(
        "--document",
        action="append",
        help="Process one manifest PDF name or path; repeat for a bounded batch",
    )
    args = parser.parse_args()
    if args.page_batch_size < 1:
        raise ValueError("--page-batch-size must be positive")

    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("ocr_performed") is not False:
        raise RuntimeError("Expected a native-only development manifest")
    health = _require_mineru(args.mineru_api_url)
    requested_documents = {Path(value).name for value in (args.document or [])}
    config = OCRConfig(
        enabled=True,
        mineru_cache_dir=args.mineru_cache_dir.expanduser().resolve(),
        mineru_api_url=args.mineru_api_url,
        mineru_api_page_batch_size=args.page_batch_size,
        mineru_backend="hybrid-engine",
        mineru_effort="medium",
    )
    documents: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    totals: Counter[str] = Counter()
    output_dir = args.output_dir.expanduser().resolve()
    for item in manifest.get("documents", []):
        if not isinstance(item, dict):
            raise ValueError("Manifest document must be an object")
        source = Path(str(item["document"])).expanduser().resolve()
        if requested_documents and source.name not in requested_documents:
            continue
        expected_sha = str(item.get("sha256", "")).lower()
        if expected_sha and _sha256(source) != expected_sha:
            raise RuntimeError(f"PDF checksum changed since manifest: {source}")
        selected = _selected_pages(item, required_reason=args.required_reason)
        if not selected:
            continue
        collected: dict[int, list[str]] = {}
        try:
            batches = _contiguous_batches(selected, args.page_batch_size)
            print(
                f"OCR document={source.name} pages={len(selected)} batches={len(batches)}",
                flush=True,
            )
            for batch_index, batch in enumerate(batches, start=1):
                print(
                    f"OCR batch={batch_index}/{len(batches)} pages={batch[0]}-{batch[-1]}",
                    flush=True,
                )
                batch_pages = set(batch)
                flags = [
                    index + 1 in batch_pages for index in range(int(item["total_pages"]))
                ]
                collected.update(_load_or_generate_mineru_pages(source, flags, config))
            destination = output_dir / f"{source.stem}.md"
            _write_markdown(destination, source, collected)
            documents.append(
                {
                    "document": source.name,
                    "requested_pages": len(selected),
                    "ocr_pages": len(collected),
                    "markdown": str(destination),
                }
            )
            totals["requested_pages"] += len(selected)
            totals["ocr_pages"] += len(collected)
        except Exception as exc:
            failures.append(
                {"document": str(source), "error_type": type(exc).__name__, "error": str(exc)}
            )
    if requested_documents:
        processed_names = {str(item["document"]) for item in documents}
        failed_names = {Path(str(item["document"])).name for item in failures}
        missing = requested_documents - processed_names - failed_names
        if missing:
            raise ValueError(f"Document not found in manifest: {sorted(missing)}")
    report = {
        "protocol": "development-financial-review-ocr-v1",
        "classification": "development_only",
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "ocr_engine": "MinerU API",
        "mineru_health": health,
        "selection": "bounded_structure_aware_discovery",
        "required_selection_reason": args.required_reason,
        "page_batch_size": args.page_batch_size,
        "documents": documents,
        "failures": failures,
        "metrics": {
            "documents_processed": len(documents),
            "pipeline_failures": len(failures),
            **dict(totals),
        },
    }
    report_path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(report_path), **report["metrics"]}, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
