from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import fitz

from hrm_finance_qa.document_page_index import DocumentPageIndex, PageRecord
from hrm_finance_qa.document_sources import OCRConfig
from generate_qp3_core_add_diversity_ocr_pool import _ocr_lines


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRECOMMIT = (
    ROOT
    / "config"
    / "finhrm_qp3_core_add_diverse_full_page_ocr_index_precommit_v1.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ocr_page(task: tuple[str, int]) -> tuple[int, str]:
    path_text, page_index = task
    document = fitz.open(path_text)
    try:
        return page_index + 1, _ocr_lines(document[page_index])
    finally:
        document.close()


def _selected_sources(precommit: dict[str, Any]) -> list[dict[str, str]]:
    parent = precommit["parents"]["reviewed_cohort_report"]
    report_path = ROOT / parent["path"]
    if _sha256(report_path) != parent["sha256"]:
        raise ValueError("Reviewed cohort report SHA-256 changed")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selected = [
        {
            "source_path": str(item["source_path"]),
            "source_sha256": str(item["source_sha256"]),
            "source_name": str(item["source_name"]),
            "company_key": str(item["company_key"]),
            "layout_family": str(item["layout_family"]),
        }
        for item in report["details"]
        if item["layout_stratum"] == "OCR_OR_HYBRID_TABLE"
    ]
    if len(selected) != 4 or len({item["source_sha256"] for item in selected}) != 4:
        raise ValueError("Expected four distinct reviewed OCR/hybrid sources")
    return selected


def _prime_source(
    source: dict[str, str],
    *,
    index: DocumentPageIndex,
    workers: int,
) -> dict[str, Any]:
    path = (ROOT / source["source_path"]).resolve()
    if _sha256(path) != source["source_sha256"]:
        raise ValueError(f"Source SHA-256 changed: {path.name}")
    document = fitz.open(path)
    try:
        page_count = len(document)
    finally:
        document.close()
    tasks = [(str(path), page_index) for page_index in range(page_count)]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        page_text = dict(executor.map(_ocr_page, tasks))
    if set(page_text) != set(range(1, page_count + 1)):
        raise RuntimeError(f"Not every physical page was OCRed: {path.name}")

    records, _ = index.load_or_build(
        path,
        ocr_config=OCRConfig(enabled=False),
        rebuild=False,
    )
    if len(records) != page_count:
        raise RuntimeError(f"Page index count differs from PDF: {path.name}")
    updates: list[PageRecord] = []
    for record in records:
        updates.append(
            replace(
                record,
                extraction_method="ocr",
                searchable_text=page_text[record.page_number],
                ocr_state="high_quality",
            )
        )
    index.update_records(updates)
    return {
        **source,
        "physical_pages": page_count,
        "pages_ocr_succeeded": len(page_text),
        "empty_ocr_pages": sum(not text.strip() for text in page_text.values()),
        "ocr_characters": sum(len(text) for text in page_text.values()),
        "cache_extraction_method": "ocr",
        "cache_ocr_state": "high_quality",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prime the frozen document page index with provenance-correct full-page "
            "Tesseract OCR for the reviewed development OCR/hybrid sources."
        )
    )
    parser.add_argument("--precommit", type=Path, default=DEFAULT_PRECOMMIT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    precommit = json.loads(args.precommit.read_text(encoding="utf-8"))
    for parent in precommit["parents"].values():
        if _sha256(ROOT / parent["path"]) != parent["sha256"]:
            raise ValueError(f"Parent SHA-256 changed: {parent['path']}")
    output_path = ROOT / precommit["output"]["report"]
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    index_path = ROOT / precommit["output"]["page_index"]
    index = DocumentPageIndex(index_path)
    source_reports: list[dict[str, Any]] = []
    for position, source in enumerate(_selected_sources(precommit), start=1):
        print(
            f"[{position}/4] OCR every page: {source['source_name']}",
            flush=True,
        )
        source_reports.append(
            _prime_source(source, index=index, workers=args.workers)
        )
    tesseract_version = subprocess.run(
        ["tesseract", "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[0]
    total_pages = sum(item["physical_pages"] for item in source_reports)
    report = {
        "schema": "finhrm-qp3-core-add-diverse-full-page-ocr-index-v1",
        "created_at": "2026-08-20",
        "classification": "development_only_full_physical_page_ocr_index",
        "gate_passed": (
            len(source_reports) == 4
            and all(
                item["physical_pages"] == item["pages_ocr_succeeded"]
                for item in source_reports
            )
        ),
        "inputs": {
            "precommit": {
                "path": args.precommit.relative_to(ROOT).as_posix(),
                "sha256": _sha256(args.precommit),
            }
        },
        "ocr": {
            "engine": tesseract_version,
            "languages": ["vie", "eng"],
            "dpi": 250,
            "workers": args.workers,
        },
        "metrics": {
            "sources": len(source_reports),
            "physical_pages": total_pages,
            "pages_ocr_succeeded": sum(
                item["pages_ocr_succeeded"] for item in source_reports
            ),
            "empty_ocr_pages": sum(
                item["empty_ocr_pages"] for item in source_reports
            ),
            "ocr_characters": sum(
                item["ocr_characters"] for item in source_reports
            ),
        },
        "sources": source_reports,
        "invariants": {
            "question_loaded_or_used": False,
            "gold_page_value_program_or_candidate_loaded_or_used": False,
            "source_filename_page_year_value_example_or_question_literal_priority": False,
            "protected_evaluation_loaded": False,
            "abw_loaded": False,
            "model_checkpoint_or_threshold_changed": False,
        },
    }
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["metrics"], ensure_ascii=False, sort_keys=True))
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
