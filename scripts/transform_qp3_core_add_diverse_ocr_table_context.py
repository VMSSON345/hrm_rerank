from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

from hrm_finance_qa.document_page_index import DocumentPageIndex
from hrm_finance_qa.document_sources import OCRConfig
from generate_qp3_core_add_diversity_ocr_pool import (
    _explicit_multilevel_header,
    _load_generator,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRECOMMIT = (
    ROOT
    / "config"
    / "finhrm_qp3_core_add_diverse_ocr_table_context_precommit_v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _table_context(raw: str, generator: Any) -> str:
    regions = [region for region in raw.split("\n\n") if region.strip()]
    return "\n\n".join(
        _explicit_multilevel_header(generator._layout_to_table_context(region))
        for region in regions
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transform every reviewed cached OCR page into generic table context."
    )
    parser.add_argument("--precommit", type=Path, default=DEFAULT_PRECOMMIT)
    args = parser.parse_args()
    precommit = json.loads(args.precommit.read_text(encoding="utf-8"))
    for parent in precommit["parents"].values():
        if _sha256(ROOT / parent["path"]) != parent["sha256"]:
            raise ValueError(f"Parent SHA-256 changed: {parent['path']}")
    output_path = ROOT / precommit["output"]["report"]
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    ocr_report = json.loads(
        (ROOT / precommit["parents"]["full_page_ocr_index_report"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    generator = _load_generator()
    index = DocumentPageIndex(ROOT / precommit["output"]["page_index"])
    source_reports: list[dict[str, Any]] = []
    for source in ocr_report["sources"]:
        source_path = (ROOT / source["source_path"]).resolve()
        if _sha256(source_path) != source["source_sha256"]:
            raise ValueError(f"Source SHA-256 changed: {source_path.name}")
        records, _ = index.load_or_build(
            source_path,
            ocr_config=OCRConfig(enabled=False),
            rebuild=False,
        )
        if len(records) != int(source["physical_pages"]):
            raise ValueError(f"Cached page count changed: {source_path.name}")
        updates = []
        before_characters = 0
        after_characters = 0
        table_pages = 0
        for record in records:
            if (
                record.extraction_method != "ocr"
                or record.ocr_state != "high_quality"
            ):
                raise ValueError(
                    f"Expected high-quality OCR cache: {source_path.name} page "
                    f"{record.page_number}"
                )
            transformed = _table_context(record.searchable_text, generator)
            before_characters += len(record.searchable_text)
            after_characters += len(transformed)
            table_pages += int("|" in transformed)
            updates.append(replace(record, searchable_text=transformed))
        index.update_records(updates)
        source_reports.append(
            {
                "source_path": source["source_path"],
                "source_name": source["source_name"],
                "source_sha256": source["source_sha256"],
                "physical_pages": len(records),
                "pages_transformed": len(updates),
                "pages_with_table_context": table_pages,
                "characters_before": before_characters,
                "characters_after": after_characters,
                "extraction_method": "ocr",
                "ocr_state": "high_quality"
            }
        )
        print(
            f"Transformed {source_path.name}: {len(updates)} pages, "
            f"{table_pages} table pages",
            flush=True,
        )
    total_pages = sum(item["pages_transformed"] for item in source_reports)
    report = {
        "schema": "finhrm-qp3-core-add-diverse-ocr-table-context-v1",
        "created_at": "2026-08-20",
        "classification": "development_only_full_page_ocr_table_context",
        "gate_passed": len(source_reports) == 4 and total_pages == 246,
        "inputs": {
            "precommit": {
                "path": args.precommit.relative_to(ROOT).as_posix(),
                "sha256": _sha256(args.precommit)
            }
        },
        "metrics": {
            "sources": len(source_reports),
            "pages_transformed": total_pages,
            "pages_with_table_context": sum(
                item["pages_with_table_context"] for item in source_reports
            ),
            "characters_before": sum(
                item["characters_before"] for item in source_reports
            ),
            "characters_after": sum(
                item["characters_after"] for item in source_reports
            )
        },
        "sources": source_reports,
        "invariants": {
            "question_loaded_or_used": False,
            "gold_page_value_program_or_candidate_loaded_or_used": False,
            "ocr_provenance_preserved": True,
            "protected_evaluation_loaded": False,
            "abw_loaded": False,
            "model_checkpoint_or_threshold_changed": False
        }
    }
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )
    print(json.dumps(report["metrics"], sort_keys=True))
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
