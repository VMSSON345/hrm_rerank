from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import fitz

from hrm_finance_qa.question_planner import compile_question_plan
from scripts.build_financial_page_manifest import score_financial_page
from scripts.evaluate_qp3_core_s4_safe_replay import _proof_decision
from scripts.generate_qp3_core_add_diversity_ocr_pool import (
    ROOT,
    DEFAULT_PRECOMMIT,
    GENERATOR_PATH,
    _digest,
    _explicit_multilevel_header,
    _load_generator,
    _ocr_lines,
)


DEFAULT_SOURCE_AUDIT = ROOT / "reports/finhrm_qp3_core_add_diverse_development_source_audit_v1.json"
DEFAULT_OUTPUT = ROOT / "data/finhrm_qp3_core_add_diversity_ocr_generated_pool_v2.jsonl"
DEFAULT_REPORT = ROOT / "reports/finhrm_qp3_core_add_diversity_ocr_generated_pool_v2.json"
_WORKER_GENERATOR: Any | None = None


def _discovery_text(page: fitz.Page, generator: Any) -> tuple[str, dict[str, Any]]:
    textpage = page.get_textpage_ocr(language="vie+eng", dpi=125, full=True)
    raw = page.get_text("text", textpage=textpage, sort=True)
    context = generator._layout_to_table_context(raw)
    return context, score_financial_page(context)


def _score_physical_page(args: tuple[str, int]) -> tuple[int, dict[str, Any]]:
    global _WORKER_GENERATOR
    path_text, page_index = args
    if _WORKER_GENERATOR is None:
        _WORKER_GENERATOR = _load_generator()
    document = fitz.open(path_text)
    try:
        _, score = _discovery_text(document[page_index], _WORKER_GENERATOR)
    finally:
        document.close()
    return page_index + 1, score


def _qualifying_add(examples: list[Any]) -> bool:
    for example in examples:
        if example.program.operation != "ADD":
            continue
        if not compile_question_plan(example.question).addition_role_hypotheses:
            continue
        decision = _proof_decision(example.question, example.program, example.candidates)
        if decision["action"] == "ANSWER" and decision["proof_contract_state"] == "VALID":
            return True
    return False


def build_pool(
    precommit_path: Path = DEFAULT_PRECOMMIT,
    source_audit_path: Path = DEFAULT_SOURCE_AUDIT,
) -> tuple[list[Any], dict[str, Any]]:
    precommit = json.loads(precommit_path.read_text(encoding="utf-8"))
    audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    generator_parent = precommit["parents"]["frozen_generic_program_generator"]
    if _digest(ROOT / generator_parent["path"]) != generator_parent["sha256"]:
        raise RuntimeError("frozen generator hash changed")
    generator = _load_generator()
    allowed_families = set(
        precommit["deterministic_selection"]["layout_strata"]["OCR_OR_HYBRID_TABLE"]
    )
    target = int(precommit["deterministic_selection"]["target_companies_per_layout"])
    maximum_pages = int(
        precommit["deterministic_selection"]["maximum_keyword_pages_per_source"]
    )
    sources = sorted(
        (row for row in audit["details"] if row["layout_family"] in allowed_families),
        key=lambda row: row["source_sha256"],
    )
    examples: list[Any] = []
    qualifying_companies: set[str] = set()
    source_rows: list[dict[str, Any]] = []
    pages_discovered = 0
    pages_materialized = 0
    skipped: Counter[str] = Counter()
    for source in sources:
        path = ROOT / source["source_path"]
        if _digest(path) != source["source_sha256"]:
            raise RuntimeError(f"source hash changed: {path.name}")
        document = fitz.open(path)
        selected_pages: list[tuple[int, dict[str, Any]]] = []
        source_examples: list[Any] = []
        source_pages_scored = 0
        try:
            discovery_inputs = [(str(path), page_index) for page_index in range(len(document))]
            with ProcessPoolExecutor(max_workers=4) as executor:
                discovery_rows = executor.map(_score_physical_page, discovery_inputs)
            for page_number, page_score in discovery_rows:
                pages_discovered += 1
                source_pages_scored += 1
                if (
                    page_score["page_class"] == "financial_summary"
                    and len(selected_pages) < maximum_pages
                ):
                    selected_pages.append((page_number, page_score))
            for page_number, _ in selected_pages:
                pages_materialized += 1
                raw_text = _ocr_lines(document[page_number - 1])
                context = "\n\n".join(
                    _explicit_multilevel_header(generator._layout_to_table_context(region))
                    for region in raw_text.split("\n\n")
                    if region.strip()
                )
                page_examples, page_skipped = generator._page_examples(
                    context,
                    source_name=path.name,
                    page=page_number,
                    max_examples=24,
                    id_start=len(examples) + len(source_examples) + 1,
                    extraction_method="tesseract_vie_250dpi",
                )
                source_examples.extend(
                    replace(
                        example,
                        supervision={
                            **dict(example.supervision or {}),
                            "company_key": source["company_key"],
                        },
                    )
                    for example in page_examples
                )
                skipped.update(page_skipped)
        finally:
            document.close()
        qualifies = _qualifying_add(source_examples)
        examples.extend(source_examples)
        if qualifies:
            qualifying_companies.add(str(source["company_key"]))
        source_rows.append(
            {
                "source_key": source["source_key"],
                "source_name": source["source_name"],
                "source_sha256": source["source_sha256"],
                "company_key": source["company_key"],
                "layout_family": source["layout_family"],
                "physical_pages_scored": source_pages_scored,
                "selected_pages": [
                    {"page": page, "score": score["score"], "page_class": score["page_class"]}
                    for page, score in selected_pages
                ],
                "examples": len(source_examples),
                "qualifying_add_answer_valid": qualifies,
            }
        )
        print(
            json.dumps(
                {
                    "source": source["source_name"],
                    "pages_scored": source_pages_scored,
                    "selected_pages": len(selected_pages),
                    "examples": len(source_examples),
                    "qualifying_add": qualifies,
                    "qualifying_companies": len(qualifying_companies),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        if len(qualifying_companies) >= target:
            break
    report = {
        "schema": "finhrm-qp3-core-add-diversity-ocr-generated-pool-v2",
        "created_at": "2026-08-20",
        "classification": "development_only_hash_ordered_full_page_ocr_discovery",
        "inputs": {
            "precommit": {"path": precommit_path.relative_to(ROOT).as_posix(), "sha256": _digest(precommit_path)},
            "source_audit": {"path": source_audit_path.relative_to(ROOT).as_posix(), "sha256": _digest(source_audit_path)},
            "generator": {"path": GENERATOR_PATH.relative_to(ROOT).as_posix(), "sha256": _digest(GENERATOR_PATH)},
        },
        "discovery": {
            "order": "ascending_source_sha256",
            "every_physical_page_scored_until_source_has_maximum_selected_pages": True,
            "scorer": "scripts.build_financial_page_manifest.score_financial_page",
            "discovery_ocr_dpi": 125,
            "materialization_ocr_dpi": 250,
            "languages": ["vie", "eng"],
            "stop_condition": "target distinct companies with explicit-two-role ADD ANSWER/VALID",
        },
        "metrics": {
            "sources_scanned": len(source_rows),
            "physical_pages_scored": pages_discovered,
            "pages_materialized": pages_materialized,
            "examples": len(examples),
            "operation_counts": dict(sorted(Counter(item.program.operation for item in examples).items())),
            "qualifying_companies": len(qualifying_companies),
            "skipped": dict(sorted(skipped.items())),
        },
        "sources": source_rows,
        "invariants": {
            "protected_evaluation_loaded": False,
            "abw_loaded": False,
            "runtime_changed": False,
            "model_checkpoint_or_threshold_changed": False,
            "source_filename_page_year_value_example_id_or_question_literal_used_for_priority": False,
        },
    }
    return examples, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite OCR v2 pool artifacts")
    examples, report = build_pool()
    generator = _load_generator()
    generator.write_program_examples(examples, args.output)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, sort_keys=True))
    return 0 if report["metrics"]["qualifying_companies"] >= 4 else 2


if __name__ == "__main__":
    raise SystemExit(main())
