from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
from typing import Any

import fitz

from hrm_finance_qa.ontology import normalize_text


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRECOMMIT = ROOT / "config/finhrm_qp3_core_add_diverse_development_cohort_precommit_v1.json"
DEFAULT_MANIFEST = ROOT / "data/finhrm_qp3_core_add_diversity_generation_manifest_v1.json"
DEFAULT_OUTPUT = ROOT / "data/finhrm_qp3_core_add_diversity_ocr_generated_pool_v1.jsonl"
DEFAULT_REPORT = ROOT / "reports/finhrm_qp3_core_add_diversity_ocr_generated_pool_v1.json"
GENERATOR_PATH = ROOT / "archive/pre_v12_cleanup_20260802/scripts/generate_document_derived_programs.py"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_generator():
    spec = importlib.util.spec_from_file_location("frozen_program_generator", GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen program generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _same_baseline(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    left_y0, left_y1 = float(left[1]), float(left[3])
    right_y0, right_y1 = float(right[1]), float(right[3])
    overlap = max(0.0, min(left_y1, right_y1) - max(left_y0, right_y0))
    minimum_height = max(min(left_y1 - left_y0, right_y1 - right_y0), 1.0)
    return overlap / minimum_height >= 0.45


def _region_lines(words: list[tuple[Any, ...]]) -> list[str]:
    rows: list[list[tuple[Any, ...]]] = []
    for word in sorted(words, key=lambda item: (float(item[1]), float(item[0]))):
        matching = next(
            (row for row in reversed(rows[-4:]) if _same_baseline(row[0], word)),
            None,
        )
        if matching is None:
            rows.append([word])
        else:
            matching.append(word)
    lines = []
    for items in rows:
        items.sort(key=lambda item: float(item[0]))
        pieces: list[str] = []
        prior_x1: float | None = None
        prior_width = 4.0
        for item in items:
            token = str(item[4]).strip()
            if not token:
                continue
            x0, x1 = float(item[0]), float(item[2])
            if prior_x1 is not None:
                gap = x0 - prior_x1
                pieces.append("  " if gap >= max(8.0, prior_width * 1.8) else " ")
            pieces.append(token)
            prior_x1 = x1
            prior_width = max((x1 - x0) / max(len(token), 1), 1.0)
        line = "".join(pieces).strip()
        if line:
            lines.append(line)
    return lines


def _ocr_lines(page: fitz.Page) -> str:
    textpage = page.get_textpage_ocr(language="vie+eng", dpi=250, full=True)
    words = list(page.get_text("words", textpage=textpage, sort=True))
    if page.rect.width > page.rect.height * 1.2:
        midpoint = page.rect.width / 2.0
        regions = [
            [word for word in words if (float(word[0]) + float(word[2])) / 2.0 < midpoint],
            [word for word in words if (float(word[0]) + float(word[2])) / 2.0 >= midpoint],
        ]
    else:
        regions = [words]
    return "\n\n".join("\n".join(_region_lines(region)) for region in regions if region)


def _explicit_multilevel_header(context: str) -> str:
    lines = context.splitlines()
    first_data = next((index for index, line in enumerate(lines) if line.startswith("|")), None)
    if first_data is None:
        return context
    header_text = " ".join(lines[:first_data])
    periods = []
    for token in re.findall(r"(?:31/12/)?(?:19|20)\d{2}", header_text):
        if token not in periods:
            periods.append(token)
    normalized = normalize_text(header_text)
    explicit_value_share_pairs = (
        len(re.findall(r"gia\s*tr", normalized)) >= 2
        and len(re.findall(r"ty\s*trong", normalized)) >= 2
    )
    first_cells = [cell.strip() for cell in lines[first_data].strip().strip("|").split("|")]
    numeric_columns = max(len(first_cells) - 1, 0)
    if len(periods) == 2 and explicit_value_share_pairs and numeric_columns == 5:
        header = (
            f"| Chỉ tiêu | {periods[0]} Giá trị | {periods[0]} Tỷ trọng | "
            f"{periods[1]} Giá trị | {periods[1]} Tỷ trọng | % Tăng giảm |"
        )
    elif len(periods) == 2 and numeric_columns == 2:
        header = f"| Chỉ tiêu | {periods[0]} | {periods[1]} |"
    else:
        return context
    return "\n".join([*lines[:first_data], header, *lines[first_data:]])


def build_pool(
    precommit_path: Path = DEFAULT_PRECOMMIT,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> tuple[list[Any], dict[str, Any]]:
    precommit = json.loads(precommit_path.read_text(encoding="utf-8"))
    generator_parent = precommit["parents"]["frozen_generic_program_generator"]
    if _digest(ROOT / generator_parent["path"]) != generator_parent["sha256"]:
        raise RuntimeError("frozen generator hash changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    generator = _load_generator()
    examples = []
    pages_processed = 0
    pages_with_examples = 0
    per_source = []
    skipped: Counter[str] = Counter()
    for source in manifest["documents"]:
        if source["layout_stratum"] != "OCR_OR_HYBRID_TABLE":
            continue
        path = Path(source["document"])
        if _digest(path) != source["sha256"]:
            raise RuntimeError(f"source hash changed: {path.name}")
        document = fitz.open(path)
        source_examples = 0
        try:
            for page_info in source["pages"]:
                page_number = int(page_info["page"])
                pages_processed += 1
                raw_text = _ocr_lines(document[page_number - 1])
                context = "\n\n".join(
                    _explicit_multilevel_header(
                        generator._layout_to_table_context(region)
                    )
                    for region in raw_text.split("\n\n")
                    if region.strip()
                )
                page_examples, page_skipped = generator._page_examples(
                    context,
                    source_name=path.name,
                    page=page_number,
                    max_examples=24,
                    id_start=len(examples) + 1,
                    extraction_method="tesseract_vie_250dpi",
                )
                page_examples = [
                    replace(
                        example,
                        supervision={
                            **dict(example.supervision or {}),
                            "company_key": source["company_key"],
                        },
                    )
                    for example in page_examples
                ]
                examples.extend(page_examples)
                source_examples += len(page_examples)
                pages_with_examples += bool(page_examples)
                skipped.update(page_skipped)
        finally:
            document.close()
        per_source.append(
            {
                "source_key": source["source_key"],
                "company_key": source["company_key"],
                "layout_family": source["layout_family"],
                "pages": len(source["pages"]),
                "examples": source_examples,
            }
        )
    tesseract_version = subprocess.run(
        ["tesseract", "--version"], check=True, capture_output=True, text=True
    ).stdout.splitlines()[0]
    report = {
        "schema": "finhrm-qp3-core-add-diversity-ocr-generated-pool-v1",
        "created_at": "2026-08-20",
        "classification": "development_only_hash_locked_tesseract_ocr_program_pool",
        "inputs": {
            "precommit": {"path": precommit_path.relative_to(ROOT).as_posix(), "sha256": _digest(precommit_path)},
            "manifest": {"path": manifest_path.relative_to(ROOT).as_posix(), "sha256": _digest(manifest_path)},
            "generator": {"path": GENERATOR_PATH.relative_to(ROOT).as_posix(), "sha256": _digest(GENERATOR_PATH)},
        },
        "ocr": {
            "engine": tesseract_version,
            "languages": ["vie", "eng"],
            "dpi": 250,
            "ocr_text_used_as_semantic_proof": False,
        },
        "metrics": {
            "pages_processed": pages_processed,
            "pages_with_examples": pages_with_examples,
            "examples": len(examples),
            "operation_counts": dict(sorted(Counter(item.program.operation for item in examples).items())),
            "companies_with_examples": len(
                {item.supervision.get("company_key") for item in examples if item.supervision}
            ),
            "skipped": dict(sorted(skipped.items())),
        },
        "sources": per_source,
        "invariants": {
            "protected_evaluation_loaded": False,
            "abw_loaded": False,
            "runtime_changed": False,
            "model_checkpoint_or_threshold_changed": False,
        },
    }
    return examples, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precommit", type=Path, default=DEFAULT_PRECOMMIT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite OCR pool artifacts")
    examples, report = build_pool(args.precommit, args.manifest)
    generator = _load_generator()
    generator.write_program_examples(examples, args.output)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
