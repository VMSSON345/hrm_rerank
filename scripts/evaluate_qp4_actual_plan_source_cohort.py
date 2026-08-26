#!/usr/bin/env python3
"""Validate and baseline the development-only QP4 actual/plan source cohort."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import AbstractContextManager
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from pathlib import Path
import re
import signal
from typing import Any

from hrm_finance_qa.question_planner import compile_question_plan
from scripts.evaluate_qp4_development_baseline import (
    _answer_signature,
    _dimension_projection,
    _semantic_projection,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "finhrm_qp4_actual_plan_ratio_direction_source_cohort_v1.json"
DEFAULT_RAW = ROOT / "reports" / "finhrm_qp4_actual_plan_source_baseline_v1.json"
DEFAULT_REPORT = ROOT / "reports" / "finhrm_qp4_actual_plan_source_cohort_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "finhrm_qp4_actual_plan_source_cohort_v1.md"
PROTECTED = re.compile(r"(?:^|[/_.-])(test|holdout|final|blind|validation)(?:[/_.-]|$)", re.I)


def _digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    if PROTECTED.search(path.resolve().as_posix()):
        raise ValueError(f"Protected input is forbidden: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_value(anchor: dict[str, Any], program: dict[str, Any]) -> Decimal:
    values = [Decimal(anchor["evidence"][name]["value"]) for name in program["operands"]]
    operation = program["operation"]
    if operation == "SELECT":
        return values[0]
    if operation == "SUBTRACT":
        return values[0] - values[1]
    if operation == "RATIO_PERCENT":
        return values[0] / values[1] * Decimal("100")
    raise ValueError(f"Unsupported gold operation: {operation}")


def _numeric_match(observed: Any, expected: Decimal, operation: str) -> bool:
    if observed is None:
        return False
    try:
        value = Decimal(str(observed))
    except InvalidOperation:
        return False
    if operation == "RATIO_PERCENT":
        return abs(value - expected) <= Decimal("0.000001")
    return value == expected


def _verify_config(config: dict[str, Any]) -> dict[str, Any]:
    for item in config["parents"].values():
        path = ROOT / item["path"]
        if _digest(path) != item["sha256"]:
            raise ValueError(f"Parent hash mismatch: {item['path']}")
    checkpoint = config["checkpoint"]
    if _digest(ROOT / checkpoint["path"]) != checkpoint["sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    for relative, expected in config["runtime_hashes"].items():
        if _digest(ROOT / relative) != expected:
            raise ValueError(f"Runtime hash mismatch: {relative}")

    intake = _load(ROOT / config["parents"]["development_pdf_intake"]["path"])
    layout_audit = _load(ROOT / config["parents"]["development_layout_audit"]["path"])
    if not str(intake["classification"]).startswith("development_only"):
        raise ValueError("PDF intake is not development-only")
    if layout_audit["classification"] != "development_only":
        raise ValueError("Layout audit is not development-only")

    dimensions = set(config["dimensions"])
    diagnosis = set(config["selection_contract"]["diagnosis_company_keys"])
    anchors = config["anchors"]
    if not anchors:
        raise ValueError("Empty source cohort")
    for anchor in anchors:
        if anchor["company_key"] in diagnosis:
            raise ValueError(f"Diagnosis-company overlap: {anchor['anchor_id']}")
        source_path = ROOT / anchor["source_path"]
        markdown_path = ROOT / anchor["markdown"]["path"]
        if PROTECTED.search(source_path.resolve().as_posix()):
            raise ValueError(f"Protected source path: {source_path}")
        if _digest(source_path) != anchor["source_sha256"]:
            raise ValueError(f"Source hash mismatch: {anchor['anchor_id']}")
        if _digest(markdown_path) != anchor["markdown"]["sha256"]:
            raise ValueError(f"Markdown hash mismatch: {anchor['anchor_id']}")
        intake_row = intake["documents"].get(anchor["source_name"])
        if not intake_row or intake_row["company_key"] != anchor["company_key"]:
            raise ValueError(f"Intake identity mismatch: {anchor['anchor_id']}")
        if intake_row["source_sha256"] != anchor["source_sha256"]:
            raise ValueError(f"Intake source mismatch: {anchor['anchor_id']}")
        layout_row = layout_audit["documents"].get(anchor["source_name"])
        if not layout_row or layout_row["layout_family"] != anchor["document_layout_family"]:
            raise ValueError(f"Layout audit mismatch: {anchor['anchor_id']}")
        if anchor["visual_review"]["decision"] != "pass":
            raise ValueError(f"Visual review is not approved: {anchor['anchor_id']}")
        if set(anchor["pairs"]) != dimensions:
            raise ValueError(f"Dimension coverage mismatch: {anchor['anchor_id']}")
        evidence = anchor["evidence"]
        if evidence["actual_2023"]["column_role"] != "ACTUAL":
            raise ValueError("Actual role mismatch")
        if evidence["plan_2023"]["column_role"] != "PLAN":
            raise ValueError("Plan role mismatch")
        if evidence["actual_2022"]["year"] != 2022:
            raise ValueError("Prior-actual year mismatch")
        for dimension, pair in anchor["pairs"].items():
            base = compile_question_plan(pair["base_question"])
            mutated = compile_question_plan(pair["mutated_question"])
            if _dimension_projection(base, dimension) == _dimension_projection(mutated, dimension):
                raise ValueError(f"Compiler did not detect {dimension}: {anchor['anchor_id']}")
            for name in ("base_program", "mutated_program"):
                _expected_value(anchor, pair[name])

    contract = config["selection_contract"]
    coverage = {
        "anchors": len(anchors),
        "companies": len({row["company_key"] for row in anchors}),
        "sources": len({row["source_sha256"] for row in anchors}),
        "document_layouts": len({row["document_layout_family"] for row in anchors}),
        "table_layouts": len({row["table_layout_family"] for row in anchors}),
    }
    minimums = {
        "anchors": contract["minimum_positive_anchors_per_dimension"],
        "companies": contract["minimum_distinct_companies_per_dimension"],
        "sources": contract["minimum_distinct_sources_per_dimension"],
        "document_layouts": contract["minimum_distinct_document_layouts_per_dimension"],
        "table_layouts": contract["minimum_distinct_table_layouts_per_dimension"],
    }
    if any(coverage[key] < minimums[key] for key in coverage):
        raise ValueError(f"Source diversity gate failed: {coverage}")
    return coverage


class _Timeout(AbstractContextManager[None]):
    def __init__(self, seconds: int) -> None:
        self.seconds = seconds
        self.previous: Any = None

    def __enter__(self) -> None:
        def handler(signum: int, frame: Any) -> None:
            raise TimeoutError(f"Source query exceeded {self.seconds} seconds")

        self.previous = signal.signal(signal.SIGALRM, handler)
        signal.alarm(self.seconds)
        return None

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self.previous)


def _run_question(engine: Any, question: str, source_path: Path, seconds: int) -> dict[str, Any]:
    try:
        with _Timeout(seconds):
            answer = engine.answer(question, source_path, mineru_auto_generate=False)
        return {"signature": _answer_signature(answer)}
    except Exception as exc:
        return {"error": {"type": type(exc).__name__, "message": str(exc)}}


def run_source_baseline(config: dict[str, Any], raw_path: Path) -> dict[str, Any]:
    from hrm_finance_qa.document_hrm import DocumentHRMBundle
    from hrm_finance_qa.document_reasoning_engine import DocumentHRMReasoningEngine

    _verify_config(config)
    checkpoint = ROOT / config["checkpoint"]["path"]
    engine = DocumentHRMReasoningEngine(DocumentHRMBundle.load(checkpoint, device="cpu"))
    seconds = int(config["selection_contract"]["per_query_timeout_seconds"])
    results: list[dict[str, Any]] = []
    total = len(config["anchors"]) * len(config["dimensions"])
    position = 0
    for anchor in config["anchors"]:
        source_path = ROOT / anchor["source_path"]
        for dimension in config["dimensions"]:
            position += 1
            print(f"[{position}/{total}] {dimension} {anchor['anchor_id']}", flush=True)
            pair = anchor["pairs"][dimension]
            base = _run_question(engine, pair["base_question"], source_path, seconds)
            mutated = _run_question(engine, pair["mutated_question"], source_path, seconds)
            for label, observed, program in (
                ("base", base, pair["base_program"]),
                ("mutated", mutated, pair["mutated_program"]),
            ):
                expected = _expected_value(anchor, program)
                signature = observed.get("signature", {})
                observed[f"expected_{label}"] = {
                    "operation": program["operation"],
                    "value": str(expected),
                    "operands": program["operands"],
                }
                observed["gold_answer_match"] = (
                    signature.get("action") == "ANSWER"
                    and signature.get("operation") == program["operation"]
                    and _numeric_match(signature.get("value"), expected, program["operation"])
                )
            base_signature = base.get("signature")
            mutated_signature = mutated.get("signature")
            same_observed_winner = bool(
                base_signature
                and mutated_signature
                and all(
                    base_signature.get(key) == mutated_signature.get(key)
                    for key in ("action", "operation", "value", "unit", "evidence")
                )
            )
            results.append(
                {
                    "anchor_id": anchor["anchor_id"],
                    "company_key": anchor["company_key"],
                    "source_name": anchor["source_name"],
                    "source_sha256": anchor["source_sha256"],
                    "document_layout_family": anchor["document_layout_family"],
                    "table_layout_family": anchor["table_layout_family"],
                    "dimension": dimension,
                    "base_question": pair["base_question"],
                    "mutated_question": pair["mutated_question"],
                    "base": base,
                    "mutated": mutated,
                    "same_observed_winner": same_observed_winner,
                    "old_winner_false_accept": bool(base["gold_answer_match"] and same_observed_winner),
                    "paired_gold_answer": bool(base["gold_answer_match"] and mutated["gold_answer_match"]),
                }
            )
    raw = {
        "schema": "finhrm-qp4-actual-plan-source-baseline-v1",
        "created_at": config["created_at"],
        "classification": "development_only_real_pdf_qp4_baseline_runtime_unchanged",
        "inputs": {
            "config": DEFAULT_CONFIG.relative_to(ROOT).as_posix(),
            "config_sha256": _digest(DEFAULT_CONFIG),
            "checkpoint_sha256": _digest(checkpoint),
        },
        "results": results,
        "runtime_hashes": {path: _digest(ROOT / path) for path in config["runtime_hashes"]},
        "invariants": config["invariants"],
    }
    raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return raw


def build_report(config_path: Path = DEFAULT_CONFIG, raw_path: Path = DEFAULT_RAW) -> dict[str, Any]:
    config = _load(config_path)
    coverage = _verify_config(config)
    raw = _load(raw_path)
    if raw["inputs"]["config_sha256"] != _digest(config_path):
        raise ValueError("Raw baseline config hash mismatch")
    if raw["runtime_hashes"] != config["runtime_hashes"]:
        raise ValueError("Runtime changed during baseline")

    by_dimension: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw["results"]:
        by_dimension[row["dimension"]].append(row)
    dimensions: dict[str, Any] = {}
    for dimension in config["dimensions"]:
        rows = by_dimension[dimension]
        base_outcomes = Counter(
            row["base"].get("signature", {}).get("action")
            or f"ERROR:{row['base'].get('error', {}).get('type', 'UNKNOWN')}"
            for row in rows
        )
        mutated_outcomes = Counter(
            row["mutated"].get("signature", {}).get("action")
            or f"ERROR:{row['mutated'].get('error', {}).get('type', 'UNKNOWN')}"
            for row in rows
        )
        dimensions[dimension] = {
            "source_evidence_coverage_confirmed": len(rows) == len(config["anchors"]),
            "anchors": len(rows),
            "companies": len({row["company_key"] for row in rows}),
            "sources": len({row["source_sha256"] for row in rows}),
            "document_layouts": len({row["document_layout_family"] for row in rows}),
            "table_layouts": len({row["table_layout_family"] for row in rows}),
            "base_gold_answers": sum(row["base"]["gold_answer_match"] for row in rows),
            "mutated_gold_answers": sum(row["mutated"]["gold_answer_match"] for row in rows),
            "paired_gold_answers": sum(row["paired_gold_answer"] for row in rows),
            "same_observed_winners": sum(row["same_observed_winner"] for row in rows),
            "old_winner_false_accepts": sum(row["old_winner_false_accept"] for row in rows),
            "base_outcomes": dict(sorted(base_outcomes.items())),
            "mutated_outcomes": dict(sorted(mutated_outcomes.items())),
        }
        dimensions[dimension]["runtime_gate_passed"] = (
            dimensions[dimension]["paired_gold_answers"] == len(rows)
            and dimensions[dimension]["old_winner_false_accepts"] == 0
        )

    failed = [name for name, row in dimensions.items() if not row["runtime_gate_passed"]]
    diagnosis = []
    for name, row in dimensions.items():
        consistent_generic_failure = (
            row["anchors"] >= 3
            and row["document_layouts"] >= 3
            and row["runtime_gate_passed"] is False
        )
        diagnosis.append(
            {
                "dimension": name,
                "compiler_dimension_detected": f"{row['anchors']}/{row['anchors']}",
                "source_runtime_gate_passed": row["runtime_gate_passed"],
                "consistent_generic_failure": consistent_generic_failure,
                "layer": "retrieval_or_binding_unisolated",
                "automatic_whitelist_eligible": False,
                "runtime_edit_authorized": False,
            }
        )
    report = {
        "schema": "finhrm-qp4-actual-plan-source-cohort-report-v1",
        "created_at": config["created_at"],
        "classification": config["classification"],
        "cohort_frozen": True,
        "source_evidence_coverage_complete": all(
            row["source_evidence_coverage_confirmed"] for row in dimensions.values()
        ),
        "coverage": coverage,
        "dimensions": dimensions,
        "runtime_gates_passed": not failed,
        "failed_runtime_dimensions": failed,
        "diagnosis": diagnosis,
        "diagnosis_conclusion": (
            "The source cohort is diverse and proof-complete. Compiler detection passes, while source "
            "execution fails closed. Retrieval or binding must be isolated before any runtime authorization; "
            "source coverage alone does not authorize an edit."
        ),
        "preauthorization_review": {
            "unit_scale_question_planner_generic_evidence_retained": True,
            "new_dimension_runtime_edit_authorized": False,
            "retrieval_or_binding_edit_authorized": False,
            "reason": "This step closes source evidence coverage and records baseline behavior only."
        },
        "runtime_changed": False,
        "runtime_hashes": config["runtime_hashes"],
        "checkpoint_sha256": config["checkpoint"]["sha256"],
        "qp4_runtime_change_authorized": False,
        "qp5_authorized": False,
        "phase_d": "locked",
        "raw_baseline": {
            "path": raw_path.relative_to(ROOT).as_posix(),
            "sha256": _digest(raw_path),
        },
    }
    return report


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# QP4 actual/plan ratio-direction source cohort v1",
        "",
        f"- Cohort frozen: `{str(report['cohort_frozen']).lower()}`",
        f"- Source evidence coverage complete: `{str(report['source_evidence_coverage_complete']).lower()}`",
        f"- Runtime gates passed: `{str(report['runtime_gates_passed']).lower()}`",
        f"- Coverage: {report['coverage']['companies']} companies / {report['coverage']['sources']} PDFs / "
        f"{report['coverage']['document_layouts']} document layouts / {report['coverage']['table_layouts']} table layouts",
        "",
        "| Dimension | Base gold | Mutated gold | Paired gold | Old-winner false accepts | Runtime gate |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, row in report["dimensions"].items():
        lines.append(
            f"| {name} | {row['base_gold_answers']}/{row['anchors']} | "
            f"{row['mutated_gold_answers']}/{row['anchors']} | {row['paired_gold_answers']}/{row['anchors']} | "
            f"{row['old_winner_false_accepts']} | {'PASS' if row['runtime_gate_passed'] else 'FAIL'} |"
        )
    lines += [
        "",
        "Source diversity is now confirmed independently of runtime behavior. Runtime remains frozen; "
        "safe abstention is not counted as an answer and synthetic layouts are not used.",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(report: dict[str, Any], report_path: Path, markdown_path: Path) -> None:
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--run-source-baseline", action="store_true")
    args = parser.parse_args()
    config = _load(args.config)
    if args.run_source_baseline:
        run_source_baseline(config, args.raw)
    report = build_report(args.config, args.raw)
    write_outputs(report, args.report, args.markdown)
    print(json.dumps({
        "coverage": report["coverage"],
        "runtime_gates_passed": report["runtime_gates_passed"],
        "failed_runtime_dimensions": report["failed_runtime_dimensions"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
