#!/usr/bin/env python3
"""Evaluate and lock the QP4 question-planner UNIT_SCALE implementation."""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

from hrm_finance_qa.question_planner import compile_question_plan
from scripts.evaluate_qp4_development_baseline import (
    _QueryTimeout,
    _answer_signature,
    _compiler_baseline,
    _dimension_projection,
    _parent_signature,
    _semantic_projection,
    _source_anchors,
    _source_counterfactual,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "finhrm_qp4_unit_scale_question_planner_implementation_v1.json"
DEFAULT_RAW = ROOT / "reports" / "finhrm_qp4_unit_scale_question_planner_source_replay_v1.json"
DEFAULT_SAFE_REPLAY = ROOT / "reports" / "finhrm_qp4_unit_scale_question_planner_safe_replay_v1.json"
DEFAULT_SAFE_MARKDOWN = ROOT / "reports" / "finhrm_qp4_unit_scale_question_planner_safe_replay_v1.md"
DEFAULT_SAFE_LOCK = ROOT / "reports" / "finhrm_qp4_unit_scale_question_planner_safe_replay_lock_v1.json"
DEFAULT_REPORT = ROOT / "reports" / "finhrm_qp4_unit_scale_question_planner_implementation_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "finhrm_qp4_unit_scale_question_planner_implementation_v1.md"
DEFAULT_LOCK = ROOT / "reports" / "finhrm_qp4_unit_scale_question_planner_implementation_lock_v1.json"
FUNCTIONAL_TEST = ROOT / "tests" / "test_qp4_unit_scale_question_planner_implementation.py"
REPORT_TEST = ROOT / "tests" / "test_qp4_unit_scale_question_planner_implementation_report.py"
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


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _digest(path),
    }


def _verify_config(config: dict[str, Any]) -> None:
    for parent in config["parents"].values():
        path = ROOT / parent["path"]
        if _digest(path) != parent["sha256"]:
            raise ValueError(f"Parent hash mismatch: {parent['path']}")
    checkpoint = config["checkpoint"]
    if _digest(ROOT / checkpoint["path"]) != checkpoint["sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    change = config["runtime_change"]
    if change["whitelist"] != ["src/hrm_finance_qa/question_planner.py"]:
        raise ValueError("Runtime whitelist is not exact")
    if _digest(ROOT / change["whitelist"][0]) != change["after_sha256"]:
        raise ValueError("Question planner postimplementation hash mismatch")
    for relative, expected in config["frozen_runtime_hashes"].items():
        if _digest(ROOT / relative) != expected:
            raise ValueError(f"Frozen runtime hash mismatch: {relative}")
    if any(config["invariants"].values()):
        raise ValueError("A release invariant was opened")


def run_source_unit_scale(config_path: Path, raw_path: Path) -> dict[str, Any]:
    from hrm_finance_qa.document_hrm import DocumentHRMBundle
    from hrm_finance_qa.document_reasoning_engine import DocumentHRMReasoningEngine

    config = _load(config_path)
    _verify_config(config)
    development = _load(ROOT / config["parents"]["development_cohort"]["path"])
    anchors = _source_anchors(development)
    checkpoint = ROOT / config["checkpoint"]["path"]
    engine = DocumentHRMReasoningEngine(DocumentHRMBundle.load(checkpoint, device="cpu"))
    timeout_seconds = int(
        development["source_backed_cohort"]["per_query_timeout_seconds"]
    )
    results: list[dict[str, Any]] = []
    for position, (_, row) in enumerate(anchors, start=1):
        question = _source_counterfactual(str(row["question"]), "UNIT_SCALE")
        plan = compile_question_plan(question)
        parent = _parent_signature(row)
        print(f"[{position}/{len(anchors)}] UNIT_SCALE {row['example_id']}", flush=True)
        try:
            with _QueryTimeout(timeout_seconds):
                answer = engine.answer(
                    question,
                    ROOT / row["source_path"],
                    mineru_auto_generate=False,
                )
            signature = _answer_signature(answer)
            same_old_winner = all(
                signature.get(key) == parent.get(key)
                for key in ("action", "operation", "value", "unit", "evidence")
            )
            observation: dict[str, Any] = {"observed_signature": signature}
        except Exception as exc:
            same_old_winner = False
            observation = {
                "error": {"type": type(exc).__name__, "message": str(exc)}
            }
        results.append(
            {
                "example_id": row["example_id"],
                "company_key": row["company_key"],
                "source_key": row["source_sha256"],
                "source_path": row["source_path"],
                "layout_family": row["layout_family"],
                "question": question,
                "requested_scale": plan.requested_scale,
                "typed_scale_detected": plan.requested_scale == "million_vnd",
                "parent_signature": parent,
                **observation,
                "old_winner_false_accept": same_old_winner,
            }
        )
    raw = {
        "schema": "finhrm-qp4-unit-scale-question-planner-source-replay-v1",
        "created_at": config["created_at"],
        "classification": "development_only_real_pdf_unit_scale_postimplementation_replay",
        "dimension": "UNIT_SCALE",
        "runs": len(results),
        "companies": len({row["company_key"] for row in results}),
        "sources": len({row["source_key"] for row in results}),
        "real_layouts": len({row["layout_family"] for row in results}),
        "results": results,
        "runtime_hashes": {
            config["runtime_change"]["whitelist"][0]: config["runtime_change"]["after_sha256"],
            **config["frozen_runtime_hashes"],
        },
        "checkpoint_sha256": config["checkpoint"]["sha256"],
        "abw_records": 0,
        "protected_records": [],
    }
    raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return raw


def _non_unit_projection(plan: Any) -> dict[str, Any]:
    projection = deepcopy(_semantic_projection(plan))
    projection.pop("requested_unit_family", None)
    projection.pop("requested_scale", None)
    projection.pop("abstention_reason", None)
    for slot in projection["operand_slots"]:
        slot.pop("unit_family", None)
    for node in projection["program_dag"]:
        node.pop("output_unit_family", None)
    return projection


def _parser_checks() -> dict[str, Any]:
    cases = (
        ("Doanh thu là bao nhiêu VND?", "vnd"),
        ("Doanh thu là bao nhiêu đồng?", "vnd"),
        ("Doanh thu theo đơn vị đồng là bao nhiêu?", "vnd"),
        ("Doanh thu theo đồng là bao nhiêu?", "vnd"),
        ("Doanh thu là bao nhiêu nghìn đồng?", "thousand_vnd"),
        ("Doanh thu là bao nhiêu triệu đồng?", "million_vnd"),
        ("Doanh thu là bao nhiêu tỷ đồng?", "billion_vnd"),
    )
    details = [
        {
            "expected": expected,
            "actual": compile_question_plan(question).requested_scale,
        }
        for question, expected in cases
    ]
    missing = compile_question_plan("Doanh thu năm 2024 là bao nhiêu?")
    conflict = compile_question_plan(
        "Doanh thu theo nghìn đồng hoặc triệu đồng là bao nhiêu?"
    )
    incompatible = compile_question_plan(
        "Biên lợi nhuận là bao nhiêu phần trăm theo triệu đồng?"
    )
    return {
        "explicit": {
            "passed": sum(row["actual"] == row["expected"] for row in details),
            "total": len(details),
            "details": details,
        },
        "missing_scale_is_none": missing.requested_scale is None,
        "conflict_fails_closed": conflict.requested_scale == "conflict"
        and conflict.abstention_reason == "conflicting_requested_scale",
        "incompatible_fails_closed": incompatible.requested_scale == "million_vnd"
        and incompatible.abstention_reason == "incompatible_requested_scale",
    }


def build_report(
    config_path: Path = DEFAULT_CONFIG,
    raw_path: Path = DEFAULT_RAW,
    safe_replay_path: Path = DEFAULT_SAFE_REPLAY,
) -> dict[str, Any]:
    config = _load(config_path)
    _verify_config(config)
    development = _load(ROOT / config["parents"]["development_cohort"]["path"])
    raw = _load(raw_path)
    safe = _load(safe_replay_path)
    compiler = _compiler_baseline(development)
    unit_rows = [
        row
        for row in compiler["counterfactual"]["details"]
        if row["dimension"] == "UNIT_SCALE"
    ]
    retained = sum(
        _non_unit_projection(compile_question_plan(row["base_question"]))
        == _non_unit_projection(compile_question_plan(row["mutated_question"]))
        for row in unit_rows
    )
    unit = compiler["counterfactual"]["by_dimension"]["UNIT_SCALE"]
    parser = _parser_checks()
    source_typed = sum(bool(row["typed_scale_detected"]) for row in raw["results"])
    source_false_accepts = sum(
        bool(row["old_winner_false_accept"]) for row in raw["results"]
    )
    safe_results = safe["results"]
    expected_safe = config["frozen_safe_replay_contract"]
    checks = {
        "compiler_unit_scale_13_of_13": unit["detected"] == unit["pairs"] == 13,
        "compiler_non_unit_semantics_13_of_13": retained == len(unit_rows) == 13,
        "canonical_13_of_13": compiler["canonical"]["exact"]
        == compiler["canonical"]["total"]
        == 13,
        "ambiguity_3_of_3": compiler["ambiguity"]["exact"]
        == compiler["ambiguity"]["total"]
        == 3,
        "surface_baseline_73_of_91": compiler["surface"]["exact"] == 73
        and compiler["surface"]["total"] == 91,
        "explicit_parser_7_of_7": parser["explicit"]["passed"]
        == parser["explicit"]["total"]
        == 7,
        "missing_scale_is_none": parser["missing_scale_is_none"],
        "conflict_fails_closed": parser["conflict_fails_closed"],
        "incompatible_fails_closed": parser["incompatible_fails_closed"],
        "source_replay_5_companies_5_sources_3_layouts": raw["runs"] == 5
        and raw["companies"] == 5
        and raw["sources"] == 5
        and raw["real_layouts"] == 3,
        "source_typed_detection_5_of_5": source_typed == raw["runs"] == 5,
        "safe_actions_exact": safe_results["gold_actions"] == expected_safe["actions"],
        "safe_execution_363_of_363": safe_results["gold_execution_correct"]
        == safe_results["gold_total"]
        == 363,
        "safe_permutation_363_of_363": safe_results["permutation_exact"]
        == safe_results["permutation_total"]
        == 363,
        "safe_counterfactual_141_zero_false_accept": safe_results[
            "counterfactual_generated"
        ]
        == expected_safe["counterfactuals"]
        == 141
        and safe_results["counterfactual_false_accepts"] == 0,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    typed_stage_passed = not failed
    source_target_met = source_false_accepts == config["source_replay_contract"][
        "old_winner_false_accept_target"
    ]
    return {
        "schema": "finhrm-qp4-unit-scale-question-planner-implementation-report-v1",
        "created_at": config["created_at"],
        "classification": config["classification"],
        "typed_intent_stage_passed": typed_stage_passed,
        "failed_gates": failed,
        "runtime_change": config["runtime_change"],
        "compiler": {
            "unit_scale_detection": {"detected": unit["detected"], "total": unit["pairs"]},
            "non_unit_semantics_retained": {"passed": retained, "total": len(unit_rows)},
            "canonical": compiler["canonical"],
            "ambiguity": compiler["ambiguity"],
            "surface": {
                "exact": compiler["surface"]["exact"],
                "total": compiler["surface"]["total"],
            },
        },
        "parser": parser,
        "source_replay": {
            "runs": raw["runs"],
            "companies": raw["companies"],
            "sources": raw["sources"],
            "real_layouts": raw["real_layouts"],
            "typed_scale_detected": source_typed,
            "old_winner_false_accepts": source_false_accepts,
            "old_winner_false_accept_target": config["source_replay_contract"][
                "old_winner_false_accept_target"
            ],
            "target_met": source_target_met,
            "target_required_to_close_typed_intent_stage": False,
        },
        "safe_replay": {
            "actions": safe_results["gold_actions"],
            "execution": f"{safe_results['gold_execution_correct']}/{safe_results['gold_total']}",
            "permutation": f"{safe_results['permutation_exact']}/{safe_results['permutation_total']}",
            "counterfactuals": safe_results["counterfactual_generated"],
            "counterfactual_false_accepts": safe_results[
                "counterfactual_false_accepts"
            ],
        },
        "checks": checks,
        "source_residual_reported": True,
        "downstream_runtime_change_authorized": False,
        "qp4_complete": False,
        "qp5_authorized": False,
        "phase_d": "locked",
        "authorized_next_step": (
            "QP4-UNIT-SCALE-SOURCE-RESIDUAL-ISOLATION-PRECOMMIT: isolate the "
            "development-only downstream failure without editing runtime; retrieval, binding, "
            "verification, conversion, QP5 and Phase D remain locked."
            if typed_stage_passed and not source_target_met
            else "Resolve failed typed-intent gates without widening the runtime whitelist."
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    compiler = report["compiler"]
    source = report["source_replay"]
    safe = report["safe_replay"]
    return "\n".join(
        [
            "# QP4 UNIT_SCALE question-planner implementation v1",
            "",
            f"- Typed-intent stage: `{'PASS' if report['typed_intent_stage_passed'] else 'FAIL'}`.",
            f"- Compiler UNIT_SCALE: `{compiler['unit_scale_detection']['detected']}/{compiler['unit_scale_detection']['total']}`.",
            f"- Non-unit semantics retained: `{compiler['non_unit_semantics_retained']['passed']}/{compiler['non_unit_semantics_retained']['total']}`.",
            f"- Source typed detection: `{source['typed_scale_detected']}/{source['runs']}` across `{source['companies']}` companies and `{source['real_layouts']}` real layouts.",
            f"- Source old-winner false accepts: `{source['old_winner_false_accepts']}`; target `{source['old_winner_false_accept_target']}`. The residual is reported and does not authorize downstream runtime edits.",
            f"- Frozen replay: actions `{safe['actions']}`, execution `{safe['execution']}`, permutation `{safe['permutation']}`, counterfactual false accepts `{safe['counterfactual_false_accepts']}`.",
            "- QP4 is not complete; QP5 and Phase D remain locked.",
            "",
            f"Next: {report['authorized_next_step']}",
            "",
        ]
    )


def write_outputs(
    report: dict[str, Any],
    report_path: Path = DEFAULT_REPORT,
    markdown_path: Path = DEFAULT_MARKDOWN,
    lock_path: Path = DEFAULT_LOCK,
) -> None:
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    artifacts = [
        DEFAULT_CONFIG,
        Path(__file__).resolve(),
        FUNCTIONAL_TEST,
        REPORT_TEST,
        DEFAULT_RAW,
        DEFAULT_SAFE_REPLAY,
        DEFAULT_SAFE_MARKDOWN,
        DEFAULT_SAFE_LOCK,
        report_path,
        markdown_path,
    ]
    missing = [path for path in artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(str(path) for path in missing))
    lock = {
        "schema": "finhrm-qp4-unit-scale-question-planner-implementation-lock-v1",
        "created_at": report["created_at"],
        "classification": report["classification"],
        "status": "complete" if report["typed_intent_stage_passed"] else "failed",
        "typed_intent_stage_passed": report["typed_intent_stage_passed"],
        "source_target_met": report["source_replay"]["target_met"],
        "source_residual_reported": report["source_residual_reported"],
        "runtime_change": report["runtime_change"],
        "downstream_runtime_change_authorized": False,
        "artifacts": [_artifact(path) for path in artifacts],
        "qp4_complete": False,
        "qp5_authorized": False,
        "phase_d": "locked",
        "authorized_next_step": report["authorized_next_step"],
    }
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--safe-replay", type=Path, default=DEFAULT_SAFE_REPLAY)
    parser.add_argument("--run-source", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.run_source:
        run_source_unit_scale(args.config, args.raw)
    report = build_report(args.config, args.raw, args.safe_replay)
    if args.write:
        write_outputs(report)
    print(
        json.dumps(
            {
                "typed_intent_stage_passed": report["typed_intent_stage_passed"],
                "failed_gates": report["failed_gates"],
                "compiler_unit_scale": report["compiler"]["unit_scale_detection"],
                "source_old_winner_false_accepts": report["source_replay"][
                    "old_winner_false_accepts"
                ],
                "source_target_met": report["source_replay"]["target_met"],
                "qp4_complete": report["qp4_complete"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report["typed_intent_stage_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
