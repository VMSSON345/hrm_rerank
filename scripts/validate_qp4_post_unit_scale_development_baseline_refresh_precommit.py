#!/usr/bin/env python3
"""Validate and lock the no-runtime QP4 post-UNIT_SCALE baseline refresh contract."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

from hrm_finance_qa.question_planner import compile_question_plan, compile_question_proof_contract


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "finhrm_qp4_post_unit_scale_development_baseline_refresh_precommit_v1.json"
DEFAULT_REPORT = ROOT / "reports" / "finhrm_qp4_post_unit_scale_development_baseline_refresh_precommit_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "finhrm_qp4_post_unit_scale_development_baseline_refresh_precommit_v1.md"
DEFAULT_LOCK = ROOT / "reports" / "finhrm_qp4_post_unit_scale_development_baseline_refresh_precommit_lock_v1.json"
TEST_PATH = ROOT / "tests" / "test_qp4_post_unit_scale_development_baseline_refresh_precommit.py"
PROTECTED = re.compile(r"(?:^|[/_.-])(test|holdout|final|blind|validation|sealed)(?:[/_.-]|$)", re.I)


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


def _check(condition: bool, name: str, failed: list[str]) -> None:
    if not condition:
        failed.append(name)


def build_report(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _load(config_path)
    failed: list[str] = []
    for name, parent in config["parents"].items():
        path = ROOT / parent["path"]
        _check(path.is_file() and _digest(path) == parent["sha256"], f"parent:{name}", failed)
    checkpoint = config["checkpoint"]
    _check(_digest(ROOT / checkpoint["path"]) == checkpoint["sha256"], "checkpoint", failed)
    for relative, expected in config["frozen_runtime_hashes"].items():
        _check(_digest(ROOT / relative) == expected, f"runtime:{relative}", failed)

    parents = config["parents"]
    reclosure_report = _load(ROOT / parents["unit_scale_reclosure_report"]["path"])
    reclosure_lock = _load(ROOT / parents["unit_scale_reclosure_lock"]["path"])
    development_config = _load(ROOT / parents["historical_development_cohort"]["path"])
    baseline = _load(ROOT / parents["historical_development_baseline_report"]["path"])
    baseline_lock = _load(ROOT / parents["historical_development_baseline_lock"]["path"])
    actual_config = _load(ROOT / parents["actual_plan_source_cohort"]["path"])
    actual_report = _load(ROOT / parents["actual_plan_source_report"]["path"])
    actual_lock = _load(ROOT / parents["actual_plan_source_lock"]["path"])
    safe = _load(ROOT / parents["post_unit_scale_safe_replay"]["path"])
    scope = config["refresh_scope"]
    gates = config["refresh_gates"]

    _check(reclosure_report["unit_scale_gate_reclosed"] is True, "unit_scale_reclosed", failed)
    _check(reclosure_report["failed_gates"] == [], "unit_scale_no_failures", failed)
    _check(reclosure_lock["status"] == "complete" and reclosure_lock["unit_scale_gate_reclosed"] is True, "reclosure_lock", failed)
    _check(reclosure_lock["runtime_file_whitelist_for_next_step"] == [], "reclosure_runtime_whitelist_empty", failed)
    _check(
        reclosure_lock["authorized_next_step"].startswith("QP4-POST-UNIT-SCALE-DEVELOPMENT-BASELINE-REFRESH-PRECOMMIT"),
        "refresh_precommit_authorized",
        failed,
    )

    _check(baseline["cohort_and_baseline_complete"] is True, "historical_baseline_complete", failed)
    _check(baseline["baseline_gates_passed"] is False, "historical_baseline_failed", failed)
    _check(
        baseline["failed_gates"]
        == [
            "compiler_counterfactual_detection_rate",
            "compiler_surface_semantic_invariance_rate",
            "source_counterfactual_old_winner_false_accepts",
            "source_surface_winner_invariance_rate",
        ],
        "historical_failed_gates",
        failed,
    )
    _check(
        baseline["source_dimensions_not_confirmed"] == ["COLUMN_ROLE", "DIRECTION", "REFERENCE_BASE"],
        "historical_unconfirmed_dimensions",
        failed,
    )
    _check(baseline_lock["status"] == "complete" and baseline_lock["runtime_changed"] is False, "historical_baseline_lock", failed)
    compiler = baseline["compiler"]
    _check(compiler["canonical"]["total"] == scope["compiler_tier"]["core_answer_families"] == 13, "compiler_families", failed)
    _check(compiler["ambiguity"]["total"] == scope["compiler_tier"]["ambiguity_controls"] == 3, "compiler_controls", failed)
    _check(compiler["surface"]["total"] == scope["compiler_tier"]["surface_variants"] == 91, "compiler_surface", failed)
    _check(compiler["counterfactual"]["total"] == scope["compiler_tier"]["semantic_minimal_pairs"] == 57, "compiler_pairs", failed)

    add = scope["add_source_tier"]
    historical_source = baseline["source_backed"]
    _check(
        (historical_source["eligible_anchors"], historical_source["companies"], historical_source["source_documents"], historical_source["real_layout_families"])
        == (add["anchors"], add["companies"], add["sources"], add["real_layouts"])
        == (5, 5, 5, 3),
        "add_source_coverage",
        failed,
    )
    _check(len(development_config["source_backed_cohort"]["meaning_preserving_categories"]) == add["surface_categories"] == 7, "add_surface_categories", failed)
    _check(development_config["source_backed_cohort"]["applicable_counterfactual_dimensions"] == add["counterfactual_dimensions"], "add_dimensions", failed)
    _check(add["surface_queries"] == 35 and add["counterfactual_queries"] == 15 and add["total_queries"] == 50, "add_query_counts", failed)

    actual = scope["actual_plan_source_tier"]
    coverage = actual_report["coverage"]
    _check(actual_report["cohort_frozen"] is True and actual_report["source_evidence_coverage_complete"] is True, "actual_cohort_complete", failed)
    _check(actual_report["runtime_gates_passed"] is False, "actual_runtime_historical_failure", failed)
    _check(actual_report["failed_runtime_dimensions"] == actual["dimensions"], "actual_failed_dimensions", failed)
    _check(actual_lock["status"] == "complete" and actual_lock["source_cohort_complete"] is True, "actual_lock_complete", failed)
    _check(
        (coverage["anchors"], coverage["companies"], coverage["sources"], coverage["document_layouts"], coverage["table_layouts"])
        == (actual["anchors"], actual["companies"], actual["sources"], actual["document_layouts"], actual["table_layouts"])
        == (4, 4, 4, 3, 4),
        "actual_coverage",
        failed,
    )
    anchors = actual_config["anchors"]
    diagnosis = set(actual_config["selection_contract"]["diagnosis_company_keys"])
    _check(len(anchors) == 4 and not diagnosis.intersection(row["company_key"] for row in anchors), "actual_company_disjoint", failed)
    _check(all(set(row["pairs"]) == set(actual["dimensions"]) for row in anchors), "actual_all_dimensions", failed)
    _check(actual["queries_per_anchor_dimension"] == 2 and actual["total_queries"] == 24, "actual_query_counts", failed)
    _check(scope["combined_source_queries"] == add["total_queries"] + actual["total_queries"] == 74, "combined_queries", failed)

    _check(scope["stage"] == "measurement_only_no_runtime_change", "measurement_only", failed)
    _check(scope["runtime_file_whitelist"] == [], "empty_runtime_whitelist", failed)
    _check(scope["evaluate_all_frozen_eligible_records"] is True, "evaluate_all", failed)
    _check(scope["source_run_boundary"] == ["question", "source PDF path"], "source_boundary", failed)
    _check(scope["per_query_timeout_seconds"] == 15 and scope["timeout_is_failure"] is True, "timeout_policy", failed)
    _check(scope["mineru_auto_generate"] is False and scope["existing_development_cache_only"] is True, "ocr_policy", failed)
    _check(scope["gold_candidate_or_program_passed_to_engine"] is False, "no_gold_engine_input", failed)
    _check(scope["safe_abstention_counts_as_answer"] is False, "no_abstention_as_answer", failed)
    _check(scope["candidate_rank_score_or_confidence_is_semantic_proof"] is False, "no_score_proof", failed)
    _check(development_config["source_backed_cohort"]["per_query_timeout_seconds"] == 15, "add_timeout_frozen", failed)
    _check(actual_config["selection_contract"]["per_query_timeout_seconds"] == 15, "actual_timeout_frozen", failed)
    _check(development_config["source_backed_cohort"]["ocr_cache_policy"]["mineru_auto_generate_during_baseline"] is False, "add_no_ocr_generate", failed)
    _check(actual_config["selection_contract"]["mineru_auto_generate_during_baseline"] is False, "actual_no_ocr_generate", failed)

    plan = compile_question_plan("Doanh thu theo triệu đồng là bao nhiêu?")
    proof = compile_question_proof_contract(plan, "SELECT")
    _check(plan.requested_scale == "million_vnd" and "unit_scale_match" in proof.required, "unit_scale_contract_active", failed)
    results = safe["results"]
    _check(results["gold_actions"] == gates["safe_actions"] == {"ABSTAIN": 92, "ANSWER": 257, "REJECT": 14}, "safe_actions", failed)
    _check(results["gold_execution_correct"] == results["gold_total"] == 363 and gates["safe_execution"] == "363/363", "safe_execution", failed)
    _check(results["permutation_exact"] == results["permutation_total"] == 363 and gates["safe_permutation"] == "363/363", "safe_permutation", failed)
    _check(results["counterfactual_generated"] == gates["safe_counterfactuals"] == 141, "safe_counterfactuals", failed)
    _check(results["counterfactual_false_accepts"] == gates["safe_counterfactual_false_accepts"] == 0, "safe_false_accepts", failed)
    _check(safe["safe_replay_gate_passed"] is True and safe["errors"] == [], "safe_replay_passed", failed)

    _check(config["runtime_changed_by_this_precommit"] is False, "runtime_unchanged", failed)
    _check(config["refresh_executed_by_this_precommit"] is False, "refresh_not_executed", failed)
    _check(config["downstream_runtime_change_authorized_in_same_step"] is False, "same_step_runtime_locked", failed)
    _check(all(value is False for value in config["invariants"].values()), "invariants", failed)

    passed = not failed
    return {
        "schema": "finhrm-qp4-post-unit-scale-development-baseline-refresh-precommit-report-v1",
        "created_at": config["created_at"],
        "classification": config["classification"],
        "precommit_gate_passed": passed,
        "failed_gates": failed,
        "refresh_scope": scope,
        "refresh_gates": gates,
        "frozen_coverage": {
            "compiler": scope["compiler_tier"],
            "add_source": add,
            "actual_plan_source": actual,
            "combined_source_queries": scope["combined_source_queries"],
        },
        "unit_scale_gate_reclosed": reclosure_report["unit_scale_gate_reclosed"],
        "historical_baselines_preserved": True,
        "refresh_executed": False,
        "runtime_changed": False,
        "runtime_hashes": config["frozen_runtime_hashes"],
        "checkpoint_sha256": checkpoint["sha256"],
        "future_refresh_outputs": config["future_refresh_outputs"],
        "qp4_complete": False,
        "qp5_authorized": False,
        "phase_d": "locked",
        "authorized_next_step": (
            config["authorized_next_step_after_lock"]
            if passed
            else "Resolve refresh-precommit failures without running the refresh or changing runtime."
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    coverage = report["frozen_coverage"]
    return "\n".join(
        [
            "# QP4 post-UNIT_SCALE development baseline refresh precommit v1",
            "",
            f"- Gate: `{'PASS' if report['precommit_gate_passed'] else 'FAIL'}`.",
            "- Stage: measurement-only precommit; the baseline refresh has not run and runtime is unchanged.",
            f"- Compiler tier: `{coverage['compiler']['core_answer_families']}` families, `{coverage['compiler']['ambiguity_controls']}` ambiguity controls, `{coverage['compiler']['surface_variants']}` surface variants, `{coverage['compiler']['semantic_minimal_pairs']}` semantic pairs.",
            f"- ADD source tier: `{coverage['add_source']['anchors']}` anchors / `{coverage['add_source']['companies']}` companies / `{coverage['add_source']['sources']}` PDFs / `{coverage['add_source']['real_layouts']}` real layouts, `{coverage['add_source']['total_queries']}` queries.",
            f"- ACTUAL/PLAN source tier: `{coverage['actual_plan_source']['anchors']}` anchors / `{coverage['actual_plan_source']['companies']}` companies / `{coverage['actual_plan_source']['sources']}` PDFs / `{coverage['actual_plan_source']['document_layouts']}` document layouts / `{coverage['actual_plan_source']['table_layouts']}` table layouts, `{coverage['actual_plan_source']['total_queries']}` queries.",
            f"- Combined source queries: `{coverage['combined_source_queries']}`; per-query timeout `15s`; existing development cache only; no OCR auto-generation.",
            "- Gold candidate/program, rank, score and confidence are forbidden as semantic proof. Safe abstention is not counted as answer; synthetic layouts are not confirmation.",
            "- Historical baselines remain immutable. QP4 is incomplete; QP5 and Phase D remain locked.",
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
    artifacts = [DEFAULT_CONFIG, Path(__file__).resolve(), TEST_PATH, report_path, markdown_path]
    missing = [path for path in artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(str(path) for path in missing))
    lock = {
        "schema": "finhrm-qp4-post-unit-scale-development-baseline-refresh-precommit-lock-v1",
        "created_at": report["created_at"],
        "classification": report["classification"],
        "status": "complete" if report["precommit_gate_passed"] else "failed",
        "precommit_gate_passed": report["precommit_gate_passed"],
        "failed_gates": report["failed_gates"],
        "refresh_executed": False,
        "runtime_changed": False,
        "runtime_hashes": report["runtime_hashes"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "runtime_file_whitelist_for_next_step": [],
        "historical_baselines_preserved": True,
        "future_refresh_outputs": report["future_refresh_outputs"],
        "artifacts": [_artifact(path) for path in artifacts],
        "qp4_complete": False,
        "qp5_authorized": False,
        "phase_d": "locked",
        "authorized_next_step": report["authorized_next_step"],
    }
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    report = build_report(args.config)
    write_outputs(report, args.report, args.markdown, args.lock)
    print(json.dumps({"precommit_gate_passed": report["precommit_gate_passed"], "failed_gates": report["failed_gates"]}, ensure_ascii=False))
    return 0 if report["precommit_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
