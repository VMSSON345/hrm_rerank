#!/usr/bin/env python3
"""Validate and lock the measurement-only post-question-surface refresh contract."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "finhrm_qp4_post_question_surface_baseline_refresh_precommit_v1.json"
DEFAULT_REPORT = ROOT / "reports" / "finhrm_qp4_post_question_surface_baseline_refresh_precommit_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "finhrm_qp4_post_question_surface_baseline_refresh_precommit_v1.md"
DEFAULT_LOCK = ROOT / "reports" / "finhrm_qp4_post_question_surface_baseline_refresh_precommit_lock_v1.json"
TEST_PATH = ROOT / "tests" / "test_qp4_post_question_surface_baseline_refresh_precommit.py"
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
    implementation = _load(ROOT / parents["question_surface_implementation_report"]["path"])
    implementation_lock = _load(ROOT / parents["question_surface_implementation_lock"]["path"])
    correction_lock = _load(ROOT / parents["question_surface_correction_lock"]["path"])
    source = _load(ROOT / parents["question_surface_source_replay"]["path"])
    safe = _load(ROOT / parents["question_surface_safe_replay"]["path"])
    safe_lock = _load(ROOT / parents["question_surface_safe_replay_lock"]["path"])
    add_config = _load(ROOT / parents["add_source_cohort"]["path"])
    actual_config = _load(ROOT / parents["actual_plan_source_cohort"]["path"])
    actual_baseline = _load(ROOT / parents["historical_actual_plan_source_baseline"]["path"])
    actual_report = _load(ROOT / parents["actual_plan_source_report"]["path"])
    actual_lock = _load(ROOT / parents["actual_plan_source_lock"]["path"])
    scope = config["refresh_scope"]
    completion = config["measurement_completion_gates"]
    semantic = config["qp4_semantic_gates"]
    generic = config["generalization_contract"]

    _check(implementation["implementation_gate_passed"] is True, "implementation_complete", failed)
    _check(implementation["failed_gates"] == [], "implementation_no_failures", failed)
    _check(implementation["runtime_after"] == config["frozen_runtime_hashes"], "implementation_runtime_exact", failed)
    _check(
        implementation["authorized_next_step"].startswith("QP4-POST-QUESTION-SURFACE-BASELINE-REFRESH-PRECOMMIT"),
        "precommit_authorized_by_report",
        failed,
    )
    _check(implementation_lock["status"] == "complete", "implementation_lock_complete", failed)
    _check(implementation_lock["implementation_gate_passed"] is True, "implementation_lock_passed", failed)
    _check(
        implementation_lock["authorized_next_step"].startswith("QP4-POST-QUESTION-SURFACE-BASELINE-REFRESH-PRECOMMIT"),
        "precommit_authorized_by_lock",
        failed,
    )
    _check(correction_lock["status"] == "complete" and correction_lock["correction_gate_passed"] is True, "correction_locked", failed)

    compiler = implementation["compiler"]
    compiler_scope = scope["compiler_tier"]
    _check(compiler["canonical"]["exact"] == compiler["canonical"]["total"] == compiler_scope["core_answer_families"] == 13, "compiler_families", failed)
    _check(compiler["ambiguity"]["exact"] == compiler["ambiguity"]["total"] == compiler_scope["ambiguity_controls"] == 3, "compiler_controls", failed)
    _check(compiler["frozen_surface"]["exact"] == compiler["frozen_surface"]["total"] == compiler_scope["frozen_surface_questions"] == 91, "compiler_surface", failed)
    _check(compiler["corrected_hold_ahead"]["exact"] == compiler["corrected_hold_ahead"]["total"] == compiler_scope["corrected_hold_ahead_questions"] == 78, "compiler_hold_ahead", failed)
    _check(compiler["minimal_pairs"]["detected"] == compiler["minimal_pairs"]["total"] == compiler_scope["semantic_minimal_pairs"] == 57, "compiler_pairs", failed)
    compiler_total = sum(
        compiler_scope[name]
        for name in (
            "core_answer_families",
            "ambiguity_controls",
            "frozen_surface_questions",
            "corrected_hold_ahead_questions",
            "semantic_minimal_pairs",
        )
    )
    _check(completion["compiler_records_scored"] == f"{compiler_total}/{compiler_total}" == "242/242", "compiler_completion", failed)

    add = scope["add_source_tier"]
    _check(
        (source["eligible_anchors"], source["companies"], source["source_documents"], source["real_layout_families"])
        == (add["anchors"], add["companies"], add["source_documents"], add["real_layout_families"])
        == (5, 5, 5, 3),
        "add_source_coverage",
        failed,
    )
    add_contract = add_config["source_backed_cohort"]
    _check(len(add_contract["meaning_preserving_categories"]) == add["surface_categories"] == 7, "add_categories", failed)
    _check(add_contract["applicable_counterfactual_dimensions"] == add["counterfactual_dimensions"], "add_dimensions", failed)
    _check(len(source["surface_results"]) == add["surface_queries"] == 35, "add_surface_query_count", failed)
    _check(len(source["counterfactual_results"]) == add["counterfactual_queries"] == 15, "add_counterfactual_query_count", failed)
    _check(add["total_queries"] == add["surface_queries"] + add["counterfactual_queries"] == 50, "add_total", failed)
    surface_passed = sum(bool(row["invariant"]) for row in source["surface_results"])
    residuals = Counter(row["category"] for row in source["surface_results"] if not row["invariant"])
    _check(surface_passed == 20, "add_current_surface_passes", failed)
    _check(dict(residuals) == add["remaining_surface_residuals"], "add_residual_inventory", failed)
    _check(implementation["source_backed"]["previously_passing_retained"] == implementation["source_backed"]["previously_passing_total"] == 20, "add_prior_passes_retained", failed)
    _check(sum(bool(row["old_winner_false_accept"]) for row in source["counterfactual_results"]) == 0, "add_false_accepts", failed)
    _check(Counter(row["dimension"] for row in source["counterfactual_results"]) == Counter({name: 5 for name in add["counterfactual_dimensions"]}), "add_dimension_balance", failed)

    actual = scope["actual_plan_source_tier"]
    coverage = actual_report["coverage"]
    _check(actual_report["cohort_frozen"] is True and actual_report["source_evidence_coverage_complete"] is True, "actual_cohort_complete", failed)
    _check(actual_lock["status"] == "complete" and actual_lock["source_cohort_complete"] is True, "actual_lock_complete", failed)
    _check(
        (coverage["anchors"], coverage["companies"], coverage["sources"], coverage["document_layouts"], coverage["table_layouts"])
        == (actual["anchors"], actual["companies"], actual["source_documents"], actual["document_layout_families"], actual["table_layout_families"])
        == (4, 4, 4, 3, 4),
        "actual_source_coverage",
        failed,
    )
    anchors = actual_config["anchors"]
    diagnosis = set(actual_config["selection_contract"]["diagnosis_company_keys"])
    _check(len(anchors) == 4 and not diagnosis.intersection(row["company_key"] for row in anchors), "actual_company_disjoint", failed)
    _check(all(set(row["pairs"]) == set(actual["dimensions"]) for row in anchors), "actual_all_dimensions", failed)
    pairs = actual_baseline["results"]
    _check(len(pairs) == actual["anchors"] * len(actual["dimensions"]) == 12, "actual_pair_count", failed)
    _check(Counter(row["dimension"] for row in pairs) == Counter({name: 4 for name in actual["dimensions"]}), "actual_dimension_balance", failed)
    _check(actual["queries_per_anchor_dimension"] == 2 and actual["total_queries"] == 24, "actual_query_count", failed)
    _check(sum(bool(row["paired_gold_answer"]) for row in pairs) == 0, "actual_historical_paired_answers", failed)
    _check(sum(bool(row["old_winner_false_accept"]) for row in pairs) == 0, "actual_historical_false_accepts", failed)
    _check(scope["combined_source_queries"] == add["total_queries"] + actual["total_queries"] == 74, "combined_source_count", failed)

    _check(scope["stage"] == "measurement_only_no_runtime_change", "measurement_only", failed)
    _check(scope["runtime_file_whitelist"] == [], "runtime_whitelist_empty", failed)
    _check(scope["historical_artifacts_are_immutable"] is True, "history_immutable", failed)
    _check(scope["source_run_boundary"] == ["question", "source PDF path"], "source_boundary", failed)
    _check(scope["per_query_timeout_seconds"] == 15, "timeout_exact", failed)
    _check(scope["every_query_must_be_attempted"] is True, "all_queries_attempted", failed)
    _check(scope["execution_error_or_timeout_is_failed_observation"] is True and scope["failed_observation_may_not_be_dropped"] is True, "errors_stay_in_denominator", failed)
    _check(scope["mineru_auto_generate"] is False and scope["existing_development_cache_only"] is True, "cache_policy", failed)
    _check(scope["gold_may_be_used_only_for_posthoc_scoring"] is True and scope["gold_candidate_or_program_passed_to_engine"] is False, "gold_boundary", failed)
    _check(scope["safe_abstention_counts_as_answer"] is False, "abstention_not_answer", failed)
    _check(scope["candidate_rank_score_or_confidence_is_semantic_proof"] is False, "no_score_proof", failed)
    _check(add_contract["per_query_timeout_seconds"] == 15, "add_timeout_frozen", failed)
    _check(add_contract["ocr_cache_policy"]["mineru_auto_generate_during_baseline"] is False, "add_no_ocr_generation", failed)
    _check(actual_config["selection_contract"]["per_query_timeout_seconds"] == 15, "actual_timeout_frozen", failed)
    _check(actual_config["selection_contract"]["mineru_auto_generate_during_baseline"] is False, "actual_no_ocr_generation", failed)

    required_generic_true = (
        "same_contract_for_every_frozen_eligible_record",
        "evaluate_all_frozen_eligible_records",
        "company_and_layout_group_metrics_required",
        "macro_and_worst_group_metrics_required",
        "errors_timeouts_and_abstentions_remain_in_denominator",
        "fail_closed_outside_proven_support",
    )
    required_generic_false = (
        "report_specific_selection_or_exclusion",
        "company_filename_page_year_value_example_or_question_literal_rules",
        "synthetic_layout_counts_as_real_confirmation",
        "unseen_or_universal_generalization_claim_authorized",
    )
    _check(all(generic[name] is True for name in required_generic_true), "generic_positive_contract", failed)
    _check(all(generic[name] is False for name in required_generic_false), "generic_negative_contract", failed)

    safe_results = safe["results"]
    _check(safe["safe_replay_gate_passed"] is True and safe["errors"] == [], "safe_replay_passed", failed)
    _check(safe_lock["status"] == "complete" and safe_lock["safe_replay_gate_passed"] is True, "safe_replay_locked", failed)
    _check(safe_results["gold_actions"] == semantic["safe_actions"] == {"ABSTAIN": 92, "ANSWER": 257, "REJECT": 14}, "safe_actions", failed)
    _check(safe_results["gold_execution_correct"] == safe_results["gold_total"] == 363 and semantic["safe_execution"] == "363/363", "safe_execution", failed)
    _check(safe_results["permutation_exact"] == safe_results["permutation_total"] == 363 and semantic["safe_permutation"] == "363/363", "safe_permutation", failed)
    _check(safe_results["counterfactual_generated"] == semantic["safe_counterfactuals"] == 141, "safe_counterfactuals", failed)
    _check(safe_results["counterfactual_false_accepts"] == semantic["safe_counterfactual_false_accepts"] == 0, "safe_false_accepts", failed)
    _check(implementation["regression"]["passed"] is True and implementation["regression"]["observed"] == 354, "compatible_regression", failed)
    _check(completion["compatible_affected_regression"] == "354/354", "future_regression_gate", failed)

    semantics = config["status_semantics"]
    _check(semantics["refresh_complete_requires_every_frozen_record_attempted_and_scored"] is True, "refresh_completion_exact", failed)
    _check(semantics["refresh_complete_does_not_imply_qp4_gates_passed"] is True, "completion_not_semantic_pass", failed)
    _check(semantics["qp4_gates_passed_requires_every_semantic_gate"] is True, "all_semantic_gates_required", failed)
    _check(semantics["residuals_must_be_reported_not_forced_to_answer"] is True, "residuals_fail_closed", failed)
    _check(semantics["missing_proof_action"] == "ABSTAIN" and semantics["contradictory_proof_action"] == "REJECT", "proof_actions", failed)
    _check(semantic["add_source_prior_passes_retained"] == "20/20", "retention_gate", failed)
    _check(all(value == 0 for value in semantic["add_source_counterfactual_old_winner_false_accepts"].values()), "add_semantic_false_accept_gate", failed)
    _check(all(value == "4/4" for value in semantic["actual_plan_paired_gold_answers"].values()), "actual_answer_targets", failed)
    _check(all(value == 0 for value in semantic["actual_plan_old_winner_false_accepts"].values()), "actual_false_accept_targets", failed)

    _check(config["runtime_changed_by_this_precommit"] is False, "runtime_unchanged", failed)
    _check(config["refresh_executed_by_this_precommit"] is False, "refresh_not_executed", failed)
    _check(config["downstream_runtime_change_authorized_in_same_step"] is False, "same_step_runtime_locked", failed)
    _check(all(value is False for value in config["invariants"].values()), "invariants", failed)

    passed = not failed
    return {
        "schema": "finhrm-qp4-post-question-surface-baseline-refresh-precommit-report-v1",
        "created_at": config["created_at"],
        "classification": config["classification"],
        "precommit_gate_passed": passed,
        "failed_gates": failed,
        "generalization_contract": generic,
        "refresh_scope": scope,
        "measurement_completion_gates": completion,
        "qp4_semantic_gates": semantic,
        "status_semantics": semantics,
        "frozen_current_observations": {
            "compiler": {
                "canonical": "13/13",
                "ambiguity": "3/3",
                "surface": "91/91",
                "corrected_hold_ahead": "78/78",
                "minimal_pairs": "57/57",
            },
            "add_source": {
                "coverage": "5 companies / 5 PDFs / 3 real layouts",
                "surface": "20/35",
                "prior_passes_retained": "20/20",
                "remaining_residuals": add["remaining_surface_residuals"],
                "counterfactual_false_accepts": 0,
            },
            "actual_plan_source": {
                "coverage": "4 companies / 4 PDFs / 3 document layouts / 4 table layouts",
                "paired_gold_answers": {name: "0/4" for name in actual["dimensions"]},
                "counterfactual_false_accepts": 0,
            },
            "safe_replay": {
                "actions": semantic["safe_actions"],
                "execution": semantic["safe_execution"],
                "permutation": semantic["safe_permutation"],
                "counterfactuals": semantic["safe_counterfactuals"],
                "false_accepts": semantic["safe_counterfactual_false_accepts"],
            },
        },
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
            else "Resolve precommit failures without running the refresh or changing runtime."
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    observations = report["frozen_current_observations"]
    return "\n".join(
        [
            "# QP4 post-question-surface baseline refresh precommit v1",
            "",
            f"- Gate: `{'PASS' if report['precommit_gate_passed'] else 'FAIL'}`.",
            "- This is a measurement-only precommit. The refresh has not run and no runtime file changed.",
            "- Generalization boundary: one frozen contract over every eligible development record; no report filtering, no report-specific rules, and no unseen/universal claim.",
            "- Compiler inventory: `13` answer families, `3` ambiguity controls, `91` frozen surfaces, `78` corrected hold-ahead surfaces, `57` semantic minimal pairs.",
            "- ADD source inventory: `50` queries over `5` companies / `5` PDFs / `3` real layouts. Current surface result is `20/35`; all `15` residuals remain in the denominator.",
            "- ACTUAL/PLAN inventory: `24` queries over `4` companies / `4` PDFs / `3` document layouts / `4` table layouts.",
            "- Measurement completion and QP4 semantic pass are separate statuses. Errors, timeouts and abstentions cannot be dropped or counted as answers.",
            f"- Frozen safe replay: actions `{observations['safe_replay']['actions']}`, execution `{observations['safe_replay']['execution']}`, permutation `{observations['safe_replay']['permutation']}`, counterfactual false accepts `{observations['safe_replay']['false_accepts']}`.",
            "- Missing proof remains ABSTAIN; contradiction remains REJECT. QP4 is incomplete; QP5 and Phase D stay locked.",
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
        "schema": "finhrm-qp4-post-question-surface-baseline-refresh-precommit-lock-v1",
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
        "same_contract_for_every_frozen_eligible_record": True,
        "report_specific_selection_or_exclusion": False,
        "historical_artifacts_preserved": True,
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
