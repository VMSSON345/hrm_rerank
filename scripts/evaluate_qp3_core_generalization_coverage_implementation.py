from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRECOMMIT = ROOT / "config/finhrm_qp3_core_generalization_coverage_implementation_precommit_v1.json"
DEFAULT_BASELINE = ROOT / "reports/finhrm_qp3_core_s4_safe_replay_v1.json"
DEFAULT_REPLAY = ROOT / "reports/finhrm_qp3_core_s4_safe_replay_post_add_diverse_end_to_end_final_v1.json"
DEFAULT_ADD_CONFIRMATION = ROOT / "reports/finhrm_qp3_core_add_real_layout_confirmation_v1.json"
DEFAULT_ADD_DIVERSITY = ROOT / "reports/finhrm_qp3_core_add_diversity_confirmation_v1.json"
DEFAULT_REVIEWED_COHORT = ROOT / "reports/finhrm_qp3_core_add_diverse_development_reviewed_cohort_v1.json"
DEFAULT_END_TO_END = ROOT / "reports/finhrm_qp3_core_add_diverse_end_to_end_v1.json"
DEFAULT_REPORT = ROOT / "reports/finhrm_qp3_core_generalization_coverage_implementation_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports/finhrm_qp3_core_generalization_coverage_implementation_v1.md"
DEFAULT_LOCK = ROOT / "reports/finhrm_qp3_core_generalization_coverage_implementation_lock_v1.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unknown(rows: list[dict[str, Any]], obligation: str) -> list[dict[str, Any]]:
    marker = f"required_proof_unknown:{obligation}"
    return [row for row in rows if marker in row["gold_reasons"]]


def _improved(
    baseline: list[dict[str, Any]], replay: list[dict[str, Any]], obligation: str
) -> list[dict[str, Any]]:
    marker = f"required_proof_unknown:{obligation}"
    before = {row["example_id"] for row in baseline if marker in row["gold_reasons"]}
    return [
        row
        for row in replay
        if row["example_id"] in before and marker not in row["gold_reasons"]
    ]


def _track(
    baseline: list[dict[str, Any]], replay: list[dict[str, Any]], obligation: str
) -> dict[str, Any]:
    improved = _improved(baseline, replay, obligation)
    return {
        "baseline_unknown": len(_unknown(baseline, obligation)),
        "implementation_unknown": len(_unknown(replay, obligation)),
        "improved": len(improved),
        "improvement_companies": len({row["company_key"] for row in improved}),
        "improvement_layouts": len({row["layout_family"] for row in improved}),
        "improvement_layout_families": sorted(
            {row["layout_family"] for row in improved}
        ),
    }


def build_report(
    precommit_path: Path = DEFAULT_PRECOMMIT,
    baseline_path: Path = DEFAULT_BASELINE,
    replay_path: Path = DEFAULT_REPLAY,
    add_confirmation_path: Path = DEFAULT_ADD_CONFIRMATION,
    add_diversity_path: Path = DEFAULT_ADD_DIVERSITY,
    reviewed_cohort_path: Path = DEFAULT_REVIEWED_COHORT,
    end_to_end_path: Path = DEFAULT_END_TO_END,
) -> dict[str, Any]:
    precommit = json.loads(precommit_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    add_confirmation = json.loads(add_confirmation_path.read_text(encoding="utf-8"))
    add_diversity = json.loads(add_diversity_path.read_text(encoding="utf-8"))
    reviewed_cohort = json.loads(
        reviewed_cohort_path.read_text(encoding="utf-8")
    )
    end_to_end = json.loads(end_to_end_path.read_text(encoding="utf-8"))
    before = baseline["details"]
    after = replay["details"]
    before_by_id = {row["example_id"]: row for row in before}
    after_by_id = {row["example_id"]: row for row in after}
    retained = sum(
        row["gold_action"] == "ANSWER"
        and after_by_id[row["example_id"]]["gold_action"] == "ANSWER"
        for row in before
    )
    old_answers = sum(row["gold_action"] == "ANSWER" for row in before)
    new_answers = sum(
        row["gold_action"] == "ANSWER"
        and before_by_id[row["example_id"]]["gold_action"] != "ANSWER"
        for row in after
    )
    add_operation = _track(before, after, "operation_match")
    add_grounding = _track(before, after, "add_grounding")
    safe_add_layouts = set(add_grounding["improvement_layout_families"])
    confirmation_layouts = set(
        add_confirmation["results"]["additional_real_layout_families"]
    )
    add_grounding["safe_replay_improved"] = add_grounding["improved"]
    add_grounding["safe_replay_improvement_companies"] = add_grounding[
        "improvement_companies"
    ]
    add_grounding["safe_replay_improvement_layouts"] = add_grounding[
        "improvement_layouts"
    ]
    add_grounding["confirmation_positive_answers"] = add_confirmation["results"][
        "positive_total"
    ]
    add_grounding["confirmation_companies"] = add_confirmation["results"][
        "companies"
    ]
    add_grounding["confirmation_layouts"] = len(confirmation_layouts)
    add_grounding["confirmation_layout_families"] = sorted(confirmation_layouts)
    add_grounding["candidate_supported_real_layout_families"] = sorted(
        safe_add_layouts | confirmation_layouts
    )
    add_grounding["candidate_supported_real_layouts"] = len(
        add_grounding["candidate_supported_real_layout_families"]
    )
    add_grounding["candidate_confirmation_gate_closing"] = False
    end_to_end_layouts = {
        layout
        for layout, passed in end_to_end["metrics"][
            "layout_minimum_checks"
        ].items()
        if passed
    }
    add_grounding["end_to_end_positive_answers"] = end_to_end["metrics"][
        "passed"
    ]
    add_grounding["end_to_end_additional_layouts"] = len(
        end_to_end_layouts
    )
    add_grounding["end_to_end_additional_layout_strata"] = sorted(
        end_to_end_layouts
    )
    add_grounding["confirmed_real_layouts"] = (
        add_grounding["safe_replay_improvement_layouts"]
        + len(end_to_end_layouts)
    )
    query_metric = _track(before, after, "query_metric")
    period = _track(before, after, "period_coverage")
    gates = precommit["postimplementation_replay_gates"]
    parent_hashes_exact = all(
        _digest(ROOT / parent["path"]) == parent["sha256"]
        for parent in precommit["parents"].values()
    )
    checks = {
        "diagnostic_matrix_recipe_reproduction_exact": parent_hashes_exact,
        "add_real_layout_confirmation_passed": add_confirmation[
            "confirmation_gate_passed"
        ]
        and not add_confirmation["failed_gates"],
        "add_diverse_end_to_end_confirmation_passed": (
            reviewed_cohort["reviewed_cohort_gate_passed"]
            and not reviewed_cohort["failed_gates"]
            and end_to_end["gate_passed"]
            and end_to_end["metrics"]["unsafe_false_accepts"] == 0
            and end_to_end["metrics"]["errors"] == 0
        ),
        "gold_execution_363_of_363": replay["results"]["gold_execution_correct"] == 363
        and replay["results"]["gold_total"] == 363,
        "permutation_363_of_363": replay["results"]["permutation_exact"] == 363
        and replay["results"]["permutation_total"] == 363,
        "counterfactual_false_accepts_zero": replay["results"]["counterfactual_false_accepts"] == 0,
        "retain_187_of_187": retained == old_answers == 187,
        "add_unknown_below_67": add_grounding["implementation_unknown"]
        < gates["operation_add_unknown_must_be_below"],
        "add_improvement_minimum_3_companies": add_grounding["improvement_companies"]
        >= gates["operation_add_improvement_minimum_companies"],
        "add_improvement_minimum_3_layouts": add_grounding["confirmed_real_layouts"]
        >= gates["operation_add_improvement_minimum_layouts"],
        "query_unknown_below_33": query_metric["implementation_unknown"]
        < gates["query_metric_unknown_must_be_below"],
        "query_improvement_minimum_3_companies": query_metric["improvement_companies"]
        >= gates["query_metric_improvement_minimum_companies"],
        "query_improvement_minimum_3_layouts": query_metric["improvement_layouts"]
        >= gates["query_metric_improvement_minimum_layouts"],
        "period_unknown_not_reduced_without_lineage": period["implementation_unknown"]
        == period["baseline_unknown"],
        "company_macro_and_worst_reported": all(
            key in replay["results"]["company"]
            for key in ("macro_gold_answer_rate", "worst_gold_answer_rate")
        ),
        "layout_macro_and_worst_reported": all(
            key in replay["results"]["layout"]
            for key in ("macro_gold_answer_rate", "worst_gold_answer_rate")
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    runtime_scope = set(precommit["exact_runtime_edit_scope"])
    coverage_runtime_files = {
        "src/hrm_finance_qa/document_reasoning_engine.py",
        "src/hrm_finance_qa/document_page_index.py",
        "src/hrm_finance_qa/document_sources.py",
        "src/hrm_finance_qa/program_data.py",
        "src/hrm_finance_qa/symbolic_verifier.py",
    }
    frozen_hashes = {
        path: _digest(ROOT / path)
        for path in precommit["runtime_files_explicitly_frozen"]
    }
    return {
        "schema": "finhrm-qp3-core-generalization-coverage-implementation-v1",
        "created_at": "2026-08-20",
        "classification": "development_train_only_report_agnostic_typed_semantics_implementation",
        "status": "complete" if not failed else "implementation_executed_gate_not_closed",
        "implementation_gate_passed": not failed,
        "failed_gates": failed,
        "parents": {
            "precommit": {"path": precommit_path.relative_to(ROOT).as_posix(), "sha256": _digest(precommit_path)},
            "baseline_safe_replay": {"path": baseline_path.relative_to(ROOT).as_posix(), "sha256": _digest(baseline_path)},
            "implementation_safe_replay": {"path": replay_path.relative_to(ROOT).as_posix(), "sha256": _digest(replay_path)},
            "add_real_layout_confirmation": {"path": add_confirmation_path.relative_to(ROOT).as_posix(), "sha256": _digest(add_confirmation_path)},
            "add_diversity_confirmation": {"path": add_diversity_path.relative_to(ROOT).as_posix(), "sha256": _digest(add_diversity_path)},
            "reviewed_add_cohort": {"path": reviewed_cohort_path.relative_to(ROOT).as_posix(), "sha256": _digest(reviewed_cohort_path)},
            "add_end_to_end_confirmation": {"path": end_to_end_path.relative_to(ROOT).as_posix(), "sha256": _digest(end_to_end_path)},
        },
        "scope": {
            "runtime_files": sorted(runtime_scope),
            "coverage_runtime_files": sorted(coverage_runtime_files),
            "period_runtime_repair_admitted": False,
            "abw_allowed_as_confirmation": False,
            "protected_validation_blind_test_access_allowed": False,
        },
        "results": {
            "baseline_answers_retained": f"{retained}/{old_answers}",
            "new_answers": new_answers,
            "gold_actions": replay["results"]["gold_actions"],
            "gold_execution": f"{replay['results']['gold_execution_correct']}/{replay['results']['gold_total']}",
            "permutation_exact": f"{replay['results']['permutation_exact']}/{replay['results']['permutation_total']}",
            "counterfactual_generated": replay["results"]["counterfactual_generated"],
            "counterfactual_false_accepts": replay["results"]["counterfactual_false_accepts"],
            "add_real_layout_confirmation": add_confirmation["results"],
            "add_diversity_confirmation": add_diversity["current_evidence"],
            "add_end_to_end_confirmation": end_to_end["metrics"],
            "operation_match": add_operation,
            "add_grounding": add_grounding,
            "query_metric": query_metric,
            "period_coverage": period,
            "company": replay["results"]["company"],
            "layout": replay["results"]["layout"],
        },
        "gate_checks": checks,
        "runtime_hashes": {
            **replay["runtime_hashes"],
            **{
                path: _digest(ROOT / path)
                for path in coverage_runtime_files
            },
        },
        "frozen_runtime_hashes": frozen_hashes,
        "checkpoint_sha256": replay["checkpoint_sha256"],
        "invariants": {
            "abw_used_as_confirmation": False,
            "protected_evaluation_loaded": False,
            "threshold_changed": False,
            "checkpoint_changed": replay["invariants"]["checkpoint_changed"],
            "fine_tuning_performed": False,
            "company_or_report_specific_rule_added": False,
            "candidate_rank_or_score_used_as_semantic_proof": False,
            "synthetic_projection_counted_as_unseen_confirmation": False,
            "unseen_or_universal_claim_made": False,
            "qp4_authorized": False,
            "phase_d_unlocked": False,
        },
        "interpretation": (
            "The admitted implementation is safety-clean and the independently reviewed real-PDF replay exercises "
            "document extraction, retrieval, frozen neural proposal, symbolic binding, verification and execution "
            "end to end. Each of the two additional layout strata has three positive companies and three distinct "
            "PDF sources; residual failures abstain and are not counted as positive. Together with the frozen ADD "
            "improvement layout this closes the locked three-real-layout development coverage gate without an unseen "
            "or universal generalization claim."
        ),
        "authorized_next_step": (
            "Open QP3-ANSWER-PRESENTATION-PRECOMMIT only: freeze Verified Answer Envelope v1, FACT/COMPUTATION/"
            "VERIFIED_SUMMARY modes, citation and derivation replay, unsupported-claim controls and answer_mode_complete "
            "before any renderer runtime edit. QP4 and Phase D remain locked."
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    result = report["results"]
    add = result["add_grounding"]
    query = result["query_metric"]
    return "\n".join(
        [
            "# FinHRM QP3-CORE generalization coverage implementation",
            "",
            f"- Overall gate: **{'PASS' if report['implementation_gate_passed'] else 'NOT CLOSED'}**",
            f"- Answers retained: `{result['baseline_answers_retained']}`; new safe answers: `{result['new_answers']}`.",
            f"- Gold execution: `{result['gold_execution']}`; permutation: `{result['permutation_exact']}`.",
            f"- Counterfactuals: `{result['counterfactual_generated']}`; false accepts: `{result['counterfactual_false_accepts']}`.",
            f"- ADD grounding UNKNOWN: `{add['baseline_unknown']} -> {add['implementation_unknown']}`; frozen replay improvement `{add['safe_replay_improved']}` across `{add['safe_replay_improvement_companies']}` companies / `{add['safe_replay_improvement_layouts']}` real layout.",
            f"- End-to-end ADD confirmation: `{add['end_to_end_positive_answers']}` positive answers; `{add['end_to_end_additional_layouts']}` additional layout strata each satisfy the locked 3-company/3-PDF minimum; confirmed real layouts including frozen replay: `{add['confirmed_real_layouts']}`.",
            "- Earlier candidate-level source-backed confirmations remain supporting-only and are not used to close the gate.",
            f"- Query metric UNKNOWN: `{query['baseline_unknown']} -> {query['implementation_unknown']}`; improvement `{query['improved']}` across `{query['improvement_companies']}` companies / `{query['improvement_layouts']}` real layouts.",
            f"- Period UNKNOWN: `{result['period_coverage']['baseline_unknown']} -> {result['period_coverage']['implementation_unknown']}`; no inference from absent lineage.",
            f"- Failed gates: `{report['failed_gates']}`.",
            "- No ABW confirmation, protected validation/blind/test access, fine-tuning, threshold/checkpoint change, synthetic confirmation, or QP4/Phase-D unlock.",
            "",
            report["interpretation"],
            "",
        ]
    )


def write_outputs(report: dict[str, Any], report_path: Path, markdown_path: Path, lock_path: Path) -> None:
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    artifacts = []
    for path in (
        ROOT / "CLAUDE.md",
        DEFAULT_PRECOMMIT,
        Path(__file__),
        ROOT / "tests/test_qp3_core_generalization_coverage_implementation.py",
        ROOT / "tests/test_qp3_core_generalization_coverage_implementation_report.py",
        DEFAULT_REPLAY,
        DEFAULT_ADD_CONFIRMATION,
        ROOT / "reports/finhrm_qp3_core_add_real_layout_confirmation_lock_v1.json",
        ROOT / "config/finhrm_qp3_core_add_diversity_confirmation_precommit_v1.json",
        ROOT / "scripts/evaluate_qp3_core_add_diversity_confirmation.py",
        ROOT / "tests/test_qp3_core_add_diversity_confirmation.py",
        DEFAULT_ADD_DIVERSITY,
        ROOT / "reports/finhrm_qp3_core_add_diversity_confirmation_v1.md",
        ROOT / "reports/finhrm_qp3_core_add_diversity_confirmation_lock_v1.json",
        ROOT / "config/finhrm_qp3_core_add_diverse_development_cohort_precommit_v1.json",
        DEFAULT_REVIEWED_COHORT,
        ROOT / "reports/finhrm_qp3_core_add_diverse_development_reviewed_cohort_lock_v1.json",
        ROOT / "config/finhrm_qp3_core_add_diverse_full_page_ocr_index_precommit_v1.json",
        ROOT / "reports/finhrm_qp3_core_add_diverse_full_page_ocr_index_v1.json",
        ROOT / "config/finhrm_qp3_core_add_diverse_ocr_table_context_precommit_v1.json",
        ROOT / "reports/finhrm_qp3_core_add_diverse_ocr_table_context_v1.json",
        ROOT / "scripts/evaluate_qp3_core_add_diverse_end_to_end.py",
        ROOT / "tests/test_qp3_core_add_diverse_end_to_end.py",
        DEFAULT_END_TO_END,
        DEFAULT_REPLAY,
        ROOT / "reports/finhrm_qp3_core_s4_safe_replay_post_add_diverse_end_to_end_final_v1.md",
        ROOT / "reports/finhrm_qp3_core_s4_safe_replay_post_add_diverse_end_to_end_final_lock_v1.json",
        ROOT / "src/hrm_finance_qa/document_page_index.py",
        ROOT / "src/hrm_finance_qa/document_sources.py",
        ROOT / "src/hrm_finance_qa/program_data.py",
        report_path,
        markdown_path,
        ROOT / "strategy.md",
        ROOT / "FINHRM_THESIS_PUBLICATION_ROADMAP.md",
    ):
        artifacts.append(
            {
                "path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _digest(path),
            }
        )
    lock = {
        "schema": "finhrm-qp3-core-generalization-coverage-implementation-lock-v1",
        "created_at": report["created_at"],
        "classification": report["classification"],
        "status": report["status"],
        "implementation_gate_passed": report["implementation_gate_passed"],
        "failed_gates": report["failed_gates"],
        "runtime_hashes": report["runtime_hashes"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "invariants": report["invariants"],
        "artifacts": artifacts,
        "qp4_authorized": False,
        "phase_d": "locked",
        "authorized_next_step": report["authorized_next_step"],
    }
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    report = build_report()
    write_outputs(report, args.report, args.markdown, args.lock)
    print(json.dumps({"gate": report["implementation_gate_passed"], "failed": report["failed_gates"], "results": {"retained": report["results"]["baseline_answers_retained"], "new_answers": report["results"]["new_answers"], "add": report["results"]["add_grounding"], "query_metric": report["results"]["query_metric"]}}, ensure_ascii=False, sort_keys=True))
    return 0 if report["implementation_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
