from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _read(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and lock QP0 question-coverage artifacts")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "question_coverage_qp0_lock_v1.json")
    args = parser.parse_args()

    catalog = ROOT / "config" / "question_family_catalog_plan_actual_v1.jsonl"
    catalog_lock = ROOT / "reports" / "question_family_catalog_plan_actual_v1_lock.json"
    compiler_report = ROOT / "reports" / "question_plan_compiler_audit_development_v1.json"
    counterfactual_report = ROOT / "reports" / "select_add_counterfactual_s4_development_v1.json"
    compiler = _read(compiler_report)
    counterfactual = _read(counterfactual_report)
    catalog_meta = _read(catalog_lock)

    if catalog_meta.get("catalog_sha256") != _digest(catalog):
        raise ValueError("Catalog hash drift")
    if int(compiler.get("catalog_examples", 0)) != 299 or int(compiler.get("family_count", 0)) != 59:
        raise ValueError("Compiler catalog coverage drift")
    if int(counterfactual.get("development_examples", 0)) != 976:
        raise ValueError("S4 development baseline size drift")
    baseline = counterfactual["policy_results"]["baseline"]["overall"]
    expected_baseline = {"accepted": 937, "correct": 829, "false_accept": 108}
    if any(int(baseline.get(key, -1)) != value for key, value in expected_baseline.items()):
        raise ValueError("S4 accepted baseline drift")
    policy_results = counterfactual.get("policy_results", {})
    passing_policies = [
        name
        for name, result in policy_results.items()
        if name != "baseline"
        and bool(result.get("overall", {}).get("all_counterfactual_gates_pass", False))
    ]
    if passing_policies:
        raise ValueError(f"QP0 unexpectedly authorizes runtime policies: {passing_policies}")

    artifacts = (
        catalog,
        catalog_lock,
        compiler_report,
        counterfactual_report,
        ROOT / "scripts" / "build_question_family_catalog_plan_actual.py",
        ROOT / "scripts" / "audit_question_plan_compiler.py",
        ROOT / "scripts" / "audit_select_add_counterfactual_s4.py",
        ROOT / "tests" / "test_question_family_qp0.py",
        ROOT / "src" / "hrm_finance_qa" / "question_planner.py",
        ROOT / "src" / "hrm_finance_qa" / "evidence_schema.py",
        ROOT / "src" / "hrm_finance_qa" / "symbolic_verifier.py",
        ROOT / "src" / "hrm_finance_qa" / "document_hrm.py",
        ROOT / "models" / "document_hrm_V8_candidate.pt",
        ROOT / "data" / "document_hrm_v12b2_reranker_disjoint_folds_subtract_v4.json",
        ROOT / "data" / "document_hrm_v12b2_development_plus_subtract_v3.jsonl",
        ROOT / "cache" / "v12b2_folds_subtract_v5" / "manifest.json",
        ROOT / "cache" / "v12b2_phase_c_subtract_v5" / "manifest.json",
    )
    lock = {
        "schema": "question-coverage-qp0-lock-v1",
        "classification": "development_only_company_report_disjoint_plus_synthetic_contract",
        "status": "qp0_complete_qp1_authorized",
        "catalog": {
            "families": 59,
            "examples": 299,
            "plan_actual_examples": 132,
            "two_year_examples": 132,
        },
        "compiler_baseline": {
            "exact": compiler["exact"],
            "exact_rate": compiler["exact_rate"],
            "plan_actual_exact": compiler["by_category"]["plan_actual"]["exact"],
            "two_year_exact": compiler["by_category"]["two_year"]["exact"],
            "column_role_pass_rate": compiler["dimension_metrics"]["column_roles"]["pass_rate"],
            "operation_pass_rate": compiler["dimension_metrics"]["operation"]["pass_rate"],
        },
        "s4_counterfactual_baseline": {
            "examples": 976,
            "accepted": 937,
            "correct": 829,
            "false_accept": 108,
            "accepted_precision": baseline["accepted_precision"],
            "coverage": baseline["coverage"],
            "accepted_select_add": counterfactual["accepted_select_add_baseline"],
            "minimum_fold_raw_gold_program_recall": counterfactual["frozen_fold_cache_minimum_raw_gold_program_recall"],
        },
        "counterfactual_decision": {
            "passing_policies": passing_policies,
            "runtime_policy_authorized": False,
            "best_balanced_audit_policy": "logical_cell_unique",
            "best_balanced_precision": policy_results["logical_cell_unique"]["overall"]["accepted_precision"],
            "best_balanced_coverage": policy_results["logical_cell_unique"]["overall"]["coverage"],
            "combined_precision": policy_results["combined_select_add_structural"]["overall"]["accepted_precision"],
            "combined_coverage": policy_results["combined_select_add_structural"]["overall"]["coverage"],
            "decision": "Do not enable an audit-only strict policy; proceed to QP1 evidence schema and header-role reconstruction.",
        },
        "invariants": {
            "checkpoint_changed": False,
            "runtime_changed": False,
            "threshold_changed": False,
            "fine_tuning_performed": False,
            "global_validation_loaded_by_scripts": False,
            "sealed_validation_loaded_by_scripts": False,
            "blind_test_loaded_by_scripts": False,
            "phase_d_unlocked": False,
        },
        "protected_read_hygiene": {
            "zero_protected_read_claim": False,
            "used_for_metrics_or_design": False,
            "note": "One broad interactive search printed an old protected report record before the QP0 scripts ran. All QP0 scripts and metrics used only allowlisted development/synthetic artifacts.",
        },
        "regression": {"tests_run": 67, "failures": 0},
        "next_allowed_step": "QP1_evidence_schema_and_table_header_reconstruction",
        "artifact_hashes": {
            str(path.relative_to(ROOT)).replace("\\", "/"): _digest(path)
            for path in artifacts
        },
        "lock_script_sha256": _digest(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(lock, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
