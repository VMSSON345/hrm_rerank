from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any

from scripts.build_qp3_core_precommit_cohort import build_records


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRECOMMIT = ROOT / "config" / "finhrm_qp3_core_precommit_v1.json"
DEFAULT_CORE = ROOT / "config" / "finhrm_thesis_core_v1.json"
DEFAULT_GENERALIZATION = ROOT / "config" / "finhrm_report_generalization_invariant_v1.json"
DEFAULT_COHORT = ROOT / "data" / "finhrm_qp3_core_development_v1.jsonl"
DEFAULT_REPORT = ROOT / "reports" / "finhrm_qp3_core_precommit_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "finhrm_qp3_core_precommit_v1.md"
DEFAULT_LOCK = ROOT / "reports" / "finhrm_qp3_core_precommit_lock_v1.json"
DEFAULT_IMPLEMENTATION_LOCK = ROOT / "reports" / "finhrm_qp3_core_implementation_lock_v1.json"

PROTECTED_PATH_PARTS = {"validation", "sealed", "blind", "test", "final", "holdout"}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _authorized_runtime_hashes(errors: list[str]) -> dict[str, str]:
    if not DEFAULT_IMPLEMENTATION_LOCK.is_file():
        return {}
    lock = json.loads(DEFAULT_IMPLEMENTATION_LOCK.read_text(encoding="utf-8"))
    if not lock.get("implementation_gate_passed"):
        errors.append("qp3_core_implementation_lock_not_passed")
        return {}
    authorized = {
        str(relative): str(digest)
        for relative, digest in lock.get("runtime_hashes", {}).items()
    }
    for relative, digest in authorized.items():
        path = ROOT / relative
        if not path.is_file() or _digest(path) != digest:
            errors.append(f"qp3_core_runtime_hash_mismatch:{Path(relative).name}")
            return {}
    return authorized


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _refuse_protected(path: Path) -> None:
    parts = {
        token.lower()
        for part in path.parts
        for token in part.replace("-", "_").replace(".", "_").split("_")
    }
    overlap = sorted(parts & PROTECTED_PATH_PARTS)
    if overlap:
        raise ValueError(f"protected evaluation path refused: {path} ({overlap})")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    _refuse_protected(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _obligation_universe(precommit: dict[str, Any]) -> list[str]:
    proof = precommit["proof_contract_addendum"]
    keys = (
        "required_for_all_answer_families",
        "required_for_directional_operations",
        "required_for_ratio_or_percent_change",
        "required_when_multi_output_or_partial_answer_is_possible",
        "required_for_multi_step",
    )
    return list(dict.fromkeys(item for key in keys for item in proof[key]))


def _applicable(
    precommit: dict[str, Any], family: dict[str, Any], record: dict[str, Any]
) -> list[str]:
    expected = family["expected"]
    if expected["behavior"] != "ANSWER":
        return []
    proof = precommit["proof_contract_addendum"]
    obligations = list(proof["required_for_all_answer_families"])
    if expected["direction_intent"] is not None:
        obligations.extend(proof["required_for_directional_operations"])
    if expected["root_operation"] in {"RATIO_PERCENT", "PERCENT_CHANGE"}:
        obligations.extend(proof["required_for_ratio_or_percent_change"])
    if int(record["requested_output_count"]) > 1:
        obligations.extend(
            proof["required_when_multi_output_or_partial_answer_is_possible"]
        )
    if len(expected["dag"]) > 1:
        obligations.extend(proof["required_for_multi_step"])
    return list(dict.fromkeys(obligations))


def _aggregate_state(states: list[str]) -> str:
    if "INVALID" in states:
        return "INVALID"
    if "UNKNOWN" in states:
        return "UNKNOWN"
    return "VALID"


def audit_record(
    record: dict[str, Any], precommit: dict[str, Any], family: dict[str, Any]
) -> dict[str, Any]:
    universe = _obligation_universe(precommit)
    applicable = _applicable(precommit, family, record)
    if family["expected"]["behavior"] == "ABSTAIN":
        states = {item: "NOT_APPLICABLE" for item in universe}
        return {
            "applicable_obligations": [],
            "proof_states": states,
            "final_proof_state": "UNKNOWN",
            "action": "ABSTAIN",
            "reason": family["expected"]["abstention_reason"],
        }

    states = {
        item: (
            str(record["base_proof_inputs"].get(item, "UNKNOWN"))
            if item in applicable
            else "NOT_APPLICABLE"
        )
        for item in universe
    }
    gold = {str(item["operand_id"]): item for item in record["gold_operands"]}
    candidates = {
        str(item["operand_id"]): item for item in record["candidate_operands"]
    }
    same_slot_set = set(gold) == set(candidates)
    if not same_slot_set:
        role_state = period_state = "INVALID"
    else:
        role_state = (
            "VALID"
            if all(candidates[key]["column_role"] == gold[key]["column_role"] for key in gold)
            else "INVALID"
        )
        period_state = (
            "VALID"
            if all(candidates[key]["period"] == gold[key]["period"] for key in gold)
            else "INVALID"
        )
    states["column_role_match"] = role_state
    states["period_coverage"] = period_state

    bindings = [binding for item in record["candidate_operands"] for binding in item["bindings"]]
    one_binding_per_slot = all(len(item["bindings"]) == 1 for item in record["candidate_operands"])
    logical_ids = [str(item["logical_cell_id"]) for item in bindings]
    states["logical_cell_unique"] = (
        "VALID"
        if one_binding_per_slot and len(logical_ids) == len(set(logical_ids))
        else "INVALID"
    )
    provenance_complete = all(
        binding.get("source_document_sha256")
        and binding.get("source_span_or_coordinate")
        for binding in bindings
    ) and all(item.get("source_span_or_coordinate") for item in record["candidate_operands"])
    states["source_provenance_complete"] = "VALID" if provenance_complete else "UNKNOWN"

    if "reference_base_match" in applicable:
        states["reference_base_match"] = (
            "VALID"
            if record["candidate_reference_base"] == record["reference_base"]
            else "INVALID"
        )
    if "answer_mode_complete" in applicable:
        states["answer_mode_complete"] = (
            "VALID"
            if int(record["available_output_count"]) == int(record["requested_output_count"])
            else "UNKNOWN"
        )
    node_inputs = record["node_proof_inputs"]
    if "all_leaf_nodes_valid" in applicable:
        states["all_leaf_nodes_valid"] = _aggregate_state(list(node_inputs["leaf_states"]))
        states["all_intermediate_nodes_valid"] = _aggregate_state(
            list(node_inputs["intermediate_states"])
        )
        states["dag_execution_valid"] = str(node_inputs["dag_execution_state"])

    required_states = [states[item] for item in applicable]
    final_state = _aggregate_state(required_states)
    action = {"VALID": "ANSWER", "UNKNOWN": "ABSTAIN", "INVALID": "REJECT"}[final_state]
    return {
        "applicable_obligations": applicable,
        "proof_states": states,
        "final_proof_state": final_state,
        "action": action,
        "reason": "all_required_valid" if action == "ANSWER" else "required_proof_not_valid",
    }


def _group_report(details: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[bool]] = defaultdict(list)
    for item in details:
        groups[str(item[key])].append(bool(item["exact"]))
    accuracy = {group: sum(values) / len(values) for group, values in sorted(groups.items())}
    return {
        "groups": len(groups),
        "exact_rate_by_group": accuracy,
        "macro_exact_rate": mean(accuracy.values()) if accuracy else 0.0,
        "worst_group_exact_rate": min(accuracy.values()) if accuracy else 0.0,
    }


def _canonical_candidate(record: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(record["candidate_operands"], key=lambda item: str(item["operand_id"]))


def build_report(
    precommit_path: Path = DEFAULT_PRECOMMIT,
    core_path: Path = DEFAULT_CORE,
    generalization_path: Path = DEFAULT_GENERALIZATION,
    cohort_path: Path = DEFAULT_COHORT,
) -> dict[str, Any]:
    for path in (precommit_path, core_path, generalization_path, cohort_path):
        _refuse_protected(path)
    precommit = json.loads(precommit_path.read_text(encoding="utf-8"))
    core = json.loads(core_path.read_text(encoding="utf-8"))
    generalization = json.loads(generalization_path.read_text(encoding="utf-8"))
    records = _load_jsonl(cohort_path)
    families = {item["family_id"]: item for item in core["families"]}
    errors: list[str] = []

    parent_mismatches = []
    for parent in precommit["parent_contracts"]:
        observed = _digest(ROOT / parent["path"])
        if observed != parent["sha256"]:
            parent_mismatches.append({"path": parent["path"], "expected": parent["sha256"], "observed": observed})
    dependency_mismatches = []
    for dependency in precommit["frozen_dependencies"]:
        observed = _digest(ROOT / dependency["path"])
        if observed != dependency["sha256"]:
            dependency_mismatches.append({"path": dependency["path"], "expected": dependency["sha256"], "observed": observed})
    runtime_hashes = {
        path: _digest(ROOT / path) for path in precommit["runtime_baseline"]
    }
    authorized_runtime_hashes = _authorized_runtime_hashes(errors)
    runtime_mismatches = [
        {"path": path, "expected": expected, "observed": runtime_hashes[path]}
        for path, expected in precommit["runtime_baseline"].items()
        if runtime_hashes[path] != expected
        and authorized_runtime_hashes.get(path) != runtime_hashes[path]
    ]
    if parent_mismatches:
        errors.append("parent_contract_hash_mismatch")
    if dependency_mismatches:
        errors.append("frozen_dependency_hash_mismatch")
    if runtime_mismatches:
        errors.append("runtime_hash_mismatch")

    rebuilt = build_records(precommit, core)
    builder_exact = rebuilt == records
    if not builder_exact:
        errors.append("builder_does_not_reproduce_frozen_cohort")

    required_fields = set(core["annotation_schema"]["required_example_fields"])
    malformed = []
    leakage = []
    details = []
    for record in records:
        missing = sorted(required_fields - set(record))
        if missing:
            malformed.append({"example_id": record.get("example_id"), "missing": missing})
        if (
            record.get("source_kind") != "synthetic_typed_proof_fixture"
            or record.get("real_document_content_used")
            or record.get("financial_numeric_value_used")
            or record.get("numeric_answer_used")
            or record.get("candidate_rank_or_score_used")
            or record.get("family_or_example_id_used_as_runtime_rule")
            or any(record.get("diagnosis_overlap", {}).values())
        ):
            leakage.append(str(record.get("example_id")))
        family = families[str(record["family_id"])]
        observed = audit_record(record, precommit, family)
        exact = (
            observed["applicable_obligations"] == record["expected_applicable_obligations"]
            and observed["proof_states"] == record["expected_proof_states"]
            and observed["final_proof_state"] == record["expected_final_proof_state"]
            and observed["action"] == record["expected_action"]
        )
        details.append(
            {
                "example_id": record["example_id"],
                "family_id": record["family_id"],
                "scenario_id": record["scenario_id"],
                "scenario_class": record["scenario_class"],
                "company_key": record["company_key"],
                "layout_family": record["layout_family"],
                "observed_final_proof_state": observed["final_proof_state"],
                "observed_action": observed["action"],
                "expected_action": record["expected_action"],
                "exact": exact,
            }
        )
    if malformed:
        errors.append("cohort_schema_invalid")
    if leakage:
        errors.append("forbidden_or_diagnosis_signal_present")

    family_counts = Counter(str(item["family_id"]) for item in records)
    company_counts = Counter(str(item["company_key"]) for item in records)
    layout_counts = Counter(str(item["layout_family"]) for item in records)
    scenario_counts = Counter(str(item["scenario_id"]) for item in records)
    class_counts = Counter(str(item["scenario_class"]) for item in records)
    action_counts = Counter(str(item["expected_action"]) for item in records)
    quota = precommit["quota"]
    answer_records = [item for item in records if item["answerability"] == "ANSWER"]
    abstain_records = [item for item in records if item["answerability"] == "ABSTAIN"]
    positive_answer_records = [
        item for item in answer_records if item["scenario_id"] in {"valid_gold", "candidate_permuted"}
    ]

    by_id = {str(item["example_id"]): item for item in records}
    permutation_exact = 0
    for family_id in precommit["scope"]["required_family_ids"]:
        base_scenario = "ambiguity_gold" if family_id in precommit["scope"]["abstain_family_ids"] else "valid_gold"
        base = by_id[f"{family_id}:{base_scenario}"]
        permuted = by_id[f"{family_id}:candidate_permuted"]
        same_evidence = (
            _canonical_candidate(base) == _canonical_candidate(permuted)
            and Counter(base["candidate_pool_order"]) == Counter(permuted["candidate_pool_order"])
            and base["expected_proof_states"] == permuted["expected_proof_states"]
            and base["expected_action"] == permuted["expected_action"]
        )
        permutation_exact += int(same_evidence)

    challenged_obligations = {
        obligation
        for record in records
        for obligation, state in record["expected_proof_states"].items()
        if obligation in precommit["scope"]["missing_obligations"] and state in {"UNKNOWN", "INVALID"}
    }
    exact_count = sum(bool(item["exact"]) for item in details)
    false_accepts = sum(
        item["expected_action"] != "ANSWER" and item["observed_action"] == "ANSWER"
        for item in details
    )
    retained = sum(
        by_id[str(item["example_id"])] ["expected_gold_retained"]
        and item["observed_action"] == "ANSWER"
        for item in details
        if str(item["example_id"]) in {str(record["example_id"]) for record in positive_answer_records}
    )
    gold_retention = retained / len(positive_answer_records) if positive_answer_records else 0.0

    confirmation = precommit["confirmation_partition"]
    company_layout_disjoint = (
        all(not any(item["diagnosis_overlap"].values()) for item in records)
        and set(company_counts) == set(confirmation["companies"])
        and set(layout_counts) == set(confirmation["layout_families"])
        and confirmation["all_companies_unseen_from_diagnosis_by_construction"]
        and confirmation["all_layouts_unseen_from_diagnosis_by_construction"]
    )
    company_report = _group_report(details, "company_key")
    layout_report = _group_report(details, "layout_family")

    quota_checks = {
        "examples": len(records) == quota["exact_examples"],
        "answer_examples": len(answer_records) == quota["exact_answer_examples"],
        "abstain_examples": len(abstain_records) == quota["exact_abstain_examples"],
        "positive_answer_examples": len(positive_answer_records) == quota["exact_positive_answer_examples"],
        "permutation_pairs": permutation_exact == quota["exact_permutation_pairs"],
        "reference_base_negatives": scenario_counts["wrong_reference_base"] == quota["exact_reference_base_negatives"],
        "multi_step_node_negatives": sum(
            scenario_counts[item]
            for item in ("leaf_unknown", "leaf_invalid", "intermediate_unknown", "intermediate_invalid", "dag_execution_invalid")
        ) == quota["exact_multi_step_node_negatives"],
        "false_accepts": false_accepts == quota["exact_false_accepts"],
        "gold_retention": gold_retention >= quota["minimum_gold_retention"],
        "company_groups": len(company_counts) == quota["required_company_groups"],
        "layout_groups": len(layout_counts) == quota["required_layout_groups"],
        "family_coverage": set(family_counts) == set(precommit["scope"]["required_family_ids"]),
        "missing_obligations_exercised": challenged_obligations == set(precommit["scope"]["missing_obligations"]),
    }
    if not all(quota_checks.values()):
        errors.append("quota_or_coverage_gate_failed")
    generalization_rule = generalization["repair_admission_rule"]
    generalization_checks = {
        "diagnosis_documents_excluded": confirmation["diagnosis_documents_excluded"],
        "minimum_companies": len(company_counts) >= generalization_rule["minimum_distinct_companies"],
        "minimum_layouts": len(layout_counts) >= generalization_rule["minimum_distinct_layout_families"],
        "company_layout_disjoint": company_layout_disjoint,
        "positive_and_hard_negative": class_counts["positive"] > 0 and sum(
            count for key, count in class_counts.items() if "negative" in key or "counterfactual" in key
        ) > 0,
        "counterfactual": class_counts["semantic_counterfactual"] > 0,
        "permutation": permutation_exact == len(precommit["scope"]["required_family_ids"]),
        "company_macro_reported": company_report["groups"] > 0,
        "layout_macro_reported": layout_report["groups"] > 0,
        "unseen_company_reported": True,
        "unseen_layout_reported": True,
    }
    if not all(generalization_checks.values()):
        errors.append("report_generalization_gate_failed")
    if exact_count != len(records):
        errors.append("oracle_state_or_action_mismatch")

    precommit_gate_passed = not errors
    return {
        "schema": "finhrm-qp3-core-precommit-audit-v1",
        "created_at": "2026-08-14",
        "classification": "development_only_synthetic_report_agnostic_proof_precommit_audit",
        "precommit_gate_passed": precommit_gate_passed,
        "errors": errors,
        "cohort": {
            "examples": len(records),
            "answer_examples": len(answer_records),
            "abstain_examples": len(abstain_records),
            "families": dict(sorted(family_counts.items())),
            "companies": dict(sorted(company_counts.items())),
            "layouts": dict(sorted(layout_counts.items())),
            "scenarios": dict(sorted(scenario_counts.items())),
            "scenario_classes": dict(sorted(class_counts.items())),
            "expected_actions": dict(sorted(action_counts.items())),
        },
        "oracle": {
            "state_and_action_exact": exact_count,
            "total": len(records),
            "false_accepts": false_accepts,
            "gold_retained": retained,
            "gold_retention_total": len(positive_answer_records),
            "gold_retention": gold_retention,
            "challenged_missing_obligations": sorted(challenged_obligations),
            "failures": [item for item in details if not item["exact"]],
            "details": details,
        },
        "counterfactuals": {
            "permutation_exact": permutation_exact,
            "permutation_total": len(precommit["scope"]["required_family_ids"]),
            "role_swaps": scenario_counts["swapped_role"],
            "year_swaps": scenario_counts["swapped_year"],
            "wrong_reference_bases": scenario_counts["wrong_reference_base"],
            "duplicate_logical_cells": scenario_counts["duplicate_logical_cell"],
            "missing_source_provenance": scenario_counts["missing_source_provenance"],
            "partial_multi_output": scenario_counts["partial_multi_output"],
            "node_proof_negatives": quota["exact_multi_step_node_negatives"],
        },
        "generalization": {
            "checks": generalization_checks,
            "company": company_report,
            "layout": layout_report,
            "unseen_company": {"examples": len(records), "oracle_exact_rate": exact_count / len(records)},
            "unseen_layout": {"examples": len(records), "oracle_exact_rate": exact_count / len(records)},
            "safe_abstention_counted_as_answered": False,
            "universal_coverage_claim_made": False,
        },
        "quota_checks": quota_checks,
        "provenance": {
            "malformed_examples": malformed,
            "forbidden_signal_examples": sorted(set(leakage)),
            "real_document_content_used": False,
            "diagnosis_document_used_as_confirmation": False,
            "financial_numeric_values_used": False,
            "numeric_answers_used": False,
            "candidate_rank_or_score_used": False,
            "family_or_example_id_used_as_runtime_rule": False,
            "builder_reproduces_frozen_cohort": builder_exact,
        },
        "hash_audit": {
            "parent_mismatches": parent_mismatches,
            "dependency_mismatches": dependency_mismatches,
            "runtime_observed": runtime_hashes,
            "runtime_mismatches": runtime_mismatches,
        },
        "artifact_hashes": {
            "precommit": _digest(precommit_path),
            "cohort": _digest(cohort_path),
            "builder": _digest(ROOT / "scripts" / "build_qp3_core_precommit_cohort.py"),
            "validator": _digest(Path(__file__)),
        },
        "invariants": {
            "runtime_implementation_performed": False,
            "runtime_acceptance_policy_changed": False,
            "threshold_changed": False,
            "checkpoint_changed": False,
            "fine_tuning_performed": False,
            "pdf_loaded": False,
            "model_loaded": False,
            "global_validation_loaded": False,
            "sealed_validation_loaded": False,
            "blind_test_loaded": False,
            "phase_d_unlocked": False,
        },
        "authorized_next_step": precommit["authorized_next_step_after_precommit"],
    }


def _render_markdown(report: dict[str, Any]) -> str:
    oracle = report["oracle"]
    generalization = report["generalization"]
    return "\n".join(
        [
            "# FinHRM QP3-CORE-PRECOMMIT",
            "",
            f"- Gate: **{'PASS' if report['precommit_gate_passed'] else 'FAIL'}**",
            f"- Oracle state/action exact: `{oracle['state_and_action_exact']}/{oracle['total']}`",
            f"- False accepts: `{oracle['false_accepts']}`",
            f"- Positive gold retention: `{oracle['gold_retained']}/{oracle['gold_retention_total']}`",
            f"- Candidate permutation: `{report['counterfactuals']['permutation_exact']}/{report['counterfactuals']['permutation_total']}`",
            f"- Company groups: `{generalization['company']['groups']}`, macro/worst `{generalization['company']['macro_exact_rate']:.3f}/{generalization['company']['worst_group_exact_rate']:.3f}`",
            f"- Layout groups: `{generalization['layout']['groups']}`, macro/worst `{generalization['layout']['macro_exact_rate']:.3f}/{generalization['layout']['worst_group_exact_rate']:.3f}`",
            "- Confirmation uses only synthetic typed proof fixtures in a dedicated company/layout namespace; no diagnosis document or real PDF content is used.",
            "- Runtime, checkpoint, threshold, training state, protected evaluation sources, and Phase D remain unchanged/locked.",
            "",
            "Authorized next step: `QP3-CORE-IMPLEMENT` under the frozen proof/action oracle; no acceptance expansion or threshold change.",
            "",
        ]
    )


def _write_outputs(
    report: dict[str, Any], report_path: Path, markdown_path: Path, lock_path: Path
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    artifact_paths = [
        DEFAULT_PRECOMMIT,
        DEFAULT_COHORT,
        ROOT / "scripts" / "build_qp3_core_precommit_cohort.py",
        Path(__file__),
        ROOT / "tests" / "test_qp3_core_precommit.py",
        report_path,
        markdown_path,
    ]
    artifacts = [
        {"path": _relative(path), "size_bytes": path.stat().st_size, "sha256": _digest(path)}
        for path in artifact_paths
        if path.exists()
    ]
    lock = {
        "schema": "finhrm-qp3-core-precommit-lock-v1",
        "created_at": "2026-08-14",
        "classification": "development_only_synthetic_report_agnostic_proof_precommit_lock",
        "status": "complete" if report["precommit_gate_passed"] else "failed",
        "phase_d": "locked",
        "precommit_gate_passed": report["precommit_gate_passed"],
        "results": {
            "oracle_state_and_action": {"exact": report["oracle"]["state_and_action_exact"], "total": report["oracle"]["total"]},
            "false_accepts": report["oracle"]["false_accepts"],
            "gold_retention": {"retained": report["oracle"]["gold_retained"], "total": report["oracle"]["gold_retention_total"]},
            "permutation": {"exact": report["counterfactuals"]["permutation_exact"], "total": report["counterfactuals"]["permutation_total"]},
            "companies": report["generalization"]["company"],
            "layouts": report["generalization"]["layout"],
            "runtime_hash_mismatches": len(report["hash_audit"]["runtime_mismatches"]),
        },
        "invariants": report["invariants"],
        "artifacts": artifacts,
        "frozen_dependencies": json.loads(DEFAULT_PRECOMMIT.read_text(encoding="utf-8"))["frozen_dependencies"],
        "authorized_next_step": report["authorized_next_step"],
    }
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and lock QP3-CORE-PRECOMMIT")
    parser.add_argument("--precommit", type=Path, default=DEFAULT_PRECOMMIT)
    parser.add_argument("--core", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--generalization", type=Path, default=DEFAULT_GENERALIZATION)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    report = build_report(args.precommit, args.core, args.generalization, args.cohort)
    _write_outputs(report, args.report, args.markdown, args.lock)
    print(
        json.dumps(
            {
                "gate": report["precommit_gate_passed"],
                "oracle_exact": report["oracle"]["state_and_action_exact"],
                "total": report["oracle"]["total"],
                "false_accepts": report["oracle"]["false_accepts"],
                "gold_retention": report["oracle"]["gold_retention"],
                "companies": report["generalization"]["company"]["groups"],
                "layouts": report["generalization"]["layout"]["groups"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["precommit_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
