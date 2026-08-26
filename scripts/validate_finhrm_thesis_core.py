"""Validate the frozen FinHRM thesis-core research contract.

The validator is deliberately compiler/data independent: it checks only the
precommitted specification, its frozen source hashes, and canonical synthetic
question-family contracts.  It does not load PDFs, models, validation, blind,
or test data.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "config" / "finhrm_thesis_core_v1.json"
DEFAULT_REPORT = ROOT / "reports" / "finhrm_thesis_core_precommit_v1.json"

EXPECTED_CORE = {
    "PA01", "PA02", "PA03", "PA04", "PA05", "PA06", "PA07", "PA08", "PA09",
    "TY01", "TY02", "TY04", "IBDS01", "C13", "C14", "C15",
}
EXPECTED_ABSTAIN = {"C13", "C14", "C15"}
ALLOWED_OPERATIONS = {
    "SELECT", "ADD", "SUBTRACT", "MULTIPLY", "DIVIDE", "RATIO_PERCENT",
    "PERCENT_CHANGE", "AVERAGE",
}
ALLOWED_COLUMN_ROLES = {"ACTUAL", "PLAN", "REVISED_PLAN", "FORECAST"}
REQUIRED_NEGATIVES = {
    "HN_ROLE_SWAP", "HN_YEAR_SWAP", "HN_ENTITY_SCOPE", "HN_UNIT_SCALE",
    "HN_SAME_VALUE_DIFFERENT_IDENTITY", "HN_PERCENT_NEAR_ABSOLUTE",
    "HN_SUBTOTAL_DETAIL", "HN_TABLE_SCOPE", "HN_COINCIDENTAL_EXECUTION",
    "HN_MISSING_QUALIFIER",
}


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _canonical_catalog(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        if row.get("variant") == "canonical":
            result[str(row["family_id"])] = row
    return result


def validate_contract(contract_path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    contract_path = contract_path.resolve()
    contract = _load_json(contract_path)
    errors: list[str] = []
    warnings: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(
        contract.get("schema") == "finhrm-thesis-core-precommit-v1",
        "unexpected contract schema",
    )
    require(
        contract.get("status") == "precommitted_before_qp2_qp3_runtime_changes",
        "contract is not marked as precommitted before runtime changes",
    )
    invariants = contract.get("invariants", {})
    for key in (
        "protected_evaluation_sources_allowed",
        "fine_tuning_allowed_during_precommit",
        "threshold_change_allowed_during_precommit",
        "runtime_acceptance_change_allowed_during_precommit",
        "phase_d_unlocked",
        "company_page_year_value_rules_allowed",
    ):
        require(invariants.get(key) is False, f"invariant must remain false: {key}")

    frozen = contract.get("frozen_source_artifacts", {})
    frozen_hash_results: dict[str, dict[str, Any]] = {}
    for name, item in frozen.items():
        path = ROOT / str(item.get("path", ""))
        exists = path.is_file()
        actual = _digest(path) if exists else ""
        expected = str(item.get("sha256", ""))
        matches = exists and actual == expected
        frozen_hash_results[name] = {
            "path": str(path.relative_to(ROOT)) if exists else str(path),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matches": matches,
        }
        require(matches, f"frozen source hash mismatch: {name}")

    families = contract.get("families", [])
    require(isinstance(families, list), "families must be a list")
    family_ids = [str(item.get("family_id", "")) for item in families]
    require(len(family_ids) == len(set(family_ids)), "family IDs must be unique")
    require(set(family_ids) == EXPECTED_CORE, "core family set differs from precommit")

    scope = contract.get("scope", {})
    require(set(scope.get("core_family_ids", [])) == EXPECTED_CORE, "scope core IDs differ")
    require(
        set(scope.get("abstain_family_ids", [])) == EXPECTED_ABSTAIN,
        "scope abstain IDs differ",
    )
    require(
        set(scope.get("answer_family_ids", [])) == EXPECTED_CORE - EXPECTED_ABSTAIN,
        "scope answer IDs differ",
    )

    for family in families:
        family_id = str(family.get("family_id", ""))
        expected = family.get("expected", {})
        behavior = expected.get("behavior")
        operands = expected.get("operands", [])
        dag = expected.get("dag", [])
        root_operation = expected.get("root_operation")
        require(behavior in {"ANSWER", "ABSTAIN"}, f"{family_id}: invalid behavior")
        if behavior == "ABSTAIN":
            require(family_id in EXPECTED_ABSTAIN, f"{family_id}: unexpected ABSTAIN family")
            require(root_operation is None, f"{family_id}: ABSTAIN cannot have root operation")
            require(not operands and not dag, f"{family_id}: ABSTAIN cannot materialize operands/DAG")
            require(bool(expected.get("abstention_reason")), f"{family_id}: missing abstention reason")
            continue

        require(family_id not in EXPECTED_ABSTAIN, f"{family_id}: ambiguity family must abstain")
        require(root_operation in ALLOWED_OPERATIONS, f"{family_id}: invalid root operation")
        require(bool(operands), f"{family_id}: answer family needs operands")
        require(bool(dag), f"{family_id}: answer family needs DAG")
        operand_ids = [str(item.get("id", "")) for item in operands]
        require(len(operand_ids) == len(set(operand_ids)), f"{family_id}: duplicate operand IDs")
        for operand in operands:
            operand_id = str(operand.get("id", ""))
            require(bool(operand_id), f"{family_id}: operand without ID")
            require(bool(operand.get("metric")), f"{family_id}/{operand_id}: missing metric")
            require(bool(operand.get("year")), f"{family_id}/{operand_id}: missing year")
            require(
                operand.get("column_role") in ALLOWED_COLUMN_ROLES,
                f"{family_id}/{operand_id}: invalid column role",
            )
            require(bool(operand.get("entity_scope")), f"{family_id}/{operand_id}: missing scope")
            require(bool(operand.get("unit_family")), f"{family_id}/{operand_id}: missing unit")

        available = set(operand_ids)
        node_ids: list[str] = []
        for node in dag:
            node_id = str(node.get("id", ""))
            operation = node.get("operation")
            inputs = [str(item) for item in node.get("inputs", [])]
            require(node_id and node_id not in available, f"{family_id}: duplicate DAG node {node_id}")
            require(operation in ALLOWED_OPERATIONS, f"{family_id}/{node_id}: invalid operation")
            require(inputs and all(item in available for item in inputs), f"{family_id}/{node_id}: unresolved input")
            available.add(node_id)
            node_ids.append(node_id)
        require(node_ids[-1:] == ["answer"], f"{family_id}: final DAG node must be answer")
        require(dag[-1].get("operation") == root_operation, f"{family_id}: root operation mismatch")
        if root_operation in {"RATIO_PERCENT", "PERCENT_CHANGE"}:
            require(bool(expected.get("reference_base")), f"{family_id}: missing reference base")

    question_catalog_path = ROOT / frozen["question_family_catalog"]["path"]
    canonical = _canonical_catalog(question_catalog_path)
    catalog_alignment: dict[str, bool] = {}
    for family in families:
        family_id = str(family["family_id"])
        if family_id == "IBDS01":
            continue
        row = canonical.get(family_id)
        aligned = row is not None
        if aligned:
            expected = family["expected"]
            existing = row["expected_ir"]
            aligned = (
                existing.get("behavior") == expected.get("behavior")
                and existing.get("root_operation") == expected.get("root_operation")
                and existing.get("answer_mode") == expected.get("answer_mode")
                and existing.get("requested_unit_family") == expected.get("requested_unit_family")
            )
        catalog_alignment[family_id] = aligned
        require(aligned, f"{family_id}: canonical catalog contract mismatch")

    negatives = contract.get("hard_negative_taxonomy", [])
    negative_ids = [str(item.get("id", "")) for item in negatives]
    require(len(negative_ids) == len(set(negative_ids)), "hard-negative IDs must be unique")
    require(set(negative_ids) == REQUIRED_NEGATIVES, "hard-negative taxonomy differs")

    annotation = contract.get("annotation_schema", {})
    example_fields = set(annotation.get("required_example_fields", []))
    operand_fields = set(annotation.get("required_operand_fields", []))
    require(
        {"program_dag", "reference_base", "source_document_sha256", "annotator_status"}
        <= example_fields,
        "annotation example schema lacks a required research field",
    )
    require(
        {"logical_cell_id", "column_role", "source_span_or_coordinate", "evidence_state"}
        <= operand_fields,
        "annotation operand schema lacks a required provenance field",
    )

    split = contract.get("split_protocol", {})
    proportions = split.get("target_company_proportions", {})
    require(abs(sum(float(value) for value in proportions.values()) - 1.0) < 1e-9, "split proportions must sum to one")
    require(split.get("no_company_overlap") is True, "company overlap must be forbidden")
    require(split.get("sealed_test_threshold_selection_allowed") is False, "sealed test cannot select threshold")

    gates = contract.get("release_gates", {})
    require(
        gates.get("compiler_contract", {}).get("core_explicit_intent_exact") == 1.0,
        "compiler exact gate must be 100%",
    )
    require(
        gates.get("counterfactual", {}).get("swapped_role_false_accepts") == 0,
        "role-swap false-accept gate must be zero",
    )
    require(
        gates.get("selective_thesis", {}).get("minimum_accepted_precision_point_estimate") == 0.90,
        "thesis accepted precision gate must remain 90%",
    )
    require(
        gates.get("selective_publication", {}).get("minimum_accepted_precision_one_sided_95_lower_bound") == 0.90,
        "publication precision lower-bound gate must remain 90%",
    )

    return {
        "schema": "finhrm-thesis-core-precommit-validation-v1",
        "classification": "development_only_contract_validation",
        "contract": str(contract_path.relative_to(ROOT)),
        "contract_sha256": _digest(contract_path),
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "research_questions": len(contract.get("research_questions", [])),
            "families": len(families),
            "answer_families": sum(item["expected"]["behavior"] == "ANSWER" for item in families),
            "abstain_families": sum(item["expected"]["behavior"] == "ABSTAIN" for item in families),
            "multi_step_families": sum(len(item["expected"].get("dag", [])) > 1 for item in families),
            "hard_negative_types": len(negatives),
        },
        "family_ids": family_ids,
        "catalog_alignment": catalog_alignment,
        "frozen_hash_results": frozen_hash_results,
        "invariants": {
            "model_loaded": False,
            "pdf_loaded": False,
            "validation_loaded": False,
            "blind_test_loaded": False,
            "fine_tuning_performed": False,
            "threshold_changed": False,
            "runtime_changed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = validate_contract(args.contract)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
