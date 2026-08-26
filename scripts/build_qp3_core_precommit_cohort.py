from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRECOMMIT = ROOT / "config" / "finhrm_qp3_core_precommit_v1.json"
DEFAULT_CORE = ROOT / "config" / "finhrm_thesis_core_v1.json"
DEFAULT_OUTPUT = ROOT / "data" / "finhrm_qp3_core_development_v1.jsonl"


def _scenario_specs(precommit: dict[str, Any], family: dict[str, Any]) -> list[dict[str, str]]:
    expected = family["expected"]
    matrix = precommit["scenario_matrix"]
    if expected["behavior"] == "ABSTAIN":
        return list(matrix["all_abstain_families"])
    scenarios = list(matrix["all_answer_families"])
    if expected["root_operation"] in {"RATIO_PERCENT", "PERCENT_CHANGE"}:
        scenarios.extend(matrix["ratio_or_percent_change_only"])
    if len(expected["dag"]) > 1:
        scenarios.extend(matrix["multi_step_only"])
    return scenarios


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


def _applicable_obligations(
    precommit: dict[str, Any], family: dict[str, Any], scenario_id: str
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
    if scenario_id == "partial_multi_output":
        obligations.extend(
            proof["required_when_multi_output_or_partial_answer_is_possible"]
        )
    if len(expected["dag"]) > 1:
        obligations.extend(proof["required_for_multi_step"])
    return list(dict.fromkeys(obligations))


def _synthetic_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _operand(
    *, raw: dict[str, Any], document_id: str, table_id: str, ordinal: int
) -> dict[str, Any]:
    operand_id = str(raw["id"])
    logical_cell_id = (
        f"{document_id}|{table_id}|row:{ordinal}|column:{ordinal}|"
        f"{raw['metric']}|{raw['year']}|{raw['column_role']}|"
        f"{raw['entity_scope']}|{raw['unit_family']}"
    )
    return {
        "operand_id": operand_id,
        "value": f"OPAQUE_VALUE::{operand_id}",
        "metric": raw["metric"],
        "period": raw["year"],
        "column_role": raw["column_role"],
        "entity_scope": raw["entity_scope"],
        "unit_scale": raw["unit_family"],
        "logical_cell_id": logical_cell_id,
        "document_id": document_id,
        "page": 1,
        "table_id": table_id,
        "row_id": f"row:{ordinal}",
        "column_id": f"column:{ordinal}",
        "source_span_or_coordinate": [
            f"synthetic-line:{10 + ordinal}",
            f"synthetic-cell:{ordinal}:{ordinal}",
        ],
        "evidence_state": "EXPLICIT",
        "bindings": [
            {
                "logical_cell_id": logical_cell_id,
                "physical_cell_id": f"physical:{ordinal}:0",
                "source_span_or_coordinate": [
                    f"synthetic-line:{10 + ordinal}",
                    f"synthetic-cell:{ordinal}:{ordinal}",
                ],
                "source_document_sha256": _synthetic_sha256(document_id),
            }
        ],
    }


def _swap_role(role: str) -> str:
    if role == "PLAN":
        return "ACTUAL"
    return "PLAN"


def _expected_states(
    precommit: dict[str, Any], family: dict[str, Any], scenario_id: str
) -> tuple[list[str], dict[str, str], str, str]:
    universe = _obligation_universe(precommit)
    applicable = _applicable_obligations(precommit, family, scenario_id)
    if family["expected"]["behavior"] == "ABSTAIN":
        return [], {item: "NOT_APPLICABLE" for item in universe}, "UNKNOWN", "ABSTAIN"
    states = {
        item: ("VALID" if item in applicable else "NOT_APPLICABLE")
        for item in universe
    }
    mutation = {
        "swapped_role": ("column_role_match", "INVALID"),
        "swapped_year": ("period_coverage", "INVALID"),
        "wrong_reference_base": ("reference_base_match", "INVALID"),
        "duplicate_logical_cell": ("logical_cell_unique", "INVALID"),
        "missing_source_provenance": ("source_provenance_complete", "UNKNOWN"),
        "partial_multi_output": ("answer_mode_complete", "UNKNOWN"),
        "leaf_unknown": ("all_leaf_nodes_valid", "UNKNOWN"),
        "leaf_invalid": ("all_leaf_nodes_valid", "INVALID"),
        "intermediate_unknown": ("all_intermediate_nodes_valid", "UNKNOWN"),
        "intermediate_invalid": ("all_intermediate_nodes_valid", "INVALID"),
        "dag_execution_invalid": ("dag_execution_valid", "INVALID"),
    }.get(scenario_id)
    if mutation is not None:
        states[mutation[0]] = mutation[1]
    required_states = [states[item] for item in applicable]
    if "INVALID" in required_states:
        return applicable, states, "INVALID", "REJECT"
    if "UNKNOWN" in required_states:
        return applicable, states, "UNKNOWN", "ABSTAIN"
    return applicable, states, "VALID", "ANSWER"


def _apply_scenario(record: dict[str, Any], scenario_id: str) -> None:
    operands = record["candidate_operands"]
    if scenario_id == "candidate_permuted":
        operands.reverse()
        record["candidate_pool_order"].reverse()
    elif scenario_id == "swapped_role":
        operands[0]["column_role"] = _swap_role(str(operands[0]["column_role"]))
    elif scenario_id == "swapped_year":
        operands[0]["period"] = f"counterfactual:{operands[0]['period']}"
    elif scenario_id == "wrong_reference_base":
        expected_base = str(record["reference_base"])
        alternatives = [
            str(item["operand_id"])
            for item in operands
            if str(item["operand_id"]) != expected_base
        ]
        record["candidate_reference_base"] = (
            alternatives[0] if alternatives else f"wrong:{expected_base}"
        )
    elif scenario_id == "duplicate_logical_cell":
        duplicate = deepcopy(operands[0]["bindings"][0])
        duplicate["physical_cell_id"] = "physical:duplicate"
        duplicate["source_span_or_coordinate"] = ["synthetic-line:duplicate"]
        operands[0]["bindings"].append(duplicate)
    elif scenario_id == "missing_source_provenance":
        operands[0]["bindings"][0]["source_span_or_coordinate"] = []
        operands[0]["source_span_or_coordinate"] = []
    elif scenario_id == "partial_multi_output":
        record["requested_output_count"] = 2
        record["available_output_count"] = 1
        record["answer_mode"] = "MULTI_OUTPUT_ALL_OR_ABSTAIN"
    elif scenario_id == "leaf_unknown":
        record["node_proof_inputs"]["leaf_states"][0] = "UNKNOWN"
    elif scenario_id == "leaf_invalid":
        record["node_proof_inputs"]["leaf_states"][0] = "INVALID"
    elif scenario_id == "intermediate_unknown":
        record["node_proof_inputs"]["intermediate_states"][0] = "UNKNOWN"
    elif scenario_id == "intermediate_invalid":
        record["node_proof_inputs"]["intermediate_states"][0] = "INVALID"
    elif scenario_id == "dag_execution_invalid":
        record["node_proof_inputs"]["dag_execution_state"] = "INVALID"
    elif scenario_id == "tempting_candidate":
        record["candidate_program_proposed"] = True
        record["candidate_program_materialized"] = False


def _base_record(
    *,
    precommit: dict[str, Any],
    family: dict[str, Any],
    family_index: int,
    scenario: dict[str, str],
) -> dict[str, Any]:
    expected = family["expected"]
    company = precommit["confirmation_partition"]["companies"][family_index % 6]
    layout = precommit["confirmation_partition"]["layout_families"][family_index % 6]
    family_id = str(family["family_id"])
    scenario_id = str(scenario["id"])
    report_id = f"synthetic:qp3-core:{company}:{layout}:{family_id}"
    document_id = f"{report_id}:document"
    table_id = f"{report_id}:table"
    gold_operands = [
        _operand(raw=item, document_id=document_id, table_id=table_id, ordinal=index)
        for index, item in enumerate(expected["operands"])
    ]
    leaf_count = max(1, len(gold_operands))
    intermediate_count = max(1, len(expected["dag"]) - 1)
    applicable, proof_states, final_state, action = _expected_states(
        precommit, family, scenario_id
    )
    answerability = str(expected["behavior"])
    record: dict[str, Any] = {
        "schema": "finhrm-qp3-core-proof-example-v1",
        "classification": "development_only_synthetic_report_agnostic_proof_precommit",
        "example_id": f"{family_id}:{scenario_id}",
        "family_id": family_id,
        "scenario_id": scenario_id,
        "scenario_class": scenario["class"],
        "question": family["canonical_question"],
        "answerability": answerability,
        "answer_or_abstention": (
            "OPAQUE_GOLD_OUTPUT"
            if answerability == "ANSWER"
            else expected["abstention_reason"]
        ),
        "answer_mode": expected["answer_mode"],
        "program_dag": deepcopy(expected["dag"]),
        "operand_order": [item["id"] for item in expected["operands"]],
        "reference_base": expected["reference_base"],
        "hard_negative_type": (
            "NONE" if scenario["class"] in {"positive", "permutation_control"}
            else scenario_id.upper()
        ),
        "company_key": company,
        "report_id": report_id,
        "source_document_sha256": _synthetic_sha256(document_id),
        "layout_family": layout,
        "annotator_status": "single_reviewed_development",
        "source_kind": "synthetic_typed_proof_fixture",
        "diagnosis_overlap": {"company": False, "layout": False, "document": False},
        "gold_operands": gold_operands,
        "candidate_operands": deepcopy(gold_operands),
        "candidate_reference_base": expected["reference_base"],
        "candidate_pool_order": [
            *(f"candidate:{item['id']}" for item in expected["operands"]),
            "distractor:semantic",
            "distractor:provenance",
        ],
        "candidate_program_proposed": answerability == "ANSWER",
        "candidate_program_materialized": answerability == "ANSWER",
        "requested_output_count": 1,
        "available_output_count": 1 if answerability == "ANSWER" else 0,
        "node_proof_inputs": {
            "leaf_states": ["VALID"] * leaf_count,
            "intermediate_states": ["VALID"] * intermediate_count,
            "dag_execution_state": "VALID",
        },
        "base_proof_inputs": {
            item: "VALID"
            for item in applicable
            if item
            not in {
                "column_role_match",
                "period_coverage",
                "logical_cell_unique",
                "source_provenance_complete",
                "reference_base_match",
                "answer_mode_complete",
                "all_leaf_nodes_valid",
                "all_intermediate_nodes_valid",
                "dag_execution_valid",
            }
        },
        "control_pair_id": (
            f"permutation:{family_id}"
            if scenario_id in {"valid_gold", "ambiguity_gold", "candidate_permuted"}
            else None
        ),
        "expected_applicable_obligations": applicable,
        "expected_proof_states": proof_states,
        "expected_final_proof_state": final_state,
        "expected_action": action,
        "expected_gold_retained": action == "ANSWER" and answerability == "ANSWER",
        "financial_numeric_value_used": False,
        "numeric_answer_used": False,
        "real_document_content_used": False,
        "candidate_rank_or_score_used": False,
        "family_or_example_id_used_as_runtime_rule": False,
    }
    _apply_scenario(record, scenario_id)
    return record


def build_records(
    precommit: dict[str, Any], core: dict[str, Any]
) -> list[dict[str, Any]]:
    families = {item["family_id"]: item for item in core["families"]}
    records: list[dict[str, Any]] = []
    for family_index, family_id in enumerate(precommit["scope"]["required_family_ids"]):
        family = families[family_id]
        for scenario in _scenario_specs(precommit, family):
            records.append(
                _base_record(
                    precommit=precommit,
                    family=family,
                    family_index=family_index,
                    scenario=scenario,
                )
            )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the frozen QP3-CORE precommit cohort")
    parser.add_argument("--precommit", type=Path, default=DEFAULT_PRECOMMIT)
    parser.add_argument("--core", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    precommit = json.loads(args.precommit.read_text(encoding="utf-8"))
    core = json.loads(args.core.read_text(encoding="utf-8"))
    records = build_records(precommit, core)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in records),
        encoding="utf-8",
    )
    print(f"examples={len(records)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
