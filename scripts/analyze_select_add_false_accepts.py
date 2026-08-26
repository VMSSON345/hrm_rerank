from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.evaluate_direction_contract_folds import (  # noqa: E402
    _current_winner,
    _digest,
    _execution_correct,
    _identity_constraint_required,
    _read_json,
    _read_jsonl_by_id,
)
from hrm_finance_qa.evidence_schema import EvidenceCell, build_evidence_graph  # noqa: E402
from hrm_finance_qa.ontology import normalize_text  # noqa: E402
from hrm_finance_qa.program_data import (  # noqa: E402
    ProgramExample,
    ReasoningProgram,
    execute_program,
    load_program_examples,
)
from hrm_finance_qa.question_planner import compile_question_plan  # noqa: E402
from hrm_finance_qa.symbolic_verifier import verify_program  # noqa: E402
from hrm_finance_qa.v12b2_supervision import (  # noqa: E402
    parse_program_key,
    supervision_integrity_issues,
)


SCOPED_OPERATIONS = {"SELECT", "ADD"}
PROTECTED_PATH_TOKENS = {"blind", "test", "holdout"}


def _guard_development_only(
    fold_manifest: dict[str, object],
    phase_b_manifest: dict[str, object],
    phase_c_manifest: dict[str, object],
) -> None:
    if fold_manifest.get("blind_test_used") is not False:
        raise ValueError("Refusing a fold manifest that touched blind/test data")
    for key in ("blind_test_used", "test_jsonl_used", "sealed_validation_loaded"):
        if key in phase_c_manifest and phase_c_manifest.get(key) is not False:
            raise ValueError(f"Refusing Phase-C manifest with {key} != false")
    if phase_c_manifest.get("reranker_training_started") is not False:
        raise ValueError("False-accept audit must not use a trained Phase-D reranker")
    classification = str(phase_b_manifest.get("classification", "")).lower()
    if classification and "development" not in classification:
        raise ValueError("Phase-B cache is not classified development-only")


def _safe_local_input(manifest: dict[str, object]) -> Path:
    raw = str(manifest.get("input", ""))
    name = Path(raw.replace("\\", "/")).name
    if not name:
        raise ValueError("Fold manifest input is missing")
    lowered = name.lower()
    if any(token in lowered for token in PROTECTED_PATH_TOKENS):
        raise ValueError(f"Refusing protected input filename: {name}")
    local = ROOT / "data" / name
    if not local.is_file():
        raise FileNotFoundError(local)
    return local


def _program_key(program: ReasoningProgram) -> str:
    return f"{program.operation}:{','.join(str(value) for value in program.operands)}"


def _raw_winner_candidate(
    record: dict[str, object], winner: ReasoningProgram
) -> dict[str, object] | None:
    for raw in record.get("candidates", ()):
        if not isinstance(raw, dict):
            continue
        try:
            parsed = parse_program_key(str(raw.get("program_key", "")))
        except (TypeError, ValueError):
            continue
        if parsed == winner:
            return raw
    return None


def _cell_summary(cell: EvidenceCell) -> dict[str, object]:
    return {
        "index": cell.index,
        "value": str(cell.value),
        "metric": cell.metric_label,
        "metric_key": cell.metric_key,
        "column": cell.column,
        "period": cell.period,
        "period_role": cell.period_role,
        "explicit_years": list(cell.explicit_years),
        "entity": cell.entity_id or cell.entity,
        "section": cell.section,
        "table_id": cell.table_id,
        "subtable_id": cell.subtable_id,
        "row_id": cell.row_id,
        "column_id": cell.column_id,
        "unit": cell.unit,
        "unit_key": cell.unit_key,
        "source_name": cell.source_name,
        "page": cell.page,
    }


def _selected_cells(
    graph_cells: tuple[EvidenceCell, ...], program: ReasoningProgram
) -> list[EvidenceCell]:
    return [graph_cells[index] for index in program.operands]


def _states(verification: object) -> dict[str, str | None]:
    return {
        item.name: item.state.value if item.state is not None else None
        for item in verification.proof_obligations
    }


def _required_states(verification: object) -> dict[str, str]:
    return {
        item.name: item.state.value
        for item in verification.proof_obligations
        if item.applicability == "required" and item.state is not None
    }


def _metadata_signature(cell: EvidenceCell) -> tuple[str, ...]:
    return (
        cell.metric_key,
        normalize_text(cell.period),
        cell.period_role,
        "|".join(cell.explicit_years),
        normalize_text(cell.entity_id or cell.entity),
        normalize_text(cell.table_id),
        normalize_text(cell.unit_key),
    )


def _physical_semantic_signature(cell: EvidenceCell) -> tuple[str, ...]:
    return (
        cell.metric_key,
        normalize_text(cell.period),
        cell.period_role,
        "|".join(cell.explicit_years),
        normalize_text(cell.entity_id or cell.entity),
        normalize_text(cell.source_name),
        str(cell.page),
        normalize_text(cell.table_id),
        normalize_text(cell.subtable_id),
        normalize_text(cell.row_id),
        normalize_text(cell.column_id),
        normalize_text(cell.unit_key),
    )


def _structural_tags(
    plan: object,
    gold: ReasoningProgram,
    predicted: ReasoningProgram,
    gold_cells: list[EvidenceCell],
    predicted_cells: list[EvidenceCell],
    required_states: dict[str, str],
) -> list[str]:
    tags: list[str] = []
    if predicted.operation != gold.operation:
        tags.append("wrong_operation")
    if any(value == "INVALID" for value in required_states.values()):
        tags.append("required_contradiction_not_enforced")
    if any(value == "UNKNOWN" for value in required_states.values()):
        tags.append("required_proof_incomplete_but_accepted")
    if required_states and all(value == "VALID" for value in required_states.values()):
        tags.append("all_required_proof_marked_valid_but_wrong")
    if (
        gold.operation == "ADD"
        and required_states.get("operation_match") == "UNKNOWN"
        and required_states.get("add_grounding") == "UNKNOWN"
    ):
        tags.append("add_intent_and_roles_unresolved")

    gold_metrics = Counter(cell.metric_key for cell in gold_cells)
    predicted_metrics = Counter(cell.metric_key for cell in predicted_cells)
    if gold_metrics != predicted_metrics:
        tags.append("wrong_metric_or_operand_role")
    gold_periods = Counter(
        (normalize_text(cell.period), cell.period_role, tuple(cell.explicit_years))
        for cell in gold_cells
    )
    predicted_periods = Counter(
        (normalize_text(cell.period), cell.period_role, tuple(cell.explicit_years))
        for cell in predicted_cells
    )
    if gold_periods != predicted_periods:
        tags.append("wrong_period_or_period_role")
    if Counter(normalize_text(cell.entity_id or cell.entity) for cell in gold_cells) != Counter(
        normalize_text(cell.entity_id or cell.entity) for cell in predicted_cells
    ):
        tags.append("wrong_entity_or_scope")
    if Counter(cell.table_id for cell in gold_cells) != Counter(
        cell.table_id for cell in predicted_cells
    ):
        tags.append("wrong_table")
    if Counter(cell.unit_key for cell in gold_cells) != Counter(
        cell.unit_key for cell in predicted_cells
    ):
        tags.append("wrong_unit")
        if gold.operation == "SELECT" and "unit_compatibility" not in required_states:
            tags.append("select_unit_not_required")
    if len(set(predicted.operands)) != len(predicted.operands):
        tags.append("duplicate_pointer")

    if len(gold_cells) == len(predicted_cells) and Counter(
        _metadata_signature(cell) for cell in gold_cells
    ) == Counter(_metadata_signature(cell) for cell in predicted_cells):
        tags.append("conflicting_value_same_typed_signature")
    if any(
        _physical_semantic_signature(gold_cell)
        == _physical_semantic_signature(predicted_cell)
        and gold_cell.index != predicted_cell.index
        for gold_cell in gold_cells
        for predicted_cell in predicted_cells
    ):
        tags.append("conflicting_value_same_physical_semantic_cell")
    if any(
        gold_cell.metric_key == predicted_cell.metric_key
        and gold_cell.table_id == predicted_cell.table_id
        and normalize_text(gold_cell.period) == normalize_text(predicted_cell.period)
        and gold_cell.period_role == predicted_cell.period_role
        and gold_cell.explicit_years == predicted_cell.explicit_years
        and gold_cell.subtable_id != predicted_cell.subtable_id
        and gold_cell.index != predicted_cell.index
        for gold_cell in gold_cells
        for predicted_cell in predicted_cells
    ):
        tags.append("same_metric_period_duplicated_across_subtables")
    if (
        gold_periods != predicted_periods
        and any(cell.explicit_years for cell in gold_cells + predicted_cells)
        and Counter(cell.explicit_years for cell in gold_cells)
        == Counter(cell.explicit_years for cell in predicted_cells)
    ):
        tags.append("year_only_period_proof_too_coarse")
    if (
        Counter(normalize_text(cell.entity_id or cell.entity) for cell in gold_cells)
        != Counter(normalize_text(cell.entity_id or cell.entity) for cell in predicted_cells)
        and "named_entity" not in required_states
    ):
        tags.append("entity_scope_not_required")

    requested_relative = any(
        phrase in normalize_text(plan.question)
        for phrase in ("so dau nam", "so cuoi nam", "dau ky", "cuoi ky")
    )
    if (
        requested_relative
        and not plan.requested_years
        and "period_coverage" not in required_states
    ):
        tags.append("relative_period_absent_from_required_contract")
    return list(dict.fromkeys(tags))


def _decimal_value(program: ReasoningProgram, example: ProgramExample) -> str | None:
    try:
        return str(execute_program(program, example.candidates))
    except (ArithmeticError, ValueError, IndexError):
        return None


def _counter_dict(counter: Counter[object]) -> dict[str, int]:
    return {str(key): value for key, value in counter.most_common()}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit development-only SELECT/ADD false accepts after S4.1"
    )
    parser.add_argument(
        "--fold-manifest",
        type=Path,
        default=ROOT / "data" / "document_hrm_v12b2_reranker_disjoint_folds_subtract_v4.json",
    )
    parser.add_argument(
        "--phase-b-root",
        type=Path,
        default=ROOT / "cache" / "v12b2_folds_subtract_v5",
    )
    parser.add_argument(
        "--phase-c-root",
        type=Path,
        default=ROOT / "cache" / "v12b2_phase_c_subtract_v5",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports" / "select_add_false_accepts_development_v1.json",
    )
    args = parser.parse_args()

    fold_manifest = _read_json(args.fold_manifest)
    phase_b_manifest = _read_json(args.phase_b_root / "manifest.json")
    phase_c_manifest = _read_json(args.phase_c_root / "manifest.json")
    _guard_development_only(fold_manifest, phase_b_manifest, phase_c_manifest)
    if int(fold_manifest.get("fold_count", 0)) != 5:
        raise ValueError("Expected exactly five development folds")

    input_path = _safe_local_input(fold_manifest)
    metadata = fold_manifest.get("examples", {})
    if not isinstance(metadata, dict):
        raise ValueError("Fold example metadata is missing")
    allowed_ids = set(str(value) for value in metadata)
    loaded = load_program_examples(input_path)
    examples = {
        example.example_id: example
        for example in loaded
        if example.example_id in allowed_ids
    }
    if set(examples) != allowed_ids:
        raise ValueError("Development example set is incomplete")

    folds = {
        int(row["fold"]): row
        for row in fold_manifest.get("folds", ())
        if isinstance(row, dict)
    }
    accepted = Counter()
    correct = Counter()
    false_accepts = Counter()
    proof_contract_states: dict[str, Counter[str]] = defaultdict(Counter)
    required_obligations: dict[str, Counter[str]] = defaultdict(Counter)
    structural_tags: dict[str, Counter[str]] = defaultdict(Counter)
    details: list[dict[str, object]] = []

    for fold in range(5):
        definition = folds[fold]
        candidate_path = args.phase_b_root / f"fold_{fold}" / "validation_candidates.jsonl"
        supervision_path = args.phase_c_root / f"fold_{fold}" / "calibration_supervision.jsonl"
        candidates = _read_jsonl_by_id(candidate_path)
        supervision = _read_jsonl_by_id(supervision_path)
        scoped_ids = {
            str(value)
            for value in definition["validation_example_ids"]
            if str(value) in examples
            and examples[str(value)].program.operation in SCOPED_OPERATIONS
        }
        for example_id in sorted(scoped_ids):
            example = examples[example_id]
            plan = compile_question_plan(example.question)
            graph = build_evidence_graph(example.candidates)
            issues = supervision_integrity_issues(example, graph=graph, plan=plan)
            stored = supervision.get(example_id)
            if stored is not None and tuple(stored.get("supervision_integrity_issues", ())) != issues:
                raise ValueError(f"Fold {fold} supervision integrity drift: {example_id}")
            if issues:
                continue
            record = candidates.get(example_id)
            if record is None:
                continue
            winner = _current_winner(example, record)
            if winner is None:
                continue
            operation = example.program.operation
            accepted[operation] += 1
            if _execution_correct(winner, example):
                correct[operation] += 1
                continue

            false_accepts[operation] += 1
            raw_winner = _raw_winner_candidate(record, winner)
            if raw_winner is None:
                raise ValueError(f"Winner not found in candidate record: {example_id}")
            identity_required = _identity_constraint_required(raw_winner)
            verification = verify_program(
                plan,
                winner,
                graph,
                financial_identity_catalog_enabled=identity_required,
            )
            required = _required_states(verification)
            contract_state = verification.proof_contract_state.value
            proof_contract_states[operation][contract_state] += 1
            for name, state in required.items():
                required_obligations[operation][f"{name}:{state}"] += 1

            gold_cells = _selected_cells(graph.cells, example.program)
            predicted_cells = _selected_cells(graph.cells, winner)
            tags = _structural_tags(
                plan,
                example.program,
                winner,
                gold_cells,
                predicted_cells,
                required,
            )
            structural_tags[operation].update(tags)
            details.append(
                {
                    "example_id": example_id,
                    "fold": fold,
                    "gold_operation": operation,
                    "question": example.question,
                    "question_plan": {
                        "operation": plan.operation,
                        "operation_candidates": list(plan.operation_candidates),
                        "operand_roles": list(plan.operand_roles),
                        "proof_operand_roles": list(plan.proof_operand_roles),
                        "metric_text": plan.metric_text,
                        "requested_years": list(plan.requested_years),
                        "requested_unit_family": plan.requested_unit_family,
                        "entity_mode": plan.entity_mode,
                    },
                    "gold_program": _program_key(example.program),
                    "predicted_program": _program_key(winner),
                    "gold_value": _decimal_value(example.program, example),
                    "predicted_value": _decimal_value(winner, example),
                    "proof_contract_state": contract_state,
                    "runtime_verification_state": verification.state.value,
                    "required_obligations": required,
                    "all_obligations": _states(verification),
                    "cached_proof_contract_state": raw_winner.get("proof_contract_state"),
                    "cached_required_valid_fraction": raw_winner.get("required_valid_fraction"),
                    "cached_required_unknown_fraction": raw_winner.get("required_unknown_fraction"),
                    "cached_runtime_eligible": raw_winner.get("runtime_eligible"),
                    "cached_rejection_reasons": raw_winner.get("rejection_reasons", []),
                    "generation_sources": raw_winner.get("generation_sources", []),
                    "structural_tags": tags,
                    "gold_cells": [_cell_summary(cell) for cell in gold_cells],
                    "predicted_cells": [_cell_summary(cell) for cell in predicted_cells],
                }
            )

    scoped_false_accepts = sum(false_accepts.values())
    report = {
        "schema": "select-add-false-accept-development-audit-v1",
        "classification": "development_only_company_report_disjoint",
        "scope": sorted(SCOPED_OPERATIONS),
        "global_validation_loaded": False,
        "sealed_validation_loaded": False,
        "blind_test_loaded": False,
        "fine_tuning_performed": False,
        "threshold_changed": False,
        "phase_d_release_gate": False,
        "inputs": {
            "fold_manifest": str(args.fold_manifest.relative_to(ROOT)),
            "development_examples": str(input_path.relative_to(ROOT)),
            "phase_b_root": str(args.phase_b_root.relative_to(ROOT)),
            "phase_c_root": str(args.phase_c_root.relative_to(ROOT)),
        },
        "artifact_hashes": {
            "fold_manifest": _digest(args.fold_manifest),
            "development_examples": _digest(input_path),
            "phase_b_manifest": _digest(args.phase_b_root / "manifest.json"),
            "phase_c_manifest": _digest(args.phase_c_root / "manifest.json"),
            "audit_script": _digest(Path(__file__)),
        },
        "counts": {
            "accepted": dict(accepted),
            "execution_correct_accepted": dict(correct),
            "false_accepts": dict(false_accepts),
            "false_accepts_total": scoped_false_accepts,
        },
        "proof_contract_state_on_false_accept": {
            operation: _counter_dict(counter)
            for operation, counter in sorted(proof_contract_states.items())
        },
        "required_obligation_state_on_false_accept": {
            operation: _counter_dict(counter)
            for operation, counter in sorted(required_obligations.items())
        },
        "structural_tag_counts": {
            operation: _counter_dict(counter)
            for operation, counter in sorted(structural_tags.items())
        },
        "details": details,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "counts": report["counts"],
                "proof_contract_state_on_false_accept": report[
                    "proof_contract_state_on_false_accept"
                ],
                "structural_tag_counts": report["structural_tag_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Saved development-only false-accept audit: {args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
