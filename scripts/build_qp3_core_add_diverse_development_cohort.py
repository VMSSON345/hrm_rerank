from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from hrm_finance_qa.ontology import normalize_text
from hrm_finance_qa.program_data import NumericCandidate, ReasoningProgram, load_program_examples, write_program_examples
from hrm_finance_qa.question_planner import compile_question_plan
from scripts.evaluate_qp3_core_s4_safe_replay import _counterfactual, _execution_correct, _proof_decision


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRECOMMIT = ROOT / "config/finhrm_qp3_core_add_diverse_development_cohort_precommit_v1.json"
DEFAULT_MANIFEST = ROOT / "data/finhrm_qp3_core_add_diversity_generation_manifest_v1.json"
DEFAULT_SOURCE_AUDIT = ROOT / "reports/finhrm_qp3_core_add_diverse_development_source_audit_v1.json"
DEFAULT_NATIVE_POOL = ROOT / "data/finhrm_qp3_core_add_diversity_generated_pool_v1.jsonl"
DEFAULT_OCR_POOL = ROOT / "data/finhrm_qp3_core_add_diversity_ocr_generated_pool_v2.jsonl"
DEFAULT_OUTPUT = ROOT / "data/finhrm_qp3_core_add_diverse_development_candidate_cohort_v2.jsonl"
DEFAULT_REPORT = ROOT / "reports/finhrm_qp3_core_add_diverse_development_candidate_cohort_v2.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selected(example: Any) -> tuple[NumericCandidate, ...]:
    by_index = {candidate.index: candidate for candidate in example.candidates}
    return tuple(by_index[index] for index in example.program.operands if index in by_index)


def _physical_key(candidate: NumericCandidate) -> tuple[Any, ...]:
    return candidate.source_name, candidate.page, candidate.source_span


def _provenance_complete(candidate: NumericCandidate) -> bool:
    return bool(
        candidate.source_name
        and candidate.page is not None
        and candidate.row_label
        and candidate.column_label
        and candidate.period
        and candidate.table_id
        and candidate.source_span
        and candidate.source_span[1] > candidate.source_span[0]
        and candidate.extraction_method
    )


def _permuted(
    candidates: Sequence[NumericCandidate], program: ReasoningProgram
) -> tuple[tuple[NumericCandidate, ...], ReasoningProgram]:
    reversed_candidates = list(reversed(candidates))
    mapping = {candidate.index: index for index, candidate in enumerate(reversed_candidates)}
    values = tuple(replace(candidate, index=index) for index, candidate in enumerate(reversed_candidates))
    return values, ReasoningProgram(program.operation, tuple(mapping[index] for index in program.operands))


def _explicit_wrong_period_counterfactual(
    example: Any,
) -> tuple[ReasoningProgram, str] | None:
    """Replace one operand with an observed same-row cell from another period."""

    selected = _selected(example)
    selected_indices = set(example.program.operands)
    options: list[tuple[tuple[Any, ...], ReasoningProgram]] = []
    for position, expected in enumerate(selected):
        expected_period = normalize_text(expected.period)
        if not expected_period:
            continue
        for alternative in example.candidates:
            alternative_period = normalize_text(alternative.period)
            if (
                alternative.index in selected_indices
                or not alternative_period
                or alternative_period == expected_period
                or normalize_text(alternative.row_label) != normalize_text(expected.row_label)
                or alternative.source_name != expected.source_name
                or alternative.page != expected.page
                or alternative.table_id != expected.table_id
            ):
                continue
            operands = list(example.program.operands)
            operands[position] = alternative.index
            if len(set(operands)) != len(operands):
                continue
            semantic_key = (
                alternative.source_name,
                alternative.page,
                alternative.table_id,
                normalize_text(alternative.row_label),
                alternative_period,
                alternative.index,
            )
            options.append(
                (
                    semantic_key,
                    ReasoningProgram(example.program.operation, tuple(operands)),
                )
            )
    if not options:
        return None
    _, program = min(options, key=lambda item: (item[0], item[1].operands))
    return program, "EXPLICIT_WRONG_PERIOD"


def _cohort_counterfactual(
    example: Any,
) -> tuple[str, ReasoningProgram, str] | None:
    """Build a deterministic semantic negative from explicitly observed cells only."""

    existing = _counterfactual(example)
    if existing is not None:
        return example.question, existing[0], existing[1]
    wrong_period = _explicit_wrong_period_counterfactual(example)
    if wrong_period is not None:
        return example.question, wrong_period[0], wrong_period[1]
    return (
        example.question,
        ReasoningProgram("SUBTRACT", example.program.operands),
        "WRONG_OPERATION",
    )


def _eligibility(example: Any, metadata: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    selected = _selected(example)
    plan = compile_question_plan(example.question)
    decision = _proof_decision(example.question, example.program, example.candidates)
    permuted_candidates, permuted_program = _permuted(example.candidates, example.program)
    permutation = _proof_decision(example.question, permuted_program, permuted_candidates)
    counterfactual = _cohort_counterfactual(example)
    counterfactual_decision = (
        _proof_decision(counterfactual[0], counterfactual[1], example.candidates)
        if counterfactual is not None
        else None
    )
    supervision = example.supervision if isinstance(example.supervision, dict) else {}
    checks = {
        "operation_add": example.program.operation == "ADD",
        "selected_arity_two": len(selected) == 2,
        "explicit_two_role_question": bool(plan.addition_role_hypotheses),
        "same_source": len({item.source_name for item in selected}) == 1,
        "same_period": len({normalize_text(item.period) for item in selected}) == 1,
        "provenance_complete": all(_provenance_complete(item) for item in selected),
        "distinct_physical_cells": len({_physical_key(item) for item in selected}) == 2,
        "hard_negative_present": bool(supervision.get("hard_negatives")),
        "gold_execution_correct": _execution_correct(example),
        "proof_complete_answer": decision["action"] == "ANSWER" and decision["proof_contract_state"] == "VALID",
        "permutation_exact": permutation["action"] == decision["action"]
        and permutation["proof_contract_state"] == decision["proof_contract_state"],
        "counterfactual_generated": counterfactual is not None,
        "counterfactual_false_accept_zero": counterfactual_decision is not None
        and counterfactual_decision["action"] != "ANSWER",
        "source_hash_exact": _digest(ROOT / metadata["source_path"]) == metadata["sha256"],
    }
    return all(checks.values()), {
        "checks": checks,
        "gold_action": decision["action"],
        "gold_proof_contract_state": decision["proof_contract_state"],
        "gold_reasons": decision["reasons"],
        "counterfactual_mutation": counterfactual[2] if counterfactual else None,
        "counterfactual_question": counterfactual[0] if counterfactual else None,
        "counterfactual_action": counterfactual_decision["action"] if counterfactual_decision else None,
    }


def build_cohort(
    precommit_path: Path = DEFAULT_PRECOMMIT,
    manifest_path: Path = DEFAULT_MANIFEST,
    source_audit_path: Path = DEFAULT_SOURCE_AUDIT,
    native_pool_path: Path = DEFAULT_NATIVE_POOL,
    ocr_pool_path: Path = DEFAULT_OCR_POOL,
) -> tuple[list[Any], dict[str, Any]]:
    precommit = json.loads(precommit_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    by_source = {
        Path(item["document"]).name: {
            **item,
            "source_path": Path(item["document"]).resolve().relative_to(ROOT.resolve()).as_posix(),
        }
        for item in manifest["documents"]
    }
    for item in source_audit["details"]:
        by_source.setdefault(
            item["source_name"],
            {
                "source_key": item["source_key"],
                "company_key": item["company_key"],
                "layout_family": item["layout_family"],
                "layout_stratum": next(
                    name
                    for name, families in precommit["deterministic_selection"]["layout_strata"].items()
                    if item["layout_family"] in families
                ),
                "sha256": item["source_sha256"],
                "source_path": item["source_path"],
            },
        )
    examples = [*load_program_examples(native_pool_path), *load_program_examples(ocr_pool_path)]
    eligible = []
    candidate_audit = []
    rejection_counts: Counter[str] = Counter()
    for example in examples:
        if example.program.operation != "ADD":
            continue
        selected = _selected(example)
        source_name = selected[0].source_name if selected else ""
        metadata = by_source.get(source_name)
        if metadata is None:
            rejection_counts["source_not_in_manifest"] += 1
            continue
        passed, audit = _eligibility(example, metadata)
        candidate_audit.append(
            {
                "example_id": example.example_id,
                "source_name": source_name,
                "company_key": metadata["company_key"],
                "layout_family": metadata["layout_family"],
                "layout_stratum": metadata["layout_stratum"],
                "page": selected[0].page if selected else None,
                "question": example.question,
                "failed_checks": [
                    name for name, value in audit["checks"].items() if not value
                ],
                "gold_action": audit["gold_action"],
                "gold_reasons": audit["gold_reasons"],
                "counterfactual_mutation": audit["counterfactual_mutation"],
                "counterfactual_action": audit["counterfactual_action"],
            }
        )
        if not passed:
            for name, value in audit["checks"].items():
                if not value:
                    rejection_counts[name] += 1
            continue
        physical = tuple(
            (item.page, item.table_id, item.row_label, item.column_label, item.source_span)
            for item in selected
        )
        eligible.append((metadata["layout_stratum"], metadata["sha256"], physical, example, audit, metadata))
    eligible.sort(key=lambda item: (item[0], item[1], item[2]))
    target = int(precommit["deterministic_selection"]["target_companies_per_layout"])
    selected_rows = []
    selected_companies: set[str] = set()
    counts: Counter[str] = Counter()
    for stratum, source_hash, physical, example, audit, metadata in eligible:
        company = str(metadata["company_key"])
        if counts[stratum] >= target or company in selected_companies:
            continue
        supervision = dict(example.supervision or {})
        supervision.update(
            {
                "company_key": company,
                "review_status": "pending_independent_review",
                "cohort_layout_stratum": stratum,
                "cohort_layout_family": metadata["layout_family"],
                "source_sha256": source_hash,
            }
        )
        selected_rows.append(replace(example, supervision=supervision))
        selected_companies.add(company)
        counts[stratum] += 1
    details = []
    for example in selected_rows:
        selected = _selected(example)
        metadata = by_source[selected[0].source_name]
        _, audit = _eligibility(example, metadata)
        details.append(
            {
                "example_id": example.example_id,
                "company_key": example.supervision["company_key"],
                "source_key": metadata["source_key"],
                "source_name": selected[0].source_name,
                "source_path": metadata["source_path"],
                "source_sha256": metadata["sha256"],
                "layout_stratum": metadata["layout_stratum"],
                "layout_family": metadata["layout_family"],
                "page": selected[0].page,
                "question": example.question,
                "answer_value": example.answer_value,
                "selected_operands": [
                    {
                        "value": str(item.value),
                        "raw": item.raw,
                        "row": item.row_label,
                        "column": item.column_label,
                        "period": item.period,
                        "table_id": item.table_id,
                        "source_span": list(item.source_span or ()),
                        "extraction_method": item.extraction_method,
                    }
                    for item in selected
                ],
                **audit,
            }
        )
    minimum = int(precommit["deterministic_selection"]["minimum_companies_per_layout"])
    strata = tuple(precommit["deterministic_selection"]["layout_strata"])
    checks = {
        "minimum_companies_per_layout": all(counts[name] >= minimum for name in strata),
        "target_companies_per_layout": all(counts[name] >= target for name in strata),
        "one_positive_per_company": len(selected_companies) == len(selected_rows),
        "company_disjoint_between_layouts": len(selected_companies)
        == sum(len({row["company_key"] for row in details if row["layout_stratum"] == name}) for name in strata),
        "all_selected_eligibility_checks_pass": all(
            all(detail["checks"].values()) for detail in details
        ),
        "counterfactual_false_accepts_zero": not any(
            detail["counterfactual_action"] == "ANSWER" for detail in details
        ),
    }
    report = {
        "schema": "finhrm-qp3-core-add-diverse-development-candidate-cohort-v2",
        "created_at": "2026-08-20",
        "classification": "development_only_hash_ordered_pending_independent_review",
        "candidate_cohort_gate_passed": all(checks.values()),
        "failed_gates": [name for name, value in checks.items() if not value],
        "inputs": {
            "precommit": {"path": precommit_path.relative_to(ROOT).as_posix(), "sha256": _digest(precommit_path)},
            "manifest": {"path": manifest_path.relative_to(ROOT).as_posix(), "sha256": _digest(manifest_path)},
            "source_audit": {"path": source_audit_path.relative_to(ROOT).as_posix(), "sha256": _digest(source_audit_path)},
            "native_pool": {"path": native_pool_path.relative_to(ROOT).as_posix(), "sha256": _digest(native_pool_path)},
            "ocr_pool": {"path": ocr_pool_path.relative_to(ROOT).as_posix(), "sha256": _digest(ocr_pool_path)},
        },
        "results": {
            "eligible_add_examples": len(eligible),
            "selected_examples": len(selected_rows),
            "selected_companies": len(selected_companies),
            "selected_by_layout": dict(sorted(counts.items())),
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "independently_reviewed": 0,
        },
        "gate_checks": checks,
        "details": details,
        "candidate_audit": candidate_audit,
        "invariants": {
            "protected_evaluation_loaded": False,
            "abw_loaded": False,
            "runtime_changed": False,
            "model_checkpoint_or_threshold_changed": False,
            "selection_used_company_filename_page_year_value_example_id_or_question_literal_priority": False,
        },
    }
    return selected_rows, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite candidate cohort artifacts")
    examples, report = build_cohort()
    write_program_examples(examples, args.output)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gate": report["candidate_cohort_gate_passed"], **report["results"]}, ensure_ascii=False, sort_keys=True))
    return 0 if report["candidate_cohort_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
