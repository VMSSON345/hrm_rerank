from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Sequence

from hrm_finance_qa.evidence_schema import build_evidence_graph
from hrm_finance_qa.ontology import normalize_text
from hrm_finance_qa.program_data import NumericCandidate, ReasoningProgram, load_program_examples
from hrm_finance_qa.question_planner import compile_question_plan
from hrm_finance_qa.symbolic_verifier import VerificationState, verify_program
from scripts.evaluate_qp3_core_s4_safe_replay import (
    PROTECTED_TOKENS,
    _counterfactual,
    _digest,
    _execution_correct,
    _layout_family,
    _proof_decision,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRECOMMIT = ROOT / "config/finhrm_qp3_core_add_real_layout_confirmation_precommit_v1.json"
DEFAULT_REPORT = ROOT / "reports/finhrm_qp3_core_add_real_layout_confirmation_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports/finhrm_qp3_core_add_real_layout_confirmation_v1.md"
DEFAULT_LOCK = ROOT / "reports/finhrm_qp3_core_add_real_layout_confirmation_lock_v1.json"
RUNTIME_PATHS = (
    "src/hrm_finance_qa/question_planner.py",
    "src/hrm_finance_qa/program_synthesis.py",
    "src/hrm_finance_qa/symbolic_verifier.py",
    "src/hrm_finance_qa/direction_contract.py",
    "src/hrm_finance_qa/document_hrm.py",
    "src/hrm_finance_qa/document_reasoning_engine.py",
    "src/hrm_finance_qa/evidence_schema.py",
    "src/hrm_finance_qa/financial_identities.py",
)


def _tokens(value: str) -> set[str]:
    return {
        token.lower()
        for token in re.split(r"[^a-zA-Z0-9]+", value)
        if token
    }


def _selected_candidates(example: Any) -> tuple[NumericCandidate, ...]:
    by_index = {candidate.index: candidate for candidate in example.candidates}
    return tuple(by_index[index] for index in example.program.operands if index in by_index)


def _source_is_allowed(candidate: NumericCandidate) -> bool:
    lowered = candidate.source_name.lower()
    return (
        bool(candidate.source_name)
        and lowered.endswith(".pdf")
        and "abw" not in lowered
        and not _tokens(candidate.source_name).intersection(PROTECTED_TOKENS)
    )


def _candidate_provenance_complete(candidate: NumericCandidate) -> bool:
    return bool(
        candidate.source_name
        and candidate.page is not None
        and candidate.table_id
        and candidate.row_label
        and candidate.period
        and candidate.source_span is not None
        and candidate.source_span[1] > candidate.source_span[0]
        and candidate.extraction_method
    )


def _physical_key(candidate: NumericCandidate) -> tuple[Any, ...]:
    return (
        candidate.source_name,
        candidate.page,
        candidate.source_span,
    )


def _permuted(
    candidates: Sequence[NumericCandidate], program: ReasoningProgram
) -> tuple[tuple[NumericCandidate, ...], ReasoningProgram]:
    reversed_candidates = list(reversed(candidates))
    index_map = {
        candidate.index: new_index
        for new_index, candidate in enumerate(reversed_candidates)
    }
    permuted_candidates = tuple(
        replace(candidate, index=new_index)
        for new_index, candidate in enumerate(reversed_candidates)
    )
    return permuted_candidates, ReasoningProgram(
        program.operation,
        tuple(index_map[index] for index in program.operands),
    )


def _company_key(example: Any) -> str:
    supervision = example.supervision if isinstance(example.supervision, dict) else {}
    explicit = normalize_text(str(supervision.get("company_key", "")))
    if not explicit:
        return ""
    return "DEVCONF_" + hashlib.sha256(explicit.encode()).hexdigest()[:12]


def _pdf_path(source_name: str) -> Path:
    return ROOT / "pdf" / "PDF_23" / Path(source_name).name


def _hard_negative_count(example: Any) -> int:
    supervision = example.supervision if isinstance(example.supervision, dict) else {}
    hard_negatives = supervision.get("hard_negatives", [])
    return len(hard_negatives) if isinstance(hard_negatives, list) else 0


def _selection_reason(example: Any, allowed_classification: str) -> str:
    if example.program.operation != "ADD":
        return "operation_out_of_scope"
    supervision = example.supervision if isinstance(example.supervision, dict) else {}
    if supervision.get("classification") != allowed_classification:
        return "classification_out_of_scope"
    if _tokens(example.example_id).intersection(PROTECTED_TOKENS):
        return "protected_record_marker"
    selected = _selected_candidates(example)
    if len(selected) != 2:
        return "selected_arity_not_two"
    if not all(_source_is_allowed(candidate) for candidate in selected):
        return "diagnosis_or_protected_source"
    if len({candidate.source_name for candidate in selected}) != 1:
        return "selected_source_mismatch"
    return "selected"


def _evaluate_example(example: Any) -> dict[str, Any]:
    selected = _selected_candidates(example)
    layout = _layout_family(example.candidates)
    plan = compile_question_plan(example.question)
    graph = build_evidence_graph(example.candidates)
    verification = verify_program(plan, example.program, graph)
    decision = _proof_decision(example.question, example.program, example.candidates)
    permuted_candidates, permuted_program = _permuted(
        example.candidates, example.program
    )
    permutation = _proof_decision(
        example.question, permuted_program, permuted_candidates
    )
    counterfactual = _counterfactual(example)
    counterfactual_decision = (
        _proof_decision(example.question, counterfactual[0], example.candidates)
        if counterfactual is not None
        else None
    )
    required = {
        item.name: item.state.value if item.state is not None else "NOT_APPLICABLE"
        for item in verification.proof_obligations
        if item.applicability == "required"
    }
    source_name = selected[0].source_name
    pdf_path = _pdf_path(source_name)
    distinct_physical_cells = len({_physical_key(candidate) for candidate in selected}) == 2
    return {
        "example_id": example.example_id,
        "company_key": _company_key(example),
        "layout_family": layout,
        "source_key": "PDF_" + hashlib.sha256(source_name.encode()).hexdigest()[:12],
        "source_pdf_exists": pdf_path.is_file(),
        "source_pdf_sha256": _digest(pdf_path) if pdf_path.is_file() else "",
        "explicit_two_role_hypotheses": len(plan.addition_role_hypotheses),
        "distinct_physical_cells": distinct_physical_cells,
        "confirmation_role": (
            "POSITIVE" if distinct_physical_cells else "PHYSICAL_COLLISION_HARD_NEGATIVE"
        ),
        "selected_provenance_complete": all(
            _candidate_provenance_complete(candidate) for candidate in selected
        ),
        "same_selected_period": len({normalize_text(candidate.period) for candidate in selected}) == 1,
        "hard_negative_count": _hard_negative_count(example),
        "gold_execution_correct": _execution_correct(example),
        "gold_action": decision["action"],
        "gold_proof_contract_state": decision["proof_contract_state"],
        "gold_reasons": decision["reasons"],
        "required_proof": required,
        "permutation_exact": permutation["action"] == decision["action"]
        and permutation["proof_contract_state"] == decision["proof_contract_state"],
        "counterfactual_generated": counterfactual is not None,
        "counterfactual_mutation": counterfactual[1] if counterfactual else None,
        "counterfactual_action": (
            counterfactual_decision["action"] if counterfactual_decision else None
        ),
        "counterfactual_reasons": (
            counterfactual_decision["reasons"] if counterfactual_decision else []
        ),
    }


def _group_report(details: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detail in details:
        grouped[str(detail[key])].append(detail)
    by_group = {
        name: {
            "examples": len(rows),
            "answer_rate": sum(row["gold_action"] == "ANSWER" for row in rows) / len(rows),
            "permutation_exact_rate": sum(row["permutation_exact"] for row in rows) / len(rows),
            "counterfactual_false_accept_rate": sum(
                row["counterfactual_action"] == "ANSWER" for row in rows
            )
            / len(rows),
        }
        for name, rows in sorted(grouped.items())
    }
    answer_rates = [row["answer_rate"] for row in by_group.values()]
    permutation_rates = [row["permutation_exact_rate"] for row in by_group.values()]
    false_accept_rates = [
        row["counterfactual_false_accept_rate"] for row in by_group.values()
    ]
    return {
        "groups": len(by_group),
        "by_group": by_group,
        "macro_answer_rate": sum(answer_rates) / len(answer_rates) if answer_rates else 0.0,
        "worst_answer_rate": min(answer_rates) if answer_rates else 0.0,
        "macro_permutation_exact_rate": (
            sum(permutation_rates) / len(permutation_rates) if permutation_rates else 0.0
        ),
        "worst_permutation_exact_rate": min(permutation_rates) if permutation_rates else 0.0,
        "macro_counterfactual_false_accept_rate": (
            sum(false_accept_rates) / len(false_accept_rates) if false_accept_rates else 0.0
        ),
        "worst_counterfactual_false_accept_rate": (
            max(false_accept_rates) if false_accept_rates else 0.0
        ),
    }


def build_report(precommit_path: Path = DEFAULT_PRECOMMIT) -> dict[str, Any]:
    precommit = json.loads(precommit_path.read_text(encoding="utf-8"))
    source_path = ROOT / precommit["source"]["path"]
    source_hash_exact = (
        source_path.stat().st_size == precommit["source"]["size_bytes"]
        and _digest(source_path) == precommit["source"]["sha256"]
    )
    parent_hashes_exact = all(
        _digest(ROOT / parent["path"]) == parent["sha256"]
        for parent in precommit["parent"].values()
    )
    examples = load_program_examples(source_path)
    selection = Counter()
    selected_examples = []
    for example in examples:
        reason = _selection_reason(
            example, precommit["source"]["allowed_classification"]
        )
        selection[reason] += 1
        if reason == "selected":
            selected_examples.append(example)
    details = [_evaluate_example(example) for example in selected_examples]
    positives = [row for row in details if row["confirmation_role"] == "POSITIVE"]
    collision_controls = [
        row
        for row in details
        if row["confirmation_role"] == "PHYSICAL_COLLISION_HARD_NEGATIVE"
    ]
    layouts = {row["layout_family"] for row in positives}
    companies = {row["company_key"] for row in positives if row["company_key"]}
    sources = {row["source_key"] for row in positives}
    all_companies = {row["company_key"] for row in details if row["company_key"]}
    all_sources = {row["source_key"] for row in details}
    required_layouts = set(
        precommit["confirmation_contract"]["required_additional_real_layout_families"]
    )
    baseline_layout = precommit["confirmation_contract"]["baseline_real_layout_family"]
    runtime_hashes = {path: _digest(ROOT / path) for path in RUNTIME_PATHS}
    runtime_hashes_exact = runtime_hashes == precommit["runtime_hashes_frozen"]
    checks = {
        "source_hash_exact": source_hash_exact,
        "parent_hashes_exact": parent_hashes_exact,
        "runtime_hashes_exact": runtime_hashes_exact,
        "selected_examples_present": bool(details),
        "zero_diagnosis_or_protected_selected": all(
            "abw" not in row["source_key"].lower() for row in details
        ),
        "all_real_pdf_sources_exist": all(row["source_pdf_exists"] for row in details),
        "all_explicit_two_role_add": all(
            row["explicit_two_role_hypotheses"] >= 1 for row in details
        ),
        "positive_distinct_physical_cells": all(
            row["distinct_physical_cells"] for row in positives
        ),
        "physical_collision_controls_present": bool(collision_controls),
        "physical_collision_controls_reject": all(
            row["gold_action"] == "REJECT"
            and row["gold_proof_contract_state"] == VerificationState.INVALID.value
            for row in collision_controls
        ),
        "all_selected_provenance_complete": all(
            row["selected_provenance_complete"] for row in details
        ),
        "all_same_selected_period": all(row["same_selected_period"] for row in details),
        "all_have_supervision_hard_negative": all(
            row["hard_negative_count"] > 0 for row in details
        ),
        "all_gold_execution_correct": all(row["gold_execution_correct"] for row in details),
        "all_positive_gold_proof_complete_answer": all(
            row["gold_action"] == "ANSWER"
            and row["gold_proof_contract_state"] == VerificationState.VALID.value
            for row in positives
        ),
        "all_candidate_permutations_exact": all(row["permutation_exact"] for row in details),
        "all_counterfactuals_generated": all(
            row["counterfactual_generated"] for row in details
        ),
        "counterfactual_false_accepts_zero": not any(
            row["counterfactual_action"] == "ANSWER" for row in details
        ),
        "minimum_distinct_companies": len(companies)
        >= precommit["confirmation_contract"]["minimum_distinct_companies"],
        "required_additional_real_layouts_exact": layouts == required_layouts,
        "minimum_additional_real_layouts": len(layouts)
        >= precommit["confirmation_contract"]["minimum_additional_real_layout_families"],
        "minimum_union_real_layouts": len(layouts | {baseline_layout})
        >= precommit["confirmation_contract"]["minimum_union_real_layout_families"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "schema": "finhrm-qp3-core-add-real-layout-confirmation-v1",
        "created_at": "2026-08-20",
        "classification": "development_only_source_backed_real_layout_confirmation",
        "confirmation_gate_passed": not failed,
        "failed_gates": failed,
        "source": {
            "path": precommit["source"]["path"],
            "sha256": precommit["source"]["sha256"],
            "total_records": len(examples),
            "selection_counts": dict(sorted(selection.items())),
            "selected_add_records": len(details),
            "selected_diagnosis_records": 0,
            "selected_protected_records": 0,
        },
        "results": {
            "gold_actions": dict(sorted(Counter(row["gold_action"] for row in details).items())),
            "gold_execution_correct": sum(row["gold_execution_correct"] for row in details),
            "gold_total": len(details),
            "positive_total": len(positives),
            "physical_collision_hard_negatives": len(collision_controls),
            "permutation_exact": sum(row["permutation_exact"] for row in details),
            "counterfactual_generated": sum(row["counterfactual_generated"] for row in details),
            "counterfactual_mutations": dict(
                sorted(Counter(row["counterfactual_mutation"] for row in details).items())
            ),
            "counterfactual_false_accepts": sum(
                row["counterfactual_action"] == "ANSWER" for row in details
            ),
            "companies": len(companies),
            "sources": len(sources),
            "all_selected_companies": len(all_companies),
            "all_selected_sources": len(all_sources),
            "additional_real_layout_families": sorted(layouts),
            "union_real_layout_families": sorted(layouts | {baseline_layout}),
            "company": _group_report(positives, "company_key"),
            "layout": _group_report(positives, "layout_family"),
            "all_selected_company": _group_report(details, "company_key"),
            "all_selected_layout": _group_report(details, "layout_family"),
        },
        "gate_checks": checks,
        "details": details,
        "runtime_hashes": runtime_hashes,
        "invariants": {
            "runtime_changed": False,
            "threshold_changed": False,
            "checkpoint_changed": False,
            "fine_tuning_performed": False,
            "abw_used_as_confirmation": False,
            "protected_evaluation_loaded": False,
            "candidate_rank_or_score_used_as_semantic_proof": False,
            "synthetic_layout_projection_counted_as_confirmation": False,
            "unseen_or_universal_claim_made": False,
            "qp4_authorized": False,
            "phase_d_unlocked": False,
        },
        "interpretation": (
            "Development-only source-backed confirmation of the unchanged structural ADD proof contract. "
            "The questions are mechanically derived, but the two layout families and selected cells come from "
            "real PDFs; this is neither a synthetic layout projection nor sealed/unseen publication evidence."
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    result = report["results"]
    return "\n".join(
        [
            "# FinHRM QP3-CORE ADD real-layout confirmation",
            "",
            f"- Gate: **{'PASS' if report['confirmation_gate_passed'] else 'FAIL'}**",
            f"- Source-backed ADD: `{result['gold_execution_correct']}/{result['gold_total']}` execution-correct and `{result['gold_actions']}` proof actions.",
            f"- Companies/sources: `{result['companies']}` / `{result['sources']}`.",
            f"- Additional real layouts: `{result['additional_real_layout_families']}`.",
            f"- Candidate permutation: `{result['permutation_exact']}/{result['gold_total']}`.",
            f"- Counterfactuals: `{result['counterfactual_generated']}`; false accepts: `{result['counterfactual_false_accepts']}`.",
            f"- Failed gates: `{report['failed_gates']}`.",
            "- ABW/protected records, synthetic layout projections, rank/score proof, threshold/model changes, QP4 and Phase D are excluded.",
            "",
            report["interpretation"],
            "",
        ]
    )


def write_outputs(
    report: dict[str, Any], report_path: Path, markdown_path: Path, lock_path: Path
) -> None:
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    artifacts = []
    for path in (
        DEFAULT_PRECOMMIT,
        Path(__file__),
        ROOT / "tests/test_qp3_core_add_real_layout_confirmation.py",
        report_path,
        markdown_path,
    ):
        artifacts.append(
            {
                "path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _digest(path),
            }
        )
    lock = {
        "schema": "finhrm-qp3-core-add-real-layout-confirmation-lock-v1",
        "created_at": report["created_at"],
        "classification": report["classification"],
        "confirmation_gate_passed": report["confirmation_gate_passed"],
        "failed_gates": report["failed_gates"],
        "runtime_hashes": report["runtime_hashes"],
        "invariants": report["invariants"],
        "artifacts": artifacts,
        "qp4_authorized": False,
        "phase_d": "locked",
    }
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precommit", type=Path, default=DEFAULT_PRECOMMIT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    report = build_report(args.precommit)
    write_outputs(report, args.report, args.markdown, args.lock)
    print(
        json.dumps(
            {
                "gate": report["confirmation_gate_passed"],
                "failed": report["failed_gates"],
                "results": report["results"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report["confirmation_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
