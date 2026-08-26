from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any, Sequence

from hrm_finance_qa.document_hrm import (
    _grounding_gate_reasons,
    _prediction_required_proof_rejection_reasons,
)
from hrm_finance_qa.evidence_schema import EvidenceCell, build_evidence_graph
from hrm_finance_qa.financial_identities import metric_alias_score
from hrm_finance_qa.ontology import normalize_text
from hrm_finance_qa.program_data import (
    NumericCandidate,
    ProgramDataError,
    ProgramExample,
    ReasoningProgram,
    execute_program,
    load_program_examples,
)
from hrm_finance_qa.question_planner import QuestionPlan, compile_question_plan
from hrm_finance_qa.symbolic_verifier import VerificationState, verify_program


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRECOMMIT = ROOT / "config" / "finhrm_qp3_core_s4_safe_replay_precommit_v1.json"
DEFAULT_REPORT = ROOT / "reports" / "finhrm_qp3_core_s4_safe_replay_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "finhrm_qp3_core_s4_safe_replay_v1.md"
DEFAULT_LOCK = ROOT / "reports" / "finhrm_qp3_core_s4_safe_replay_lock_v1.json"
PROTECTED_TOKENS = {"validation", "sealed", "blind", "test", "holdout", "final"}
SCOPED_OPERATIONS = {"SELECT", "ADD"}
RUNTIME_PATHS = (
    "src/hrm_finance_qa/question_planner.py",
    "src/hrm_finance_qa/program_synthesis.py",
    "src/hrm_finance_qa/symbolic_verifier.py",
    "src/hrm_finance_qa/direction_contract.py",
    "src/hrm_finance_qa/document_hrm.py",
    "src/hrm_finance_qa/document_reasoning_engine.py",
)
MUTATION_PRIORITY = {
    "WRONG_YEAR": 0,
    "WRONG_COLUMN_ROLE": 1,
    "WRONG_ENTITY_SCOPE": 2,
    "WRONG_METRIC": 3,
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refuse_protected_path(path: Path) -> None:
    tokens = {
        token.lower()
        for part in path.parts
        for token in re.split(r"[^a-zA-Z0-9]+", part)
        if token
    }
    overlap = sorted(tokens.intersection(PROTECTED_TOKENS))
    if overlap:
        raise ValueError(f"protected path refused: {path} ({overlap})")


def _protected_record_reason(example: ProgramExample) -> str:
    identifiers = [example.example_id]
    if isinstance(example.supervision, dict):
        identifiers.extend(
            str(example.supervision.get(key, ""))
            for key in ("split", "partition", "subset", "source_split")
        )
    normalized = " ".join(identifiers).lower()
    return next((token for token in PROTECTED_TOKENS if token in normalized), "")


def _company_key(example: ProgramExample) -> str:
    if isinstance(example.supervision, dict):
        explicit = str(example.supervision.get("company_key", "")).strip()
        if explicit:
            normalized = normalize_text(explicit)
            return "DEVCO_" + hashlib.sha256(normalized.encode()).hexdigest()[:12]
    match = re.search(
        r"(?im)^\s*(?:c[oô]ng ty|company)\s*:\s*(.+?)\s*$",
        example.context,
    )
    value = match.group(1) if match else "unknown-development-company"
    normalized = normalize_text(value)
    return "DEVCO_" + hashlib.sha256(normalized.encode()).hexdigest()[:12]


def _layout_family(candidates: Sequence[NumericCandidate]) -> str:
    structured = [candidate for candidate in candidates if candidate.table_id]
    if not structured:
        return "DEVELOPMENT_PLAINTEXT_SPARSE"
    tables = {candidate.table_id for candidate in structured}
    methods = {
        normalize_text(candidate.extraction_method)
        for candidate in structured
        if candidate.extraction_method
    }
    if any(method in {"ocr", "mineru", "hybrid"} for method in methods):
        return "DEVELOPMENT_OCR_OR_HYBRID_TABLE"
    periods = {normalize_text(candidate.period) for candidate in structured}
    explicit_year = any(re.fullmatch(r"(?:19|20)\d{2}", period) for period in periods)
    relative = any(
        period in {"dau nam", "cuoi nam", "dau ky", "cuoi ky", "prior", "current"}
        for period in periods
    )
    if len(tables) > 1:
        return "DEVELOPMENT_NATIVE_MULTI_TABLE"
    if explicit_year and relative:
        return "DEVELOPMENT_NATIVE_MIXED_PERIOD_TABLE"
    if explicit_year:
        return "DEVELOPMENT_NATIVE_EXPLICIT_YEAR_TABLE"
    if relative:
        return "DEVELOPMENT_NATIVE_RELATIVE_PERIOD_TABLE"
    return "DEVELOPMENT_NATIVE_STRUCTURED_SPARSE"


def _proof_decision(
    question: str,
    program: ReasoningProgram,
    candidates: Sequence[NumericCandidate],
) -> dict[str, Any]:
    plan = compile_question_plan(question)
    graph = build_evidence_graph(candidates)
    verification = verify_program(plan, program, graph)
    prediction = SimpleNamespace(
        program=program,
        program_dag=None,
        slot_bindings=(),
        grounding_constraints_applied=(),
    )
    reasons = tuple(
        dict.fromkeys(
            (
                *_grounding_gate_reasons(
                    question,
                    program,
                    candidates,
                    verification,
                ),
                *_prediction_required_proof_rejection_reasons(
                    question,
                    prediction,
                    candidates,
                ),
            )
        )
    )
    required = [
        obligation
        for obligation in verification.proof_obligations
        if obligation.applicability == "required"
    ]
    invalid = any(item.state is VerificationState.INVALID for item in required)
    unknown = any(item.state is VerificationState.UNKNOWN for item in required)
    if not reasons and not invalid and not unknown:
        action = "ANSWER"
    elif invalid or any("invalid" in reason or "mismatch" in reason for reason in reasons):
        action = "REJECT"
    else:
        action = "ABSTAIN"
    return {
        "action": action,
        "proof_contract_state": verification.proof_contract_state.value,
        "reasons": list(reasons),
        "required": {
            item.name: item.state.value if item.state is not None else "NOT_APPLICABLE"
            for item in required
        },
    }


def _execution_correct(example: ProgramExample) -> bool:
    try:
        observed = execute_program(example.program, example.candidates)
        expected = Decimal(str(example.answer_value))
    except (InvalidOperation, ProgramDataError, ValueError, ZeroDivisionError):
        return False
    tolerance = max(Decimal("0.01"), abs(expected) * Decimal("0.000001"))
    return abs(observed - expected) <= tolerance


def _permuted(
    candidates: Sequence[NumericCandidate],
    program: ReasoningProgram,
) -> tuple[tuple[NumericCandidate, ...], ReasoningProgram]:
    old = list(reversed(candidates))
    index_map = {candidate.index: new_index for new_index, candidate in enumerate(old)}
    permuted = tuple(
        replace(candidate, index=new_index)
        for new_index, candidate in enumerate(old)
    )
    return permuted, ReasoningProgram(
        program.operation,
        tuple(index_map[index] for index in program.operands),
    )


def _mutation_type(
    plan: QuestionPlan,
    expected: EvidenceCell,
    alternative: EvidenceCell,
    *,
    operand_position: int,
) -> str:
    requested_years = set(plan.requested_years)
    alternative_years = alternative.effective_years(plan)
    if requested_years and alternative_years and requested_years.isdisjoint(alternative_years):
        return "WRONG_YEAR"
    if operand_position < len(plan.column_roles):
        expected_role = plan.column_roles[operand_position].lower()
        if alternative.column_role and alternative.column_role != expected_role:
            return "WRONG_COLUMN_ROLE"
    if not expected.embedded_entity and alternative.embedded_entity:
        return "WRONG_ENTITY_SCOPE"
    roles = plan.proof_operand_roles or plan.operand_roles
    expected_metric = (
        plan.metric_text
        if plan.operation == "SELECT"
        else roles[operand_position]
        if operand_position < len(roles)
        else ""
    )
    if expected_metric and expected.metric_label and alternative.metric_label:
        gold_score = metric_alias_score(expected.metric_label, (expected_metric,))
        alternative_score = metric_alias_score(
            alternative.metric_label, (expected_metric,)
        )
        # A different normalized row label is not automatically a negative:
        # aliases and equivalent disclosures are common across layouts. Only
        # a strong gold match paired with an explicit semantic contradiction
        # is admitted as WRONG_METRIC.
        if gold_score >= 0.72 and alternative_score <= 0.10:
            return "WRONG_METRIC"
    return ""


def _counterfactual(
    example: ProgramExample,
) -> tuple[ReasoningProgram, str] | None:
    plan = compile_question_plan(example.question)
    graph = build_evidence_graph(example.candidates)
    by_index = {cell.index: cell for cell in graph.cells}
    selected = set(example.program.operands)
    options: list[tuple[int, tuple[str, ...], ReasoningProgram, str]] = []
    for position, expected_index in enumerate(example.program.operands):
        expected = by_index.get(expected_index)
        if expected is None:
            continue
        for alternative in graph.cells:
            if alternative.index in selected:
                continue
            mutation = _mutation_type(
                plan,
                expected,
                alternative,
                operand_position=position,
            )
            if not mutation:
                continue
            operands = list(example.program.operands)
            operands[position] = alternative.index
            if len(set(operands)) != len(operands):
                continue
            semantic_key = (
                alternative.metric_key,
                alternative.period,
                alternative.column_role,
                alternative.entity_id,
                alternative.source_name,
                str(alternative.page),
            )
            options.append(
                (
                    MUTATION_PRIORITY[mutation],
                    semantic_key,
                    ReasoningProgram(example.program.operation, tuple(operands)),
                    mutation,
                )
            )
    if not options:
        return None
    _, _, program, mutation = min(options, key=lambda item: (item[0], item[1], item[2].operands))
    return program, mutation


def _group_report(details: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detail in details:
        grouped[str(detail[key])].append(detail)
    rows: dict[str, dict[str, Any]] = {}
    for name, values in sorted(grouped.items()):
        counterfactuals = [item for item in values if item["counterfactual_generated"]]
        rows[name] = {
            "examples": len(values),
            "gold_answer_rate": sum(item["gold_action"] == "ANSWER" for item in values) / len(values),
            "permutation_exact_rate": sum(item["permutation_exact"] for item in values) / len(values),
            "counterfactual_false_accept_rate": (
                sum(item["counterfactual_action"] == "ANSWER" for item in counterfactuals)
                / len(counterfactuals)
                if counterfactuals
                else None
            ),
        }
    answer_rates = [value["gold_answer_rate"] for value in rows.values()]
    permutation_rates = [value["permutation_exact_rate"] for value in rows.values()]
    false_accept_rates = [
        value["counterfactual_false_accept_rate"]
        for value in rows.values()
        if value["counterfactual_false_accept_rate"] is not None
    ]
    return {
        "groups": len(rows),
        "by_group": rows,
        "macro_gold_answer_rate": sum(answer_rates) / len(answer_rates) if answer_rates else 0.0,
        "worst_gold_answer_rate": min(answer_rates) if answer_rates else 0.0,
        "macro_permutation_exact_rate": sum(permutation_rates) / len(permutation_rates) if permutation_rates else 0.0,
        "worst_permutation_exact_rate": min(permutation_rates) if permutation_rates else 0.0,
        "macro_counterfactual_false_accept_rate": (
            sum(false_accept_rates) / len(false_accept_rates) if false_accept_rates else None
        ),
        "worst_counterfactual_false_accept_rate": max(false_accept_rates) if false_accept_rates else None,
    }


def build_report(precommit_path: Path = DEFAULT_PRECOMMIT) -> dict[str, Any]:
    _refuse_protected_path(precommit_path)
    precommit = json.loads(precommit_path.read_text(encoding="utf-8"))
    source_path = ROOT / precommit["source"]["path"]
    _refuse_protected_path(source_path)
    errors: list[str] = []
    source_bytes = source_path.read_bytes()
    source_text_lower = source_bytes.lower()
    if len(source_bytes) != int(precommit["source"]["size_bytes"]):
        errors.append("source_size_mismatch")
    if hashlib.sha256(source_bytes).hexdigest() != precommit["source"]["sha256"]:
        errors.append("source_hash_mismatch")
    if b"abw" in source_text_lower:
        errors.append("abw_record_present")
    parent = precommit["parent_implementation_lock"]
    parent_path = ROOT / parent["path"]
    if _digest(parent_path) != parent["sha256"]:
        errors.append("parent_implementation_lock_mismatch")

    examples = load_program_examples(source_path)
    protected_records = [
        (example.example_id, reason)
        for example in examples
        if (reason := _protected_record_reason(example))
    ]
    if protected_records:
        errors.append("protected_record_present")
    eligible = [
        example
        for example in examples
        if example.program.operation in SCOPED_OPERATIONS
        and not _protected_record_reason(example)
    ]
    details: list[dict[str, Any]] = []
    for example in eligible:
        company = _company_key(example)
        layout = _layout_family(example.candidates)
        gold = _proof_decision(example.question, example.program, example.candidates)
        permuted_candidates, permuted_program = _permuted(example.candidates, example.program)
        permutation = _proof_decision(
            example.question,
            permuted_program,
            permuted_candidates,
        )
        counterfactual = _counterfactual(example)
        counterfactual_decision = (
            _proof_decision(example.question, counterfactual[0], example.candidates)
            if counterfactual is not None
            else None
        )
        details.append(
            {
                "example_id": example.example_id,
                "operation": example.program.operation,
                "company_key": company,
                "layout_family": layout,
                "gold_action": gold["action"],
                "gold_proof_contract_state": gold["proof_contract_state"],
                "gold_execution_correct": _execution_correct(example),
                "gold_reasons": gold["reasons"],
                "permuted_action": permutation["action"],
                "permutation_exact": permutation["action"] == gold["action"]
                and permutation["proof_contract_state"] == gold["proof_contract_state"],
                "counterfactual_generated": counterfactual is not None,
                "counterfactual_mutation": counterfactual[1] if counterfactual else None,
                "counterfactual_action": counterfactual_decision["action"] if counterfactual_decision else None,
                "counterfactual_reasons": counterfactual_decision["reasons"] if counterfactual_decision else [],
            }
        )

    company = _group_report(details, "company_key")
    layout = _group_report(details, "layout_family")
    counterfactuals = [item for item in details if item["counterfactual_generated"]]
    false_accepts = sum(item["counterfactual_action"] == "ANSWER" for item in counterfactuals)
    permutation_exact = sum(item["permutation_exact"] for item in details)
    gold_answered = sum(item["gold_action"] == "ANSWER" for item in details)
    gold_execution_correct = sum(item["gold_execution_correct"] for item in details)
    actions = Counter(item["gold_action"] for item in details)
    mutations = Counter(item["counterfactual_mutation"] for item in counterfactuals)

    minimum_company = int(precommit["scope"]["minimum_company_groups"])
    minimum_layout = int(precommit["scope"]["minimum_layout_groups"])
    if not eligible:
        errors.append("no_select_add_examples")
    if company["groups"] < minimum_company:
        errors.append("insufficient_company_groups")
    if layout["groups"] < minimum_layout:
        errors.append("insufficient_layout_groups")
    if permutation_exact != len(details):
        errors.append("candidate_permutation_regression")
    if not counterfactuals:
        errors.append("no_explicit_counterfactuals")
    if false_accepts:
        errors.append("counterfactual_false_accept")

    checkpoint_hash = _digest(ROOT / "models" / "document_hrm_V8_candidate.pt")
    runtime_hashes = {path: _digest(ROOT / path) for path in RUNTIME_PATHS}
    return {
        "schema": "finhrm-qp3-core-s4-safe-replay-v1",
        "created_at": "2026-08-14",
        "classification": "development_train_only_report_agnostic_s4_stress_replay",
        "safe_replay_gate_passed": not errors,
        "errors": errors,
        "source": {
            "path": precommit["source"]["path"],
            "sha256": precommit["source"]["sha256"],
            "total_examples": len(examples),
            "eligible_select_add_examples": len(eligible),
            "protected_records": protected_records,
            "abw_records": 0,
        },
        "results": {
            "operations": dict(sorted(Counter(item["operation"] for item in details).items())),
            "gold_actions": dict(sorted(actions.items())),
            "gold_answered": gold_answered,
            "gold_total": len(details),
            "gold_execution_correct": gold_execution_correct,
            "permutation_exact": permutation_exact,
            "permutation_total": len(details),
            "counterfactual_generated": len(counterfactuals),
            "counterfactual_mutations": dict(sorted(mutations.items())),
            "counterfactual_false_accepts": false_accepts,
            "company": company,
            "layout": layout,
        },
        "details": details,
        "runtime_hashes": runtime_hashes,
        "checkpoint_sha256": checkpoint_hash,
        "invariants": {
            "abw_used": False,
            "protected_evaluation_loaded": False,
            "validation_loaded": False,
            "blind_loaded": False,
            "test_data_loaded": False,
            "candidate_rank_or_score_used_as_semantic_proof": False,
            "threshold_changed": False,
            "checkpoint_changed": checkpoint_hash
            != "3a33276ded0d2ff2dd3d200c4c774a59eec0f8cad7102bc273cb263d983944de",
            "fine_tuning_performed": False,
            "runtime_changed_by_replay": False,
            "phase_d_unlocked": False,
            "safe_abstention_counted_as_answered": False,
            "unseen_confirmation_claim_made": False,
            "universal_coverage_claim_made": False,
        },
        "interpretation": (
            "Broad development/train regression stress only. Company/layout macro and worst-group "
            "results are reported, but this source cannot establish unseen or sealed generalization."
        ),
        "authorized_next_step": (
            "If the gate passes, close the validation-free S4 stress replay only; keep empirical unseen "
            "generalization and any release claim locked."
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    result = report["results"]
    return "\n".join(
        [
            "# FinHRM QP3-CORE S4 safe replay",
            "",
            f"- Gate: **{'PASS' if report['safe_replay_gate_passed'] else 'FAIL'}**",
            f"- Eligible SELECT/ADD: `{result['gold_total']}` (`{result['operations']}`)",
            f"- Gold actions: `{result['gold_actions']}`; execution-correct gold: `{result['gold_execution_correct']}/{result['gold_total']}`",
            f"- Permutation exact: `{result['permutation_exact']}/{result['permutation_total']}`",
            f"- Explicit counterfactuals: `{result['counterfactual_generated']}`; false accepts: `{result['counterfactual_false_accepts']}`",
            f"- Company groups: `{result['company']['groups']}`; macro/worst gold answer rate: `{result['company']['macro_gold_answer_rate']:.3f}/{result['company']['worst_gold_answer_rate']:.3f}`",
            f"- Layout groups: `{result['layout']['groups']}`; macro/worst gold answer rate: `{result['layout']['macro_gold_answer_rate']:.3f}/{result['layout']['worst_gold_answer_rate']:.3f}`",
            "- No ABW, validation/blind/test data, threshold/checkpoint change, fine-tuning, or Phase-D unlock.",
            "- This is development/train stress evidence, not unseen or universal report coverage.",
            "",
        ]
    )


def write_outputs(
    report: dict[str, Any],
    *,
    report_path: Path = DEFAULT_REPORT,
    markdown_path: Path = DEFAULT_MARKDOWN,
    lock_path: Path = DEFAULT_LOCK,
) -> None:
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    artifacts = []
    for path in (
        DEFAULT_PRECOMMIT,
        Path(__file__),
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
        "schema": "finhrm-qp3-core-s4-safe-replay-lock-v1",
        "created_at": "2026-08-14",
        "classification": "development_train_only_report_agnostic_s4_stress_lock",
        "status": "complete" if report["safe_replay_gate_passed"] else "failed",
        "safe_replay_gate_passed": report["safe_replay_gate_passed"],
        "results": report["results"],
        "runtime_hashes": report["runtime_hashes"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "invariants": report["invariants"],
        "artifacts": artifacts,
        "qp4_authorized": False,
        "authorized_next_step": report["authorized_next_step"],
    }
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validation-free report-agnostic S4 stress replay")
    parser.add_argument("--precommit", type=Path, default=DEFAULT_PRECOMMIT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    report = build_report(args.precommit)
    write_outputs(
        report,
        report_path=args.report,
        markdown_path=args.markdown,
        lock_path=args.lock,
    )
    print(
        json.dumps(
            {
                "gate": report["safe_replay_gate_passed"],
                "errors": report["errors"],
                **{
                    key: report["results"][key]
                    for key in (
                        "operations",
                        "gold_actions",
                        "gold_answered",
                        "gold_total",
                        "permutation_exact",
                        "permutation_total",
                        "counterfactual_generated",
                        "counterfactual_false_accepts",
                    )
                },
                "company_groups": report["results"]["company"]["groups"],
                "layout_groups": report["results"]["layout"]["groups"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report["safe_replay_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
