from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.evaluate_direction_contract_folds import (  # noqa: E402
    _digest,
    _execution_correct,
    _feature,
    _identity_constraint_required,
    _read_json,
    _read_jsonl_by_id,
)
from hrm_finance_qa.document_hrm import (  # noqa: E402
    _compile_symbolic_query_spec,
    _program_search_rejection_reasons,
    _symbolic_program_search_key,
)
from hrm_finance_qa.evidence_schema import EvidenceCell, EvidenceGraph, build_evidence_graph  # noqa: E402
from hrm_finance_qa.ontology import normalize_text  # noqa: E402
from hrm_finance_qa.program_data import (  # noqa: E402
    ProgramExample,
    ReasoningProgram,
    canonical_program_equal,
    load_program_examples,
)
from hrm_finance_qa.question_planner import QuestionPlan, compile_question_plan  # noqa: E402
from hrm_finance_qa.symbolic_verifier import VerificationState, verify_program  # noqa: E402
from hrm_finance_qa.v12b2_supervision import parse_program_key, supervision_integrity_issues  # noqa: E402


SCOPED_OPERATIONS = {"SELECT", "ADD"}
_DIRECTIONAL_OPERATIONS = {"SUBTRACT", "PERCENT_CHANGE"}
_STALE_DIRECTION_REASONS = {"temporal_direction", "graph_temporal_order"}
PROTECTED_PATH_TOKENS = {"blind", "test", "holdout", "validation"}


def _guard_path(path: Path) -> None:
    lowered = {part.lower() for part in path.parts}
    if lowered.intersection(PROTECTED_PATH_TOKENS):
        raise ValueError(f"Refusing protected path: {path}")


def _guard_manifests(
    fold_manifest: dict[str, object],
    phase_b_manifest: dict[str, object],
    phase_c_manifest: dict[str, object],
) -> None:
    if fold_manifest.get("blind_test_used") is not False:
        raise ValueError("Fold manifest is not development-only")
    for manifest in (phase_b_manifest, phase_c_manifest):
        for key in ("blind_test_used", "test_jsonl_used", "sealed_validation_loaded"):
            if key in manifest and manifest.get(key) is not False:
                raise ValueError(f"Refusing manifest with {key} != false")
    if phase_c_manifest.get("reranker_training_started") is not False:
        raise ValueError("QP0 counterfactual refuses trained Phase-D reranker")


def _input_path(fold_manifest: dict[str, object]) -> Path:
    name = Path(str(fold_manifest.get("input", "")).replace("\\", "/")).name
    if not name or any(token in name.lower() for token in PROTECTED_PATH_TOKENS):
        raise ValueError(f"Unsafe development input: {name}")
    path = ROOT / "data" / name
    _guard_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _states(verification: object) -> dict[str, VerificationState]:
    return {
        item.name: item.state
        for item in verification.proof_obligations
        if item.state is not None
    }


def _selected(graph: EvidenceGraph, program: ReasoningProgram) -> tuple[EvidenceCell, ...]:
    return tuple(graph.cells[index] for index in program.operands)


def _required_role_from_question(plan: QuestionPlan) -> str:
    normalized = f" {plan.normalized} "
    actual = any(phrase in normalized for phrase in (" thuc hien ", " thuc te ", " ket qua dat duoc "))
    planned = any(phrase in normalized for phrase in (" ke hoach ", " muc tieu "))
    if actual == planned:
        return ""
    return "ACTUAL" if actual else "PLAN"


def _evidence_column_role(cell: EvidenceCell) -> str:
    normalized = f" {normalize_text(f'{cell.column} {cell.period}')} "
    actual = any(phrase in normalized for phrase in (" thuc hien ", " thuc te "))
    planned = any(phrase in normalized for phrase in (" ke hoach ", " muc tieu "))
    if actual == planned:
        return ""
    return "ACTUAL" if actual else "PLAN"


def _relative_role_from_question(plan: QuestionPlan) -> str:
    normalized = f" {plan.normalized} "
    prior = any(phrase in normalized for phrase in (" dau nam ", " dau ky ", " nam truoc ", " 01 01 "))
    current = any(phrase in normalized for phrase in (" cuoi nam ", " cuoi ky ", " nam nay ", " 31 12 "))
    if prior == current:
        return ""
    return "prior" if prior else "current"


def _physical_signature(cell: EvidenceCell) -> tuple[object, ...]:
    return (
        cell.metric_key,
        normalize_text(cell.period),
        cell.period_role,
        tuple(cell.explicit_years),
        normalize_text(cell.entity_id or cell.entity),
        normalize_text(cell.source_name),
        cell.page,
        normalize_text(cell.table_id),
        normalize_text(cell.subtable_id),
        normalize_text(cell.row_id),
        normalize_text(cell.column_id),
        normalize_text(cell.unit_key),
    )


def _logical_conflict(graph: EvidenceGraph, selected: tuple[EvidenceCell, ...]) -> bool:
    groups: dict[tuple[object, ...], set[object]] = defaultdict(set)
    for cell in graph.cells:
        groups[_physical_signature(cell)].add(cell.value)
    return any(len(groups[_physical_signature(cell)]) > 1 for cell in selected)


def _unit_complete(plan: QuestionPlan, selected: tuple[EvidenceCell, ...]) -> bool:
    requested = plan.requested_unit_family
    if not requested:
        return True
    if not selected:
        return False
    return all(cell.unit_key == requested for cell in selected)


def _policy_reasons(
    policy: str,
    plan: QuestionPlan,
    program: ReasoningProgram,
    graph: EvidenceGraph,
    verification: object,
) -> tuple[str, ...]:
    if program.operation not in SCOPED_OPERATIONS:
        return ()
    selected = _selected(graph, program)
    states = _states(verification)
    reasons: list[str] = []

    def required_proof_complete() -> None:
        if verification.proof_contract_state is not VerificationState.VALID:
            reasons.append("required_proof_not_complete")

    def add_strict() -> None:
        if program.operation != "ADD":
            return
        for name in ("operation_match", "operand_roles", "add_grounding"):
            if states.get(name) is not VerificationState.VALID:
                reasons.append(f"{name}_not_valid")

    def select_year() -> None:
        if program.operation == "SELECT" and plan.requested_years and states.get("period_coverage") is not VerificationState.VALID:
            reasons.append("requested_year_not_valid")

    def select_relative() -> None:
        required = _relative_role_from_question(plan)
        if program.operation == "SELECT" and required:
            if not selected or any(cell.period_role != required for cell in selected):
                reasons.append("relative_period_role_not_valid")

    def select_unit() -> None:
        if program.operation == "SELECT" and plan.requested_unit_family and not _unit_complete(plan, selected):
            reasons.append("requested_unit_not_valid")

    def select_entity() -> None:
        if program.operation != "SELECT":
            return
        name = "named_entity" if plan.entity_mode == "named" else "reporting_entity"
        if states.get(name) is not VerificationState.VALID:
            reasons.append(f"{name}_not_valid")

    def select_column_role() -> None:
        required = _required_role_from_question(plan)
        if program.operation == "SELECT" and required:
            if not selected or any(_evidence_column_role(cell) != required for cell in selected):
                reasons.append("column_role_not_valid")

    def uniqueness() -> None:
        if _logical_conflict(graph, selected):
            reasons.append("logical_cell_conflict")

    if policy == "required_proof_complete_all":
        required_proof_complete()
    elif policy == "add_strict_required_proof":
        add_strict()
    elif policy == "select_requested_year_complete":
        select_year()
    elif policy == "select_relative_period_complete":
        select_relative()
    elif policy == "select_requested_unit_complete":
        select_unit()
    elif policy == "select_entity_scope_complete":
        select_entity()
    elif policy == "select_column_role_complete":
        select_column_role()
    elif policy == "logical_cell_unique":
        uniqueness()
    elif policy == "select_qualifier_bundle":
        select_year()
        select_relative()
        select_unit()
        select_entity()
        select_column_role()
    elif policy == "combined_select_add_structural":
        add_strict()
        select_year()
        select_relative()
        select_unit()
        select_entity()
        select_column_role()
        uniqueness()
    elif policy != "baseline":
        raise ValueError(f"Unknown policy: {policy}")
    return tuple(dict.fromkeys(reasons))


POLICIES = (
    "baseline",
    "required_proof_complete_all",
    "add_strict_required_proof",
    "select_requested_year_complete",
    "select_relative_period_complete",
    "select_requested_unit_complete",
    "select_entity_scope_complete",
    "select_column_role_complete",
    "logical_cell_unique",
    "select_qualifier_bundle",
    "combined_select_add_structural",
)


def _candidate_pool(
    example: ProgramExample,
    record: dict[str, object] | None,
) -> list[tuple[tuple[float | int, ...], ReasoningProgram, object, tuple[str, ...]]]:
    if record is None or not isinstance(record.get("candidates"), list):
        return []
    plan = compile_question_plan(example.question)
    graph = build_evidence_graph(example.candidates)
    spec = _compile_symbolic_query_spec(example.question, example.candidates)
    pool: list[tuple[tuple[float | int, ...], ReasoningProgram, object, tuple[str, ...]]] = []
    for raw in record["candidates"]:
        if not isinstance(raw, dict):
            continue
        try:
            program = parse_program_key(str(raw.get("program_key", "")))
        except (TypeError, ValueError):
            continue
        identity_required = _identity_constraint_required(raw)
        verification = verify_program(
            plan,
            program,
            graph,
            financial_identity_catalog_enabled=identity_required,
        )
        if program.operation in _DIRECTIONAL_OPERATIONS:
            base_reasons = set(
                _program_search_rejection_reasons(
                    spec,
                    example.question,
                    program,
                    example.candidates,
                    verification,
                    financial_identity_catalog_enabled=identity_required,
                )
            )
        else:
            base_reasons = {
                str(reason)
                for reason in raw.get("rejection_reasons", ())
                if str(reason) not in _STALE_DIRECTION_REASONS
            } | set(verification.reasons)
        if base_reasons:
            continue
        neural_score = _feature(raw, "neural_log_score")
        key = _symbolic_program_search_key(
            spec,
            example.question,
            program,
            example.candidates,
            neural_score,
            verification,
            proof_guided=True,
        )
        pool.append((key, program, verification, tuple(sorted(base_reasons))))
    return pool


def _winner_from_pool(
    policy: str,
    plan: QuestionPlan,
    graph: EvidenceGraph,
    pool: list[tuple[tuple[float | int, ...], ReasoningProgram, object, tuple[str, ...]]],
) -> ReasoningProgram | None:
    eligible = [
        item
        for item in pool
        if not _policy_reasons(policy, plan, item[1], graph, item[2])
    ]
    return max(eligible, key=lambda item: item[0])[1] if eligible else None


def _empty() -> Counter[str]:
    return Counter({"examples": 0, "accepted": 0, "correct": 0, "false_accept": 0, "pointer_exact": 0})


def _rates(counts: Counter[str]) -> dict[str, object]:
    return {
        **dict(counts),
        "coverage": counts["accepted"] / counts["examples"] if counts["examples"] else 0.0,
        "accepted_precision": counts["correct"] / counts["accepted"] if counts["accepted"] else None,
        "signed_execution_rate": counts["correct"] / counts["examples"] if counts["examples"] else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="QP0 counterfactual audit over frozen S4 development Top-64")
    parser.add_argument("--fold-manifest", type=Path, default=ROOT / "data" / "document_hrm_v12b2_reranker_disjoint_folds_subtract_v4.json")
    parser.add_argument("--phase-b-root", type=Path, default=ROOT / "cache" / "v12b2_folds_subtract_v5")
    parser.add_argument("--phase-c-root", type=Path, default=ROOT / "cache" / "v12b2_phase_c_subtract_v5")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "select_add_counterfactual_s4_development_v1.json")
    args = parser.parse_args()

    for path in (args.fold_manifest, args.phase_b_root, args.phase_c_root):
        _guard_path(path)
    fold_manifest = _read_json(args.fold_manifest)
    phase_b_manifest = _read_json(args.phase_b_root / "manifest.json")
    phase_c_manifest = _read_json(args.phase_c_root / "manifest.json")
    _guard_manifests(fold_manifest, phase_b_manifest, phase_c_manifest)
    input_path = _input_path(fold_manifest)
    metadata = fold_manifest.get("examples")
    if not isinstance(metadata, dict):
        raise ValueError("Missing fold metadata")
    allowed_ids = set(str(value) for value in metadata)
    source_examples = load_program_examples(input_path)
    examples = {example.example_id: example for example in source_examples if example.example_id in allowed_ids}
    if set(examples) != allowed_ids:
        raise ValueError("Development example set incomplete")
    training_operations = {str(value) for value in phase_b_manifest.get("training_operations", ())}
    fold_defs = {int(row["fold"]): row for row in fold_manifest.get("folds", ()) if isinstance(row, dict)}

    totals = {policy: _empty() for policy in POLICIES}
    by_operation = {policy: defaultdict(_empty) for policy in POLICIES}
    by_fold = {policy: defaultdict(_empty) for policy in POLICIES}
    transition = {policy: Counter() for policy in POLICIES if policy != "baseline"}
    integrity_excluded = Counter()
    details: list[dict[str, object]] = []
    gold_policy_retained = {policy: Counter() for policy in POLICIES if policy != "baseline"}
    raw_gold_present = Counter()

    for fold in range(5):
        definition = fold_defs[fold]
        candidate_path = args.phase_b_root / f"fold_{fold}" / "validation_candidates.jsonl"
        supervision_path = args.phase_c_root / f"fold_{fold}" / "calibration_supervision.jsonl"
        candidates = _read_jsonl_by_id(candidate_path)
        supervision = _read_jsonl_by_id(supervision_path)
        scoped_ids = {
            str(value)
            for value in definition["validation_example_ids"]
            if str(value) in examples and examples[str(value)].program.operation in training_operations
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
                integrity_excluded.update(issues)
                continue
            record = candidates.get(example_id)
            raw_candidates = record.get("candidates", ()) if isinstance(record, dict) else ()
            if any(
                isinstance(raw, dict)
                and canonical_program_equal(parse_program_key(str(raw.get("program_key", ""))), example.program)
                for raw in raw_candidates
            ):
                raw_gold_present["present"] += 1
            raw_gold_present["total"] += 1

            pool = _candidate_pool(example, record)
            winners: dict[str, ReasoningProgram | None] = {}
            for policy in POLICIES:
                winner = _winner_from_pool(policy, plan, graph, pool)
                winners[policy] = winner
                counts = totals[policy]
                op_counts = by_operation[policy][example.program.operation]
                fold_counts = by_fold[policy][str(fold)]
                for target in (counts, op_counts, fold_counts):
                    target["examples"] += 1
                    if winner is not None:
                        target["accepted"] += 1
                        correct = _execution_correct(winner, example)
                        target["correct"] += int(correct)
                        target["false_accept"] += int(not correct)
                        target["pointer_exact"] += int(canonical_program_equal(winner, example.program))

            baseline = winners["baseline"]
            baseline_state = "abstain" if baseline is None else ("correct" if _execution_correct(baseline, example) else "false_accept")
            row_transitions = {}
            gold_verification = verify_program(plan, example.program, graph)
            for policy in POLICIES[1:]:
                winner = winners[policy]
                state = "abstain" if winner is None else ("correct" if _execution_correct(winner, example) else "false_accept")
                transition[policy][f"{baseline_state}->{state}"] += 1
                row_transitions[policy] = state
                retained = not _policy_reasons(policy, plan, example.program, graph, gold_verification)
                gold_policy_retained[policy]["total"] += 1
                gold_policy_retained[policy]["retained"] += int(retained)
            if example.program.operation in SCOPED_OPERATIONS and baseline is not None:
                details.append(
                    {
                        "example_id": example_id,
                        "fold": fold,
                        "gold_operation": example.program.operation,
                        "question": example.question,
                        "baseline_program": f"{baseline.operation}:{','.join(map(str, baseline.operands))}",
                        "baseline_state": baseline_state,
                        "policy_states": row_transitions,
                    }
                )

    baseline = totals["baseline"]
    expected_baseline = {
        "examples": 976,
        "accepted": 937,
        "correct": 829,
        "false_accept": 108,
    }
    if any(baseline[key] != value for key, value in expected_baseline.items()):
        raise ValueError(
            f"Frozen S4 baseline drift: {dict(baseline)} != {expected_baseline}"
        )
    policy_reports: dict[str, object] = {}
    for policy in POLICIES:
        counts = totals[policy]
        metrics = _rates(counts)
        if policy == "baseline":
            metrics.update(
                {
                    "false_accept_removed_F": 0,
                    "true_accept_removed_T": 0,
                    "new_false_accepts": 0,
                    "new_true_accepts": 0,
                    "net_false_accept_reduction": 0,
                    "net_true_accept_reduction": 0,
                    "nine_F_minus_T": 0,
                }
            )
        else:
            transitions = transition[policy]
            f_removed = transitions["false_accept->abstain"] + transitions["false_accept->correct"]
            t_removed = transitions["correct->abstain"] + transitions["correct->false_accept"]
            new_false = transitions["correct->false_accept"] + transitions["abstain->false_accept"]
            new_true = transitions["false_accept->correct"] + transitions["abstain->correct"]
            gold_retention = (
                gold_policy_retained[policy]["retained"] / gold_policy_retained[policy]["total"]
                if gold_policy_retained[policy]["total"] else 0.0
            )
            metrics.update(
                {
                    "false_accept_removed_F": f_removed,
                    "true_accept_removed_T": t_removed,
                    "new_false_accepts": new_false,
                    "new_true_accepts": new_true,
                    "net_false_accept_reduction": baseline["false_accept"] - counts["false_accept"],
                    "net_true_accept_reduction": baseline["correct"] - counts["correct"],
                    "nine_F_minus_T": 9 * f_removed - t_removed,
                    "gold_policy_retention": gold_retention,
                    "transitions": dict(transitions),
                }
            )
            gates = {
                "accepted_precision_90": metrics["accepted_precision"] is not None and float(metrics["accepted_precision"]) >= 0.90,
                "coverage_88_5": float(metrics["coverage"]) >= 0.885,
                "nine_F_minus_T_143": 9 * f_removed - t_removed >= 143,
                "gold_policy_retention_95": gold_retention >= 0.95,
                "new_false_accepts_zero": new_false == 0,
            }
            metrics["counterfactual_gates"] = gates
            metrics["all_counterfactual_gates_pass"] = all(gates.values())
        policy_reports[policy] = {
            "overall": metrics,
            "per_operation": {operation: _rates(value) for operation, value in sorted(by_operation[policy].items())},
            "per_fold": {fold: _rates(value) for fold, value in sorted(by_fold[policy].items())},
        }

    report = {
        "schema": "select-add-counterfactual-s4-development-v1",
        "classification": "development_only_company_report_disjoint",
        "evaluation_mode": "counterfactual_candidate_reselection_over_frozen_top64",
        "policies_are_audit_only": True,
        "runtime_changed": False,
        "threshold_changed": False,
        "fine_tuning_performed": False,
        "global_validation_loaded": False,
        "sealed_validation_loaded": False,
        "blind_test_loaded": False,
        "development_examples": baseline["examples"],
        "accepted_select_add_baseline": sum(
            by_operation["baseline"][operation]["accepted"] for operation in SCOPED_OPERATIONS
        ),
        "top64_canonical_pointer_gold_presence": raw_gold_present["present"] / raw_gold_present["total"] if raw_gold_present["total"] else 0.0,
        "frozen_fold_cache_minimum_raw_gold_program_recall": phase_b_manifest.get("minimum_fold_raw_gold_program_recall"),
        "frozen_fold_cache_mean_raw_gold_program_recall": phase_b_manifest.get("mean_fold_raw_gold_program_recall"),
        "integrity_excluded": sum(integrity_excluded.values()),
        "integrity_issue_counts": dict(integrity_excluded),
        "policy_results": policy_reports,
        "release_conditions": {
            "target_accepted_precision": 0.90,
            "minimum_coverage": 0.885,
            "required_nine_F_minus_T": 143,
            "baseline_phase_d_locked": True,
            "runtime_policy_authorized": False,
        },
        "protected_read_hygiene": {
            "zero_protected_read_claim": False,
            "used_for_metrics_or_policy": False,
            "note": "A broad interactive search before this script printed one old protected report record. This audit itself is path-guarded and loads only the allowlisted development fold/cache artifacts.",
        },
        "artifact_hashes": {
            "fold_manifest": _digest(args.fold_manifest),
            "development_examples": _digest(input_path),
            "phase_b_manifest": _digest(args.phase_b_root / "manifest.json"),
            "phase_c_manifest": _digest(args.phase_c_root / "manifest.json"),
            "checkpoint": _digest(ROOT / "models" / "document_hrm_V8_candidate.pt"),
            "question_planner": _digest(ROOT / "src" / "hrm_finance_qa" / "question_planner.py"),
            "symbolic_verifier": _digest(ROOT / "src" / "hrm_finance_qa" / "symbolic_verifier.py"),
            "document_hrm": _digest(ROOT / "src" / "hrm_finance_qa" / "document_hrm.py"),
            "audit_script": _digest(Path(__file__)),
        },
        "select_add_accepted_details": details,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"development_examples": report["development_examples"], "accepted_select_add_baseline": report["accepted_select_add_baseline"], "top64_canonical_pointer_gold_presence": report["top64_canonical_pointer_gold_presence"], "frozen_fold_cache_minimum_raw_gold_program_recall": report["frozen_fold_cache_minimum_raw_gold_program_recall"], "policies": {name: value["overall"] for name, value in policy_reports.items()}}, ensure_ascii=False, indent=2))
    print(f"Saved QP0 counterfactual audit: {args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
