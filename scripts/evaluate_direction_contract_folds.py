from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrm_finance_qa.direction_contract import assess_program_direction  # noqa: E402
from hrm_finance_qa.document_hrm import (  # noqa: E402
    _compile_symbolic_query_spec,
    _program_search_rejection_reasons,
    _symbolic_program_search_key,
)
from hrm_finance_qa.evidence_schema import build_evidence_graph  # noqa: E402
from hrm_finance_qa.program_data import (  # noqa: E402
    ProgramExample,
    ReasoningProgram,
    canonical_program_equal,
    execute_program,
    load_program_examples,
)
from hrm_finance_qa.program_reranker import FEATURE_INDEX  # noqa: E402
from hrm_finance_qa.question_planner import compile_question_plan  # noqa: E402
from hrm_finance_qa.symbolic_verifier import (  # noqa: E402
    VerificationState,
    verify_program,
)
from hrm_finance_qa.v12b2_supervision import (  # noqa: E402
    parse_program_key,
    supervision_integrity_issues,
)


COMMON_OPERATION_MINIMUM = 50
MAX_OPERATION_DROP = 0.01
MINIMUM_GOLD_RETENTION = 0.95
MINIMUM_ACCEPTED_PRECISION = 0.90
MINIMUM_COVERAGE = 0.885
_DIRECTIONAL_OPERATIONS = {"SUBTRACT", "PERCENT_CHANGE"}
_STALE_DIRECTION_REASONS = {"temporal_direction", "graph_temporal_order"}


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object in {path}")
    return value


def _read_jsonl_by_id(path: Path) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            example_id = str(row.get("example_id", ""))
            if not example_id or example_id in output:
                raise ValueError(f"{path}:{line_number}: duplicate/missing example_id")
            output[example_id] = row
    return output


def _execution_correct(
    predicted: ReasoningProgram, example: ProgramExample
) -> bool:
    try:
        predicted_value = execute_program(predicted, example.candidates)
        expected_value = execute_program(example.program, example.candidates)
    except (ArithmeticError, ValueError, IndexError):
        return False
    tolerance = max(Decimal("0.035"), abs(expected_value) * Decimal("1e-9"))
    return abs(predicted_value - expected_value) <= tolerance


def _feature(candidate: dict[str, object], name: str) -> float:
    values = candidate.get("features")
    index = FEATURE_INDEX[name]
    if not isinstance(values, list) or index >= len(values):
        return 0.0
    return float(values[index])


def _identity_constraint_required(candidate: dict[str, object]) -> bool:
    sources = candidate.get("generation_sources", ())
    return isinstance(sources, list) and any(
        str(source).startswith("financial_identity:") for source in sources
    )


def _current_winner(
    example: ProgramExample,
    record: dict[str, object] | None,
) -> ReasoningProgram | None:
    if record is None:
        return None
    raw_candidates = record.get("candidates", ())
    if not isinstance(raw_candidates, list):
        return None
    plan = compile_question_plan(example.question)
    graph = build_evidence_graph(example.candidates)
    spec = _compile_symbolic_query_spec(example.question, example.candidates)
    eligible: list[tuple[tuple[float | int, ...], ReasoningProgram]] = []
    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, dict):
            continue
        try:
            program = parse_program_key(str(raw_candidate.get("program_key", "")))
        except (ValueError, TypeError):
            continue
        identity_required = _identity_constraint_required(raw_candidate)
        verification = verify_program(
            plan,
            program,
            graph,
            financial_identity_catalog_enabled=identity_required,
        )
        if program.operation in _DIRECTIONAL_OPERATIONS:
            reasons = set(
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
            cached_reasons = {
                str(reason)
                for reason in raw_candidate.get("rejection_reasons", ())
                if str(reason) not in _STALE_DIRECTION_REASONS
            }
            reasons = cached_reasons | set(verification.reasons)
        if reasons:
            continue
        neural_score = _feature(raw_candidate, "neural_log_score")
        key = _symbolic_program_search_key(
            spec,
            example.question,
            program,
            example.candidates,
            neural_score,
            verification,
            proof_guided=True,
        )
        eligible.append((key, program))
    return max(eligible, key=lambda item: item[0])[1] if eligible else None


def _frozen_winner(record: dict[str, object] | None) -> ReasoningProgram | None:
    if record is None:
        return None
    candidates = record.get("candidates", ())
    if not isinstance(candidates, list):
        return None
    winners = [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict)
        and _feature(candidate, "legacy_winner") >= 0.5
        and bool(candidate.get("runtime_eligible"))
    ]
    if not winners:
        return None
    return parse_program_key(str(winners[0].get("program_key", "")))


def _empty_counts() -> Counter[str]:
    return Counter(
        {
            "examples": 0,
            "accepted": 0,
            "pointer_exact": 0,
            "signed_execution": 0,
            "frozen_accepted": 0,
            "frozen_signed_execution": 0,
            "direction_examples": 0,
            "direction_accepted": 0,
            "direction_correct": 0,
            "gold_retained": 0,
        }
    )


def _rates(counts: Counter[str]) -> dict[str, object]:
    total = counts["examples"]
    accepted = counts["accepted"]
    frozen_accepted = counts["frozen_accepted"]
    direction_total = counts["direction_examples"]
    direction_accepted = counts["direction_accepted"]
    return {
        **dict(counts),
        "pointer_exact_rate": counts["pointer_exact"] / total if total else 0.0,
        "signed_execution_rate": counts["signed_execution"] / total if total else 0.0,
        "coverage": accepted / total if total else 0.0,
        "accepted_precision": counts["signed_execution"] / accepted if accepted else 0.0,
        "frozen_coverage": frozen_accepted / total if total else 0.0,
        "frozen_signed_execution_rate": counts["frozen_signed_execution"] / total if total else 0.0,
        "signed_execution_delta_vs_frozen": (
            counts["signed_execution"] - counts["frozen_signed_execution"]
        ) / total if total else 0.0,
        "direction_coverage": direction_accepted / direction_total if direction_total else 0.0,
        "direction_accuracy": counts["direction_correct"] / direction_accepted if direction_accepted else 0.0,
        "end_to_end_direction_accuracy": counts["direction_correct"] / direction_total if direction_total else 0.0,
        "verifier_gold_retention": counts["gold_retained"] / total if total else 0.0,
    }


def _evaluate_fold(
    fold: int,
    *,
    fold_definition: dict[str, object],
    phase_b_root: Path,
    phase_c_root: Path,
    examples: dict[str, ProgramExample],
    training_operations: set[str],
) -> tuple[dict[str, object], Counter[str], dict[str, Counter[str]]]:
    phase_b_manifest = _read_json(phase_b_root / f"fold_{fold}" / "manifest.json")
    phase_c_manifest = _read_json(phase_c_root / f"fold_{fold}" / "manifest.json")
    if any(
        int(phase_b_manifest.get(key, -1)) != 0
        for key in ("company_overlap", "report_overlap", "group_overlap")
    ):
        raise ValueError(f"Fold {fold} is not company/report/group disjoint")
    candidate_path = phase_b_root / f"fold_{fold}" / "validation_candidates.jsonl"
    supervision_path = phase_c_root / f"fold_{fold}" / "calibration_supervision.jsonl"
    if _digest(candidate_path) != str(phase_b_manifest["validation_cache_sha256"]):
        raise ValueError(f"Fold {fold} candidate hash mismatch")
    if _digest(supervision_path) != str(phase_c_manifest["calibration_supervision_sha256"]):
        raise ValueError(f"Fold {fold} supervision hash mismatch")
    candidates = _read_jsonl_by_id(candidate_path)
    supervision = _read_jsonl_by_id(supervision_path)
    calibration_ids = {
        str(value) for value in fold_definition["validation_example_ids"]
    }
    scoped_ids = {
        example_id
        for example_id in calibration_ids
        if example_id in examples
        and examples[example_id].program.operation in training_operations
    }
    totals = _empty_counts()
    by_operation: dict[str, Counter[str]] = {}
    integrity_issues: Counter[str] = Counter()
    excluded = 0
    candidate_missing = 0
    for example_id in sorted(scoped_ids):
        example = examples[example_id]
        plan = compile_question_plan(example.question)
        graph = build_evidence_graph(example.candidates)
        issues = supervision_integrity_issues(example, graph=graph, plan=plan)
        stored = supervision.get(example_id)
        if stored is not None and tuple(stored.get("supervision_integrity_issues", ())) != issues:
            raise ValueError(f"Fold {fold} S3 integrity drift for {example_id}")
        if issues:
            excluded += 1
            integrity_issues.update(issues)
            continue
        operation = example.program.operation
        counts = by_operation.setdefault(operation, _empty_counts())
        totals["examples"] += 1
        counts["examples"] += 1
        gold_verification = verify_program(plan, example.program, graph)
        if gold_verification.state is not VerificationState.INVALID:
            totals["gold_retained"] += 1
            counts["gold_retained"] += 1
        record = candidates.get(example_id)
        candidate_missing += int(record is None)
        current = _current_winner(example, record)
        frozen = _frozen_winner(record)
        if current is not None:
            totals["accepted"] += 1
            counts["accepted"] += 1
            if canonical_program_equal(current, example.program):
                totals["pointer_exact"] += 1
                counts["pointer_exact"] += 1
            if _execution_correct(current, example):
                totals["signed_execution"] += 1
                counts["signed_execution"] += 1
        if frozen is not None:
            totals["frozen_accepted"] += 1
            counts["frozen_accepted"] += 1
            if _execution_correct(frozen, example):
                totals["frozen_signed_execution"] += 1
                counts["frozen_signed_execution"] += 1
        gold_direction = assess_program_direction(plan, example.program, graph)
        if gold_direction.outcome == "direction_supported":
            totals["direction_examples"] += 1
            counts["direction_examples"] += 1
            if current is not None and current.operation in _DIRECTIONAL_OPERATIONS:
                totals["direction_accepted"] += 1
                counts["direction_accepted"] += 1
                predicted_direction = assess_program_direction(plan, current, graph)
                if predicted_direction.outcome == "direction_supported":
                    totals["direction_correct"] += 1
                    counts["direction_correct"] += 1
    eligibility = phase_c_manifest.get("operation_eligibility", {})
    subtract_eligibility = (
        eligibility.get("SUBTRACT", {}) if isinstance(eligibility, dict) else {}
    )
    fold_report = {
        "fold": fold,
        "classification": "development_calibration_fold",
        "company_overlap": int(phase_b_manifest["company_overlap"]),
        "report_overlap": int(phase_b_manifest["report_overlap"]),
        "group_overlap": int(phase_b_manifest["group_overlap"]),
        "scoped_examples_before_integrity": len(scoped_ids),
        "integrity_excluded": excluded,
        "integrity_issue_counts": dict(integrity_issues),
        "candidate_records_missing": candidate_missing,
        "metrics": _rates(totals),
        "per_operation": {
            operation: _rates(counts)
            for operation, counts in sorted(by_operation.items())
        },
        "subtract_group_gate": {
            "train_groups": int(subtract_eligibility.get("train_groups", 0)),
            "calibration_groups": int(subtract_eligibility.get("calibration_groups", 0)),
            "enabled": bool(subtract_eligibility.get("enabled", False)),
        },
    }
    return fold_report, totals, by_operation


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Direction Contract on five development folds"
    )
    parser.add_argument(
        "--fold-manifest",
        type=Path,
        default=ROOT / "data" / "document_hrm_v12b2_reranker_disjoint_folds_v1.json",
    )
    parser.add_argument(
        "--phase-b-root",
        type=Path,
        default=ROOT / "cache" / "v12b2_folds_candidate_recall",
    )
    parser.add_argument(
        "--phase-c-root",
        type=Path,
        default=ROOT / "cache" / "v12b2_phase_c_v3",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports" / "subtract_direction_s4_development_folds_v1.json",
    )
    args = parser.parse_args()

    fold_manifest = _read_json(args.fold_manifest)
    if int(fold_manifest.get("fold_count", 0)) != 5:
        raise ValueError("S4 requires exactly five folds")
    if fold_manifest.get("blind_test_used") is not False:
        raise ValueError("S4 refuses a manifest that touched blind test")
    metadata = fold_manifest.get("examples", {})
    if not isinstance(metadata, dict):
        raise ValueError("Fold example metadata is missing")
    allowed_ids = set(str(value) for value in metadata)
    source_examples = load_program_examples(Path(str(fold_manifest["input"])))
    examples = {
        example.example_id: example
        for example in source_examples
        if example.example_id in allowed_ids
    }
    if set(examples) != allowed_ids:
        raise ValueError("Development fold example set is incomplete")
    phase_b_index = _read_json(args.phase_b_root / "manifest.json")
    training_operations = {
        str(value) for value in phase_b_index.get("training_operations", ())
    }
    fold_definitions = {
        int(row["fold"]): row
        for row in fold_manifest.get("folds", ())
        if isinstance(row, dict)
    }
    aggregate = _empty_counts()
    aggregate_operations: dict[str, Counter[str]] = {}
    folds: list[dict[str, object]] = []
    for fold in range(5):
        fold_report, totals, operation_counts = _evaluate_fold(
            fold,
            fold_definition=fold_definitions[fold],
            phase_b_root=args.phase_b_root,
            phase_c_root=args.phase_c_root,
            examples=examples,
            training_operations=training_operations,
        )
        folds.append(fold_report)
        aggregate.update(totals)
        for operation, counts in operation_counts.items():
            aggregate_operations.setdefault(operation, _empty_counts()).update(counts)
    overall = _rates(aggregate)
    per_operation = {
        operation: _rates(counts)
        for operation, counts in sorted(aggregate_operations.items())
    }
    common_operation_drops = {
        operation: values["signed_execution_delta_vs_frozen"]
        for operation, values in per_operation.items()
        if int(values["examples"]) >= COMMON_OPERATION_MINIMUM
        and float(values["signed_execution_delta_vs_frozen"]) < -MAX_OPERATION_DROP
    }
    minimum_fold_retention = min(
        float(fold["metrics"]["verifier_gold_retention"]) for fold in folds
    )
    subtract_group_gate = all(
        bool(fold["subtract_group_gate"]["enabled"]) for fold in folds
    )
    gates = {
        "five_company_report_disjoint_folds": len(folds) == 5 and all(
            fold["company_overlap"] == fold["report_overlap"] == fold["group_overlap"] == 0
            for fold in folds
        ),
        "minimum_fold_verifier_gold_retention_95": minimum_fold_retention >= MINIMUM_GOLD_RETENTION,
        "no_common_operation_drop_over_1pp": not common_operation_drops,
        "accepted_precision_90": float(overall["accepted_precision"]) >= MINIMUM_ACCEPTED_PRECISION,
        "coverage_88_5": float(overall["coverage"]) >= MINIMUM_COVERAGE,
        "subtract_groups_30_train_15_calibration_all_folds": subtract_group_gate,
    }
    phase_d_release_gate = all(gates.values())
    code_paths = (
        ROOT / "src" / "hrm_finance_qa" / "question_planner.py",
        ROOT / "src" / "hrm_finance_qa" / "direction_contract.py",
        ROOT / "src" / "hrm_finance_qa" / "symbolic_verifier.py",
        ROOT / "src" / "hrm_finance_qa" / "program_synthesis.py",
        ROOT / "src" / "hrm_finance_qa" / "v12b2_supervision.py",
        ROOT / "src" / "hrm_finance_qa" / "document_hrm.py",
    )
    report = {
        "schema": "subtract-direction-s4-development-fold-evaluation-v1",
        "classification": "development_only_company_report_disjoint",
        "evaluation_mode": "current_contract_replay_over_frozen_top64_candidates",
        "global_validation_loaded": False,
        "blind_test_loaded": False,
        "fine_tuning_performed": False,
        "threshold_changed": False,
        "development_manifest_examples": len(examples),
        "protected_or_untracked_input_records_excluded": len(source_examples) - len(examples),
        "training_operations": sorted(training_operations),
        "overall": overall,
        "per_operation": per_operation,
        "folds": folds,
        "common_operation_minimum_examples": COMMON_OPERATION_MINIMUM,
        "common_operation_drops_over_1pp": common_operation_drops,
        "minimum_fold_verifier_gold_retention": minimum_fold_retention,
        "gates": gates,
        "phase_d_release_gate": phase_d_release_gate,
        "next_allowed_step": "phase_d" if phase_d_release_gate else "frozen_fallback_and_collect_data",
        "artifact_hashes": {
            "fold_manifest": _digest(args.fold_manifest),
            "phase_b_manifest": _digest(args.phase_b_root / "manifest.json"),
            "phase_c_manifest": _digest(args.phase_c_root / "manifest.json"),
            "evaluation_script": _digest(Path(__file__)),
            "code": {str(path.relative_to(ROOT)): _digest(path) for path in code_paths},
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "overall": overall,
        "common_operation_drops_over_1pp": common_operation_drops,
        "minimum_fold_verifier_gold_retention": minimum_fold_retention,
        "subtract_group_gate": subtract_group_gate,
        "gates": gates,
        "phase_d_release_gate": phase_d_release_gate,
    }, ensure_ascii=False, indent=2))
    print(f"Saved S4 development report: {args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
