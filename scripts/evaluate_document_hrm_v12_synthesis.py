from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrm_finance_qa.document_hrm import (  # noqa: E402
    DocumentHRMBundle,
    evaluate_document_hrm,
)
from hrm_finance_qa.program_data import load_program_examples  # noqa: E402
from hrm_finance_qa.program_reranker import ProgramRerankerBundle  # noqa: E402
from hrm_finance_qa.v11_pipeline import build_reranker_records  # noqa: E402


COMMON_OPERATIONS = (
    "SELECT",
    "ADD",
    "SUBTRACT",
    "MULTIPLY",
    "DIVIDE",
    "RATIO_PERCENT",
    "PERCENT_CHANGE",
    "AVERAGE",
)


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _reject_protected(path: Path) -> None:
    normalized = str(path.expanduser()).lower().replace("\\", "/")
    if any(
        marker in normalized
        for marker in ("test.jsonl", "blind", "final_eval", "/final.jsonl")
    ):
        raise ValueError(f"V12 development refuses protected input: {path}")


def _important(metrics: dict[str, float]) -> dict[str, float]:
    names = (
        "operation_accuracy",
        "pointer_pair_accuracy",
        "program_exact_accuracy",
        "execution_accuracy",
        "accepted_coverage",
        "accepted_execution_precision",
        "gold_program_recall_at_10",
        "gold_program_generated_recall",
        "gold_eligible_program_recall",
        "gold_symbolic_program_recall_at_10",
        "typed_synthesis_gold_generation_rate",
        "typed_synthesis_example_rate",
        "program_search_trace_coverage",
        "single_eligible_program_rate",
        "search_pipeline_gold_rejection_rate",
        "mean_program_candidates_considered",
        "mean_synthesized_program_candidates",
    )
    output = {name: metrics[name] for name in names}
    for operation in COMMON_OPERATIONS:
        prefix = f"program_operation_{operation.lower()}"
        output[f"{operation}_program_exact"] = metrics[
            f"{prefix}_program_exact_accuracy"
        ]
        output[f"{operation}_execution"] = metrics[
            f"{prefix}_execution_accuracy"
        ]
    return output


def _runtime_bundle(
    path: Path,
    *,
    device: str,
    enabled: bool,
    financial_identity_catalog_enabled: bool = False,
    candidate_recall_v12_4_enabled: bool = False,
    program_reranker_path: Path | None = None,
) -> DocumentHRMBundle:
    bundle = DocumentHRMBundle.load(path, device=device)
    bundle.metadata = dict(bundle.metadata)
    bundle.metadata["typed_program_synthesis_enabled"] = enabled
    bundle.metadata["typed_program_synthesis_version"] = (
        "typed-slot-synthesis-v1" if enabled else "disabled"
    )
    bundle.metadata["financial_identity_catalog_enabled"] = (
        financial_identity_catalog_enabled
    )
    bundle.metadata["candidate_recall_v12_4_enabled"] = (
        candidate_recall_v12_4_enabled
    )
    if program_reranker_path is not None:
        reranker = ProgramRerankerBundle.load(
            program_reranker_path, device=device
        )
        reranker.validate_base_model(path)
        bundle.attach_program_reranker(reranker)
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validation-only V12 typed program synthesis ablation"
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=ROOT / "models" / "document_hrm_V8_candidate.pt",
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=ROOT / "data" / "validation.jsonl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT
        / "reports"
        / "document_hrm_V12_typed_synthesis_validation.json",
    )
    parser.add_argument(
        "--reuse-baseline-report",
        type=Path,
        help=(
            "Reuse the typed-off baseline/audit from a prior validation-only "
            "report after verifying the frozen V8 hash and safety flags"
        ),
    )
    parser.add_argument(
        "--reuse-reference-section",
        choices=("baseline", "v12"),
        default="baseline",
        help=(
            "Select the comparison arm from --reuse-baseline-report. Use "
            "'v12' when a later V12 revision is compared with its immediate "
            "predecessor instead of the typed-off V8 arm."
        ),
    )
    parser.add_argument(
        "--reuse-baseline-is-same-runtime",
        action="store_true",
        help=(
            "Declare that a verified reused baseline was produced with the "
            "same V12 runtime flags; rejected unless those flags match"
        ),
    )
    parser.add_argument(
        "--reuse-candidate-report",
        type=Path,
        help=(
            "Reuse the candidate metrics/audit from an identical completed "
            "validation-only run; useful for recomputing comparison gates "
            "without another model forward pass."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--financial-identity-catalog",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate the experimental YAML identity catalog in the V12 arm",
    )
    parser.add_argument(
        "--candidate-recall-v12-4",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate V12.4 candidate union and trace-complete fallback",
    )
    parser.add_argument(
        "--program-reranker-path",
        type=Path,
        help="Optional V12-B sidecar; the V8 base checkpoint stays read-only",
    )
    parser.add_argument(
        "--same-runtime-reference",
        action="store_true",
        help=(
            "Recompute the no-sidecar reference with the same typed/catalog/"
            "V12.4 flags; required for a controlled reranker A/B comparison"
        ),
    )
    parser.add_argument("--target-precision", type=float, default=0.90)
    args = parser.parse_args()

    if (
        args.program_reranker_path is not None
        and args.reuse_candidate_report is not None
    ):
        raise ValueError(
            "A sidecar evaluation cannot reuse candidate predictions"
        )
    if args.same_runtime_reference and args.reuse_baseline_report is not None:
        raise ValueError(
            "--same-runtime-reference cannot reuse a historical baseline"
        )
    if (
        args.reuse_baseline_is_same_runtime
        and args.reuse_baseline_report is None
    ):
        raise ValueError(
            "--reuse-baseline-is-same-runtime requires --reuse-baseline-report"
        )

    _reject_protected(args.validation)
    required_paths = [args.base_model, args.validation]
    if args.program_reranker_path is not None:
        required_paths.append(args.program_reranker_path)
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    base_hash_before = _digest(args.base_model)
    examples = load_program_examples(args.validation)

    baseline_reused_from: str | None = None
    comparison_runtime_equivalent = bool(args.same_runtime_reference)
    if args.reuse_baseline_report is not None:
        source = args.reuse_baseline_report.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        prior = json.loads(source.read_text(encoding="utf-8"))
        if (
            prior.get("validation_only") is not True
            or prior.get("blind_test_used") is not False
            or prior.get("test_jsonl_used") is not False
            or prior.get("base_model_sha256_after") != base_hash_before
        ):
            raise RuntimeError("Reusable baseline report failed safety checks")
        section = args.reuse_reference_section
        baseline_important = dict(prior[section])
        baseline_audit = dict(prior["candidate_audit"][section])
        latency_key = "baseline_seconds" if section == "baseline" else "v12_seconds"
        baseline_seconds = float(
            prior.get("latency", {}).get(latency_key, 0.0)
        )
        baseline_reused_from = str(source)
        comparison_runtime_equivalent = bool(
            section == "v12"
            or (
                section == "baseline"
                and prior.get("same_runtime_reference") is True
            )
        )
        if args.reuse_baseline_is_same_runtime:
            if (
                bool(prior.get("financial_identity_catalog_enabled"))
                != bool(args.financial_identity_catalog)
                or bool(prior.get("candidate_recall_v12_4_enabled"))
                != bool(args.candidate_recall_v12_4)
            ):
                raise RuntimeError(
                    "Reused same-runtime baseline flags do not match"
                )
            comparison_runtime_equivalent = True
    else:
        baseline = _runtime_bundle(
            args.base_model,
            device=args.device,
            enabled=args.same_runtime_reference,
            financial_identity_catalog_enabled=(
                args.financial_identity_catalog
                if args.same_runtime_reference
                else False
            ),
            candidate_recall_v12_4_enabled=(
                args.candidate_recall_v12_4
                if args.same_runtime_reference
                else False
            ),
        )
        started = perf_counter()
        baseline_metrics = evaluate_document_hrm(
            baseline, examples, batch_size=args.batch_size
        )
        baseline_seconds = perf_counter() - started
        _, baseline_audit = build_reranker_records(
            baseline, examples, top_k=args.top_k, batch_size=args.batch_size
        )
        baseline_important = _important(baseline_metrics)

    candidate_reused_from: str | None = None
    if args.reuse_candidate_report is not None:
        source = args.reuse_candidate_report.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        prior = json.loads(source.read_text(encoding="utf-8"))
        if (
            prior.get("validation_only") is not True
            or prior.get("blind_test_used") is not False
            or prior.get("test_jsonl_used") is not False
            or prior.get("base_model_sha256_after") != base_hash_before
            or bool(prior.get("financial_identity_catalog_enabled"))
            != bool(args.financial_identity_catalog)
            or bool(prior.get("candidate_recall_v12_4_enabled"))
            != bool(args.candidate_recall_v12_4)
        ):
            raise RuntimeError("Reusable candidate report failed safety checks")
        v12_important = dict(prior["v12"])
        v12_audit = dict(prior["candidate_audit"]["v12"])
        v12_seconds = float(prior.get("latency", {}).get("v12_seconds", 0.0))
        candidate_reused_from = str(source)
    else:
        candidate = _runtime_bundle(
            args.base_model,
            device=args.device,
            enabled=True,
            financial_identity_catalog_enabled=args.financial_identity_catalog,
            candidate_recall_v12_4_enabled=args.candidate_recall_v12_4,
            program_reranker_path=args.program_reranker_path,
        )
        started = perf_counter()
        v12_metrics = evaluate_document_hrm(
            candidate, examples, batch_size=args.batch_size
        )
        v12_seconds = perf_counter() - started
        if candidate.program_reranker is None:
            _, v12_audit = build_reranker_records(
                candidate, examples, top_k=args.top_k, batch_size=args.batch_size
            )
        else:
            # The sidecar changes selection only, not V12.4 generation. Reuse
            # the frozen-reference audit rather than constructing a training
            # cache from predictions that have already seen a reranker.
            v12_audit = dict(baseline_audit)
        v12_important = _important(v12_metrics)
    deltas = {
        key: v12_important[key] - baseline_important[key]
        for key in v12_important
    }
    operation_execution_deltas = {
        operation: deltas[f"{operation}_execution"]
        for operation in COMMON_OPERATIONS
    }
    candidate_gate_applicable = not (
        args.program_reranker_path is not None
        and comparison_runtime_equivalent
    )
    candidate_gate_checks = {
        "gold_generated_recall_gain_at_least_5pp": (
            deltas["gold_program_generated_recall"] >= 0.05
        ),
        "gold_eligible_recall_not_lower": (
            deltas["gold_eligible_program_recall"] >= 0.0
        ),
        "no_trace_count_lower": v12_audit["no_trace"] < baseline_audit["no_trace"],
        "pipeline_gold_rejection_not_higher": (
            deltas["search_pipeline_gold_rejection_rate"] <= 0.0
        ),
    }
    runtime_gate_checks = {
        "execution_not_lower": deltas["execution_accuracy"] >= 0.0,
        "program_exact_not_lower": deltas["program_exact_accuracy"] >= 0.0,
        "pointer_pair_not_lower": deltas["pointer_pair_accuracy"] >= 0.0,
        "no_common_operation_loses_over_1pp": (
            min(operation_execution_deltas.values()) >= -0.01
        ),
    }
    reranker_release_checks: dict[str, bool] = {}
    if args.program_reranker_path is not None:
        assert candidate.program_reranker is not None
        reranker_release_checks = {
            "runtime_safety_gate": all(runtime_gate_checks.values()),
            "accepted_precision_at_least_target": (
                v12_important["accepted_execution_precision"]
                >= args.target_precision
            ),
            "accepted_coverage_not_lower": (
                deltas["accepted_coverage"] >= 0.0
            ),
            "calibration_target_met": bool(
                candidate.program_reranker.metadata.get(
                    "calibration_target_met", False
                )
            ),
            "calibration_regression_guard": bool(
                candidate.program_reranker.metadata.get(
                    "regression_guard_passed", False
                )
            ),
            "ranking_regression_guard": bool(
                candidate.program_reranker.metadata.get(
                    "ranking_regression_guard_passed", False
                )
            ),
            "confidence_coverage_guard": bool(
                candidate.program_reranker.metadata.get(
                    "confidence_coverage_guard_passed", False
                )
            ),
        }
    base_hash_after = _digest(args.base_model)
    if base_hash_after != base_hash_before:
        raise RuntimeError("Frozen V8 changed during V12 validation ablation")

    report = {
        "architecture": (
            str(
                candidate.program_reranker.metadata.get(
                    "architecture",
                    "v12b-program-reranker",
                )
            )
            if args.program_reranker_path is not None
            else (
                "v12.4-operation-conditioned-candidate-recall"
                if args.candidate_recall_v12_4
                else (
                    "v12.3-safe-additive-financial-identity-overlay"
                    if args.financial_identity_catalog
                    else "v12.1-typed-slot-generic-financial-role-guards"
                )
            )
        ),
        "financial_identity_catalog_enabled": bool(
            args.financial_identity_catalog
        ),
        "candidate_recall_v12_4_enabled": bool(
            args.candidate_recall_v12_4
        ),
        "base_model": str(args.base_model.resolve()),
        "base_model_sha256_before": base_hash_before,
        "base_model_sha256_after": base_hash_after,
        "base_model_unchanged": True,
        "blind_test_used": False,
        "test_jsonl_used": False,
        "validation_only": True,
        "baseline_reused_from": baseline_reused_from,
        "comparison_reference_section": args.reuse_reference_section,
        "same_runtime_reference": bool(args.same_runtime_reference),
        "comparison_runtime_equivalent": comparison_runtime_equivalent,
        "candidate_reused_from": candidate_reused_from,
        "program_reranker_path": (
            str(args.program_reranker_path.resolve())
            if args.program_reranker_path is not None
            else None
        ),
        "program_reranker_sha256": (
            _digest(args.program_reranker_path)
            if args.program_reranker_path is not None
            else None
        ),
        "candidate_generation_gate": {
            "applicable": candidate_gate_applicable,
            "passed": (
                all(candidate_gate_checks.values())
                if candidate_gate_applicable
                else True
            ),
            "checks": candidate_gate_checks,
        },
        "runtime_safety_gate": {
            "passed": all(runtime_gate_checks.values()),
            "checks": runtime_gate_checks,
            "common_operation_execution_deltas": operation_execution_deltas,
        },
        "reranker_release_gate": {
            "passed": bool(reranker_release_checks)
            and all(reranker_release_checks.values()),
            "target_precision": args.target_precision,
            "checks": reranker_release_checks,
        },
        "latency": {
            "baseline_seconds": baseline_seconds,
            "v12_seconds": v12_seconds,
            "v12_over_baseline": (
                v12_seconds / baseline_seconds if baseline_seconds else 0.0
            ),
        },
        "candidate_audit": {
            "baseline": baseline_audit,
            "v12": v12_audit,
        },
        "baseline": baseline_important,
        "v12": v12_important,
        "v12_minus_baseline": deltas,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved validation-only report: {args.report}")
    print("Frozen V8 SHA-256 verified unchanged.")
    print("Blind test was not touched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
