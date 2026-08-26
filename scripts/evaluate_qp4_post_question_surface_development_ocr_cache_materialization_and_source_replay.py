#!/usr/bin/env python3
"""Materialize lazy development OCR caches and replay the frozen QP4 source stratum."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import signal
import time
from typing import Any

from scripts.evaluate_qp4_post_question_surface_baseline_diagnostic_review import (
    _CandidateCaptureBundle,
    _actual_result,
    _source_summary,
)
from scripts.evaluate_qp4_post_question_surface_residual_implementation import (
    _compiler_measurement,
    _run_regression,
)
from scripts.evaluate_qp4_post_question_surface_residual_implementation_diagnostic_refresh import (
    _required_proof_action_passed,
)


ROOT = Path(__file__).resolve().parents[1]
PRECOMMIT = ROOT / "config/finhrm_qp4_post_question_surface_source_retrieval_roadmap_review_precommit_v1.json"
PRECOMMIT_LOCK = ROOT / "reports/finhrm_qp4_post_question_surface_source_retrieval_roadmap_review_precommit_lock_v1.json"
REPORT = ROOT / "reports/finhrm_qp4_post_question_surface_development_ocr_cache_materialization_and_source_replay_v1.json"
MARKDOWN = ROOT / "reports/finhrm_qp4_post_question_surface_development_ocr_cache_materialization_and_source_replay_v1.md"
LOCK = ROOT / "reports/finhrm_qp4_post_question_surface_development_ocr_cache_materialization_and_source_replay_lock_v1.json"
CACHE_ROOT = ROOT / "cache/mineru"
PROTECTED = re.compile(r"(?:^|[/_.-])(test|holdout|final|blind|validation|sealed)(?:[/_.-]|$)", re.I)
EARLY_ACCEPTED_WRONG_LAYERS = {
    "QUESTION_PLAN",
    "CANDIDATE_POPULATION_OR_RETRIEVAL",
    "SEMANTIC_BINDING_OR_PROGRAM_SYNTHESIS",
}


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    if PROTECTED.search(path.resolve().as_posix()):
        raise ValueError(f"Protected input is forbidden: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _digest(path),
    }


def _verify_authorization(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    lock = _load(PRECOMMIT_LOCK)
    execution = config["authorized_execution_contract"]
    if not lock.get("precommit_gate_passed"):
        raise ValueError("Source/retrieval roadmap-review precommit is not closed")
    if lock.get("authorized_next_step") != execution["step"]:
        raise ValueError("Materialization authorization mismatch")
    if lock.get("runtime_file_whitelist") != [] or execution["materialization_phase"]["runtime_code_change_allowed"]:
        raise ValueError("Runtime changes are forbidden for cache materialization")
    for name, parent in config["parents"].items():
        path = ROOT / parent["path"]
        if not path.is_file() or _digest(path) != parent["sha256"]:
            raise ValueError(f"Frozen parent hash mismatch: {name}")
    for relative, expected in config["frozen_runtime_hashes"].items():
        if _digest(ROOT / relative) != expected:
            raise ValueError(f"Runtime hash mismatch before materialization: {relative}")
    checkpoint = ROOT / config["checkpoint"]["path"]
    if _digest(checkpoint) != config["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint hash mismatch")

    diagnostic = _load(ROOT / config["parents"]["diagnostic_report"]["path"])
    baseline = _load(ROOT / config["parents"]["actual_plan_source_baseline"]["path"])
    cohort = _load(ROOT / config["parents"]["actual_plan_source_cohort"]["path"])
    predicate = config["review_evidence"]["selection_predicate"]
    matching_strata = [
        row
        for row in diagnostic["source_or_retrieval_strata_outside_current_scope"]
        if row["track"] == predicate["track"]
        and row["group"] == predicate["group"]
        and row["first_failing_layer"] == predicate["first_failing_layer"]
    ]
    if len(matching_strata) != 1 or not matching_strata[0]["generic_evidence"]:
        raise ValueError("Frozen generic source stratum is missing")
    baseline_rows = [
        row for row in baseline["results"] if row["dimension"] == predicate["group"]
    ]
    anchor_by_source = {row["source_sha256"]: row for row in cohort["anchors"]}
    selected: list[dict[str, Any]] = []
    expected_error = config["review_evidence"]["observed_source_error_message"]
    for row in baseline_rows:
        anchor = anchor_by_source.get(row["source_sha256"])
        if anchor is None:
            raise ValueError("Frozen baseline source is absent from the cohort")
        source = ROOT / anchor["source_path"]
        if PROTECTED.search(source.resolve().as_posix()):
            raise ValueError(f"Protected source is forbidden: {source}")
        if _digest(source) != anchor["source_sha256"]:
            raise ValueError("Frozen source hash mismatch")
        error = row["base"].get("error", {})
        if error.get("type") != config["review_evidence"]["observed_source_error_type"] or error.get("message") != expected_error:
            raise ValueError("Selection attempted to include a non-matching source failure")
        selected.append({"baseline": row, "anchor": anchor})
    if len(selected) != execution["expected_source_documents"]:
        raise ValueError("Selected source count does not match the precommit")
    if len(selected) * 2 != execution["expected_observations"]:
        raise ValueError("Selected observation count does not match the precommit")
    return diagnostic, cohort, selected


class _Timeout:
    def __init__(self, seconds: int) -> None:
        self.seconds = seconds
        self.previous: Any = None

    def __enter__(self) -> None:
        def handler(signum: int, frame: Any) -> None:
            raise TimeoutError(f"Source query exceeded {self.seconds} seconds")

        self.previous = signal.signal(signal.SIGALRM, handler)
        signal.alarm(self.seconds)
        return None

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self.previous)


def _materialize_question(engine: Any, question: str, source: Path, seconds: int) -> dict[str, Any]:
    events: list[list[Any]] = []

    def progress(stage: str, elapsed: float) -> None:
        events.append([stage, elapsed])

    started = time.perf_counter()
    try:
        with _Timeout(seconds):
            answer = engine.answer(
                question,
                source,
                mineru_auto_generate=True,
                progress_callback=progress,
            )
        return {
            "outcome": "RETURNED",
            "source_action": str(answer.source_action),
            "elapsed_seconds": time.perf_counter() - started,
            "stages": [name for name, _ in answer.stage_timings],
        }
    except Exception as exc:
        return {
            "outcome": "ERROR",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "elapsed_seconds": time.perf_counter() - started,
            "stages": [name for name, _ in events],
        }


def _source_cache_artifacts(source: Path) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for directory in sorted(CACHE_ROOT.glob(f"{source.stem}-*")):
        if not directory.is_dir():
            continue
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            payload_pages: list[int] = []
            if path.name.endswith("_content_list.json"):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(payload, list):
                        payload_pages = sorted({
                            int(item["page_idx"])
                            for item in payload
                            if isinstance(item, dict)
                            and isinstance(item.get("page_idx"), int)
                            and item["page_idx"] >= 0
                        })
                except (OSError, ValueError, TypeError):
                    payload_pages = []
            artifacts.append({
                **_artifact(path),
                "content_list": path.name.endswith("_content_list.json"),
                "zero_based_pages": payload_pages,
            })
    return artifacts


def _trace_signature(observation: dict[str, Any]) -> tuple[Any, ...] | None:
    trace = observation["trace"]
    if trace["source_outcome"] != "RETURNED":
        return None
    program = trace["predicted_operation_and_program"] or {}
    return (
        trace["source_action"],
        program.get("operation"),
        program.get("value"),
        json.dumps(trace["selected_physical_evidence_signature"], ensure_ascii=False, sort_keys=True),
    )


def run(
    materialization_timeout_seconds: int = 1800,
    *,
    replay_only: bool = False,
) -> dict[str, Any]:
    from hrm_finance_qa.document_hrm import DocumentHRMBundle
    from hrm_finance_qa.document_reasoning_engine import DocumentHRMReasoningEngine
    from scripts.evaluate_qp3_core_s4_safe_replay import build_report as build_safe

    config = _load(PRECOMMIT)
    diagnostic, cohort, selected = _verify_authorization(config)
    checkpoint = ROOT / config["checkpoint"]["path"]
    bundle = DocumentHRMBundle.load(checkpoint, device="cpu")
    engine = DocumentHRMReasoningEngine(bundle)
    materialization_rows: list[dict[str, Any]] = []
    cache_manifest: list[dict[str, Any]] = []

    if replay_only:
        previous = _load(REPORT)
        materialization_rows = previous["materialization"]["queries"]
    else:
        for position, item in enumerate(selected, start=1):
            anchor = item["anchor"]
            source = ROOT / anchor["source_path"]
            pair = anchor["pairs"][config["review_evidence"]["selection_predicate"]["group"]]
            print(f"[materialize {position}/{len(selected)}] development source", flush=True)
            query_rows = []
            for label in ("base", "mutated"):
                result = _materialize_question(
                    engine,
                    pair[f"{label}_question"],
                    source,
                    materialization_timeout_seconds,
                )
                query_rows.append({"variant": label, **result})
            materialization_rows.append({
                "source_sha256": anchor["source_sha256"],
                "company_key": anchor["company_key"],
                "document_layout_family": anchor["document_layout_family"],
                "queries": query_rows,
            })
    for item in selected:
        anchor = item["anchor"]
        source = ROOT / anchor["source_path"]
        artifacts = _source_cache_artifacts(source)
        cache_manifest.append({
            "source_sha256": anchor["source_sha256"],
            "artifacts": artifacts,
        })
        matching = next(
            row for row in materialization_rows
            if row["source_sha256"] == anchor["source_sha256"]
        )
        matching["cache_artifacts"] = len(artifacts)

    capture = _CandidateCaptureBundle(bundle)
    replay_engine = DocumentHRMReasoningEngine(capture)
    replay_seconds = int(cohort["selection_contract"]["per_query_timeout_seconds"])
    observations: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    false_accepts = 0
    paired_pass = 0
    for position, item in enumerate(selected, start=1):
        anchor = item["anchor"]
        baseline = item["baseline"]
        print(f"[artifact-only replay {position}/{len(selected)}] development source", flush=True)
        pair_observations: list[dict[str, Any]] = []
        for label in ("base", "mutated"):
            observation = _actual_result(
                replay_engine,
                capture,
                baseline,
                anchor,
                label,
                replay_seconds,
            )
            if _required_proof_action_passed(observation, "ANSWER"):
                observation["first_failing_layer"] = "PASS"
            observations.append(observation)
            pair_observations.append(observation)
            replay_rows.append({
                "source_sha256": anchor["source_sha256"],
                "company_key": anchor["company_key"],
                "document_layout_family": anchor["document_layout_family"],
                "variant": label,
                "first_failing_layer": observation["first_failing_layer"],
                "source_outcome": observation["trace"]["source_outcome"],
                "source_action": observation["trace"].get("source_action"),
                "verification_state": observation["trace"].get("verification_state"),
            })
        same_winner = (
            _trace_signature(pair_observations[0]) is not None
            and _trace_signature(pair_observations[0]) == _trace_signature(pair_observations[1])
        )
        false_accepts += int(pair_observations[0]["first_failing_layer"] == "PASS" and same_winner)
        paired_pass += int(all(row["first_failing_layer"] == "PASS" for row in pair_observations))

    source_summary = _source_summary(observations)
    source_errors = sum(row["trace"]["source_outcome"] == "ERROR" for row in observations)
    accepted_wrong = sum(
        row["trace"].get("source_action") == "ANSWER"
        and row["first_failing_layer"] in EARLY_ACCEPTED_WRONG_LAYERS
        for row in observations
    )
    cache_files = [artifact for row in cache_manifest for artifact in row["artifacts"]]
    compiler = _compiler_measurement()
    safe = build_safe()
    safe_results = safe["results"]
    regression = _run_regression()
    runtime_after = {
        relative: _digest(ROOT / relative)
        for relative in config["frozen_runtime_hashes"]
    }
    required = config["authorized_execution_contract"]["required_gates"]
    checks = {
        "all_selected_source_hashes_match": len(selected) == config["authorized_execution_contract"]["expected_source_documents"],
        "all_sources_have_structured_cache": all(
            row["artifacts"] and all(item["content_list"] for item in row["artifacts"])
            for row in cache_manifest
        ),
        "all_materialized_artifacts_are_reusable_with_auto_generation_disabled": source_errors == required["source_load_or_extraction_failures"],
        "source_load_or_extraction_failures": source_errors == required["source_load_or_extraction_failures"],
        "accepted_wrong": accepted_wrong == required["accepted_wrong"],
        "counterfactual_false_accepts": false_accepts == required["counterfactual_false_accepts"],
        "compiler_inventory": compiler["full_inventory"] == required["compiler_inventory"],
        "regression": regression["passed"],
        "gold_execution": f"{safe_results['gold_execution_correct']}/{safe_results['gold_total']}" == required["gold_execution"],
        "candidate_permutation": f"{safe_results['permutation_exact']}/{safe_results['permutation_total']}" == required["candidate_permutation"],
        "frozen_counterfactuals": safe_results["counterfactual_generated"] == required["frozen_counterfactuals"],
        "frozen_counterfactual_false_accepts": safe_results["counterfactual_false_accepts"] == required["frozen_counterfactual_false_accepts"],
        "frozen_safe_replay_gate": safe["safe_replay_gate_passed"] == required["frozen_safe_replay_gate"],
        "runtime_unchanged": runtime_after == config["frozen_runtime_hashes"],
        "checkpoint_unchanged": _digest(checkpoint) == config["checkpoint"]["sha256"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    gate = not failed
    generic_downstream = [
        row
        for row in source_summary["strata"]
        if row["generic_evidence"] and row["first_failing_layer"] != "PASS"
    ]
    source_availability_failures = {
        "all_materialized_artifacts_are_reusable_with_auto_generation_disabled",
        "source_load_or_extraction_failures",
    }
    if not gate and set(failed).issubset(source_availability_failures):
        next_step = "QP4-POST-QUESTION-SURFACE-POST-CACHE-SOURCE-ERROR-DIAGNOSTIC-REVIEW-PRECOMMIT: review the remaining lazy page-selection/materialization failures without changing runtime or treating one source/layout as generic evidence."
    elif not gate:
        next_step = "Reclose QP4 development OCR-cache materialization/source replay without changing runtime."
    elif generic_downstream:
        next_step = "QP4-POST-QUESTION-SURFACE-POST-CACHE-RESIDUAL-DIAGNOSTIC-REVIEW-PRECOMMIT: freeze only generic downstream earliest-layer strata before any runtime edit."
    else:
        next_step = "QP4-POST-QUESTION-SURFACE-RESIDUAL-GATE-RECLOSURE-PRECOMMIT: review QP4 completion gates without changing runtime."
    return {
        "schema": "finhrm-qp4-post-question-surface-development-ocr-cache-materialization-and-source-replay-v1",
        "created_at": "2026-08-23",
        "classification": "development_only_qp4_lazy_page_selective_ocr_cache_materialization_runtime_unchanged",
        "status": "COMPLETE" if gate else "GATE_FAILED",
        "implementation_gate_passed": gate,
        "failed_gates": failed,
        "checks": checks,
        "selection": {
            "predicate": config["review_evidence"]["selection_predicate"],
            "observations": len(observations),
            "companies": len({item["anchor"]["company_key"] for item in selected}),
            "sources": len({item["anchor"]["source_sha256"] for item in selected}),
            "real_layouts": len({item["anchor"]["document_layout_family"] for item in selected}),
            "question_triggered": True,
            "whole_corpus_ocr": False,
        },
        "materialization": {
            "queries": materialization_rows,
            "cache_artifacts": len(cache_files),
            "content_list_artifacts": sum(item["content_list"] for item in cache_files),
        },
        "artifact_only_replay": {
            "queries": len(observations),
            "source_errors": source_errors,
            "paired_pass": paired_pass,
            "accepted_wrong": accepted_wrong,
            "counterfactual_false_accepts": false_accepts,
            "by_first_failing_layer": dict(sorted(Counter(row["first_failing_layer"] for row in observations).items())),
            "rows": replay_rows,
            "groups": source_summary["groups"],
            "strata": source_summary["strata"],
            "generic_downstream_strata": generic_downstream,
        },
        "compiler_control": compiler["full_inventory"],
        "regression": regression,
        "safe_replay": {
            "gate": safe["safe_replay_gate_passed"],
            "actions": safe_results["gold_actions"],
            "execution": f"{safe_results['gold_execution_correct']}/{safe_results['gold_total']}",
            "permutation": f"{safe_results['permutation_exact']}/{safe_results['permutation_total']}",
            "counterfactuals": safe_results["counterfactual_generated"],
            "counterfactual_false_accepts": safe_results["counterfactual_false_accepts"],
        },
        "cache_manifest": cache_manifest,
        "runtime_changed": False,
        "runtime_edit_authorized": False,
        "runtime_hashes": runtime_after,
        "checkpoint_sha256": config["checkpoint"]["sha256"],
        "artifact_policy": config["artifact_policy"],
        "invariants": config["invariants"],
        "qp4_complete": False,
        "qp5_authorized": False,
        "phase_d": "locked",
        "authorized_next_step": next_step,
        "parent_diagnostic_safe_replay_retained": diagnostic["retained_safe_replay"],
    }


def _markdown(report: dict[str, Any]) -> str:
    replay = report["artifact_only_replay"]
    selection = report["selection"]
    return (
        "# QP4 lazy development OCR-cache materialization and source replay v1\n\n"
        f"- Gate: `{'PASS' if report['implementation_gate_passed'] else 'FAIL'}`; runtime unchanged.\n"
        f"- Selection: `{selection['observations']}` question-triggered observations, `{selection['companies']}` companies, `{selection['sources']}` PDFs, `{selection['real_layouts']}` real layouts; whole-corpus OCR is false.\n"
        f"- Structured cache artifacts: `{report['materialization']['content_list_artifacts']}`; artifact-only replay source errors: `{replay['source_errors']}`.\n"
        f"- Replay earliest layers: `{replay['by_first_failing_layer']}`; paired PASS `{replay['paired_pass']}/4`; accepted-wrong `{replay['accepted_wrong']}`; false accepts `{replay['counterfactual_false_accepts']}`.\n"
        f"- Compiler `{report['compiler_control']}`; regression `{report['regression']['observed']}/{report['regression']['expected']}`; frozen execution/permutation `{report['safe_replay']['execution']}/{report['safe_replay']['permutation']}`; counterfactual false accepts `{report['safe_replay']['counterfactual_false_accepts']}`.\n"
        "- Cache availability is not semantic proof. Missing proof remains ABSTAIN and contradiction remains REJECT. QP4 remains incomplete; QP5 and Phase D remain locked.\n"
    )


def write_outputs(report: dict[str, Any]) -> None:
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN.write_text(_markdown(report), encoding="utf-8")
    artifacts = [_artifact(PRECOMMIT), _artifact(Path(__file__).resolve()), _artifact(REPORT), _artifact(MARKDOWN)]
    lock = {
        "schema": "finhrm-qp4-post-question-surface-development-ocr-cache-materialization-and-source-replay-lock-v1",
        "created_at": report["created_at"],
        "classification": report["classification"],
        "status": "complete" if report["implementation_gate_passed"] else "failed",
        "implementation_gate_passed": report["implementation_gate_passed"],
        "failed_gates": report["failed_gates"],
        "selection": report["selection"],
        "artifact_only_replay": {
            key: report["artifact_only_replay"][key]
            for key in (
                "queries",
                "source_errors",
                "paired_pass",
                "accepted_wrong",
                "counterfactual_false_accepts",
                "by_first_failing_layer",
                "generic_downstream_strata",
            )
        },
        "safe_replay": report["safe_replay"],
        "cache_manifest": report["cache_manifest"],
        "runtime_changed": False,
        "runtime_edit_authorized": False,
        "runtime_hashes": report["runtime_hashes"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "artifact_policy": report["artifact_policy"],
        "artifacts": artifacts,
        "qp4_complete": False,
        "qp5_authorized": False,
        "phase_d": "locked",
        "authorized_next_step": report["authorized_next_step"],
    }
    LOCK.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-only", action="store_true")
    parser.add_argument("--materialization-timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    report = run(
        materialization_timeout_seconds=args.materialization_timeout_seconds,
        replay_only=args.replay_only,
    )
    write_outputs(report)
    print(json.dumps({
        "gate": report["implementation_gate_passed"],
        "failed_gates": report["failed_gates"],
        "selection": report["selection"],
        "cache_artifacts": report["materialization"]["cache_artifacts"],
        "artifact_only_replay": {
            key: report["artifact_only_replay"][key]
            for key in (
                "source_errors",
                "paired_pass",
                "accepted_wrong",
                "counterfactual_false_accepts",
                "by_first_failing_layer",
            )
        },
        "authorized_next_step": report["authorized_next_step"],
    }, ensure_ascii=False, sort_keys=True))
    return 0 if report["implementation_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
