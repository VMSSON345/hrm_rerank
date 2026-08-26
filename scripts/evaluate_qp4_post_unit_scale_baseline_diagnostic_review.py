#!/usr/bin/env python3
"""Run the frozen no-runtime QP4 residual diagnostic review."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import signal
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

PRECOMMIT = ROOT / "config/finhrm_qp4_post_unit_scale_baseline_diagnostic_review_precommit_v1.json"
PRECOMMIT_REPORT = ROOT / "reports/finhrm_qp4_post_unit_scale_baseline_diagnostic_review_precommit_v1.json"
PRECOMMIT_LOCK = ROOT / "reports/finhrm_qp4_post_unit_scale_baseline_diagnostic_review_precommit_lock_v1.json"
ADD_CONFIG = ROOT / "config/finhrm_qp4_development_cohort_v1.json"
ACTUAL_CONFIG = ROOT / "config/finhrm_qp4_actual_plan_ratio_direction_source_cohort_v1.json"
ADD_BASELINE_RAW = ROOT / "reports/finhrm_qp4_post_unit_scale_source_backed_baseline_v1.json"
ACTUAL_BASELINE_RAW = ROOT / "reports/finhrm_qp4_post_unit_scale_actual_plan_source_baseline_v1.json"
REFRESH_REPORT = ROOT / "reports/finhrm_qp4_post_unit_scale_development_baseline_refresh_v1.json"
SAFE_REPLAY = ROOT / "reports/finhrm_qp4_unit_scale_proof_contract_safe_replay_v1.json"
RAW = ROOT / "reports/finhrm_qp4_post_unit_scale_baseline_diagnostic_review_raw_v1.json"
REPORT = ROOT / "reports/finhrm_qp4_post_unit_scale_baseline_diagnostic_review_v1.json"
MARKDOWN = ROOT / "reports/finhrm_qp4_post_unit_scale_baseline_diagnostic_review_v1.md"
LOCK = ROOT / "reports/finhrm_qp4_post_unit_scale_baseline_diagnostic_review_lock_v1.json"
TEST = ROOT / "tests/test_qp4_post_unit_scale_baseline_diagnostic_review.py"

LAYERS = (
    "QUESTION_PLAN",
    "SOURCE_LOAD_OR_EXTRACTION",
    "CANDIDATE_POPULATION_OR_RETRIEVAL",
    "SEMANTIC_BINDING_OR_PROGRAM_SYNTHESIS",
    "SYMBOLIC_PROOF_OR_ACTION",
)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
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


def _verify_precommit(config: dict[str, Any]) -> None:
    report = _load(PRECOMMIT_REPORT)
    lock = _load(PRECOMMIT_LOCK)
    if not report.get("precommit_gate_passed") or not lock.get("precommit_gate_passed"):
        raise ValueError("Diagnostic precommit is not closed")
    if lock.get("authorized_next_step") != config.get("authorized_next_step_after_lock"):
        raise ValueError("Diagnostic authorization mismatch")
    contract = config["diagnostic_contract"]
    if contract["runtime_file_whitelist"] or lock["runtime_file_whitelist_for_next_step"]:
        raise ValueError("Diagnostic runtime whitelist must be empty")
    if contract["allowed_new_evaluator"] != Path(__file__).resolve().relative_to(ROOT).as_posix():
        raise ValueError("This evaluator is not pre-authorized")
    for name, item in config["parents"].items():
        path = ROOT / item["path"]
        if not path.is_file() or _digest(path) != item["sha256"]:
            raise ValueError(f"Frozen parent hash mismatch: {name}")
    checkpoint = ROOT / config["checkpoint"]["path"]
    if _digest(checkpoint) != config["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    for relative, expected in config["frozen_runtime_hashes"].items():
        if _digest(ROOT / relative) != expected:
            raise ValueError(f"Runtime hash mismatch: {relative}")


class _Timeout:
    def __init__(self, seconds: int) -> None:
        self.seconds = seconds
        self.previous: Any = None

    def __enter__(self) -> None:
        def handler(signum: int, frame: Any) -> None:
            raise TimeoutError(f"Diagnostic query exceeded {self.seconds} seconds")

        self.previous = signal.signal(signal.SIGALRM, handler)
        signal.alarm(self.seconds)
        return None

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self.previous)


class _CandidateCaptureBundle:
    """Observe the exact candidate population passed to the frozen predictor."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.last_candidates: tuple[Any, ...] = ()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def predict(self, question: str, candidates: Sequence[Any], **kwargs: Any) -> Any:
        self.last_candidates = tuple(candidates)
        return self.inner.predict(question, candidates, **kwargs)


def _candidate_trace(candidate: Any) -> dict[str, Any]:
    return {
        "source_name": str(candidate.source_name),
        "page": candidate.page,
        "row_label": str(candidate.row_label),
        "column_label": str(candidate.column_label),
        "period": str(candidate.period),
        "source_span": list(candidate.source_span or ()),
        "value": str(candidate.value),
        "unit": str(candidate.unit),
        "extraction_method": str(candidate.extraction_method),
        "table_id": str(candidate.table_id),
        "scope": str(candidate.scope),
        "entity_id": str(candidate.entity_id),
    }


def _answer_trace(answer: Any) -> dict[str, Any]:
    selected = [
        _candidate_trace(item.candidate)
        for item in sorted(answer.evidence, key=lambda item: item.operand_number)
    ]
    dag = None
    if answer.program_dag is not None:
        dag = {
            "output_node": answer.program_dag.output_node,
            "nodes": [
                {
                    "node_id": node.node_id,
                    "operation": node.operation,
                    "inputs": [
                        {"kind": item.kind, "reference": item.reference}
                        for item in node.inputs
                    ],
                }
                for node in answer.program_dag.nodes
            ],
        }
    return {
        "source_outcome": "RETURNED",
        "retrieved_chunk_count": len(answer.retrieved_chunks),
        "retrieved_chunk_citations": [chunk.citation() for chunk in answer.retrieved_chunks],
        "candidate_count": int(answer.candidate_count),
        "predicted_operation_and_program": {
            "operation": str(answer.program.operation),
            "operands": list(answer.program.operands),
            "value": None if answer.value is None else str(answer.value),
            "unit": str(answer.unit),
            "dag": dag,
        },
        "selected_physical_evidence_signature": selected,
        "verification_state": str(answer.verification_state),
        "proof_obligations": [list(item) for item in answer.proof_obligations],
        "verification_reasons": list(answer.verification_reasons),
        "source_action": str(answer.source_action),
        "stage_timings": [[name, seconds] for name, seconds in answer.stage_timings],
    }


def _error_trace(exc: Exception, events: list[list[Any]]) -> dict[str, Any]:
    return {
        "source_outcome": "ERROR",
        "error": {"type": type(exc).__name__, "message": str(exc)},
        "retrieved_chunk_count": None,
        "retrieved_chunk_citations": [],
        "candidate_count": None,
        "predicted_operation_and_program": None,
        "selected_physical_evidence_signature": [],
        "verification_state": None,
        "proof_obligations": [],
        "verification_reasons": [],
        "source_action": None,
        "stage_timings": events,
    }


def _run_question(
    engine: Any,
    capture: _CandidateCaptureBundle,
    question: str,
    source_path: Path,
    seconds: int,
) -> tuple[Any | None, tuple[Any, ...], dict[str, Any]]:
    capture.last_candidates = ()
    events: list[list[Any]] = []

    def progress(stage: str, elapsed: float) -> None:
        events.append([stage, elapsed])

    try:
        with _Timeout(seconds):
            answer = engine.answer(
                question,
                source_path,
                mineru_auto_generate=False,
                progress_callback=progress,
            )
        return answer, capture.last_candidates, _answer_trace(answer)
    except Exception as exc:
        return None, capture.last_candidates, _error_trace(exc, events)


def _decimal_equal(left: Any, right: Any) -> bool:
    try:
        return Decimal(str(left)) == Decimal(str(right))
    except Exception:
        return False


def _add_support(candidates: Sequence[Any], expected: list[list[Any]]) -> tuple[bool, list[dict[str, Any]]]:
    from scripts.evaluate_qp4_development_baseline import _physical_signature

    signatures = {_physical_signature(candidate): candidate for candidate in candidates}
    matches: list[dict[str, Any]] = []
    for raw in expected:
        signature = tuple(raw[:5]) + (tuple(raw[5]),) + tuple(raw[6:])
        candidate = signatures.get(signature)
        if candidate is not None:
            matches.append(_candidate_trace(candidate))
    return len(matches) == len(expected), matches


def _actual_operand_match(candidate: Any, anchor: dict[str, Any], operand: str) -> bool:
    expected = anchor["evidence"][operand]
    return (
        str(candidate.source_name) == str(anchor["source_name"])
        and candidate.page == anchor["page"]
        and _decimal_equal(candidate.value, expected["value"])
    )


def _actual_support(
    candidates: Sequence[Any], anchor: dict[str, Any], operands: Sequence[str]
) -> tuple[bool, list[dict[str, Any]]]:
    matches: list[dict[str, Any]] = []
    for operand in operands:
        candidate = next(
            (item for item in candidates if _actual_operand_match(item, anchor, operand)),
            None,
        )
        if candidate is not None:
            matches.append(_candidate_trace(candidate))
    return len(matches) == len(operands), matches


def _expected_role_year(operand: str) -> tuple[str, str]:
    role, year = operand.rsplit("_", 1)
    return ("PLAN" if role == "plan" else "ACTUAL", year)


def _actual_plan_passes(plan_projection: dict[str, Any], program: dict[str, Any]) -> bool:
    if plan_projection["operation"] != program["operation"]:
        return False
    slots = plan_projection["operand_slots"]
    if len(slots) != len(program["operands"]):
        return False
    for slot, operand in zip(slots, program["operands"]):
        role, year = _expected_role_year(operand)
        if str(slot["column_role"]).upper() != role or str(slot["year"]) != year:
            return False
    return True


def _selected_add_matches(trace: dict[str, Any], expected: dict[str, Any]) -> bool:
    from scripts.evaluate_qp4_development_baseline import _physical_signature

    observed = sorted(
        _physical_signature(item) for item in trace["selected_physical_evidence_signature"]
    )
    frozen = sorted(
        tuple(item[:5]) + (tuple(item[5]),) + tuple(item[6:])
        for item in expected["evidence"]
    )
    return observed == frozen


def _selected_actual_matches(
    answer: Any, anchor: dict[str, Any], operands: Sequence[str]
) -> bool:
    selected = [
        item.candidate for item in sorted(answer.evidence, key=lambda item: item.operand_number)
    ]
    return len(selected) == len(operands) and all(
        _actual_operand_match(candidate, anchor, operand)
        for candidate, operand in zip(selected, operands)
    )


def _classify_add(
    plan_passed: bool,
    answer: Any | None,
    trace: dict[str, Any],
    candidate_support: bool,
    expected: dict[str, Any],
) -> str:
    if not plan_passed:
        return "QUESTION_PLAN"
    if trace["source_outcome"] == "ERROR":
        return "SOURCE_LOAD_OR_EXTRACTION"
    if not candidate_support or not trace["candidate_count"]:
        return "CANDIDATE_POPULATION_OR_RETRIEVAL"
    predicted = trace["predicted_operation_and_program"]
    binding_passed = (
        predicted["operation"] == expected["operation"]
        and _decimal_equal(predicted["value"], expected["value"])
        and predicted["unit"] == expected["unit"]
        and _selected_add_matches(trace, expected)
    )
    if not binding_passed:
        return "SEMANTIC_BINDING_OR_PROGRAM_SYNTHESIS"
    if trace["source_action"] != expected["action"] or trace["verification_state"] != "VALID":
        return "SYMBOLIC_PROOF_OR_ACTION"
    return "PASS"


def _classify_actual(
    plan_passed: bool,
    answer: Any | None,
    trace: dict[str, Any],
    candidate_support: bool,
    anchor: dict[str, Any],
    program: dict[str, Any],
    expected_value: Decimal,
) -> str:
    if not plan_passed:
        return "QUESTION_PLAN"
    if trace["source_outcome"] == "ERROR":
        return "SOURCE_LOAD_OR_EXTRACTION"
    if not candidate_support or not trace["candidate_count"]:
        return "CANDIDATE_POPULATION_OR_RETRIEVAL"
    predicted = trace["predicted_operation_and_program"]
    binding_passed = (
        answer is not None
        and predicted["operation"] == program["operation"]
        and _decimal_equal(predicted["value"], expected_value)
        and _selected_actual_matches(answer, anchor, program["operands"])
    )
    if not binding_passed:
        return "SEMANTIC_BINDING_OR_PROGRAM_SYNTHESIS"
    if trace["source_action"] != "ANSWER" or trace["verification_state"] != "VALID":
        return "SYMBOLIC_PROOF_OR_ACTION"
    return "PASS"


def run_diagnostic(config: dict[str, Any], raw_path: Path = RAW) -> dict[str, Any]:
    from hrm_finance_qa.document_hrm import DocumentHRMBundle
    from hrm_finance_qa.document_reasoning_engine import DocumentHRMReasoningEngine
    from hrm_finance_qa.question_planner import compile_question_plan
    from scripts.evaluate_qp4_actual_plan_source_cohort import _expected_value
    from scripts.evaluate_qp4_development_baseline import _compiler_baseline, _semantic_projection

    _verify_precommit(config)
    add_config = _load(ADD_CONFIG)
    add_config["runtime_hashes"] = {
        path: config["frozen_runtime_hashes"][path] for path in add_config["runtime_hashes"]
    }
    compiler = _compiler_baseline(add_config)
    canonical = {row["family_id"]: row["question"] for row in compiler["canonical"]["details"]}
    compiler_results: list[dict[str, Any]] = []
    for row in compiler["surface"]["details"]:
        if row["semantic_invariant"]:
            continue
        expected = _semantic_projection(compile_question_plan(canonical[row["family_id"]]))
        observed = _semantic_projection(compile_question_plan(row["question"]))
        compiler_results.append({
            "track": "COMPILER_SURFACE",
            "group": row["category"],
            "family_id": row["family_id"],
            "selection_basis": "all records in a frozen generic failed surface category",
            "semantic_plan_projection": {"expected": expected, "observed": observed},
            "first_failing_layer": "QUESTION_PLAN" if observed != expected else "PASS",
        })

    checkpoint = ROOT / config["checkpoint"]["path"]
    capture = _CandidateCaptureBundle(DocumentHRMBundle.load(checkpoint, device="cpu"))
    engine = DocumentHRMReasoningEngine(capture)
    seconds = int(config["diagnostic_contract"]["per_query_timeout_seconds"])
    source_results: list[dict[str, Any]] = []

    add_raw = _load(ADD_BASELINE_RAW)
    add_rows = [row for row in add_raw["surface_results"] if not row["invariant"]]
    position = 0
    for row in add_rows:
        position += 1
        print(f"[{position}/39] ADD_SURFACE {row['category']}", flush=True)
        expected_plan = _semantic_projection(compile_question_plan(row["parent_question"]))
        observed_plan = _semantic_projection(compile_question_plan(row["question"]))
        plan_passed = observed_plan == expected_plan
        answer, candidates, trace = _run_question(
            engine, capture, row["question"], ROOT / row["source_path"], seconds
        )
        support, matches = _add_support(candidates, row["parent_signature"]["evidence"])
        layer = _classify_add(
            plan_passed, answer, trace, support, row["parent_signature"]
        )
        plan_trace = {
            "expected": expected_plan,
            "observed": observed_plan,
            "passed": plan_passed,
        }
        trace["semantic_plan_projection"] = plan_trace
        source_results.append({
            "track": "ADD_SURFACE",
            "group": row["category"],
            "company_key": row["company_key"],
            "source_key": row["source_key"],
            "layout_family": row["layout_family"],
            "selection_basis": "all records in a frozen generic failed ADD surface category",
            "semantic_plan_projection": plan_trace,
            "candidate_support": {
                "all_expected_physical_evidence_observed": support,
                "matched_expected_candidates": matches,
                "gold_used_post_hoc_only": True,
            },
            "trace": trace,
            "first_failing_layer": layer,
        })

    actual_config = _load(ACTUAL_CONFIG)
    anchors = {row["source_sha256"]: row for row in actual_config["anchors"]}
    actual_raw = _load(ACTUAL_BASELINE_RAW)
    for baseline_row in actual_raw["results"]:
        anchor = anchors[baseline_row["source_sha256"]]
        pair = anchor["pairs"][baseline_row["dimension"]]
        for label in ("base", "mutated"):
            position += 1
            print(f"[{position}/39] ACTUAL_PLAN {baseline_row['dimension']} {label}", flush=True)
            question = pair[f"{label}_question"]
            program = pair[f"{label}_program"]
            projection = _semantic_projection(compile_question_plan(question))
            plan_passed = _actual_plan_passes(projection, program)
            answer, candidates, trace = _run_question(
                engine, capture, question, ROOT / anchor["source_path"], seconds
            )
            support, matches = _actual_support(candidates, anchor, program["operands"])
            expected_value = _expected_value(anchor, program)
            layer = _classify_actual(
                plan_passed,
                answer,
                trace,
                support,
                anchor,
                program,
                expected_value,
            )
            plan_trace = {
                "observed": projection,
                "expected_operation": program["operation"],
                "expected_operands": program["operands"],
                "passed": plan_passed,
            }
            trace["semantic_plan_projection"] = plan_trace
            source_results.append({
                "track": "ACTUAL_PLAN",
                "group": baseline_row["dimension"],
                "query_variant": label,
                "company_key": baseline_row["company_key"],
                "source_key": baseline_row["source_sha256"],
                "layout_family": baseline_row["document_layout_family"],
                "table_layout_family": baseline_row["table_layout_family"],
                "selection_basis": "all base and mutated queries in a frozen failed ACTUAL/PLAN dimension",
                "semantic_plan_projection": plan_trace,
                "candidate_support": {
                    "all_expected_value_page_support_observed": support,
                    "matched_expected_candidates": matches,
                    "gold_used_post_hoc_only": True,
                },
                "expected_execution": {
                    "operation": program["operation"],
                    "value": str(expected_value),
                    "action": "ANSWER",
                },
                "trace": trace,
                "first_failing_layer": layer,
            })

    raw = {
        "schema": "finhrm-qp4-post-unit-scale-baseline-diagnostic-review-raw-v1",
        "created_at": config["created_at"],
        "classification": "development_only_qp4_residual_diagnostic_runtime_unchanged",
        "inputs": {
            "precommit": PRECOMMIT.relative_to(ROOT).as_posix(),
            "precommit_sha256": _digest(PRECOMMIT),
            "checkpoint_sha256": _digest(checkpoint),
            "source_run_boundary": ["question", "source PDF path"],
            "gold_candidate_or_program_passed_to_engine": False,
            "gold_used_only_after_observation": True,
            "mineru_auto_generate": False,
            "per_query_timeout_seconds": seconds,
        },
        "compiler_results": compiler_results,
        "source_results": source_results,
        "runtime_hashes": {
            path: _digest(ROOT / path) for path in config["frozen_runtime_hashes"]
        },
        "invariants": config["invariants"],
    }
    raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return raw


def _source_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["track"], row["group"])].append(row)
    groups: dict[str, Any] = {}
    strata: list[dict[str, Any]] = []
    for (track, group), values in sorted(grouped.items()):
        key = f"{track}:{group}"
        layers = Counter(row["first_failing_layer"] for row in values)
        groups[key] = {
            "queries": len(values),
            "companies": len({row["company_key"] for row in values}),
            "sources": len({row["source_key"] for row in values}),
            "real_layouts": len({row["layout_family"] for row in values}),
            "by_first_failing_layer": dict(sorted(layers.items())),
            "mixed_first_failure_group": len(layers) > 1,
        }
        for layer in sorted(layers):
            layer_rows = [row for row in values if row["first_failing_layer"] == layer]
            companies = len({row["company_key"] for row in layer_rows})
            sources = len({row["source_key"] for row in layer_rows})
            layouts = len({row["layout_family"] for row in layer_rows})
            strata.append({
                "track": track,
                "group": group,
                "first_failing_layer": layer,
                "queries": len(layer_rows),
                "companies": companies,
                "sources": sources,
                "real_layouts": layouts,
                "generic_evidence": companies >= 3 and sources >= 3 and layouts >= 3,
            })
    return {
        "queries": len(rows),
        "source_errors": sum(row["trace"]["source_outcome"] == "ERROR" for row in rows),
        "by_first_failing_layer": dict(sorted(Counter(row["first_failing_layer"] for row in rows).items())),
        "groups": groups,
        "strata": strata,
    }


def build_report(config_path: Path = PRECOMMIT, raw_path: Path = RAW) -> dict[str, Any]:
    config = _load(config_path)
    _verify_precommit(config)
    raw = _load(raw_path)
    if raw.get("schema") != "finhrm-qp4-post-unit-scale-baseline-diagnostic-review-raw-v1":
        raise ValueError("Unexpected diagnostic raw schema")
    if raw["inputs"]["precommit_sha256"] != _digest(config_path):
        raise ValueError("Diagnostic raw precommit hash mismatch")
    if raw["runtime_hashes"] != {
        path: _digest(ROOT / path) for path in config["frozen_runtime_hashes"]
    }:
        raise ValueError("Runtime changed during diagnostic")

    compiler_rows = raw["compiler_results"]
    source_rows = raw["source_results"]
    checks = {
        "all_18_compiler_residuals_classified": len(compiler_rows) == 18,
        "all_39_source_residual_queries_replayed_and_classified": len(source_rows) == 39,
        "every_query_has_exactly_one_first_failing_layer_or_pass": all(
            row["first_failing_layer"] in {*LAYERS, "PASS"}
            for row in [*compiler_rows, *source_rows]
        ),
        "source_errors_not_relabelled_as_abstention_pass": all(
            row["trace"]["source_outcome"] != "ERROR"
            or not row["semantic_plan_projection"]["passed"]
            or row["first_failing_layer"] == "SOURCE_LOAD_OR_EXTRACTION"
            for row in source_rows
        ),
        "post_hoc_gold_never_changes_engine_input_or_winner": (
            raw["inputs"]["gold_candidate_or_program_passed_to_engine"] is False
            and raw["inputs"]["gold_used_only_after_observation"] is True
            and all(row["candidate_support"]["gold_used_post_hoc_only"] for row in source_rows)
        ),
        "per_group_company_source_layout_counts_reported": True,
        "mixed_failure_groups_split_before_runtime_nomination": True,
        "runtime_changed": False,
        "runtime_edit_authorized": False,
        "qp4_complete": False,
        "qp5_authorized": False,
        "phase_d_unlocked": False,
    }
    failed = [name for name, value in checks.items() if value != config["diagnostic_gates"][name]]
    compiler_summary = {
        "residuals": len(compiler_rows),
        "by_group": {
            group: {
                "records": len(values),
                "families": len({row["family_id"] for row in values}),
                "by_first_failing_layer": dict(sorted(Counter(row["first_failing_layer"] for row in values).items())),
            }
            for group, values in sorted(
                (
                    (group, [row for row in compiler_rows if row["group"] == group])
                    for group in {row["group"] for row in compiler_rows}
                )
            )
        },
    }
    source_summary = _source_summary(source_rows)
    minimum = config["diagnostic_contract"]["minimum_generic_evidence"]
    planner_groups = [
        {
            "group": group,
            "families": row["families"],
            "first_failing_layer": "QUESTION_PLAN",
        }
        for group, row in compiler_summary["by_group"].items()
        if row["families"] >= minimum["compiler_family_groups"]
        and row["by_first_failing_layer"] == {"QUESTION_PLAN": row["records"]}
    ]
    planner_nomination = bool(planner_groups)
    source_outside_whitelist = [
        row for row in source_summary["strata"]
        if row["generic_evidence"]
        and row["first_failing_layer"] in {
            "SOURCE_LOAD_OR_EXTRACTION",
            "CANDIDATE_POPULATION_OR_RETRIEVAL",
        }
    ]
    next_step = (
        "QP4-QUESTION-SURFACE-IMPLEMENTATION-PRECOMMIT: freeze the smallest generic question_planner symbol scope for the isolated compiler surface groups; no runtime change is authorized by this diagnostic. Source/retrieval strata remain separately locked."
        if planner_nomination and not failed
        else "QP4-BASELINE-DIAGNOSTIC-REVIEW-RECLOSURE-PRECOMMIT: resolve diagnostic contract failures without changing runtime."
    )
    safe = _load(SAFE_REPLAY)["results"]
    return {
        "schema": "finhrm-qp4-post-unit-scale-baseline-diagnostic-review-report-v1",
        "created_at": config["created_at"],
        "classification": "development_only_qp4_residual_diagnostic_runtime_unchanged",
        "status": "DIAGNOSTIC_COMPLETE_RUNTIME_UNCHANGED" if not failed else "DIAGNOSTIC_GATE_FAILED_RUNTIME_UNCHANGED",
        "diagnostic_complete": not failed,
        "diagnostic_gate_passed": not failed,
        "failed_gates": failed,
        "checks": checks,
        "compiler": compiler_summary,
        "source": source_summary,
        "first_failing_layer_order": list(LAYERS),
        "implementation_precommit_candidate": {
            "nominated": planner_nomination and not failed,
            "target_file": "src/hrm_finance_qa/question_planner.py" if planner_nomination and not failed else None,
            "generic_failure_groups": planner_groups,
            "runtime_edit_authorized": False,
            "new_precommit_required": True,
        },
        "source_or_retrieval_strata_outside_current_whitelist": source_outside_whitelist,
        "safe_replay_retained": {
            "actions": safe["gold_actions"],
            "execution": f"{safe['gold_execution_correct']}/{safe['gold_total']}",
            "permutation": f"{safe['permutation_exact']}/{safe['permutation_total']}",
            "counterfactuals": safe["counterfactual_generated"],
            "counterfactual_false_accepts": safe["counterfactual_false_accepts"],
        },
        "runtime_changed": False,
        "runtime_edit_authorized": False,
        "runtime_hashes": config["frozen_runtime_hashes"],
        "checkpoint_sha256": config["checkpoint"]["sha256"],
        "invariants": config["invariants"],
        "qp4_complete": False,
        "qp5_authorized": False,
        "phase_d": "locked",
        "raw_diagnostic": _artifact(raw_path),
        "authorized_next_step": next_step,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# QP4 post-UNIT_SCALE baseline diagnostic review v1",
        "",
        f"- Diagnostic gate: `{'PASS' if report['diagnostic_gate_passed'] else 'FAIL'}`; runtime unchanged.",
        f"- Compiler residuals: `{report['compiler']['residuals']}`.",
        f"- Source residual queries: `{report['source']['queries']}`; observed source errors `{report['source']['source_errors']}`.",
        f"- First failing layers: `{report['source']['by_first_failing_layer']}`.",
        "",
        "| Source residual group | Queries | Companies | PDFs | Real layouts | First-layer strata |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, row in report["source"]["groups"].items():
        lines.append(
            f"| {name} | {row['queries']} | {row['companies']} | {row['sources']} | {row['real_layouts']} | {row['by_first_failing_layer']} |"
        )
    lines.extend([
        "",
        "Gold was used only after observation. DocumentSourceError and timeout are failures; safe abstention is not an answer. Rank, score and confidence are not proof.",
        "Runtime edit remains unauthorized. QP4 is incomplete; QP5 and Phase D remain locked.",
        "",
        "Next: " + report["authorized_next_step"],
        "",
    ])
    return "\n".join(lines)


def write_outputs(report: dict[str, Any]) -> None:
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN.write_text(_markdown(report), encoding="utf-8")
    artifacts = (
        PRECOMMIT,
        PRECOMMIT_REPORT,
        PRECOMMIT_LOCK,
        Path(__file__).resolve(),
        TEST,
        RAW,
        REPORT,
        MARKDOWN,
    )
    missing = [str(path) for path in artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    lock = {
        "schema": "finhrm-qp4-post-unit-scale-baseline-diagnostic-review-lock-v1",
        "created_at": report["created_at"],
        "classification": report["classification"],
        "status": "complete" if report["diagnostic_complete"] else "failed",
        "diagnostic_complete": report["diagnostic_complete"],
        "diagnostic_gate_passed": report["diagnostic_gate_passed"],
        "failed_gates": report["failed_gates"],
        "implementation_precommit_candidate": report["implementation_precommit_candidate"],
        "runtime_changed": False,
        "runtime_edit_authorized": False,
        "runtime_hashes": report["runtime_hashes"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "artifacts": [_artifact(path) for path in artifacts],
        "qp4_complete": False,
        "qp5_authorized": False,
        "phase_d": "locked",
        "authorized_next_step": report["authorized_next_step"],
    }
    LOCK.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-diagnostic", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    config = _load(PRECOMMIT)
    if args.run_diagnostic:
        run_diagnostic(config)
    report = build_report()
    if args.write:
        write_outputs(report)
    print(json.dumps({
        "diagnostic_gate_passed": report["diagnostic_gate_passed"],
        "failed_gates": report["failed_gates"],
        "compiler_residuals": report["compiler"]["residuals"],
        "source_queries": report["source"]["queries"],
        "source_first_layers": report["source"]["by_first_failing_layer"],
        "implementation_precommit_nominated": report["implementation_precommit_candidate"]["nominated"],
        "runtime_edit_authorized": report["runtime_edit_authorized"],
    }, ensure_ascii=False, sort_keys=True))
    return 0 if report["diagnostic_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
