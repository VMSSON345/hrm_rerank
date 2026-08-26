from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "5da2ff00bf58496a36be7c08d5bbdb1959a9542e"

TRUST_AUDIT = ROOT / "reports/post_contamination_trust_baseline_audit_v1.json"
INVARIANT = ROOT / "config/finhrm_report_generalization_invariant_v1.json"
INVARIANT_LOCK = ROOT / "reports/finhrm_report_generalization_invariant_lock_v1.json"
THESIS_CORE = ROOT / "config/finhrm_thesis_core_v1.json"
COHORT = ROOT / "reports/finhrm_qp4_post_question_surface_post_cache_residual_coverage_cohort_build_v1.json"
RAW_REPLAY = ROOT / "reports/finhrm_qp4_full_source_semantic_replay_refresh_v1.json"
PLANNER = ROOT / "src/hrm_finance_qa/question_planner.py"

PRECOMMIT = ROOT / "config/finhrm_qp4_positive_target_question_plan_diagnosis_instrumentation_trust_baseline_v2.json"
PRECOMMIT_MD = ROOT / "reports/finhrm_qp4_positive_target_question_plan_diagnosis_instrumentation_trust_baseline_v2.md"
PRECOMMIT_LOCK = ROOT / "reports/finhrm_qp4_positive_target_question_plan_diagnosis_instrumentation_trust_baseline_lock_v2.json"
TRACE = ROOT / "reports/finhrm_qp4_positive_target_question_plan_diagnosis_instrumentation_trust_baseline_replay_v2.jsonl"
REPLAY = ROOT / "reports/finhrm_qp4_positive_target_question_plan_diagnosis_instrumentation_trust_baseline_replay_v2.json"
REPLAY_MD = ROOT / "reports/finhrm_qp4_positive_target_question_plan_diagnosis_instrumentation_trust_baseline_replay_v2.md"
REPLAY_LOCK = ROOT / "reports/finhrm_qp4_positive_target_question_plan_diagnosis_instrumentation_trust_baseline_replay_lock_v2.json"

SURFACES = {
    "canonical": 0,
    "paraphrase_1": 1,
    "long_clause": 2,
    "clause_reordered": 3,
    "no_diacritics": 4,
}
COLLECTIONS = {
    "base_target": "base_surfaces",
    "reference_base_mutation_target": "reference_base_mutation_surfaces",
}
PROTECTED_MARKERS = ("validation", "blind", "test", "sealed", "current_best_validation", "unseen", "holdout")
ALLOWED_TRACE_SYMBOLS = {
    "compile_question_plan",
    "_partition_column_role_context",
    "_compile_column_roles",
    "_align_column_roles",
    "_compile_operand_slots",
    "_compile_program_intent_dag",
    "_compile_abstention_reason",
}
TRACE_STATE_KEYS = (
    "operation",
    "operation_candidates",
    "asks_percent",
    "change_wording",
    "temporal_change",
    "ratio_roles",
    "lexical_column_roles",
    "contextual_column_roles",
    "column_roles",
    "operand_roles",
    "years",
    "requested_years",
    "roles",
    "arity",
    "slot_ids",
    "reference_base",
    "reference_base_named",
    "abstention_reason",
    "direction_intent",
    "requested_column_roles",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": digest(path),
    }


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def value_digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    command = ["git", "-c", f"safe.directory={ROOT.as_posix()}", *args]
    return subprocess.check_output(command, cwd=ROOT, text=True, encoding="utf-8").strip()


def check(condition: bool, name: str, failures: list[str]) -> None:
    if not condition:
        failures.append(name)


def expected_projection(anchor: dict[str, Any], category: str) -> dict[str, Any]:
    program_key = "base" if category == "base_target" else "reference_base_mutation"
    program = anchor["programs"][program_key]
    operands = list(program["operands"])
    evidence = anchor["evidence"]
    roles = [str(evidence[item]["role"]) for item in operands]
    years = [str(evidence[item]["period"]) for item in operands]
    denominator = evidence[operands[1]]
    reference_base = str(denominator["role"]).lower()
    if reference_base == "actual" and years[0] != years[1]:
        reference_base = f"actual_{years[1]}"
    operation = str(program["operation"])
    return {
        "abstention_reason_present": False,
        "operation": operation,
        "operation_candidates": [operation],
        "reference_base": reference_base,
        "roles": roles,
        "years": years,
    }


def normalize_projection(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "abstention_reason_present": bool(value.get("abstention_reason_present")),
        "operation": value.get("operation"),
        "operation_candidates": list(value.get("operation_candidates") or []),
        "reference_base": value.get("reference_base"),
        "roles": list(value.get("roles") or []),
        "years": [str(item) for item in (value.get("years") or [])],
    }


def question_for(anchor: dict[str, Any], category: str, surface: str) -> tuple[str, dict[str, Any]]:
    collection = COLLECTIONS[category]
    index = SURFACES[surface]
    question = str(anchor["questions"][collection][index])
    return question, {"collection": collection, "index": index}


def diversity(cases: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "rows": len(cases),
        "companies": len({row["company_key"] for row in cases}),
        "sources": len({row["source_sha256"] for row in cases}),
        "real_document_layouts": len({row["document_layout_family"] for row in cases}),
        "real_table_layouts": len({row["table_layout_family"] for row in cases}),
        "extraction_families": len({row["extraction_family"] for row in cases}),
    }


def derive_precommit() -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    trust = load(TRUST_AUDIT)
    invariant = load(INVARIANT)
    invariant_lock = load(INVARIANT_LOCK)
    cohort = load(COHORT)
    raw = load(RAW_REPLAY)

    head = git("rev-parse", "--verify", "HEAD")
    baseline_line = git("rev-list", "--parents", "-n", "1", BASELINE_COMMIT).split()
    check(head == BASELINE_COMMIT, "head_is_governing_baseline", failures)
    check(len(baseline_line) == 1, "baseline_has_no_parent", failures)
    check(trust.get("status") == "PASS", "trust_audit_status", failures)
    semantics = trust.get("baseline_semantics", {})
    check(semantics.get("forward_anchor_only") is True, "forward_anchor_only", failures)
    check(semantics.get("certifies_pre_baseline_artifacts_as_clean") is False, "no_false_prebaseline_certification", failures)

    locked_invariant = invariant_lock.get("artifacts", {}).get("generalization_invariant", {})
    locked_thesis = invariant_lock.get("artifacts", {}).get("parent_thesis_core", {})
    check(locked_invariant.get("path") == INVARIANT.relative_to(ROOT).as_posix(), "invariant_lock_path", failures)
    check(locked_invariant.get("sha256") == digest(INVARIANT), "invariant_lock_hash", failures)
    check(locked_thesis.get("path") == THESIS_CORE.relative_to(ROOT).as_posix(), "thesis_lock_path", failures)
    check(locked_thesis.get("sha256") == digest(THESIS_CORE), "thesis_lock_hash", failures)
    admission = invariant.get("repair_admission_rule", {})
    check(admission.get("minimum_distinct_companies") == 3, "minimum_companies_is_3", failures)
    check(admission.get("minimum_distinct_layout_families") == 3, "minimum_layouts_is_3", failures)
    check(invariant.get("invariants", {}).get("protected_evaluation_sources_allowed") is False, "protected_sources_forbidden", failures)
    check(invariant.get("invariants", {}).get("threshold_change_allowed") is False, "threshold_change_forbidden", failures)

    check(str(cohort.get("classification", "")).startswith("development_only_"), "cohort_development_only", failures)
    check(cohort.get("status") == "complete", "cohort_complete", failures)
    cohort_invariants = cohort.get("invariants", {})
    for key in (
        "abw_used",
        "protected_evaluation_loaded",
        "validation_loaded",
        "blind_loaded",
        "test_data_loaded",
        "sealed_or_holdout_loaded",
        "fine_tuning_performed",
        "threshold_changed",
        "checkpoint_changed",
        "model_changed",
        "runtime_changed",
        "ocr_run",
        "source_replay_run",
    ):
        check(cohort_invariants.get(key) is False, f"cohort_invariant_{key}_false", failures)
    check(str(raw.get("classification", "")).startswith("development_only_"), "raw_replay_development_only", failures)
    raw_invariants = raw.get("invariants", {})
    for key in ("abw_used", "protected_evaluation_loaded", "fine_tuning_performed", "runtime_changed", "model_changed", "threshold_changed", "checkpoint_changed", "new_ocr_generated"):
        check(raw_invariants.get(key) is False, f"raw_replay_invariant_{key}_false", failures)

    anchors = list(cohort.get("anchors") or [])
    check(len(anchors) == 6, "six_anchors", failures)
    for anchor in anchors:
        source_path = str(anchor.get("source_path", "")).lower()
        check(bool(source_path), f"source_path_present:{anchor.get('anchor_id')}", failures)
        check(not any(marker in source_path for marker in PROTECTED_MARKERS), f"source_path_not_protected:{anchor.get('anchor_id')}", failures)
        check(anchor.get("visual_review", {}).get("decision") == "PASS", f"visual_review_approved:{anchor.get('anchor_id')}", failures)
        check(bool(anchor.get("provenance")), f"provenance_present:{anchor.get('anchor_id')}", failures)
        check(bool(anchor.get("programs", {}).get("base")), f"base_program_present:{anchor.get('anchor_id')}", failures)
        check(bool(anchor.get("programs", {}).get("reference_base_mutation")), f"mutation_program_present:{anchor.get('anchor_id')}", failures)

    source_diversity = {
        "companies": len({row.get("company_key") for row in anchors}),
        "sources": len({row.get("source_sha256") for row in anchors}),
        "real_document_layouts": len({row.get("document_layout_family") for row in anchors}),
    }
    check(source_diversity == {"companies": 6, "sources": 6, "real_document_layouts": 3}, "cohort_diversity_6_6_3", failures)

    rows = list(raw.get("rows") or [])
    check(len(rows) == 90, "raw_rows_90", failures)
    row_by_id: dict[str, tuple[int, dict[str, Any]]] = {}
    for ordinal, row in enumerate(rows, 1):
        row_id = str(row.get("oracle_row_id", ""))
        check(row_id not in row_by_id, f"unique_raw_row:{row_id}", failures)
        row_by_id[row_id] = (ordinal, row)

    targets: list[dict[str, Any]] = []
    positive_rows = 0
    for anchor in anchors:
        for category in ("base_target", "reference_base_mutation_target"):
            expected = expected_projection(anchor, category)
            for surface in SURFACES:
                positive_rows += 1
                row_id = f"{anchor['anchor_id']}:{category}:{surface}"
                found = row_by_id.get(row_id)
                if found is None:
                    failures.append(f"missing_raw_row:{row_id}")
                    continue
                ordinal, row = found
                observed = normalize_projection(row.get("question_plan") or {})
                if observed == expected:
                    continue
                question, selector = question_for(anchor, category, surface)
                divergence_fields = [key for key in expected if observed.get(key) != expected.get(key)]
                targets.append(
                    {
                        "order": len(targets) + 1,
                        "raw_row_ordinal": ordinal,
                        "oracle_row_id": row_id,
                        "anchor_id": anchor["anchor_id"],
                        "company_key": anchor["company_key"],
                        "source_sha256": anchor["source_sha256"],
                        "document_layout_family": anchor["document_layout_family"],
                        "table_layout_family": anchor["table_layout_family"],
                        "extraction_family": anchor["extraction_family"],
                        "category": category,
                        "surface_label": surface,
                        "question_selector": selector,
                        "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
                        "expected_projection": expected,
                        "observed_frozen_projection": observed,
                        "divergence_fields": divergence_fields,
                    }
                )

    check(positive_rows == 60, "positive_rows_60", failures)
    check(len(targets) == 19, "independently_derived_targets_19", failures)
    check(len({row["oracle_row_id"] for row in targets}) == 19, "target_ids_unique_19", failures)

    observed_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in targets:
        signature = canonical(
            {
                "category": row["category"],
                "surface_label": row["surface_label"],
            }
        )
        observed_groups[signature].append(row)
    group_summary = []
    for signature, group in sorted(observed_groups.items(), key=lambda item: min(row["order"] for row in item[1])):
        group_summary.append(
            {
                "observed_signature_sha256": hashlib.sha256(signature.encode("utf-8")).hexdigest(),
                "category": group[0]["category"],
                "surface_label": group[0]["surface_label"],
                "observed_projection_sha256s": sorted({value_digest(row["observed_frozen_projection"]) for row in group}),
                "divergence_field_sets": sorted({canonical(row["divergence_fields"]) for row in group}),
                "diversity": diversity(group),
                "causal_claim_before_instrumentation": "NONE",
            }
        )
    check(sorted(item["diversity"]["rows"] for item in group_summary) == [1, 6, 6, 6], "observed_groups_6_6_6_1", failures)

    current_planner_hash = digest(PLANNER)
    runtime_before = raw.get("integrity", {}).get("runtime_before", {})
    runtime_after = raw.get("integrity", {}).get("runtime_after", {})
    planner_rel = PLANNER.relative_to(ROOT).as_posix()
    check(runtime_before.get(planner_rel) == current_planner_hash, "raw_runtime_before_matches_planner", failures)
    check(runtime_after.get(planner_rel) == current_planner_hash, "raw_runtime_after_matches_planner", failures)
    check(not git("diff", "--name-only", BASELINE_COMMIT, "--", planner_rel), "planner_unchanged_from_baseline", failures)

    replay_script = artifact(Path(__file__).resolve())
    gates = {
        "baseline_head_exact": head == BASELINE_COMMIT,
        "baseline_root_commit": len(baseline_line) == 1,
        "forward_anchor_semantics_acknowledged": semantics.get("forward_anchor_only") is True,
        "old_precommit_not_an_input": True,
        "old_evaluator_not_an_input": True,
        "raw_question_plan_passed_flag_ignored": True,
        "protected_paths_not_read": True,
        "governing_invariant_hashes_reverified": locked_invariant.get("sha256") == digest(INVARIANT) and locked_thesis.get("sha256") == digest(THESIS_CORE),
        "development_scope_reverified": not any(name.startswith("cohort_invariant_") or name.startswith("raw_replay_invariant_") or name.startswith("source_path_not_protected") for name in failures),
        "oracle_materialized_from_anchor_program_and_evidence": positive_rows == 60,
        "selection_independently_recomputed": len(targets) == 19,
        "all_question_selectors_and_hashes_materialized": len(targets) == 19,
        "unobserved_frozen_projection_available_19": len(targets) == 19,
        "persistent_replay_harness_present": Path(__file__).is_file(),
        "runtime_whitelist_empty": True,
        "compiler_calls_during_precommit_zero": True,
    }
    for name, passed in gates.items():
        check(bool(passed), f"gate:{name}", failures)

    status = "PASS" if not failures else "HOLD"
    result = {
        "schema": "finhrm-qp4-question-plan-diagnosis-instrumentation-trust-baseline-precommit-v2",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "classification": "development_only_new_trust_baseline_question_plan_diagnosis_instrumentation_precommit",
        "status": status,
        "precommit_gate_passed": status == "PASS",
        "governing_baseline": {
            "commit": BASELINE_COMMIT,
            "head_at_precommit": head,
            "parent_count": len(baseline_line) - 1,
            "semantics": "forward anchor only; no pre-baseline cleanliness claim",
            "user_authorized_qp4_restart_from_this_baseline": True,
        },
        "inputs": {
            "trust_baseline_audit": artifact(TRUST_AUDIT),
            "generalization_invariant": artifact(INVARIANT),
            "generalization_invariant_lock": artifact(INVARIANT_LOCK),
            "thesis_core": artifact(THESIS_CORE),
            "development_source_cohort": artifact(COHORT),
            "raw_development_replay_diagnostic_evidence_only": artifact(RAW_REPLAY),
            "question_planner_source": artifact(PLANNER),
            "persistent_replay_harness": replay_script,
        },
        "explicitly_excluded_as_inputs": {
            "old_precommits_and_locks": True,
            "old_corrected_evaluator_and_its_selection": True,
            "old_question_plan_instrumentation_replay": True,
            "old_candidate_or_binding_diagnosis": True,
            "validation_blind_test_sealed_current_best_validation": True,
            "source_pdf_content": True,
            "model_checkpoint_page_index_ocr": True,
        },
        "development_scope": {
            "anchors": len(anchors),
            "positive_rows_rederived": positive_rows,
            "source_diversity": source_diversity,
            "source_paths_checked_by_metadata_only": True,
            "source_pdf_bytes_read": 0,
        },
        "oracle_contract": {
            "correctness_source": "each frozen development anchor's programs, evidence roles, and evidence periods",
            "raw_replay_use": "diagnostic observed projection only; question_plan_passed is ignored",
            "comparison": "exact operation, operation_candidates, roles, years, reference_base, and abstention presence",
            "gold_or_expected_plan_passed_to_compiler": False,
            "oracle_applied_post_hoc_only": True,
        },
        "independent_diagnosis": {
            "target_count": len(targets),
            "target_order": "source cohort anchor order, then base_target/reference_base_mutation_target, then frozen surface order",
            "observed_projection_groups": group_summary,
            "root_cause_hypotheses_carried_from_old_precommit": 0,
            "root_cause_claim_before_instrumentation": "NONE",
            "targets": targets,
        },
        "instrumentation_contract": {
            "module": "hrm_finance_qa.question_planner",
            "entry_symbol": "compile_question_plan",
            "mechanism": "Python frame tracing limited to named Question Planner symbols; no monkeypatch, argument rewrite, return rewrite, model, source, candidate, or binding access",
            "allowed_symbols": sorted(ALLOWED_TRACE_SYMBOLS),
            "question_content_in_trace": False,
            "question_sha256_only": True,
            "source_line_identity": "line number plus SHA-256 of source line under frozen question_planner hash",
            "observer_noninterference_gate": "19/19 exact baseline projection equals instrumented projection",
        },
        "replay_contract": {
            "command": f"py -3.13 {replay_script['path']} replay",
            "execution_order": "for each target in frozen order: one baseline call, then one traced call",
            "baseline_compiler_calls": 19,
            "instrumented_compiler_calls": 19,
            "total_compiler_calls": 38,
            "candidate_or_binding_calls": 0,
            "model_inferences": 0,
            "pdf_reads": 0,
            "ocr_calls": 0,
            "page_index_calls": 0,
            "baseline_frozen_match_required": "19/19",
            "output_noninterference_required": "19/19",
            "direct_trace": TRACE.relative_to(ROOT).as_posix(),
            "trace_durability": "one JSON object per case, flush plus os.fsync",
            "final_outputs": [REPLAY.relative_to(ROOT).as_posix(), REPLAY_MD.relative_to(ROOT).as_posix(), REPLAY_LOCK.relative_to(ROOT).as_posix()],
        },
        "precommit_feasibility_gates": gates,
        "failures": failures,
        "runtime_freeze": {
            "runtime_file_whitelist": [],
            "question_planner_sha256": current_planner_hash,
            "model_inference_authorized": False,
            "checkpoint_read_or_change_authorized": False,
            "threshold_change_authorized": False,
            "runtime_repair_authorized": False,
        },
        "authorized_next_action": "Run replay mode exactly once" if status == "PASS" else "HOLD",
    }
    return result, failures


def precommit_markdown(result: dict[str, Any]) -> str:
    diagnosis = result["independent_diagnosis"]
    lines = [
        "# QP4 Question Plan diagnosis/instrumentation — new trust baseline",
        "",
        f"Status: **{result['status']}**.",
        "",
        f"Governing baseline: `{BASELINE_COMMIT}` (root/forward anchor only).",
        "",
        "The cohort was rebuilt from the six-anchor development artifact and the raw 90-row development replay. Old precommits, corrected-evaluator selection, candidate/binding traces, and protected evaluation artifacts are not inputs. The raw `question_plan_passed` flag is ignored; correctness is recomputed from anchor programs and evidence roles/periods.",
        "",
        f"Independent result: {diagnosis['target_count']} Question Plan divergences. No old root-cause hypothesis is carried forward.",
        "",
        "Observed projection groups (descriptive only, not causal proof):",
        "",
    ]
    for group in diagnosis["observed_projection_groups"]:
        d = group["diversity"]
        lines.append(f"- `{group['category']}/{group['surface_label']}`: {d['rows']} rows, {d['companies']} companies, {d['sources']} sources, {d['real_document_layouts']} real document layouts; causal claim: NONE.")
    lines += [
        "",
        "The replay harness is persistent and hash-locked before PASS. Precommit mode performs zero compiler calls. Replay mode is fail-closed unless this precommit is PASS, then performs exactly 19 baseline and 19 traced calls with per-case fsynced JSONL and exact noninterference checks.",
        "",
        "Runtime/model/threshold/checkpoint changes are not authorized.",
        "",
    ]
    if result["failures"]:
        lines.append("Failures: " + ", ".join(result["failures"]))
        lines.append("")
    return "\n".join(lines)


def run_precommit() -> int:
    result, failures = derive_precommit()
    write_json(PRECOMMIT, result)
    PRECOMMIT_MD.write_text(precommit_markdown(result), encoding="utf-8")
    lock = {
        "schema": "finhrm-qp4-question-plan-diagnosis-instrumentation-trust-baseline-precommit-lock-v2",
        "created_at": result["created_at"],
        "classification": "development_only_new_trust_baseline_question_plan_precommit_lock",
        "status": result["status"],
        "governing_baseline": BASELINE_COMMIT,
        "artifacts": {"precommit": artifact(PRECOMMIT), "markdown": artifact(PRECOMMIT_MD), "persistent_replay_harness": artifact(Path(__file__).resolve())},
        "gates": result["precommit_feasibility_gates"],
        "compiler_calls": 0,
        "runtime_changed": False,
        "repair_authorized": False,
        "authorized_next_action": result["authorized_next_action"],
    }
    write_json(PRECOMMIT_LOCK, lock)
    print(f"precommit_status={result['status']}")
    print(f"targets={result['independent_diagnosis']['target_count']}")
    print("compiler_calls=0")
    if failures:
        print("failures=" + ",".join(failures))
        return 2
    return 0


def trace_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value):
        return {field.name: trace_safe(getattr(value, field.name)) for field in dataclasses.fields(value) if field.name not in {"question", "normalized", "metric_text", "entity_text"}}
    if isinstance(value, (tuple, list)):
        return [trace_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): trace_safe(item) for key, item in value.items()}
    if isinstance(value, set):
        return sorted(trace_safe(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return type(value).__name__


def plan_projection(plan: Any) -> dict[str, Any]:
    slots = list(getattr(plan, "operand_slots", ()))
    roles = [str(slot.column_role) for slot in slots]
    years = [str(slot.year) for slot in slots]
    if not roles:
        roles = [str(item) for item in getattr(plan, "column_roles", ())]
    if not years:
        years = [str(item) for item in getattr(plan, "requested_years", ())]
    return {
        "abstention_reason_present": bool(getattr(plan, "abstention_reason", "")),
        "operation": getattr(plan, "operation", None),
        "operation_candidates": list(getattr(plan, "operation_candidates", ())),
        "reference_base": getattr(plan, "reference_base", None),
        "roles": roles,
        "years": years,
    }


def return_projection(symbol: str, value: Any) -> Any:
    if symbol == "compile_question_plan":
        return plan_projection(value)
    if symbol in {"_compile_column_roles", "_align_column_roles"}:
        return list(value)
    if symbol == "_partition_column_role_context":
        return [{"sha256": hashlib.sha256(str(item).encode("utf-8")).hexdigest(), "length": len(str(item))} for item in value]
    if symbol == "_compile_operand_slots":
        return [{"id": slot.id, "role": slot.column_role, "year": str(slot.year), "unit_family": slot.unit_family} for slot in value]
    if symbol == "_compile_program_intent_dag":
        reference_base, nodes = value
        return {"reference_base": reference_base, "nodes": [{"id": node.id, "operation": node.operation, "inputs": list(node.inputs)} for node in nodes]}
    if symbol == "_compile_abstention_reason":
        return value
    return trace_safe(value)


def traced_compile(module: Any, question: str) -> tuple[Any, list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    planner_path = PLANNER.resolve()
    source_lines = PLANNER.read_text(encoding="utf-8").splitlines()

    def tracer(frame: Any, event: str, arg: Any) -> Any:
        if Path(frame.f_code.co_filename).resolve() != planner_path or frame.f_code.co_name not in ALLOWED_TRACE_SYMBOLS:
            return tracer
        symbol = frame.f_code.co_name
        record: dict[str, Any] = {"event": event, "symbol": symbol, "line": frame.f_lineno}
        if 1 <= frame.f_lineno <= len(source_lines):
            record["source_line_sha256"] = hashlib.sha256(source_lines[frame.f_lineno - 1].encode("utf-8")).hexdigest()
        if event == "line":
            state = {key: trace_safe(frame.f_locals[key]) for key in TRACE_STATE_KEYS if key in frame.f_locals}
            if state:
                record["state"] = state
        elif event == "return":
            record["return"] = return_projection(symbol, arg)
        elif event == "exception":
            record["exception_type"] = getattr(arg[0], "__name__", str(arg[0]))
        events.append(record)
        return tracer

    sys.settrace(tracer)
    try:
        plan = module.compile_question_plan(question)
    finally:
        sys.settrace(None)
    return plan, events


def operation_transition(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    prior_value: Any = None
    prior_line: dict[str, Any] | None = None
    transitions: list[dict[str, Any]] = []
    for event in events:
        if event.get("event") != "line" or event.get("symbol") != "compile_question_plan":
            continue
        state = event.get("state") or {}
        if "operation" in state and state["operation"] != prior_value:
            if prior_line is not None:
                transitions.append({"producer_line": prior_line["line"], "source_line_sha256": prior_line.get("source_line_sha256"), "from": prior_value, "to": state["operation"]})
            prior_value = state["operation"]
        prior_line = event
    return transitions[-1] if transitions else None


def causal_signature(case: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    divergence = list(case["divergence_fields"])
    if "operation" in divergence or "operation_candidates" in divergence:
        producer_symbol = "compile_question_plan"
        transition = operation_transition(events)
        producer_line = transition.get("producer_line") if transition else None
        source_line_sha256 = transition.get("source_line_sha256") if transition else None
        causal_transition = transition
    elif "roles" in divergence or "years" in divergence:
        producer_symbol = "_compile_operand_slots"
        returned = next((event for event in reversed(events) if event.get("event") == "return" and event.get("symbol") == producer_symbol), {})
        producer_line = returned.get("line")
        source_line_sha256 = returned.get("source_line_sha256")
        causal_transition = returned.get("return")
    else:
        producer_symbol = "_compile_program_intent_dag"
        returned = next((event for event in reversed(events) if event.get("event") == "return" and event.get("symbol") == producer_symbol), {})
        producer_line = returned.get("line")
        source_line_sha256 = returned.get("source_line_sha256")
        causal_transition = returned.get("return")
    signature = {
        "module": "hrm_finance_qa.question_planner",
        "producer_symbol": producer_symbol,
        "producer_line": producer_line,
        "source_line_sha256": source_line_sha256,
        "causal_transition": causal_transition,
        "downstream_incorrect_fields": divergence,
    }
    signature["sha256"] = value_digest(signature)
    return signature


def run_replay() -> int:
    precommit = load(PRECOMMIT)
    lock = load(PRECOMMIT_LOCK)
    if precommit.get("status") != "PASS" or precommit.get("precommit_gate_passed") is not True or lock.get("status") != "PASS":
        raise RuntimeError("HOLD: new-trust-baseline precommit is not PASS")
    if lock.get("artifacts", {}).get("precommit", {}).get("sha256") != digest(PRECOMMIT):
        raise RuntimeError("HOLD: precommit lock hash mismatch")
    if precommit.get("governing_baseline", {}).get("commit") != BASELINE_COMMIT:
        raise RuntimeError("HOLD: governing baseline mismatch")
    if digest(PLANNER) != precommit.get("runtime_freeze", {}).get("question_planner_sha256"):
        raise RuntimeError("HOLD: Question Planner source hash mismatch")

    cohort = load(COHORT)
    anchors = {row["anchor_id"]: row for row in cohort["anchors"]}
    sys.path.insert(0, str(ROOT / "src"))
    import hrm_finance_qa.question_planner as planner_module

    targets = list(precommit["independent_diagnosis"]["targets"])
    if len(targets) != 19:
        raise RuntimeError("HOLD: target count is not 19")

    baseline_calls = 0
    instrumented_calls = 0
    frozen_matches = 0
    noninterference = 0
    records: list[dict[str, Any]] = []
    TRACE.parent.mkdir(parents=True, exist_ok=True)
    with TRACE.open("w", encoding="utf-8", newline="\n") as handle:
        for case in targets:
            anchor = anchors[case["anchor_id"]]
            selector = case["question_selector"]
            question = str(anchor["questions"][selector["collection"]][selector["index"]])
            question_sha256 = hashlib.sha256(question.encode("utf-8")).hexdigest()
            if question_sha256 != case["question_sha256"]:
                raise RuntimeError(f"HOLD: question hash mismatch: {case['oracle_row_id']}")

            baseline_plan = planner_module.compile_question_plan(question)
            baseline_calls += 1
            baseline_projection = plan_projection(baseline_plan)
            traced_plan, events = traced_compile(planner_module, question)
            instrumented_calls += 1
            instrumented_projection = plan_projection(traced_plan)

            frozen_exact = baseline_projection == case["observed_frozen_projection"]
            observer_exact = baseline_projection == instrumented_projection
            frozen_matches += int(frozen_exact)
            noninterference += int(observer_exact)
            signature = causal_signature(case, events)
            flow = [(event.get("event"), event.get("symbol"), event.get("line")) for event in events]
            record = {
                "schema": "finhrm-qp4-question-plan-trust-baseline-instrumentation-trace-case-v2",
                "classification": "development_only_question_plan_trace_no_question_text_no_source_no_model",
                "order": case["order"],
                "oracle_row_id": case["oracle_row_id"],
                "anchor_id": case["anchor_id"],
                "company_key": case["company_key"],
                "source_sha256": case["source_sha256"],
                "document_layout_family": case["document_layout_family"],
                "question_sha256": question_sha256,
                "expected_projection": case["expected_projection"],
                "frozen_observed_projection": case["observed_frozen_projection"],
                "baseline_projection": baseline_projection,
                "instrumented_projection": instrumented_projection,
                "baseline_matches_frozen": frozen_exact,
                "instrumentation_noninterference": observer_exact,
                "control_flow_sha256": value_digest(flow),
                "causal_signature": signature,
                "trace_events": events,
            }
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            records.append(record)
            print(f"case={case['order']}/19 baseline=1 instrumented=1 frozen={int(frozen_exact)} observer={int(observer_exact)}")

    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        clusters[record["causal_signature"]["sha256"]].append(record)
    cluster_rows = []
    for signature_sha256, group in sorted(clusters.items(), key=lambda item: min(row["order"] for row in item[1])):
        d = {
            "rows": len(group),
            "companies": len({row["company_key"] for row in group}),
            "sources": len({row["source_sha256"] for row in group}),
            "real_document_layouts": len({row["document_layout_family"] for row in group}),
        }
        generic = d["companies"] >= 3 and d["sources"] >= 3 and d["real_document_layouts"] >= 3
        cluster_rows.append(
            {
                "causal_signature_sha256": signature_sha256,
                "causal_signature": group[0]["causal_signature"],
                "case_ids": [row["oracle_row_id"] for row in group],
                "diversity": d,
                "conclusion": "GENERIC_ROOT_CAUSE" if generic else "LOCAL_INSUFFICIENT_FOR_GENERICITY",
            }
        )

    status = "PASS" if baseline_calls == 19 and instrumented_calls == 19 and frozen_matches == 19 and noninterference == 19 and sum(len(group) for group in clusters.values()) == 19 else "HOLD"
    summary = {
        "schema": "finhrm-qp4-question-plan-diagnosis-instrumentation-trust-baseline-replay-v2",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "classification": "development_only_new_trust_baseline_question_plan_diagnosis_instrumentation_replay",
        "status": status,
        "governing_baseline": BASELINE_COMMIT,
        "inputs": {"precommit": artifact(PRECOMMIT), "precommit_lock": artifact(PRECOMMIT_LOCK), "question_planner": artifact(PLANNER), "trace": artifact(TRACE)},
        "execution": {
            "target_cases": len(targets),
            "baseline_compiler_calls": baseline_calls,
            "instrumented_compiler_calls": instrumented_calls,
            "total_compiler_calls": baseline_calls + instrumented_calls,
            "baseline_matches_frozen": f"{frozen_matches}/19",
            "instrumentation_noninterference": f"{noninterference}/19",
            "candidate_or_binding_calls": 0,
            "model_inferences": 0,
            "pdf_reads": 0,
            "ocr_calls": 0,
            "page_index_calls": 0,
        },
        "causal_clusters": cluster_rows,
        "all_19_cases_disposed": sum(len(group) for group in clusters.values()) == 19,
        "runtime_changed": digest(PLANNER) != precommit["runtime_freeze"]["question_planner_sha256"],
        "repair_authorized": False,
        "authorized_next_action": "Review causal clusters and, if desired, create a separate runtime-repair precommit" if status == "PASS" else "HOLD",
    }
    write_json(REPLAY, summary)
    markdown = [
        "# QP4 Question Plan diagnosis/instrumentation replay — new trust baseline",
        "",
        f"Status: **{status}**.",
        "",
        f"Executed {baseline_calls} baseline and {instrumented_calls} instrumented compiler calls. Frozen baseline match: {frozen_matches}/19. Observer noninterference: {noninterference}/19.",
        "",
        "Exact code-derived causal clusters:",
        "",
    ]
    for cluster in cluster_rows:
        sig = cluster["causal_signature"]
        d = cluster["diversity"]
        markdown.append(f"- `{sig['producer_symbol']}:{sig['producer_line']}` / `{cluster['causal_signature_sha256'][:12]}`: {d['rows']} rows, {d['companies']} companies, {d['sources']} sources, {d['real_document_layouts']} real layouts — {cluster['conclusion']}.")
    markdown += ["", "No runtime, model, threshold, checkpoint, candidate, binding, PDF, OCR, or page-index change/call was authorized or performed.", ""]
    REPLAY_MD.write_text("\n".join(markdown), encoding="utf-8")
    replay_lock = {
        "schema": "finhrm-qp4-question-plan-diagnosis-instrumentation-trust-baseline-replay-lock-v2",
        "created_at": summary["created_at"],
        "classification": "development_only_new_trust_baseline_question_plan_replay_lock",
        "status": status,
        "governing_baseline": BASELINE_COMMIT,
        "artifacts": {"trace": artifact(TRACE), "summary": artifact(REPLAY), "markdown": artifact(REPLAY_MD)},
        "execution": summary["execution"],
        "runtime_changed": summary["runtime_changed"],
        "repair_authorized": False,
        "authorized_next_action": summary["authorized_next_action"],
    }
    write_json(REPLAY_LOCK, replay_lock)
    print(f"replay_status={status}")
    print(f"baseline_calls={baseline_calls}")
    print(f"instrumented_calls={instrumented_calls}")
    return 0 if status == "PASS" else 3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("precommit", "replay"))
    args = parser.parse_args()
    return run_precommit() if args.mode == "precommit" else run_replay()


if __name__ == "__main__":
    raise SystemExit(main())
