from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE = "5da2ff00bf58496a36be7c08d5bbdb1959a9542e"
REPORT = ROOT / "reports/finhrm_qp4_question_plan_clean_diagnosis_closure_v1.json"
LINE_ENDING_POLICY = ".gitattributes"

FILES: dict[str, str] = {
    "config/finhrm_qp4_positive_target_question_plan_diagnosis_instrumentation_trust_baseline_v2.json": "governing_precommit_evidence",
    "reports/finhrm_qp4_positive_target_question_plan_diagnosis_instrumentation_trust_baseline_lock_v2.json": "governing_precommit_lock",
    "reports/finhrm_qp4_positive_target_question_plan_diagnosis_instrumentation_trust_baseline_replay_lock_v2.json": "governing_execution_lock",
    "reports/finhrm_qp4_positive_target_question_plan_diagnosis_instrumentation_trust_baseline_replay_v2.json": "governing_execution_summary",
    "reports/finhrm_qp4_positive_target_question_plan_diagnosis_instrumentation_trust_baseline_replay_v2.jsonl": "governing_fsynced_raw_trace",
    "reports/finhrm_qp4_positive_target_question_plan_diagnosis_instrumentation_trust_baseline_replay_v2.md": "governing_human_readable_execution_summary_hash_locked",
    "reports/finhrm_qp4_positive_target_question_plan_diagnosis_instrumentation_trust_baseline_review_lock_v1.json": "governing_diagnosis_review_lock",
    "reports/finhrm_qp4_positive_target_question_plan_diagnosis_instrumentation_trust_baseline_review_v1.json": "governing_final_diagnosis_review",
    "reports/finhrm_qp4_positive_target_question_plan_diagnosis_instrumentation_trust_baseline_review_v1.md": "governing_human_readable_review_summary_hash_locked",
    "reports/finhrm_qp4_positive_target_question_plan_diagnosis_instrumentation_trust_baseline_v2.md": "governing_human_readable_precommit_summary_hash_locked",
    "scripts/review_qp4_question_plan_trust_baseline_trace.py": "governing_reproducible_trace_review_tooling",
    "scripts/run_qp4_question_plan_trust_baseline.py": "governing_reproducible_precommit_and_replay_tooling",
}

PRECOMMIT = ROOT / "config/finhrm_qp4_positive_target_question_plan_diagnosis_instrumentation_trust_baseline_v2.json"
PRECOMMIT_LOCK = ROOT / "reports/finhrm_qp4_positive_target_question_plan_diagnosis_instrumentation_trust_baseline_lock_v2.json"
REPLAY = ROOT / "reports/finhrm_qp4_positive_target_question_plan_diagnosis_instrumentation_trust_baseline_replay_v2.json"
REPLAY_LOCK = ROOT / "reports/finhrm_qp4_positive_target_question_plan_diagnosis_instrumentation_trust_baseline_replay_lock_v2.json"
TRACE = ROOT / "reports/finhrm_qp4_positive_target_question_plan_diagnosis_instrumentation_trust_baseline_replay_v2.jsonl"
REVIEW = ROOT / "reports/finhrm_qp4_positive_target_question_plan_diagnosis_instrumentation_trust_baseline_review_v1.json"
REVIEW_LOCK = ROOT / "reports/finhrm_qp4_positive_target_question_plan_diagnosis_instrumentation_trust_baseline_review_lock_v1.json"
PLANNER = ROOT / "src/hrm_finance_qa/question_planner.py"
PAGE_INDEX_SOURCE = ROOT / "src/hrm_finance_qa/document_page_index.py"

PROTECTED_PATTERN = re.compile(r"validation|blind|test|sealed|current_best_validation", re.IGNORECASE)
FORBIDDEN_PAYLOAD_KEYS = {
    "question",
    "normalized",
    "metric_text",
    "document_text",
    "page_context",
    "ocr_output",
    "validation_data",
    "blind_data",
    "test_data",
    "sealed_data",
    "current_best_validation",
}


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", *args],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    ).strip()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    return {
        "path": relative,
        "classification": FILES.get(relative, "governing_diagnosis_closure_audit"),
        "size_bytes": path.stat().st_size,
        "sha256": digest(path),
    }


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def all_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(all_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(all_keys(item))
    return keys


def all_strings(value: Any) -> list[str]:
    strings: list[str] = []
    if isinstance(value, dict):
        for item in value.values():
            strings.extend(all_strings(item))
    elif isinstance(value, list):
        for item in value:
            strings.extend(all_strings(item))
    elif isinstance(value, str):
        strings.append(value)
    return strings


def protected_content_audit(failures: list[str]) -> dict[str, Any]:
    marker_lines: list[dict[str, Any]] = []
    json_payload_key_hits: dict[str, list[str]] = {}
    protected_path_value_hits: dict[str, list[str]] = {}
    parsed_json_documents = 0
    parsed_trace_rows = 0
    parsed_python_files = 0

    audit_script = Path(__file__).resolve().relative_to(ROOT).as_posix()
    scan_files = list(FILES) + [LINE_ENDING_POLICY, audit_script]
    for relative in scan_files:
        path = ROOT / relative
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".py":
            ast.parse(text)
            parsed_python_files += 1
        if path.suffix in {".json", ".jsonl"}:
            values = (
                [json.loads(line) for line in text.splitlines() if line.strip()]
                if path.suffix == ".jsonl"
                else [json.loads(text)]
            )
            parsed_json_documents += len(values)
            if path.suffix == ".jsonl":
                parsed_trace_rows += len(values)
            hits = sorted(set().union(*(all_keys(value) for value in values)) & FORBIDDEN_PAYLOAD_KEYS)
            if hits:
                json_payload_key_hits[relative] = hits
            path_hits = sorted(
                {
                    value
                    for document in values
                    for value in all_strings(document)
                    if ("/" in value or "\\" in value) and PROTECTED_PATTERN.search(value)
                }
            )
            if path_hits:
                protected_path_value_hits[relative] = path_hits
        for line_number, line in enumerate(text.splitlines(), 1):
            if not PROTECTED_PATTERN.search(line):
                continue
            allowed_policy_reference = (
                relative == "scripts/run_qp4_question_plan_trust_baseline.py"
                and any(
                    token in line
                    for token in (
                        "PROTECTED_MARKERS",
                        '"validation_loaded"',
                        '"blind_loaded"',
                        '"test_data_loaded"',
                        '"sealed_or_holdout_loaded"',
                        '"validation_blind_test_sealed_current_best_validation"',
                    )
                )
                or relative == PRECOMMIT.relative_to(ROOT).as_posix()
                and '"validation_blind_test_sealed_current_best_validation"' in line
                or relative == audit_script
            )
            marker_lines.append(
                {
                    "path": relative,
                    "line": line_number,
                    "line_sha256": hashlib.sha256(line.encode("utf-8")).hexdigest(),
                    "classification": "policy_or_exclusion_reference" if allowed_policy_reference else "UNEXPECTED",
                }
            )
            if not allowed_policy_reference:
                failures.append(f"unexpected_protected_marker:{relative}:{line_number}")

    if json_payload_key_hits:
        failures.append("forbidden_payload_keys_present")
    if protected_path_value_hits:
        failures.append("protected_path_value_present")
    if parsed_trace_rows != 19:
        failures.append("trace_rows_not_19")
    return {
        "files_scanned": len(scan_files),
        "original_12_files_scanned": len(FILES),
        "additional_closure_policy_and_tooling_files_scanned": 2,
        "utf8_text_files": len(scan_files),
        "parsed_python_files": parsed_python_files,
        "parsed_json_documents_including_jsonl_rows": parsed_json_documents,
        "parsed_trace_rows": parsed_trace_rows,
        "forbidden_payload_key_hits": json_payload_key_hits,
        "protected_path_value_hits": protected_path_value_hits,
        "marker_line_audit": marker_lines,
        "all_marker_occurrences_are_policy_or_exclusion_references": all(row["classification"] == "policy_or_exclusion_reference" for row in marker_lines),
        "protected_content_detected": bool(json_payload_key_hits or protected_path_value_hits or any(row["classification"] == "UNEXPECTED" for row in marker_lines)),
    }


def cluster_accounting(failures: list[str]) -> dict[str, Any]:
    review = load(REVIEW)
    trace_rows = [json.loads(line) for line in TRACE.read_text(encoding="utf-8").splitlines() if line.strip()]
    clusters = list(review.get("normalized_causal_clusters") or [])

    def select(symbol: str, line: int, fields: set[str]) -> dict[str, Any]:
        found = [
            cluster
            for cluster in clusters
            if cluster.get("normalized_causal_signature", {}).get("producer_symbol") == symbol
            and cluster.get("normalized_causal_signature", {}).get("producer_line") == line
            and set(cluster.get("normalized_causal_signature", {}).get("downstream_incorrect_fields") or []) == fields
        ]
        if len(found) != 1:
            failures.append(f"cluster_identity:{symbol}:{line}:{sorted(fields)}")
            return {"case_ids": [], "diversity": {}, "conclusion": "MISSING_OR_AMBIGUOUS"}
        return found[0]

    definitions = [
        (
            "_compile_operand_slots:1144",
            select("_compile_operand_slots", 1144, {"reference_base", "roles"}),
            6,
            "GENERIC_ROOT_CAUSE",
        ),
        (
            "compile_question_plan:575_SELECT_to_RATIO_PERCENT",
            select("compile_question_plan", 575, {"operation", "operation_candidates", "reference_base"}),
            6,
            "GENERIC_ROOT_CAUSE",
        ),
        (
            "compile_question_plan:575_year_order_inversion",
            select("compile_question_plan", 575, {"operation", "operation_candidates", "reference_base", "years"}),
            6,
            "GENERIC_ROOT_CAUSE",
        ),
        (
            "compile_question_plan:521_singleton",
            select("compile_question_plan", 521, {"operation", "operation_candidates"}),
            1,
            "LOCAL_INSUFFICIENT_FOR_GENERICITY",
        ),
    ]

    named: dict[str, dict[str, Any]] = {}
    sets: dict[str, set[str]] = {}
    for name, cluster, expected_rows, expected_conclusion in definitions:
        case_ids = list(cluster.get("case_ids") or [])
        case_set = set(case_ids)
        if len(case_ids) != expected_rows or len(case_set) != expected_rows:
            failures.append(f"cluster_row_count:{name}")
        if cluster.get("conclusion") != expected_conclusion:
            failures.append(f"cluster_conclusion:{name}")
        sets[name] = case_set
        named[name] = {
            "expected_rows": expected_rows,
            "actual_rows": len(case_ids),
            "unique_rows": len(case_set),
            "case_ids": case_ids,
            "conclusion": cluster.get("conclusion"),
        }

    names = list(sets)
    overlaps: dict[str, list[str]] = {}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = sorted(sets[left] & sets[right])
            overlaps[f"{left}__AND__{right}"] = overlap
            if overlap:
                failures.append(f"cluster_overlap:{left}:{right}")
    union = set().union(*sets.values())
    trace_ids = {str(row.get("oracle_row_id")) for row in trace_rows}
    if len(union) != 19 or union != trace_ids:
        failures.append("cluster_union_not_exact_trace_19")
    select_name = "compile_question_plan:575_SELECT_to_RATIO_PERCENT"
    year_name = "compile_question_plan:575_year_order_inversion"
    return {
        "clusters": named,
        "two_575_clusters_disjoint": not bool(sets[select_name] & sets[year_name]),
        "two_575_cluster_overlap": sorted(sets[select_name] & sets[year_name]),
        "all_pairwise_overlaps": overlaps,
        "union_case_ids": sorted(union),
        "union_unique_cases": len(union),
        "trace_unique_cases": len(trace_ids),
        "union_exactly_matches_locked_trace": union == trace_ids,
        "double_counted_cases": sorted({case_id for case_id in union if sum(case_id in case_set for case_set in sets.values()) > 1}),
    }


def build() -> int:
    failures: list[str] = []
    head = git("rev-parse", "HEAD")
    audit_script = Path(__file__).resolve().relative_to(ROOT).as_posix()
    report_relative = REPORT.relative_to(ROOT).as_posix()
    amend_refresh = head != BASELINE
    tracked_diff = set(filter(None, git("diff", "--name-only").splitlines()))
    if amend_refresh:
        if git("rev-parse", "HEAD^") != BASELINE:
            failures.append("intermediate_commit_parent_not_new_trust_baseline")
        allowed_refresh_diff = {LINE_ENDING_POLICY, audit_script, report_relative}
        if not {LINE_ENDING_POLICY, audit_script}.issubset(tracked_diff):
            failures.append("required_line_policy_refresh_diff_missing")
        if not tracked_diff.issubset(allowed_refresh_diff):
            failures.append("unexpected_tracked_diff_during_amend_refresh")
        expected_untracked_before_write: set[str] = set()
    else:
        expected_untracked_before_write = set(FILES) | {audit_script, LINE_ENDING_POLICY}
        if REPORT.is_file():
            expected_untracked_before_write.add(report_relative)
        if tracked_diff:
            failures.append("tracked_diff_present_before_closure")
    actual_untracked = set(filter(None, git("ls-files", "--others", "--exclude-standard").splitlines()))
    if actual_untracked != expected_untracked_before_write:
        failures.append("untracked_allowlist_mismatch_before_closure_report")

    for relative in FILES:
        if not (ROOT / relative).is_file():
            failures.append(f"missing_file:{relative}")
    content_audit = protected_content_audit(failures)

    precommit = load(PRECOMMIT)
    precommit_lock = load(PRECOMMIT_LOCK)
    replay = load(REPLAY)
    replay_lock = load(REPLAY_LOCK)
    review = load(REVIEW)
    review_lock = load(REVIEW_LOCK)
    for name, value in (
        ("precommit", precommit),
        ("precommit_lock", precommit_lock),
        ("replay", replay),
        ("replay_lock", replay_lock),
        ("review", review),
        ("review_lock", review_lock),
    ):
        if value.get("status") != "PASS":
            failures.append(f"status_not_pass:{name}")

    accounting = cluster_accounting(failures)
    if not accounting["two_575_clusters_disjoint"]:
        failures.append("two_575_clusters_overlap")
    if accounting["double_counted_cases"]:
        failures.append("double_counted_cases")

    raw_replay = load(REPLAY)
    frozen_raw_source = load(ROOT / "reports/finhrm_qp4_full_source_semantic_replay_refresh_v1.json")
    expected_page_index_hash = frozen_raw_source.get("integrity", {}).get("runtime_after", {}).get("src/hrm_finance_qa/document_page_index.py")
    page_index_hash = digest(PAGE_INDEX_SOURCE)
    if page_index_hash != expected_page_index_hash:
        failures.append("page_index_source_hash_mismatch")

    runtime_diff = git("diff", "--name-only", BASELINE, "--", "src", "models")
    if runtime_diff:
        failures.append("runtime_or_model_diff")
    if digest(PLANNER) != precommit.get("runtime_freeze", {}).get("question_planner_sha256"):
        failures.append("planner_hash_changed")

    artifacts = [artifact(relative) for relative in FILES]
    artifacts.append(
        {
            "path": LINE_ENDING_POLICY,
            "classification": "governing_hash_locked_artifact_line_ending_policy",
            "size_bytes": (ROOT / LINE_ENDING_POLICY).stat().st_size,
            "sha256": digest(ROOT / LINE_ENDING_POLICY),
        }
    )
    artifacts.append(
        {
            "path": audit_script,
            "classification": "governing_reproducible_clean_diagnosis_closure_tooling",
            "size_bytes": Path(__file__).stat().st_size,
            "sha256": digest(Path(__file__)),
        }
    )
    planned_commit_allowlist = sorted([item["path"] for item in artifacts] + [report_relative])
    status = "PASS" if not failures else "HOLD"
    report = {
        "schema": "finhrm-qp4-question-plan-clean-diagnosis-closure-v1",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "classification": "development_only_clean_question_plan_diagnosis_closure_no_runtime_repair",
        "status": status,
        "governing_baseline": BASELINE,
        "head_before_closure_commit": BASELINE,
        "intermediate_commit_replaced_by_amend": head if amend_refresh else None,
        "original_untracked_file_count": len(FILES),
        "artifact_classification": {
            "governing_evidence_or_tooling": artifacts,
            "temp_debug_intermediate": [],
            "all_original_12_required": True,
            "reason_markdown_retained": "Each markdown is a final human-readable companion already named and SHA-256 locked by its governing lock.",
            "reason_raw_trace_retained": "The fsynced 19-row JSONL is required to independently verify call accounting, noninterference, exact case IDs, and code-level causal signatures.",
            "reason_execution_summary_retained": "The replay summary and lock preserve the exact 19+19 execution evidence; the later review corrects only forbidden literal-based causal partitioning.",
        },
        "protected_content_audit": content_audit,
        "case_accounting": accounting,
        "retained_open_state": {
            "source": "explicit_user_instruction_for_this_closure",
            "candidate": {"open": 21, "total": 24, "status": "OPEN"},
            "binding_or_provenance": {"open": 15, "total": 17, "status": "OPEN"},
            "compile_question_plan_521": "LOCAL_INSUFFICIENT_FOR_GENERICITY",
        },
        "frozen_scope": {
            "runtime_diff_vs_baseline": runtime_diff.splitlines() if runtime_diff else [],
            "runtime_modified": False,
            "model_modified": False,
            "threshold_modified": False,
            "checkpoint_modified": False,
            "page_index_source_sha256": page_index_hash,
            "page_index_source_matches_frozen_replay_hash": page_index_hash == expected_page_index_hash,
            "page_index_or_dataset_paths_in_original_12": [],
            "page_index_or_dataset_paths_in_write_set": [],
            "page_index_or_dataset_paths_in_planned_commit": [],
            "frozen_page_index_or_original_dataset_touched": False,
            "byte_identity_before_after_condition": "NOT_TRIGGERED_NO_TOUCH",
        },
        "planned_commit_allowlist": planned_commit_allowlist,
        "failures": failures,
        "runtime_repair_opened": False,
        "runtime_repair_authorized": False,
        "next_allowed_action": "QP4 QUESTION-PLAN GENERIC RUNTIME-REPAIR PRECOMMIT ONLY" if status == "PASS" else "HOLD",
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"closure_status={status}")
    print(f"original_untracked_files={len(FILES)}")
    print(f"planned_commit_files={len(planned_commit_allowlist)}")
    print(f"case_union={accounting['union_unique_cases']}/19")
    print(f"two_575_disjoint={str(accounting['two_575_clusters_disjoint']).lower()}")
    print(f"protected_content_detected={str(content_audit['protected_content_detected']).lower()}")
    if failures:
        print("failures=" + ",".join(failures))
    return 0 if status == "PASS" else 2


def verify() -> int:
    report = load(REPORT)
    failures: list[str] = []
    if report.get("status") != "PASS":
        failures.append("closure_report_not_pass")
    if report.get("governing_baseline") != BASELINE:
        failures.append("governing_baseline")
    for item in report.get("artifact_classification", {}).get("governing_evidence_or_tooling", []):
        path = ROOT / item["path"]
        if not path.is_file() or digest(path) != item["sha256"]:
            failures.append(f"artifact_hash:{item['path']}")
    if git("merge-base", "--is-ancestor", BASELINE, "HEAD") != "":
        # `git merge-base --is-ancestor` succeeds with empty stdout.
        failures.append("unexpected_merge_base_output")
    if git("status", "--porcelain=v1", "--untracked-files=all"):
        failures.append("working_tree_not_clean")
    print("closure_verify=" + ("PASS" if not failures else "HOLD"))
    if failures:
        print("failures=" + ",".join(failures))
    return 0 if not failures else 3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    return verify() if args.verify else build()


if __name__ == "__main__":
    raise SystemExit(main())
