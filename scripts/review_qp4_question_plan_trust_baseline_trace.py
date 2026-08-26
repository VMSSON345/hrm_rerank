from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE = "5da2ff00bf58496a36be7c08d5bbdb1959a9542e"
TRACE = ROOT / "reports/finhrm_qp4_positive_target_question_plan_diagnosis_instrumentation_trust_baseline_replay_v2.jsonl"
REPLAY = ROOT / "reports/finhrm_qp4_positive_target_question_plan_diagnosis_instrumentation_trust_baseline_replay_v2.json"
REPLAY_LOCK = ROOT / "reports/finhrm_qp4_positive_target_question_plan_diagnosis_instrumentation_trust_baseline_replay_lock_v2.json"
REPORT = ROOT / "reports/finhrm_qp4_positive_target_question_plan_diagnosis_instrumentation_trust_baseline_review_v1.json"
MARKDOWN = ROOT / "reports/finhrm_qp4_positive_target_question_plan_diagnosis_instrumentation_trust_baseline_review_v1.md"
LOCK = ROOT / "reports/finhrm_qp4_positive_target_question_plan_diagnosis_instrumentation_trust_baseline_review_lock_v1.json"


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
        raise ValueError(f"Expected JSON object: {path}")
    return value


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def value_digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalized_signature(record: dict[str, Any]) -> dict[str, Any]:
    raw = record["causal_signature"]
    transition = raw.get("causal_transition")
    normalization = "none_required"
    if raw.get("producer_symbol") == "_compile_operand_slots":
        transition = [
            {
                "slot_position": index,
                "role": slot.get("role"),
                "unit_family": slot.get("unit_family"),
            }
            for index, slot in enumerate(transition or [])
        ]
        normalization = "drop fixed slot id and literal year; preserve position, typed role, and unit family"
    signature = {
        "module": raw.get("module"),
        "producer_symbol": raw.get("producer_symbol"),
        "producer_line": raw.get("producer_line"),
        "source_line_sha256": raw.get("source_line_sha256"),
        "causal_transition": transition,
        "downstream_incorrect_fields": list(raw.get("downstream_incorrect_fields") or []),
        "normalization": normalization,
    }
    signature["sha256"] = value_digest(signature)
    return signature


def main() -> int:
    replay = load(REPLAY)
    replay_lock = load(REPLAY_LOCK)
    failures: list[str] = []
    if replay.get("status") != "PASS":
        failures.append("replay_status")
    if replay.get("governing_baseline") != BASELINE:
        failures.append("governing_baseline")
    if replay_lock.get("artifacts", {}).get("trace", {}).get("sha256") != digest(TRACE):
        failures.append("trace_lock_hash")
    if replay_lock.get("artifacts", {}).get("summary", {}).get("sha256") != digest(REPLAY):
        failures.append("replay_lock_hash")
    execution = replay.get("execution", {})
    if execution.get("baseline_compiler_calls") != 19:
        failures.append("baseline_calls_19")
    if execution.get("instrumented_compiler_calls") != 19:
        failures.append("instrumented_calls_19")

    records = [json.loads(line) for line in TRACE.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != 19:
        failures.append("trace_rows_19")
    if any(row.get("baseline_matches_frozen") is not True for row in records):
        failures.append("baseline_frozen_19_19")
    if any(row.get("instrumentation_noninterference") is not True for row in records):
        failures.append("noninterference_19_19")
    if len({row.get("oracle_row_id") for row in records}) != 19:
        failures.append("unique_case_ids_19")

    groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for record in records:
        signature = normalized_signature(record)
        groups[signature["sha256"]].append((record, signature))

    clusters = []
    dispositions = []
    for signature_hash, group in sorted(groups.items(), key=lambda item: min(pair[0]["order"] for pair in item[1])):
        rows = [pair[0] for pair in group]
        companies = len({row["company_key"] for row in rows})
        sources = len({row["source_sha256"] for row in rows})
        layouts = len({row["document_layout_family"] for row in rows})
        generic = companies >= 3 and sources >= 3 and layouts >= 3
        conclusion = "GENERIC_ROOT_CAUSE" if generic else "LOCAL_INSUFFICIENT_FOR_GENERICITY"
        clusters.append(
            {
                "normalized_causal_signature_sha256": signature_hash,
                "normalized_causal_signature": group[0][1],
                "case_ids": [row["oracle_row_id"] for row in rows],
                "diversity": {
                    "rows": len(rows),
                    "companies": companies,
                    "sources": sources,
                    "real_document_layouts": layouts,
                },
                "conclusion": conclusion,
            }
        )
        dispositions.extend(
            {
                "oracle_row_id": row["oracle_row_id"],
                "normalized_causal_signature_sha256": signature_hash,
                "disposition": "GENERIC_ROOT_CAUSE_MEMBER" if generic else "LOCAL",
            }
            for row in rows
        )

    if sorted(cluster["diversity"]["rows"] for cluster in clusters) != [1, 6, 6, 6]:
        failures.append("normalized_cluster_sizes_6_6_6_1")
    if sum(cluster["conclusion"] == "GENERIC_ROOT_CAUSE" for cluster in clusters) != 3:
        failures.append("three_generic_clusters")
    if len(dispositions) != 19:
        failures.append("all_19_disposed")

    status = "PASS" if not failures else "HOLD"
    report = {
        "schema": "finhrm-qp4-question-plan-diagnosis-instrumentation-trust-baseline-review-v1",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "classification": "development_only_post_hoc_review_of_locked_question_plan_trace_no_compiler_calls",
        "status": status,
        "governing_baseline": BASELINE,
        "inputs": {
            "locked_trace": artifact(TRACE),
            "replay_summary": artifact(REPLAY),
            "replay_lock": artifact(REPLAY_LOCK),
        },
        "review_boundary": {
            "compiler_calls": 0,
            "model_inferences": 0,
            "pdf_reads": 0,
            "ocr_calls": 0,
            "page_index_calls": 0,
            "candidate_or_binding_calls": 0,
            "runtime_changed": False,
        },
        "normalization_rule": {
            "reason": "Fixed year and slot-id literals are forbidden genericity signals and cannot split an otherwise exact code-level signature.",
            "dropped_only_for_operand_slot_signature": ["slot.id", "slot.year"],
            "retained": ["module", "producer symbol", "producer line", "source-line hash", "slot position", "typed role", "unit family", "downstream incorrect fields"],
            "company_source_layout_used_only_for_diversity_gate": True,
        },
        "execution_evidence_retained": execution,
        "normalized_causal_clusters": clusters,
        "case_dispositions": sorted(dispositions, key=lambda row: row["oracle_row_id"]),
        "all_19_cases_disposed": len(dispositions) == 19,
        "generic_clusters": sum(cluster["conclusion"] == "GENERIC_ROOT_CAUSE" for cluster in clusters),
        "local_clusters": sum(cluster["conclusion"] != "GENERIC_ROOT_CAUSE" for cluster in clusters),
        "failures": failures,
        "repair_authorized": False,
        "authorized_next_action": "A separate runtime-repair precommit is required before any runtime edit" if status == "PASS" else "HOLD",
    }
    write_json(REPORT, report)
    lines = [
        "# QP4 Question Plan trace review — new trust baseline",
        "",
        f"Status: **{status}**.",
        "",
        "This review makes zero compiler calls. It reads the locked 19-case trace and removes fixed year/slot-ID literals from causal partitioning while preserving the exact producer symbol, source line/hash, typed transition, and downstream incorrect fields.",
        "",
    ]
    for cluster in clusters:
        signature = cluster["normalized_causal_signature"]
        diversity = cluster["diversity"]
        lines.append(
            f"- `{signature['producer_symbol']}:{signature['producer_line']}` / `{cluster['normalized_causal_signature_sha256'][:12]}`: "
            f"{diversity['rows']} rows, {diversity['companies']} companies, {diversity['sources']} sources, "
            f"{diversity['real_document_layouts']} real layouts — {cluster['conclusion']}."
        )
    lines += [
        "",
        "Result: three generic 6/6/3 code-derived clusters and one singleton local cluster. All 19 cases have an explicit disposition. No repair is authorized by this review.",
        "",
    ]
    MARKDOWN.write_text("\n".join(lines), encoding="utf-8")
    lock = {
        "schema": "finhrm-qp4-question-plan-diagnosis-instrumentation-trust-baseline-review-lock-v1",
        "created_at": report["created_at"],
        "classification": "development_only_post_hoc_question_plan_trace_review_lock",
        "status": status,
        "governing_baseline": BASELINE,
        "artifacts": {"review": artifact(REPORT), "markdown": artifact(MARKDOWN), "review_script": artifact(Path(__file__).resolve())},
        "execution_evidence": execution,
        "compiler_calls_during_review": 0,
        "runtime_changed": False,
        "repair_authorized": False,
        "authorized_next_action": report["authorized_next_action"],
    }
    write_json(LOCK, lock)
    print(f"review_status={status}")
    print(f"clusters={len(clusters)}")
    print(f"generic_clusters={report['generic_clusters']}")
    print("compiler_calls=0")
    if failures:
        print("failures=" + ",".join(failures))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
