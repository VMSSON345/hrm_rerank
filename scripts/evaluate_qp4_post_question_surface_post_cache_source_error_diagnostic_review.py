#!/usr/bin/env python3
"""Review frozen QP4 post-cache residuals without OCR or runtime execution."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PRECOMMIT = ROOT / "config/finhrm_qp4_post_question_surface_post_cache_source_error_diagnostic_review_precommit_v1.json"
PRECOMMIT_LOCK = ROOT / "reports/finhrm_qp4_post_question_surface_post_cache_source_error_diagnostic_review_precommit_lock_v1.json"
REPORT = ROOT / "reports/finhrm_qp4_post_question_surface_post_cache_source_error_diagnostic_review_v1.json"
MARKDOWN = ROOT / "reports/finhrm_qp4_post_question_surface_post_cache_source_error_diagnostic_review_v1.md"
LOCK = ROOT / "reports/finhrm_qp4_post_question_surface_post_cache_source_error_diagnostic_review_lock_v1.json"


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


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


def _coverage(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["first_failing_layer"]].append(row)
    return {
        layer: {
            "queries": len(layer_rows),
            "companies": len({row["company_key"] for row in layer_rows}),
            "sources": len({row["source_sha256"] for row in layer_rows}),
            "real_layouts": len({row["document_layout_family"] for row in layer_rows}),
        }
        for layer, layer_rows in sorted(grouped.items())
    }


def _strata(
    rows: list[dict[str, Any]], minimum: dict[str, int]
) -> list[dict[str, Any]]:
    coverage = _coverage(rows)
    result: list[dict[str, Any]] = []
    for layer, counts in coverage.items():
        missing = [
            key
            for key in ("companies", "sources", "real_layouts")
            if counts[key] < minimum[key]
        ]
        layer_rows = [row for row in rows if row["first_failing_layer"] == layer]
        result.append(
            {
                "first_failing_layer": layer,
                **counts,
                "source_actions": dict(
                    sorted(Counter(row["source_action"] or "NO_ACTION" for row in layer_rows).items())
                ),
                "verification_states": dict(
                    sorted(Counter(row["verification_state"] or "NO_STATE" for row in layer_rows).items())
                ),
                "missing_generic_dimensions": missing,
                "generic_evidence": not missing,
                "runtime_candidate": False,
            }
        )
    return result


def _verify_authorization(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    lock = _load(PRECOMMIT_LOCK)
    execution = config["authorized_execution_contract"]
    if lock.get("status") != "complete" or not lock.get("precommit_gate_passed"):
        raise ValueError("Post-cache diagnostic-review precommit is not closed")
    if lock.get("failed_gates") != []:
        raise ValueError("Post-cache diagnostic-review precommit has failed gates")
    if lock.get("authorized_next_step") != execution["step"]:
        raise ValueError("Post-cache diagnostic-review authorization mismatch")
    if lock.get("runtime_file_whitelist") != [] or lock.get("runtime_edit_authorized"):
        raise ValueError("Diagnostic-review runtime whitelist must remain empty")
    if lock.get("ocr_rerun_performed") or lock.get("source_replay_rerun_performed"):
        raise ValueError("Precommit unexpectedly reran OCR or source replay")
    config_artifacts = {
        row["path"]: row["sha256"] for row in lock.get("artifacts", [])
    }
    relative_config = PRECOMMIT.relative_to(ROOT).as_posix()
    if config_artifacts.get(relative_config) != _digest(PRECOMMIT):
        raise ValueError("Precommit config hash is not locked")

    for name, parent in config["parents"].items():
        path = ROOT / parent["path"]
        if not path.is_file() or _digest(path) != parent["sha256"]:
            raise ValueError(f"Frozen parent hash mismatch: {name}")
    for relative, expected in config["frozen_runtime_hashes"].items():
        if _digest(ROOT / relative) != expected:
            raise ValueError(f"Runtime changed before diagnostic review: {relative}")
    checkpoint = ROOT / config["checkpoint"]["path"]
    if _digest(checkpoint) != config["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint changed before diagnostic review")

    report = _load(ROOT / config["parents"]["materialization_report"]["path"])
    parent_lock = _load(ROOT / config["parents"]["materialization_lock"]["path"])
    return report, parent_lock


def run() -> dict[str, Any]:
    config = _load(PRECOMMIT)
    parent, parent_lock = _verify_authorization(config)
    frozen = config["frozen_parent_state"]
    execution = config["authorized_execution_contract"]
    required = execution["required_checks"]
    rows = parent["artifact_only_replay"]["rows"]
    coverage = _coverage(rows)
    minimum = config["coverage_review"]["generic_evidence_minimum"]
    strata = _strata(rows, minimum)
    generic = [row for row in strata if row["generic_evidence"]]

    source_rows = [
        row for row in rows if row["first_failing_layer"] == "SOURCE_LOAD_OR_EXTRACTION"
    ]
    materialization_errors: list[dict[str, Any]] = []
    for source in parent["materialization"]["queries"]:
        for query in source["queries"]:
            if query["outcome"] == "ERROR":
                materialization_errors.append(
                    {
                        "source_sha256": source["source_sha256"],
                        "company_key": source["company_key"],
                        "document_layout_family": source["document_layout_family"],
                        "variant": query["variant"],
                        "error_type": query.get("error_type"),
                        "terminal_stage": query.get("stages", [None])[-1],
                        "worker_disconnect_visible": "Server disconnected without sending a response"
                        in query.get("error_message", ""),
                    }
                )

    cache_artifacts = [
        artifact
        for source in parent["cache_manifest"]
        for artifact in source["artifacts"]
    ]
    cache_hashes_exact = True
    for artifact in cache_artifacts:
        path = ROOT / artifact["path"]
        if (
            not path.is_file()
            or path.stat().st_size != artifact["size_bytes"]
            or _digest(path) != artifact["sha256"]
        ):
            cache_hashes_exact = False
            break

    expected_coverage = {
        row["first_failing_layer"]: {
            key: row[key]
            for key in ("queries", "companies", "sources", "real_layouts")
        }
        for row in config["coverage_review"]["strata"]
    }
    runtime_hashes_exact = all(
        _digest(ROOT / relative) == expected
        and parent["runtime_hashes"].get(relative) == expected
        and parent_lock["runtime_hashes"].get(relative) == expected
        for relative, expected in config["frozen_runtime_hashes"].items()
    )
    source_signature_matches = (
        len(materialization_errors) == len(source_rows) == 2
        and {
            (
                row["source_sha256"],
                row["company_key"],
                row["document_layout_family"],
                row["variant"],
            )
            for row in materialization_errors
        }
        == {
            (
                row["source_sha256"],
                row["company_key"],
                row["document_layout_family"],
                row["variant"],
            )
            for row in source_rows
        }
    )
    checks = {
        "parent_hashes_exact": True,
        "parent_failed_gates_exact": (
            parent["status"] == frozen["report_status"]
            and parent_lock["status"] == frozen["lock_status"]
            and parent["implementation_gate_passed"] is False
            and parent_lock["implementation_gate_passed"] is False
            and parent["failed_gates"] == frozen["failed_gates"]
            and parent_lock["failed_gates"] == frozen["failed_gates"]
        ),
        "all_eight_observations_partitioned_once": (
            len(rows) == frozen["selection"]["observations"]
            and len({(row["source_sha256"], row["variant"]) for row in rows}) == len(rows)
        ),
        "coverage_recomputed_from_rows": coverage == expected_coverage,
        "source_error_coverage": coverage.get("SOURCE_LOAD_OR_EXTRACTION")
        == {"queries": 2, "companies": 1, "sources": 1, "real_layouts": 1},
        "candidate_retrieval_coverage": coverage.get("CANDIDATE_POPULATION_OR_RETRIEVAL")
        == {"queries": 2, "companies": 2, "sources": 2, "real_layouts": 2},
        "semantic_binding_coverage": coverage.get("SEMANTIC_BINDING_OR_PROGRAM_SYNTHESIS")
        == {"queries": 4, "companies": 3, "sources": 3, "real_layouts": 2},
        "generic_residual_strata": len(generic) == required["generic_residual_strata"],
        "source_error_signature_matches_frozen_rows": source_signature_matches,
        "source_error_repeatable_only_within_one_source": (
            {row["variant"] for row in materialization_errors} == {"base", "mutated"}
            and len({row["company_key"] for row in materialization_errors}) == 1
            and len({row["source_sha256"] for row in materialization_errors}) == 1
            and len({row["document_layout_family"] for row in materialization_errors}) == 1
        ),
        "source_error_is_document_source_error": all(
            row["error_type"] == "DocumentSourceError" for row in materialization_errors
        ),
        "source_error_stops_after_mineru_retrieval": all(
            row["terminal_stage"] == "mineru_retrieval" for row in materialization_errors
        ),
        "worker_disconnect_signature_visible": all(
            row["worker_disconnect_visible"] for row in materialization_errors
        ),
        "cache_manifest_hashes_exact": (
            cache_hashes_exact
            and len(cache_artifacts)
            == frozen["cache_manifest"]["structured_content_list_artifacts"]
            and all(row["content_list"] for row in cache_artifacts)
        ),
        "accepted_wrong": parent["artifact_only_replay"]["accepted_wrong"]
        == required["accepted_wrong"],
        "counterfactual_false_accepts": parent["artifact_only_replay"]["counterfactual_false_accepts"]
        == required["counterfactual_false_accepts"],
        "compiler_inventory": parent["compiler_control"] == required["compiler_inventory"],
        "regression": (
            parent["regression"]["passed"] is True
            and f"{parent['regression']['observed']}/{parent['regression']['expected']}"
            == required["regression"]
        ),
        "gold_execution": parent["safe_replay"]["execution"] == required["gold_execution"],
        "candidate_permutation": parent["safe_replay"]["permutation"]
        == required["candidate_permutation"],
        "frozen_counterfactuals": parent["safe_replay"]["counterfactuals"]
        == required["frozen_counterfactuals"],
        "frozen_counterfactual_false_accepts": parent["safe_replay"]["counterfactual_false_accepts"]
        == required["frozen_counterfactual_false_accepts"],
        "frozen_safe_replay_gate": parent["safe_replay"]["gate"]
        is required["frozen_safe_replay_gate"],
        "runtime_hashes_exact": runtime_hashes_exact,
        "checkpoint_hash_exact": (
            _digest(ROOT / config["checkpoint"]["path"])
            == config["checkpoint"]["sha256"]
            == parent["checkpoint_sha256"]
            == parent_lock["checkpoint_sha256"]
        ),
        "runtime_whitelist_empty": config["review_decision"]["runtime_file_whitelist"] == [],
        "runtime_edit_not_authorized": config["review_decision"]["runtime_edit_authorized"] is False,
        "ocr_and_source_replay_not_rerun": True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    complete = not failed
    next_step = (
        "QP4-POST-QUESTION-SURFACE-POST-CACHE-RESIDUAL-COVERAGE-COHORT-PRECOMMIT: freeze a development-safe, semantically coherent company/source/layout-disjoint cohort before any further OCR, source replay, or runtime proposal."
        if complete
        else "Resolve the same artifact-only diagnostic-review gate without OCR, source replay, or runtime change."
    )

    return {
        "schema": "finhrm-qp4-post-question-surface-post-cache-source-error-diagnostic-review-report-v1",
        "created_at": config["created_at"],
        "classification": "development_only_qp4_post_cache_source_error_diagnostic_review_runtime_unchanged",
        "status": "DIAGNOSTIC_COMPLETE" if complete else "DIAGNOSTIC_GATE_FAILED",
        "diagnostic_gate_passed": complete,
        "failed_gates": failed,
        "checks": checks,
        "selection": frozen["selection"],
        "residual_strata": strata,
        "generic_residual_strata": generic,
        "source_error_diagnostic": {
            "queries": len(materialization_errors),
            "companies": len({row["company_key"] for row in materialization_errors}),
            "sources": len({row["source_sha256"] for row in materialization_errors}),
            "real_layouts": len({row["document_layout_family"] for row in materialization_errors}),
            "variants": sorted({row["variant"] for row in materialization_errors}),
            "error_types": dict(sorted(Counter(row["error_type"] for row in materialization_errors).items())),
            "terminal_stages": dict(sorted(Counter(row["terminal_stage"] for row in materialization_errors).items())),
            "observed_signature": "MINERU_WORKER_DISCONNECT_AFTER_RETRIEVAL_STAGE",
            "repeatable_within_one_source": True,
            "generic_across_company_source_layout": False,
            "root_cause_established": False,
            "interpretation": "The wrapper error and worker-disconnect signature are reproducible for two variants of one development source. This is insufficient to attribute a generic OCR, page-selection, source-loader, retrieval, or layout defect.",
            "fail_closed_policy": "Retain both failures in the denominator with no ANSWER action; do not infer period, provenance, or intended pages from the question or nearby context.",
        },
        "cache_manifest": {
            "sources": len(parent["cache_manifest"]),
            "structured_content_list_artifacts": len(cache_artifacts),
            "hashes_exact": cache_hashes_exact,
            "whole_corpus_ocr_performed": False,
            "new_ocr_artifacts_created": 0,
        },
        "retained_controls": {
            "compiler": parent["compiler_control"],
            "regression": f"{parent['regression']['observed']}/{parent['regression']['expected']}",
            "safe_replay_gate": parent["safe_replay"]["gate"],
            "actions": parent["safe_replay"]["actions"],
            "execution": parent["safe_replay"]["execution"],
            "permutation": parent["safe_replay"]["permutation"],
            "counterfactuals": parent["safe_replay"]["counterfactuals"],
            "counterfactual_false_accepts": parent["safe_replay"]["counterfactual_false_accepts"],
            "accepted_wrong": parent["artifact_only_replay"]["accepted_wrong"],
        },
        "decision": {
            "attribution": config["review_decision"]["attribution"],
            "generic_runtime_evidence_sufficient": False,
            "runtime_changed": False,
            "runtime_edit_authorized": False,
            "runtime_file_whitelist": [],
            "implementation_precommit_authorized": False,
            "development_coverage_cohort_precommit_authorized": complete,
        },
        "runtime_hashes": config["frozen_runtime_hashes"],
        "checkpoint_sha256": config["checkpoint"]["sha256"],
        "artifact_policy": config["artifact_policy"],
        "invariants": config["invariants"],
        "ocr_rerun_performed": False,
        "source_replay_rerun_performed": False,
        "qp4_complete": False,
        "qp5_authorized": False,
        "phase_d": "locked",
        "authorized_next_step": next_step,
    }


def _markdown(report: dict[str, Any]) -> str:
    source = report["source_error_diagnostic"]
    coverage = {
        row["first_failing_layer"]: row for row in report["residual_strata"]
    }
    retrieval = coverage["CANDIDATE_POPULATION_OR_RETRIEVAL"]
    binding = coverage["SEMANTIC_BINDING_OR_PROGRAM_SYNTHESIS"]
    return (
        "# QP4 post-cache source-error diagnostic review v1\n\n"
        f"- Review gate: `{'PASS' if report['diagnostic_gate_passed'] else 'FAIL'}`; parent QP4 implementation gate remains failed.\n"
        f"- Source error: `{source['queries']}` queries, `{source['companies']}` company, `{source['sources']}` PDF, `{source['real_layouts']}` real layout. The worker-disconnect signature repeats within one source but is not generic evidence.\n"
        f"- Residual coverage: source `1/1/1`, candidate/retrieval `{retrieval['companies']}/{retrieval['sources']}/{retrieval['real_layouts']}`, binding/program synthesis `{binding['companies']}/{binding['sources']}/{binding['real_layouts']}`. No stratum reaches `3/3/3`.\n"
        "- Decision: runtime whitelist remains empty. No OCR/layout, page-selection, retrieval, binding, verifier, model, checkpoint, or threshold change is authorized.\n"
        f"- Frozen controls: compiler `{report['retained_controls']['compiler']}`, regression `{report['retained_controls']['regression']}`, execution/permutation `{report['retained_controls']['execution']}/{report['retained_controls']['permutation']}`, counterfactual false accepts `{report['retained_controls']['counterfactual_false_accepts']}`.\n"
        "- No PDF/OCR content or raw trace was persisted. QP4 remains incomplete; QP5 and Phase D remain locked.\n"
    )


def write_outputs(report: dict[str, Any]) -> None:
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN.write_text(_markdown(report), encoding="utf-8")
    artifacts = [
        _artifact(PRECOMMIT),
        _artifact(PRECOMMIT_LOCK),
        _artifact(Path(__file__).resolve()),
        _artifact(REPORT),
        _artifact(MARKDOWN),
    ]
    lock = {
        "schema": "finhrm-qp4-post-question-surface-post-cache-source-error-diagnostic-review-lock-v1",
        "created_at": report["created_at"],
        "classification": report["classification"],
        "status": "complete" if report["diagnostic_gate_passed"] else "failed",
        "diagnostic_gate_passed": report["diagnostic_gate_passed"],
        "failed_gates": report["failed_gates"],
        "residual_strata": report["residual_strata"],
        "generic_residual_strata": report["generic_residual_strata"],
        "source_error_diagnostic": report["source_error_diagnostic"],
        "decision": report["decision"],
        "retained_controls": report["retained_controls"],
        "runtime_changed": False,
        "runtime_edit_authorized": False,
        "runtime_file_whitelist": [],
        "ocr_rerun_performed": False,
        "source_replay_rerun_performed": False,
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
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("--run is required")
    report = run()
    write_outputs(report)
    print(
        json.dumps(
            {
                "diagnostic_gate_passed": report["diagnostic_gate_passed"],
                "failed_gates": report["failed_gates"],
                "generic_residual_strata": len(report["generic_residual_strata"]),
                "source_error_coverage": {
                    key: report["source_error_diagnostic"][key]
                    for key in ("queries", "companies", "sources", "real_layouts")
                },
                "runtime_edit_authorized": report["decision"]["runtime_edit_authorized"],
                "authorized_next_step": report["authorized_next_step"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report["diagnostic_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
