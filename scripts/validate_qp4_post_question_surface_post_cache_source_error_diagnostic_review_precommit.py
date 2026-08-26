#!/usr/bin/env python3
"""Validate the artifact-only QP4 post-cache source-error review precommit."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/finhrm_qp4_post_question_surface_post_cache_source_error_diagnostic_review_precommit_v1.json"
MARKDOWN = ROOT / "reports/finhrm_qp4_post_question_surface_post_cache_source_error_diagnostic_review_precommit_v1.md"
LOCK = ROOT / "reports/finhrm_qp4_post_question_surface_post_cache_source_error_diagnostic_review_precommit_lock_v1.json"


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


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
        for layer, layer_rows in grouped.items()
    }


def validate() -> dict[str, Any]:
    config = _load(CONFIG)
    failed: list[str] = []

    def check(value: bool, name: str) -> None:
        if not value:
            failed.append(name)

    for name, parent in config["parents"].items():
        path = ROOT / parent["path"]
        check(path.is_file() and _digest(path) == parent["sha256"], f"parent:{name}")

    report = _load(ROOT / config["parents"]["materialization_report"]["path"])
    parent_lock = _load(ROOT / config["parents"]["materialization_lock"]["path"])
    frozen = config["frozen_parent_state"]
    expected_failed = frozen["failed_gates"]

    check(report["status"] == frozen["report_status"], "parent_report_status")
    check(parent_lock["status"] == frozen["lock_status"], "parent_lock_status")
    check(report["implementation_gate_passed"] is frozen["implementation_gate_passed"], "parent_report_gate")
    check(parent_lock["implementation_gate_passed"] is frozen["implementation_gate_passed"], "parent_lock_gate")
    check(report["failed_gates"] == expected_failed, "parent_report_failed_gates")
    check(parent_lock["failed_gates"] == expected_failed, "parent_lock_failed_gates")
    check(
        sorted(name for name, passed in report["checks"].items() if not passed) == sorted(expected_failed),
        "only_expected_parent_checks_failed",
    )
    check(
        parent_lock["authorized_next_step"].startswith(
            "QP4-POST-QUESTION-SURFACE-POST-CACHE-SOURCE-ERROR-DIAGNOSTIC-REVIEW-PRECOMMIT"
        ),
        "authorized_parent_step",
    )

    selection = frozen["selection"]
    for key in ("observations", "companies", "sources", "real_layouts", "question_triggered", "whole_corpus_ocr"):
        check(report["selection"][key] == selection[key], f"selection:{key}")

    replay = report["artifact_only_replay"]
    for key, expected in frozen["replay"].items():
        check(replay[key] == expected, f"replay:{key}")
    rows = replay["rows"]
    check(len(rows) == frozen["selection"]["observations"], "eight_observations")
    check(
        len({(row["source_sha256"], row["variant"]) for row in rows}) == len(rows),
        "observations_partitioned_once",
    )

    observed_coverage = _coverage(rows)
    expected_coverage = {
        row["first_failing_layer"]: {
            key: row[key]
            for key in ("queries", "companies", "sources", "real_layouts")
        }
        for row in config["coverage_review"]["strata"]
    }
    check(observed_coverage == expected_coverage, "coverage_recomputed_from_rows")
    check(sum(row["queries"] for row in expected_coverage.values()) == len(rows), "coverage_partitions_all_rows")

    minimum = config["coverage_review"]["generic_evidence_minimum"]
    generic_layers = [
        layer
        for layer, coverage in observed_coverage.items()
        if coverage["companies"] >= minimum["companies"]
        and coverage["sources"] >= minimum["sources"]
        and coverage["real_layouts"] >= minimum["real_layouts"]
    ]
    check(generic_layers == config["coverage_review"]["generic_downstream_strata"], "no_generic_residual_stratum")
    check(config["coverage_review"]["runtime_evidence_sufficient"] is False, "runtime_evidence_insufficient")

    source_rows = [row for row in rows if row["first_failing_layer"] == "SOURCE_LOAD_OR_EXTRACTION"]
    check({row["variant"] for row in source_rows} == set(config["coverage_review"]["source_error_variants"]), "source_variants")
    check(all(row["source_outcome"] == "ERROR" for row in source_rows), "source_outcomes")
    check(len({row["company_key"] for row in source_rows}) == 1, "source_error_one_company")
    check(len({row["source_sha256"] for row in source_rows}) == 1, "source_error_one_source")
    check(len({row["document_layout_family"] for row in source_rows}) == 1, "source_error_one_layout")
    check(replay["generic_downstream_strata"] == [], "parent_generic_strata_empty")

    manifest = [artifact for source in report["cache_manifest"] for artifact in source["artifacts"]]
    expected_cache = frozen["cache_manifest"]
    check(len(report["cache_manifest"]) == expected_cache["source_documents"], "cache_source_count")
    check(len(manifest) == expected_cache["structured_content_list_artifacts"], "cache_artifact_count")
    check(
        sum(not artifact["content_list"] for artifact in manifest) == expected_cache["non_content_list_artifacts"],
        "cache_content_list_only",
    )
    for artifact in manifest:
        path = ROOT / artifact["path"]
        check(path.is_file(), f"cache_exists:{artifact['sha256']}")
        if path.is_file():
            check(path.stat().st_size == artifact["size_bytes"], f"cache_size:{artifact['sha256']}")
            check(_digest(path) == artifact["sha256"], f"cache_hash:{artifact['sha256']}")

    decision = config["review_decision"]
    execution = config["authorized_execution_contract"]
    check(decision["runtime_edit_authorized"] is False, "runtime_not_authorized")
    check(decision["runtime_file_whitelist"] == [], "runtime_whitelist_empty")
    check(all(decision[key] is False for key in (
        "ocr_or_layout_algorithm_change_authorized",
        "page_selection_change_authorized",
        "retrieval_change_authorized",
        "semantic_binding_change_authorized",
        "symbolic_verifier_change_authorized",
        "model_or_checkpoint_change_authorized",
    )), "no_algorithm_change_authorized")
    check("run MinerU or create additional OCR artifacts" in execution["forbidden_actions"], "ocr_forbidden")
    check("rerun the completed materialization/source replay" in execution["forbidden_actions"], "replay_forbidden")

    check(report["compiler_control"] == execution["required_checks"]["compiler_inventory"], "compiler_control")
    check(report["regression"]["passed"] is True, "parent_regression_passed")
    check(
        f"{report['regression']['observed']}/{report['regression']['expected']}" == execution["required_checks"]["regression"],
        "parent_regression_count",
    )
    safe = report["safe_replay"]
    check(safe["gate"] is True, "parent_safe_replay")
    check(safe["execution"] == execution["required_checks"]["gold_execution"], "gold_execution")
    check(safe["permutation"] == execution["required_checks"]["candidate_permutation"], "candidate_permutation")
    check(safe["counterfactuals"] == execution["required_checks"]["frozen_counterfactuals"], "counterfactual_count")
    check(safe["counterfactual_false_accepts"] == 0, "counterfactual_false_accepts")

    for relative, expected in config["frozen_runtime_hashes"].items():
        check(_digest(ROOT / relative) == expected, f"runtime:{relative}")
        check(report["runtime_hashes"].get(relative) == expected, f"parent_runtime:{relative}")
    check(_digest(ROOT / config["checkpoint"]["path"]) == config["checkpoint"]["sha256"], "checkpoint")
    check(report["checkpoint_sha256"] == config["checkpoint"]["sha256"], "parent_checkpoint")
    check(all(value is False for value in config["invariants"].values()), "invariants")
    check(config["artifact_policy"]["raw_or_intermediate_artifacts_persisted"] is False, "artifact_policy")

    return {
        "precommit_gate_passed": not failed,
        "failed_gates": failed,
        "coverage": observed_coverage,
        "generic_residual_strata": generic_layers,
        "runtime_changed": False,
        "runtime_edit_authorized": False,
        "runtime_file_whitelist": [],
        "ocr_rerun_performed": False,
        "source_replay_rerun_performed": False,
        "authorized_next_step": (
            execution["step"]
            if not failed
            else "Resolve post-cache source-error diagnostic-review precommit failures without changing runtime or rerunning OCR."
        ),
    }


def main() -> int:
    config = _load(CONFIG)
    result = validate()
    coverage = result["coverage"]
    MARKDOWN.write_text(
        "# QP4 post-cache source-error diagnostic-review precommit v1\n\n"
        f"- Gate: `{'PASS' if result['precommit_gate_passed'] else 'FAIL'}`; runtime unchanged, OCR/source replay not rerun, and runtime whitelist empty.\n"
        f"- Source residual: `{coverage.get('SOURCE_LOAD_OR_EXTRACTION', {}) .get('queries', 0)}` queries over `1` company / `1` PDF / `1` real layout. Repeatable within one source, but not generic.\n"
        "- Other residual coverage: candidate/retrieval `2/2/2`; semantic binding/program synthesis `3/3/2` (companies/PDFs/real layouts). Neither reaches the frozen `3/3/3` minimum.\n"
        "- Decision: authorize only an artifact-only diagnostic review from the final parent report and lock. No OCR, PDF read, source replay, runtime edit, threshold/model/checkpoint change, or report-specific rule is authorized.\n"
        "- Frozen parent controls remain: compiler `242/242`, regression `360/360`, execution `363/363`, permutation `363/363`, and `141` counterfactuals with zero false accepts.\n"
        "- QP4 remains incomplete; QP5 and Phase D remain locked.\n",
        encoding="utf-8",
    )
    artifacts = [_artifact(CONFIG), _artifact(Path(__file__).resolve()), _artifact(MARKDOWN)]
    lock = {
        "schema": "finhrm-qp4-post-question-surface-post-cache-source-error-diagnostic-review-precommit-lock-v1",
        "created_at": config["created_at"],
        "classification": config["classification"],
        "status": "complete" if result["precommit_gate_passed"] else "failed",
        **result,
        "parent_report_sha256": config["parents"]["materialization_report"]["sha256"],
        "parent_lock_sha256": config["parents"]["materialization_lock"]["sha256"],
        "runtime_hashes": config["frozen_runtime_hashes"],
        "checkpoint_sha256": config["checkpoint"]["sha256"],
        "artifact_policy": config["artifact_policy"],
        "artifacts": artifacts,
        "qp4_complete": False,
        "qp5_authorized": False,
        "phase_d": "locked",
    }
    LOCK.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["precommit_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
