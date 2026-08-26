#!/usr/bin/env python3
"""Validate the artifact-minimal QP4 source/retrieval roadmap-review precommit."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/finhrm_qp4_post_question_surface_source_retrieval_roadmap_review_precommit_v1.json"
MARKDOWN = ROOT / "reports/finhrm_qp4_post_question_surface_source_retrieval_roadmap_review_precommit_v1.md"
LOCK = ROOT / "reports/finhrm_qp4_post_question_surface_source_retrieval_roadmap_review_precommit_lock_v1.json"


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


def validate() -> dict[str, Any]:
    config = _load(CONFIG)
    failed: list[str] = []

    def check(value: bool, name: str) -> None:
        if not value:
            failed.append(name)

    for name, parent in config["parents"].items():
        path = ROOT / parent["path"]
        check(path.is_file() and _digest(path) == parent["sha256"], f"parent:{name}")

    diagnostic = _load(ROOT / config["parents"]["diagnostic_report"]["path"])
    diagnostic_lock = _load(ROOT / config["parents"]["diagnostic_lock"]["path"])
    baseline = _load(ROOT / config["parents"]["actual_plan_source_baseline"]["path"])
    evidence = config["review_evidence"]
    expected_error = evidence["observed_source_error_message"]

    strata = [
        row
        for row in diagnostic["source_or_retrieval_strata_outside_current_scope"]
        if row["track"] == "ACTUAL_PLAN"
        and row["group"] == "REFERENCE_BASE"
        and row["first_failing_layer"] == "SOURCE_LOAD_OR_EXTRACTION"
    ]
    check(len(strata) == 1, "one_generic_source_stratum")
    if strata:
        stratum = strata[0]
        check(stratum["generic_evidence"] is True, "stratum_generic")
        for key in ("queries", "companies", "sources", "real_layouts"):
            expected_key = {
                "queries": "observations",
                "companies": "companies",
                "sources": "source_documents",
                "real_layouts": "real_layout_families",
            }[key]
            check(stratum[key] == evidence[expected_key], f"coverage:{key}")

    reference_rows = [row for row in baseline["results"] if row["dimension"] == "REFERENCE_BASE"]
    check(len(reference_rows) == 4, "reference_base_anchor_count")
    check(len({row["company_key"] for row in reference_rows}) == 4, "company_count")
    check(len({row["source_sha256"] for row in reference_rows}) == 4, "source_count")
    check(len({row["document_layout_family"] for row in reference_rows}) == 3, "layout_count")
    check(
        all(
            row["base"].get("error", {}).get("type") == evidence["observed_source_error_type"]
            and row["base"].get("error", {}).get("message") == expected_error
            for row in reference_rows
        ),
        "uniform_cache_availability_error",
    )
    minimum = evidence["generic_evidence_minimum"]
    check(evidence["companies"] >= minimum["companies"], "minimum_companies")
    check(evidence["source_documents"] >= minimum["source_documents"], "minimum_sources")
    check(evidence["real_layout_families"] >= minimum["real_layout_families"], "minimum_layouts")

    review = config["roadmap_review"]
    execution = config["authorized_execution_contract"]
    check(review["runtime_edit_authorized"] is False, "runtime_not_authorized")
    check(review["runtime_file_whitelist"] == [], "empty_runtime_whitelist")
    check(review["broad_ocr_or_layout_algorithm_extension_authorized"] is False, "no_broad_ocr_extension")
    check(review["retrieval_algorithm_change_authorized"] is False, "no_retrieval_change")
    check(execution["materialization_phase"]["runtime_code_change_allowed"] is False, "materialization_no_runtime_change")
    check(execution["materialization_phase"]["protected_evaluation_sources_allowed"] is False, "development_only")
    check(execution["replay_phase"]["mineru_auto_generate"] is False, "artifact_only_replay")
    check(execution["replay_phase"]["gold_candidate_or_program_passed_to_engine"] is False, "no_gold_engine_input")
    check(execution["required_gates"]["accepted_wrong"] == 0, "accepted_wrong_gate")
    check(execution["required_gates"]["counterfactual_false_accepts"] == 0, "false_accept_gate")
    check(diagnostic["retained_safe_replay"]["gate"] is True, "parent_safe_replay")
    check(diagnostic_lock["authorized_next_step"].startswith("QP4-POST-QUESTION-SURFACE-SOURCE-RETRIEVAL-ROADMAP-REVIEW-PRECOMMIT"), "authorized_parent_step")

    for relative, expected in config["frozen_runtime_hashes"].items():
        check(_digest(ROOT / relative) == expected, f"runtime:{relative}")
    check(_digest(ROOT / config["checkpoint"]["path"]) == config["checkpoint"]["sha256"], "checkpoint")
    check(all(value is False for value in config["invariants"].values()), "invariants")
    check(config["artifact_policy"]["raw_or_intermediate_artifacts_persisted"] is False, "artifact_policy")

    return {
        "precommit_gate_passed": not failed,
        "failed_gates": failed,
        "generic_source_stratum": {
            "observations": evidence["observations"],
            "companies": evidence["companies"],
            "sources": evidence["source_documents"],
            "real_layouts": evidence["real_layout_families"],
            "attribution": review["attribution"],
        },
        "runtime_changed": False,
        "runtime_edit_authorized": False,
        "runtime_file_whitelist": [],
        "authorized_next_step": (
            execution["step"]
            if not failed
            else "Resolve source/retrieval roadmap-review precommit failures without changing runtime."
        ),
    }


def main() -> int:
    config = _load(CONFIG)
    result = validate()
    stratum = result["generic_source_stratum"]
    MARKDOWN.write_text(
        "# QP4 source/retrieval roadmap-review precommit v1\n\n"
        f"- Gate: `{'PASS' if result['precommit_gate_passed'] else 'FAIL'}`; runtime unchanged and whitelist empty.\n"
        f"- Generic evidence: `{stratum['observations']}` REFERENCE_BASE observations, `{stratum['companies']}` companies, `{stratum['sources']}` PDFs, `{stratum['real_layouts']}` real layouts; first failure is source load/extraction.\n"
        "- Decision: authorize development OCR-cache materialization with the frozen existing pipeline, followed by artifact-only replay. This does not authorize an OCR/layout/retrieval algorithm change.\n"
        "- Replay must use no gold engine input, retain source failures in the denominator, keep accepted-wrong and false accepts at zero, and pass all frozen replay gates.\n"
        "- QP4 remains incomplete; QP5 and Phase D remain locked.\n",
        encoding="utf-8",
    )
    artifacts = [_artifact(CONFIG), _artifact(Path(__file__).resolve()), _artifact(MARKDOWN)]
    lock = {
        "schema": "finhrm-qp4-post-question-surface-source-retrieval-roadmap-review-precommit-lock-v1",
        "created_at": config["created_at"],
        "classification": config["classification"],
        "status": "complete" if result["precommit_gate_passed"] else "failed",
        **result,
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
