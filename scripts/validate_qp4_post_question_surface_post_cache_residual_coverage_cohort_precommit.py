#!/usr/bin/env python3
"""Validate the QP4 post-cache residual coverage-cohort precommit."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/finhrm_qp4_post_question_surface_post_cache_residual_coverage_cohort_precommit_v1.json"
MARKDOWN = ROOT / "reports/finhrm_qp4_post_question_surface_post_cache_residual_coverage_cohort_precommit_v1.md"
LOCK = ROOT / "reports/finhrm_qp4_post_question_surface_post_cache_residual_coverage_cohort_precommit_lock_v1.json"


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


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


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
    basis = config["authorization_basis"]
    check(diagnostic["status"] == "DIAGNOSTIC_COMPLETE", "diagnostic_status")
    check(diagnostic["diagnostic_gate_passed"] is basis["diagnostic_gate_passed"], "diagnostic_gate")
    check(diagnostic["failed_gates"] == [], "diagnostic_failed_gates")
    check(diagnostic_lock["status"] == "complete", "diagnostic_lock_status")
    check(diagnostic_lock["diagnostic_gate_passed"] is True, "diagnostic_lock_gate")
    check(diagnostic["generic_residual_strata"] == [], "no_generic_residual_strata")
    check(len(diagnostic["generic_residual_strata"]) == basis["generic_residual_strata"], "generic_strata_count")
    check(
        diagnostic["decision"]["generic_runtime_evidence_sufficient"]
        is basis["generic_runtime_evidence_sufficient"],
        "generic_runtime_evidence_insufficient",
    )
    check(
        diagnostic["decision"]["development_coverage_cohort_precommit_authorized"]
        is basis["development_coverage_cohort_precommit_authorized"],
        "cohort_precommit_authorized",
    )
    check(diagnostic["decision"]["runtime_edit_authorized"] is False, "diagnostic_runtime_not_authorized")
    check(diagnostic["decision"]["runtime_file_whitelist"] == [], "diagnostic_runtime_whitelist_empty")
    check(
        diagnostic_lock["authorized_next_step"].startswith(
            "QP4-POST-QUESTION-SURFACE-POST-CACHE-RESIDUAL-COVERAGE-COHORT-PRECOMMIT"
        ),
        "authorized_parent_step",
    )

    observed_coverage = {
        row["first_failing_layer"]: f"{row['companies']}/{row['sources']}/{row['real_layouts']}"
        for row in diagnostic["residual_strata"]
    }
    expected_gaps = {
        row["first_failing_layer"]: row["observed_coverage"]
        for row in basis["residual_gaps"]
    }
    check(observed_coverage == expected_gaps, "residual_gap_coverage")
    check(
        all(not row["generic_evidence"] and not row["runtime_candidate"] for row in diagnostic["residual_strata"]),
        "residuals_not_runtime_candidates",
    )

    catalog = _load(ROOT / config["parents"]["development_company_catalog"]["path"])
    universe = config["source_universe"]
    documents = catalog["documents"]
    check(catalog["classification"] == universe["classification_required"], "catalog_development_only")
    check(catalog["metrics"]["documents"] == universe["catalog_documents"], "catalog_document_metric")
    check(catalog["metrics"]["company_groups"] == universe["catalog_company_groups"], "catalog_company_metric")
    check(len(documents) == universe["catalog_documents"], "catalog_document_count")
    check(
        len({row["company_key"] for row in documents.values()}) == universe["catalog_company_groups"],
        "catalog_company_count",
    )
    check(universe["evaluate_all_catalog_entries_for_eligibility"] is True, "full_catalog_audit")
    check(universe["protected_evaluation_sources_allowed"] is False, "protected_sources_forbidden")
    check(universe["whole_corpus_ocr_allowed"] is False, "whole_corpus_ocr_forbidden")
    check(universe["new_ocr_during_cohort_build_allowed"] is False, "cohort_build_ocr_forbidden")

    diagnosis = _load(ROOT / config["parents"]["diagnosis_source_cohort"]["path"])
    diagnosis_anchors = diagnosis["anchors"]
    check(len(diagnosis_anchors) == 4, "diagnosis_anchor_count")
    check(len({row["company_key"] for row in diagnosis_anchors}) == 4, "diagnosis_company_count")
    check(len({row["source_sha256"] for row in diagnosis_anchors}) == 4, "diagnosis_source_count")
    check(len({row["document_layout_family"] for row in diagnosis_anchors}) == 3, "diagnosis_document_layout_count")
    check(len({row["table_layout_family"] for row in diagnosis_anchors}) == 4, "diagnosis_table_layout_count")

    disjoint = config["disjointness_contract"]
    check(disjoint["selected_company_disjoint_from_diagnosis"] is True, "company_disjoint")
    check(disjoint["selected_source_sha256_disjoint_from_diagnosis"] is True, "source_disjoint")
    check(disjoint["selected_layout_instance_disjoint_from_diagnosis"] is True, "layout_instance_disjoint")
    check(len(disjoint["layout_instance_fingerprint_fields"]) == 5, "layout_fingerprint_fields")
    check(disjoint["maximum_selected_anchors_per_company"] == 1, "one_anchor_per_company")
    check(disjoint["maximum_selected_anchors_per_source"] == 1, "one_anchor_per_source")
    check(disjoint["duplicate_layout_instance_fingerprint_allowed"] is False, "no_duplicate_layout_instance")
    check(disjoint["synthetic_layout_counts_as_confirmation"] is False, "no_synthetic_layout_confirmation")

    question_rows = _load_jsonl(ROOT / config["parents"]["question_family_catalog"]["path"])
    question = config["question_diversity_contract"]
    check(len(question_rows) == 299, "question_catalog_count")
    check(
        all(str(row.get("classification", "")).startswith("development_only") for row in question_rows),
        "question_catalog_development_only",
    )
    for family in question["family_ids"]:
        variants = {row["variant"] for row in question_rows if row.get("family_id") == family}
        check(set(question["meaning_preserving_surface_variants"]).issubset(variants), f"surface_variants:{family}")
        check(
            set(question["catalog_control_variants"]).issubset(variants),
            f"semantic_controls:{family}",
        )
    check(
        "reference-base ACTUAL-versus-PLAN to ACTUAL-versus-prior-ACTUAL mutation"
        in question["required_semantic_controls"],
        "reference_base_control",
    )
    check(question["question_literal_used_for_source_selection"] is False, "no_question_literal_selection")
    check(question["minimum_surface_variants_per_semantic_program"] >= 4, "surface_variant_minimum")
    check(question["surface_forms_count_as_layout_confirmation"] is False, "surface_not_layout")
    check(question["counterfactual_false_accepts_required"] == 0, "counterfactual_false_accept_gate")
    check(question["candidate_permutation_exact_required"] is True, "permutation_gate")

    semantic = config["semantic_stratum"]
    check(
        (semantic["track"], semantic["group"], semantic["operation"])
        == ("ACTUAL_PLAN", "REFERENCE_BASE", "RATIO_PERCENT"),
        "semantic_stratum_exact",
    )
    check(len(semantic["required_evidence"]) >= 8, "explicit_evidence_contract")
    check(semantic["nearby_context_or_question_used_to_infer_period_or_provenance"] is False, "no_inferred_provenance")
    check(semantic["missing_proof_policy"] == "UNKNOWN_TO_ABSTAIN", "fail_closed_unknown")
    check(semantic["contradiction_policy"] == "REJECT", "reject_contradiction")

    selection = config["deterministic_selection"]
    forbidden = set(selection["forbidden_selection_or_priority_inputs"])
    check(selection["eligibility_evaluated_before_runtime_outcome"] is True, "eligibility_before_outcome")
    check(
        {
            "company identity",
            "report filename",
            "page number",
            "year literal",
            "numeric value",
            "example ID",
            "question literal",
            "candidate rank",
            "neural score",
            "confidence",
            "eventual earliest failing layer",
            "eventual ANSWER, ABSTAIN or REJECT action",
        }.issubset(forbidden),
        "forbidden_selection_inputs",
    )
    check(selection["minimum_selected_anchors"] >= 3, "anchor_minimum")
    check(selection["target_selected_anchors"] >= selection["minimum_selected_anchors"], "anchor_target")
    for key in (
        "minimum_distinct_companies",
        "minimum_distinct_sources",
        "minimum_distinct_real_document_layout_families",
        "minimum_distinct_real_table_layout_families",
        "minimum_distinct_layout_instance_fingerprints",
        "minimum_distinct_normalized_metric_identities",
    ):
        check(selection[key] >= 3, f"selection:{key}")
    check(selection["minimum_extraction_families"] >= 2, "extraction_diversity")
    check(selection["continue_scan_after_ineligible_or_rejected_source"] is True, "continue_after_rejection")

    build = config["authorized_build_contract"]
    required = build["required_build_gates"]
    check(build["step"] == "QP4-POST-QUESTION-SURFACE-POST-CACHE-RESIDUAL-COVERAGE-COHORT-BUILD", "authorized_build_step")
    check("run MinerU or generate OCR artifacts" in build["forbidden_actions"], "mineru_forbidden")
    check("run source replay or the reasoning engine" in build["forbidden_actions"], "source_replay_forbidden")
    check("change runtime, threshold, checkpoint or model" in build["forbidden_actions"], "runtime_change_forbidden")
    check(required["diagnosis_company_overlap"] == 0, "company_overlap_gate")
    check(required["diagnosis_source_overlap"] == 0, "source_overlap_gate")
    check(required["diagnosis_layout_instance_overlap"] == 0, "layout_overlap_gate")
    check(required["protected_sources_selected"] == 0, "protected_source_gate")
    check(required["ocr_generated"] == 0, "ocr_generation_gate")
    check(required["source_replay_queries"] == 0, "source_replay_gate")

    check(diagnostic["retained_controls"] == config["retained_controls"], "retained_controls_exact")
    for relative, expected in config["frozen_runtime_hashes"].items():
        check(_digest(ROOT / relative) == expected, f"runtime:{relative}")
        check(diagnostic["runtime_hashes"].get(relative) == expected, f"diagnostic_runtime:{relative}")
    check(_digest(ROOT / config["checkpoint"]["path"]) == config["checkpoint"]["sha256"], "checkpoint")
    check(diagnostic["checkpoint_sha256"] == config["checkpoint"]["sha256"], "diagnostic_checkpoint")
    check(all(value is False for value in config["invariants"].values()), "invariants")
    check(config["artifact_policy"]["raw_or_intermediate_artifacts_persisted"] is False, "artifact_policy")

    return {
        "precommit_gate_passed": not failed,
        "failed_gates": failed,
        "source_universe": {
            "catalog_documents": len(documents),
            "catalog_company_groups": len({row["company_key"] for row in documents.values()}),
            "classification": catalog["classification"],
        },
        "diagnosis_exclusion": {
            "companies": len({row["company_key"] for row in diagnosis_anchors}),
            "sources": len({row["source_sha256"] for row in diagnosis_anchors}),
            "document_layouts": len({row["document_layout_family"] for row in diagnosis_anchors}),
            "table_layouts": len({row["table_layout_family"] for row in diagnosis_anchors}),
        },
        "minimum_cohort": {
            "anchors": selection["minimum_selected_anchors"],
            "companies": selection["minimum_distinct_companies"],
            "sources": selection["minimum_distinct_sources"],
            "document_layouts": selection["minimum_distinct_real_document_layout_families"],
            "table_layouts": selection["minimum_distinct_real_table_layout_families"],
            "layout_instances": selection["minimum_distinct_layout_instance_fingerprints"],
            "metric_identities": selection["minimum_distinct_normalized_metric_identities"],
            "extraction_families": selection["minimum_extraction_families"],
        },
        "runtime_changed": False,
        "runtime_edit_authorized": False,
        "runtime_file_whitelist": [],
        "ocr_authorized": False,
        "source_replay_authorized": False,
        "authorized_next_step": (
            build["step"]
            if not failed
            else "Resolve residual coverage-cohort precommit failures without OCR, source replay, or runtime change."
        ),
    }


def main() -> int:
    config = _load(CONFIG)
    result = validate()
    minimum = result["minimum_cohort"]
    universe = result["source_universe"]
    MARKDOWN.write_text(
        "# QP4 post-cache residual coverage-cohort precommit v1\n\n"
        f"- Gate: `{'PASS' if result['precommit_gate_passed'] else 'FAIL'}`; runtime whitelist empty, OCR and source replay still locked.\n"
        f"- Universe: all `{universe['catalog_documents']}` hash-locked development PDFs across `{universe['catalog_company_groups']}` company groups are audited for eligibility; protected markers are excluded.\n"
        "- Semantic scope remains `ACTUAL_PLAN / REFERENCE_BASE / RATIO_PERCENT`; operation mixing is forbidden because it would destroy residual attribution.\n"
        f"- Minimum selected cohort: `{minimum['anchors']}` anchors over `{minimum['companies']}` companies, `{minimum['sources']}` PDFs, `{minimum['document_layouts']}` real document layouts, `{minimum['table_layouts']}` real table layouts, `{minimum['metric_identities']}` normalized metric identities and `{minimum['extraction_families']}` extraction families.\n"
        "- Every selected company, source and layout-instance fingerprint must be disjoint from the four-source diagnosis cohort; at most one anchor per company/source.\n"
        "- Each semantic program requires at least four meaning-preserving surfaces plus role, year, missing-role and ambiguity controls. Synthetic surfaces never count as layout evidence.\n"
        "- The next build may read catalog-matched development PDFs and existing caches for review only; it may not run MinerU, source replay, the reasoning engine, or modify runtime. QP4 remains incomplete; QP5 and Phase D remain locked.\n",
        encoding="utf-8",
    )
    artifacts = [_artifact(CONFIG), _artifact(Path(__file__).resolve()), _artifact(MARKDOWN)]
    lock = {
        "schema": "finhrm-qp4-post-question-surface-post-cache-residual-coverage-cohort-precommit-lock-v1",
        "created_at": config["created_at"],
        "classification": config["classification"],
        "status": "complete" if result["precommit_gate_passed"] else "failed",
        **result,
        "semantic_stratum": config["semantic_stratum"],
        "disjointness_contract": config["disjointness_contract"],
        "question_diversity_contract": config["question_diversity_contract"],
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
