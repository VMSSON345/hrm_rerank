from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRECOMMIT = ROOT / "config" / "finhrm_qp4_precommit_v1.json"
DEFAULT_REPORT = ROOT / "reports" / "finhrm_qp4_precommit_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "finhrm_qp4_precommit_v1.md"
DEFAULT_LOCK = ROOT / "reports" / "finhrm_qp4_precommit_lock_v1.json"
TEST_MODULE = ROOT / "tests" / "test_qp4_precommit.py"
PROTECTED_PARTS = {"validation", "blind", "test", "sealed", "holdout", "final"}


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _digest(path),
    }


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _protected_path(path: str) -> bool:
    parts = {part.lower() for part in Path(path).parts}
    return bool(parts & PROTECTED_PARTS)


def _check(condition: bool, gate: str, failed: list[str]) -> None:
    if not condition:
        failed.append(gate)


def evaluate(precommit_path: Path = DEFAULT_PRECOMMIT) -> dict[str, Any]:
    config = _load(precommit_path)
    failed: list[str] = []

    _check(config.get("schema") == "finhrm-qp4-precommit-v1", "schema", failed)
    _check(
        config.get("classification")
        == "development_only_counterfactual_long_question_contract_runtime_unchanged",
        "classification",
        failed,
    )

    claude = config.get("claude_md", {})
    claude_path = ROOT / str(claude.get("path", ""))
    _check(bool(claude.get("read_before_scope_design")), "claude_read", failed)
    _check(
        claude_path.is_file() and _digest(claude_path) == claude.get("sha256"),
        "claude_hash",
        failed,
    )

    parent_count = 0
    for name, parent in config.get("parents", {}).items():
        relative = str(parent.get("path", ""))
        path = ROOT / relative
        _check(not _protected_path(relative), f"parent_{name}_not_protected", failed)
        _check(
            path.is_file() and _digest(path) == parent.get("sha256"),
            f"parent_{name}_hash",
            failed,
        )
        parent_count += 1

    parent_lock_path = ROOT / config["parents"][
        "qp3_answer_presentation_implementation_lock"
    ]["path"]
    parent_lock = _load(parent_lock_path)
    entry = config.get("entry_condition", {})
    _check(parent_lock.get("implementation_gate_passed") is True, "qp3_gate", failed)
    _check(parent_lock.get("failed_gates") == [], "qp3_failed_gates", failed)
    _check(parent_lock.get("qp3_complete") is True, "qp3_complete", failed)
    _check(
        parent_lock.get("qp4_precommit_authorized") is True,
        "parent_qp4_precommit_authorized",
        failed,
    )
    _check(
        parent_lock.get("qp4_runtime_change_authorized") is False,
        "parent_qp4_runtime_locked",
        failed,
    )
    _check(parent_lock.get("phase_d") == "locked", "parent_phase_d_locked", failed)
    _check(entry.get("qp3_complete") is True, "entry_qp3_complete", failed)
    _check(
        entry.get("qp4_runtime_change_authorized_by_parent") is False,
        "entry_qp4_runtime_locked",
        failed,
    )

    allowed = config.get("allowed_evidence", {})
    _check(allowed.get("split") == "development_only", "development_only", failed)
    _check(allowed.get("abw_allowed") is False, "abw_forbidden", failed)
    _check(
        allowed.get(
            "protected_validation_blind_test_sealed_holdout_final_access_allowed"
        )
        is False,
        "protected_data_forbidden",
        failed,
    )
    _check(
        allowed.get("synthetic_layout_allowed_as_generalization_confirmation")
        is False,
        "synthetic_layout_forbidden",
        failed,
    )
    _check(
        allowed.get("source_backed_real_pdf_required_for_end_to_end_confirmation")
        is True,
        "real_pdf_required",
        failed,
    )
    _check(
        allowed.get("safe_abstention_counts_as_answered") is False,
        "safe_abstention_not_answer",
        failed,
    )

    scope = config.get("qp4_scope", {})
    _check(len(scope.get("core_answer_families", [])) == 13, "core_families", failed)
    _check(
        set(scope.get("ambiguity_control_families", [])) == {"C13", "C14", "C15"},
        "ambiguity_families",
        failed,
    )

    projection = config.get("semantic_projection_v1", {})
    forbidden_projection = set(projection.get("forbidden_projection_fields", []))
    _check(
        {"candidate_rank", "neural_score", "confidence"}.issubset(
            forbidden_projection
        ),
        "projection_ignores_scores",
        failed,
    )
    _check(
        {"question", "normalized", "intent_sources"}.issubset(
            set(projection.get("excluded_surface_fields", []))
        ),
        "projection_excludes_surface",
        failed,
    )
    _check(
        "preserve order" in projection.get("commutative_policy", ""),
        "ordered_noncommutative_program",
        failed,
    )

    counterfactual = config.get("minimal_pair_counterfactual_contract", {})
    required_dimensions = {
        "COLUMN_ROLE",
        "OPERAND_YEAR",
        "ENTITY_SCOPE",
        "UNIT_SCALE",
        "DIRECTION",
        "REFERENCE_BASE",
    }
    _check(
        set(counterfactual.get("required_dimensions", [])) == required_dimensions,
        "counterfactual_dimensions",
        failed,
    )
    taxonomy = counterfactual.get("mutation_taxonomy", [])
    _check(len(taxonomy) == len(required_dimensions), "counterfactual_taxonomy_size", failed)
    _check(
        {item.get("dimension") for item in taxonomy} == required_dimensions,
        "counterfactual_taxonomy_dimensions",
        failed,
    )
    _check(
        "exactly one" in counterfactual.get("construction_rule", "").lower(),
        "minimal_pair_exactly_one_dimension",
        failed,
    )
    _check(
        counterfactual.get("score_only_rejection_allowed") is False,
        "score_only_rejection_forbidden",
        failed,
    )
    _check(
        counterfactual.get("same_numeric_result_counts_as_correct_old_program")
        is False,
        "coincidental_execution_forbidden",
        failed,
    )
    _check(
        "INVALID" in counterfactual.get("old_program_rule", "")
        and "UNKNOWN" in counterfactual.get("old_program_rule", ""),
        "tri_state_old_program_rule",
        failed,
    )

    surface = config.get("long_question_and_surface_contract", {})
    surface_categories = {
        "SHORT_LONG_EQUIVALENT",
        "DISCOURSE_PREFIX_SUFFIX",
        "CLAUSE_PERMUTATION",
        "PUNCTUATION_VARIANT",
        "NO_DIACRITICS",
        "BILINGUAL_PARENTHETICAL",
        "IRRELEVANT_DISTRACTOR_CLAUSE",
    }
    _check(
        set(surface.get("meaning_preserving_categories", [])) == surface_categories,
        "surface_categories",
        failed,
    )
    invariance = surface.get("required_invariance", {})
    _check(
        invariance
        and all(value == 1.0 for value in invariance.values()),
        "surface_invariance_exact",
        failed,
    )
    _check(
        surface.get("surface_length_as_proof_or_acceptance_feature_allowed") is False,
        "length_not_proof",
        failed,
    )
    _check(
        surface.get("question_text_may_fill_missing_period_provenance_or_evidence_role")
        is False,
        "surface_cannot_fill_proof",
        failed,
    )

    evidence_stress = config.get("evidence_stress_contract", {})
    _check(
        {"REPEATED_METRIC", "CONFLICTING_CELL", "CANDIDATE_PERMUTATION"}.issubset(
            set(evidence_stress.get("categories", []))
        ),
        "evidence_stress_categories",
        failed,
    )
    _check(
        evidence_stress.get("missing_proof_action") == "ABSTAIN"
        and evidence_stress.get("contradiction_action") == "REJECT",
        "evidence_fail_closed",
        failed,
    )

    audit_tracks = config.get("separate_audit_tracks", {})
    _check(
        set(audit_tracks)
        == {"MULTI_OUTPUT", "QUALITATIVE_FOLLOW_UP", "ANAPHORA", "CROSS_DOCUMENT"},
        "separate_audit_tracks",
        failed,
    )
    _check(
        "No current QP4 coverage claim"
        in audit_tracks.get("CROSS_DOCUMENT", {}).get("answered_definition", ""),
        "cross_document_not_claimed",
        failed,
    )

    cohort = config.get("development_cohort_contract", {})
    compiler = cohort.get("compiler_stress", {})
    source = cohort.get("source_backed_end_to_end", {})
    _check(
        compiler.get("minimum_answer_family_count") == 13
        and compiler.get("minimum_ambiguity_family_count") == 3,
        "compiler_cohort_family_coverage",
        failed,
    )
    _check(
        compiler.get("company_or_layout_coverage_claim_allowed") is False,
        "compiler_not_layout_confirmation",
        failed,
    )
    for field in (
        "minimum_distinct_companies_per_applicable_counterfactual_dimension",
        "minimum_distinct_source_documents_per_applicable_counterfactual_dimension",
        "minimum_distinct_real_layout_families_per_applicable_counterfactual_dimension",
        "minimum_distinct_companies_for_long_surface_stress",
        "minimum_distinct_source_documents_for_long_surface_stress",
        "minimum_distinct_real_layout_families_for_long_surface_stress",
    ):
        _check(source.get(field, 0) >= 3, f"source_cohort_{field}", failed)
    _check(
        source.get("synthetic_fixture_counts_toward_coverage") is False,
        "source_cohort_no_synthetic",
        failed,
    )
    _check(
        source.get("safe_abstention_counts_as_answer") is False,
        "source_cohort_abstention_not_answer",
        failed,
    )
    _check(
        source.get("company_disjoint_from_diagnosis") is True
        and source.get("independent_review_required") is True,
        "source_cohort_independent",
        failed,
    )

    gates = config.get("implementation_replay_gates", {})
    false_accept_gates = [
        key
        for key in gates
        if key.endswith("false_accept_count")
        or key == "semantic_counterfactual_old_program_false_accept_count"
    ]
    _check(
        false_accept_gates and all(gates[key] == 0 for key in false_accept_gates),
        "false_accept_gates_zero",
        failed,
    )
    _check(
        gates.get("short_long_semantic_ir_exact_rate") == 1.0
        and gates.get("short_long_eligible_program_set_exact_rate") == 1.0
        and gates.get("short_long_winner_exact_rate") == 1.0,
        "short_long_gates_exact",
        failed,
    )
    _check(
        gates.get("candidate_permutation_exact_rate") == 1.0
        and gates.get("hash_seed_invariant") is True,
        "determinism_gates",
        failed,
    )
    _check(gates.get("frozen_safe_replay_gold_execution") == "363/363", "replay_execution", failed)
    _check(gates.get("frozen_safe_replay_candidate_permutation") == "363/363", "replay_permutation", failed)
    _check(gates.get("frozen_safe_replay_counterfactual_false_accepts") == 0, "replay_counterfactual", failed)
    _check(
        gates.get("retain_core_actions")
        == {"ANSWER": 259, "ABSTAIN": 90, "REJECT": 14},
        "replay_actions",
        failed,
    )

    whitelist = config.get("exact_runtime_edit_scope_for_qp4_implementation", {})
    expected_whitelist = {
        "src/hrm_finance_qa/question_planner.py",
        "src/hrm_finance_qa/program_synthesis.py",
        "src/hrm_finance_qa/symbolic_verifier.py",
        "src/hrm_finance_qa/direction_contract.py",
    }
    _check(set(whitelist) == expected_whitelist, "runtime_whitelist", failed)
    _check(
        set(config.get("runtime_files_explicitly_frozen", [])).isdisjoint(whitelist),
        "frozen_runtime_disjoint",
        failed,
    )
    admission = config.get("runtime_change_admission", {})
    _check(
        admission.get("baseline_diagnostic_required_before_any_runtime_edit") is True,
        "baseline_before_runtime",
        failed,
    )
    _check(
        "smallest" in admission.get("conditional_scope_rule", ""),
        "smallest_failed_layer_only",
        failed,
    )

    runtime_hashes: dict[str, str] = {}
    for relative, expected in config.get("runtime_hashes_at_precommit", {}).items():
        path = ROOT / relative
        actual = _digest(path) if path.is_file() else ""
        runtime_hashes[relative] = actual
        _check(actual == expected, f"runtime_hash_{relative}", failed)

    forbidden = "\n".join(config.get("forbidden_implementation_actions", [])).lower()
    for token, gate in (
        ("abw", "forbidden_abw"),
        ("validation", "forbidden_validation"),
        ("candidate rank", "forbidden_rank"),
        ("safe abstention", "forbidden_abstention_count"),
        ("phase d", "forbidden_phase_d"),
    ):
        _check(token in forbidden, gate, failed)

    frozen = config.get("frozen_invariants", {})
    _check(frozen and all(value is False for value in frozen.values()), "frozen_invariants", failed)
    _check(
        "No QP4 runtime change is authorized or performed by this precommit"
        in config.get("authorized_next_step", ""),
        "runtime_remains_locked",
        failed,
    )
    _check(
        "QP5 and Phase D remain locked" in config.get("authorized_next_step", ""),
        "later_phases_locked",
        failed,
    )

    failed = sorted(set(failed))
    return {
        "schema": "finhrm-qp4-precommit-report-v1",
        "created_at": config.get("created_at"),
        "classification": config.get("classification"),
        "status": "PASS" if not failed else "FAIL",
        "precommit_gate_passed": not failed,
        "failed_gates": failed,
        "parents_verified": parent_count + 1,
        "runtime_changed": False,
        "runtime_hashes": runtime_hashes,
        "checkpoint_sha256": entry.get("checkpoint_sha256"),
        "contract": {
            "core_answer_families": len(scope.get("core_answer_families", [])),
            "ambiguity_controls": len(scope.get("ambiguity_control_families", [])),
            "counterfactual_dimensions": counterfactual.get("required_dimensions", []),
            "meaning_preserving_surface_categories": surface.get(
                "meaning_preserving_categories", []
            ),
            "separate_audit_tracks": sorted(audit_tracks),
            "minimum_companies_per_counterfactual_dimension": source.get(
                "minimum_distinct_companies_per_applicable_counterfactual_dimension"
            ),
            "minimum_real_layouts_per_counterfactual_dimension": source.get(
                "minimum_distinct_real_layout_families_per_applicable_counterfactual_dimension"
            ),
            "runtime_whitelist": sorted(whitelist),
        },
        "frozen_safe_replay": {
            "actions": gates.get("retain_core_actions"),
            "gold_execution": gates.get("frozen_safe_replay_gold_execution"),
            "candidate_permutation": gates.get(
                "frozen_safe_replay_candidate_permutation"
            ),
            "counterfactual_false_accepts": gates.get(
                "frozen_safe_replay_counterfactual_false_accepts"
            ),
        },
        "invariants": frozen,
        "qp4_precommit_complete": not failed,
        "qp4_runtime_change_authorized": False,
        "qp5_authorized": False,
        "phase_d": "locked",
        "authorized_next_step": config.get("authorized_next_step"),
    }


def _markdown(report: dict[str, Any]) -> str:
    contract = report["contract"]
    replay = report["frozen_safe_replay"]
    return "\n".join(
        [
            "# QP4 Precommit v1",
            "",
            f"- Status: **{report['status']}**",
            "- Runtime changed: **no**",
            f"- Core scope: `{contract['core_answer_families']}` answer families + `{contract['ambiguity_controls']}` ambiguity controls",
            f"- Minimal-pair dimensions: `{', '.join(contract['counterfactual_dimensions'])}`",
            f"- Meaning-preserving surface categories: `{len(contract['meaning_preserving_surface_categories'])}`",
            f"- Source-backed minimum per applicable counterfactual dimension: `{contract['minimum_companies_per_counterfactual_dimension']}` companies / `{contract['minimum_real_layouts_per_counterfactual_dimension']}` real layouts / `3` source documents",
            f"- Frozen replay: actions `{replay['actions']}`, execution `{replay['gold_execution']}`, permutation `{replay['candidate_permutation']}`, counterfactual false accepts `{replay['counterfactual_false_accepts']}`",
            "- Missing proof remains `ABSTAIN`; explicit contradiction remains `REJECT`.",
            "- Synthetic question variants do not count as company/layout confirmation.",
            "- QP4 runtime, QP5 and Phase D remain locked in this step.",
            "",
            "Next: build/freeze the QP4 development cohort and baseline diagnostic; runtime implementation requires a separate explicit request and must stay inside the conditional four-file whitelist.",
            "",
        ]
    )


def write_artifacts(report: dict[str, Any]) -> None:
    DEFAULT_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    DEFAULT_MARKDOWN.write_text(_markdown(report), encoding="utf-8")
    artifacts = [
        DEFAULT_PRECOMMIT,
        Path(__file__).resolve(),
        TEST_MODULE,
        DEFAULT_REPORT,
        DEFAULT_MARKDOWN,
        ROOT / "strategy.md",
        ROOT / "FINHRM_THESIS_PUBLICATION_ROADMAP.md",
    ]
    missing = [path for path in artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(str(path) for path in missing))
    lock = {
        "schema": "finhrm-qp4-precommit-lock-v1",
        "created_at": report["created_at"],
        "classification": report["classification"],
        "status": "complete" if report["precommit_gate_passed"] else "failed",
        "precommit_complete": report["precommit_gate_passed"],
        "failed_gates": report["failed_gates"],
        "runtime_changed": False,
        "checkpoint_sha256": report["checkpoint_sha256"],
        "artifacts": [_artifact(path) for path in artifacts],
        "qp4_scoped_implementation_contract_ready": report["precommit_gate_passed"],
        "qp4_runtime_change_authorized": False,
        "qp5_authorized": False,
        "phase_d": "locked",
        "authorized_next_step": report["authorized_next_step"],
    }
    DEFAULT_LOCK.write_text(
        json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precommit", type=Path, default=DEFAULT_PRECOMMIT)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    report = evaluate(args.precommit)
    if args.write:
        write_artifacts(report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["precommit_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
