from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRECOMMIT = ROOT / "config" / "finhrm_qp3_answer_presentation_precommit_v1.json"
DEFAULT_REPORT = ROOT / "reports" / "finhrm_qp3_answer_presentation_precommit_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "finhrm_qp3_answer_presentation_precommit_v1.md"
DEFAULT_LOCK = ROOT / "reports" / "finhrm_qp3_answer_presentation_precommit_lock_v1.json"
TEST_MODULE = ROOT / "tests" / "test_qp3_answer_presentation_precommit.py"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _digest(path),
    }


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(precommit_path: Path = DEFAULT_PRECOMMIT) -> dict[str, Any]:
    config = _load(precommit_path)
    errors: list[str] = []

    if config.get("schema") != "finhrm-qp3-answer-presentation-precommit-v1":
        errors.append("wrong_precommit_schema")
    if config.get("status") != "precommitted_runtime_unchanged_exact_scope_only":
        errors.append("wrong_precommit_status")

    claude = config.get("claude_md", {})
    if not claude.get("read_before_scope_design"):
        errors.append("claude_not_read_before_scope_design")
    for name, record in {"claude_md": claude, **config.get("parents", {})}.items():
        path = ROOT / str(record.get("path", ""))
        if not path.is_file():
            errors.append(f"missing_parent:{name}")
        elif _digest(path) != record.get("sha256"):
            errors.append(f"parent_hash_mismatch:{name}")

    core_lock_record = config.get("parents", {}).get("qp3_core_implementation_lock", {})
    core_lock_path = ROOT / str(core_lock_record.get("path", ""))
    core_lock = _load(core_lock_path) if core_lock_path.is_file() else {}
    entry = config.get("entry_condition", {})
    if not core_lock.get("implementation_gate_passed"):
        errors.append("qp3_core_implementation_gate_not_passed")
    if core_lock.get("failed_gates"):
        errors.append("qp3_core_has_failed_gates")
    if core_lock.get("checkpoint_sha256") != entry.get("checkpoint_sha256"):
        errors.append("checkpoint_parent_mismatch")
    if entry.get("authorized_parent_next_step_contains", "") not in core_lock.get(
        "authorized_next_step", ""
    ):
        errors.append("parent_does_not_authorize_presentation_precommit")
    if core_lock.get("qp4_authorized") or core_lock.get("phase_d") != "locked":
        errors.append("parent_release_order_not_locked")

    envelope = config.get("verified_answer_envelope_v1", {})
    if envelope.get("status_values") != ["ANSWER", "ABSTAIN", "REJECT"]:
        errors.append("wrong_status_values")
    if envelope.get("answer_mode_values") != [
        "FACT",
        "COMPUTATION",
        "VERIFIED_SUMMARY",
    ]:
        errors.append("wrong_answer_mode_values")
    proof = envelope.get("proof", {})
    if proof.get("candidate_rank_neural_score_confidence_as_semantic_proof_allowed"):
        errors.append("score_or_confidence_semantic_proof_allowed")
    if proof.get("warning_text_parsing_as_status_authority_allowed"):
        errors.append("warning_text_status_authority_allowed")
    if "never become ANSWER" not in proof.get("promotion_forbidden", ""):
        errors.append("parent_action_promotion_not_forbidden")

    citations = envelope.get("citations", {})
    if citations.get("operand_citation_coverage_required") != 1.0:
        errors.append("operand_citation_coverage_not_exact")
    if not citations.get("source_span_must_resolve_to_selected_evidence"):
        errors.append("citation_source_span_not_required")
    if "never" not in citations.get("page_render_policy", ""):
        errors.append("page_invention_not_forbidden")

    program = envelope.get("program", {})
    if not program.get("each_intermediate_must_replay_exactly"):
        errors.append("intermediate_replay_not_exact")
    if not program.get("final_replay_must_equal_canonical_value"):
        errors.append("final_derivation_replay_not_exact")
    if program.get("free_text_formula_as_authority_allowed"):
        errors.append("free_text_formula_allowed")

    summary = envelope.get("summary_coverage", {})
    if "overall status is ABSTAIN" not in summary.get("partial_rule", ""):
        errors.append("partial_summary_not_fail_closed")
    if "REJECT" not in summary.get("conflict_rule", ""):
        errors.append("summary_conflict_not_rejected")
    if not summary.get("qualitative_claim_requires_direct_cited_text_span"):
        errors.append("uncited_qualitative_claim_allowed")

    renderer = config.get("deterministic_renderer_contract", {})
    if not renderer.get("templates_only") or not renderer.get("same_envelope_same_bytes"):
        errors.append("renderer_not_deterministic")
    if renderer.get("renderer_may_create_number_page_citation_formula_or_claim"):
        errors.append("renderer_invention_allowed")
    if renderer.get("renderer_may_consume_confidence_or_rank"):
        errors.append("renderer_score_consumption_allowed")

    cohort = config.get("development_generalization_cohort", {})
    for dimension in (
        "minimum_verified_examples_per_mode",
        "minimum_distinct_companies_per_mode",
        "minimum_distinct_real_layout_families_per_mode",
        "minimum_distinct_source_documents_per_mode",
    ):
        if cohort.get(dimension, 0) < 3:
            errors.append(f"insufficient_generalization_gate:{dimension}")
    if cohort.get("synthetic_fixture_counts_toward_coverage"):
        errors.append("synthetic_fixture_counts_toward_coverage")
    if not cohort.get("company_and_layout_disjoint_confirmation_required"):
        errors.append("company_layout_disjoint_not_required")

    gates = config.get("implementation_replay_gates", {})
    exact_one = (
        "numeric_value_unit_scale_exact_rate",
        "citation_exact_rate",
        "operand_citation_coverage",
        "derivation_replay_exact_rate",
        "candidate_permutation_exact_rate",
        "citation_permutation_exact_rate",
        "summary_input_permutation_exact_rate",
    )
    for name in exact_one:
        if gates.get(name) != 1.0:
            errors.append(f"gate_not_exact:{name}")
    exact_zero = (
        "unsupported_claim_count",
        "summary_complete_false_claim_count",
        "answer_promotion_from_parent_abstain_or_reject_count",
        "hard_negative_false_accept_count",
        "frozen_safe_replay_counterfactual_false_accepts",
    )
    for name in exact_zero:
        if gates.get(name) != 0:
            errors.append(f"gate_not_zero:{name}")
    if gates.get("retain_core_actions") != {
        "ANSWER": 259,
        "ABSTAIN": 90,
        "REJECT": 14,
    }:
        errors.append("core_actions_not_frozen")
    if gates.get("frozen_safe_replay_gold_execution") != "363/363":
        errors.append("gold_execution_not_frozen")
    if gates.get("frozen_safe_replay_candidate_permutation") != "363/363":
        errors.append("candidate_permutation_not_frozen")

    scope = config.get("exact_runtime_edit_scope_for_implementation", {})
    if set(scope) != {
        "src/hrm_finance_qa/answer_presentation.py",
        "src/hrm_finance_qa/document_reasoning_engine.py",
    }:
        errors.append("runtime_scope_not_exact")
    new_module = ROOT / "src" / "hrm_finance_qa" / "answer_presentation.py"
    if new_module.exists():
        errors.append("presentation_runtime_created_before_precommit_lock")
    if set(scope) & set(config.get("runtime_files_explicitly_frozen", [])):
        errors.append("runtime_scope_overlaps_frozen_files")

    current_hashes: dict[str, str] = {}
    for relative, expected in config.get("runtime_hashes_before_implementation", {}).items():
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"missing_runtime:{relative}")
            continue
        actual = _digest(path)
        current_hashes[relative] = actual
        if actual != expected:
            errors.append(f"runtime_hash_mismatch:{relative}")

    frozen = config.get("frozen_invariants", {})
    if any(bool(value) for value in frozen.values()):
        errors.append("frozen_invariant_opened")
    if "QP3-ANSWER-PRESENTATION-IMPLEMENTATION only" not in config.get(
        "authorized_next_step", ""
    ):
        errors.append("next_step_not_exact_presentation_implementation")
    if "QP4 and Phase D remain locked" not in config.get("authorized_next_step", ""):
        errors.append("next_step_opens_locked_phase")

    return {
        "schema": "finhrm-qp3-answer-presentation-precommit-report-v1",
        "created_at": config.get("created_at"),
        "classification": config.get("classification"),
        "status": "PASS" if not errors else "FAIL",
        "precommit_gate_passed": not errors,
        "failed_gates": errors,
        "parents_verified": len(config.get("parents", {})) + 1,
        "runtime_changed": False,
        "runtime_hashes": current_hashes,
        "checkpoint_sha256": entry.get("checkpoint_sha256"),
        "contract": {
            "schema_version": envelope.get("schema_version"),
            "status_values": envelope.get("status_values"),
            "answer_mode_values": envelope.get("answer_mode_values"),
            "deterministic_templates_only": renderer.get("templates_only"),
            "unsupported_claim_gate": gates.get("unsupported_claim_count"),
            "derivation_replay_exact_rate": gates.get("derivation_replay_exact_rate"),
            "minimum_companies_per_mode": cohort.get("minimum_distinct_companies_per_mode"),
            "minimum_real_layouts_per_mode": cohort.get("minimum_distinct_real_layout_families_per_mode"),
        },
        "frozen_safe_replay": {
            "actions": gates.get("retain_core_actions"),
            "gold_execution": gates.get("frozen_safe_replay_gold_execution"),
            "candidate_permutation": gates.get("frozen_safe_replay_candidate_permutation"),
            "counterfactual_false_accepts": gates.get("frozen_safe_replay_counterfactual_false_accepts"),
        },
        "invariants": frozen,
        "qp4_authorized": False,
        "phase_d": "locked",
        "authorized_next_step": config.get("authorized_next_step"),
    }


def _markdown(report: dict[str, Any]) -> str:
    contract = report["contract"]
    replay = report["frozen_safe_replay"]
    return "\n".join(
        [
            "# QP3 Answer Presentation Precommit v1",
            "",
            f"- Status: **{report['status']}**",
            "- Runtime changed: **no**",
            f"- Modes: `{', '.join(contract['answer_mode_values'])}`",
            f"- Minimum development coverage per mode: `{contract['minimum_companies_per_mode']}` companies / `{contract['minimum_real_layouts_per_mode']}` real layouts / `3` source documents",
            f"- Frozen core actions: `{replay['actions']}`",
            f"- Frozen replay: execution `{replay['gold_execution']}`, permutation `{replay['candidate_permutation']}`, counterfactual false accepts `{replay['counterfactual_false_accepts']}`",
            "- Unsupported claims allowed: `0`",
            "- Missing proof remains `ABSTAIN`; contradiction remains `REJECT`.",
            "- QP4 and Phase D remain locked.",
            "",
            "Authorized next step: `QP3-ANSWER-PRESENTATION-IMPLEMENTATION` within the exact two-file scope in the precommit.",
            "",
        ]
    )


def write_artifacts(report: dict[str, Any]) -> None:
    DEFAULT_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    DEFAULT_MARKDOWN.write_text(_markdown(report), encoding="utf-8")
    artifacts = [DEFAULT_PRECOMMIT, Path(__file__).resolve(), TEST_MODULE, DEFAULT_REPORT, DEFAULT_MARKDOWN]
    missing = [path for path in artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(str(path) for path in missing))
    lock = {
        "schema": "finhrm-qp3-answer-presentation-precommit-lock-v1",
        "created_at": report["created_at"],
        "classification": report["classification"],
        "status": "complete" if report["precommit_gate_passed"] else "failed",
        "precommit_complete": report["precommit_gate_passed"],
        "failed_gates": report["failed_gates"],
        "runtime_changed": False,
        "checkpoint_sha256": report["checkpoint_sha256"],
        "artifacts": [_artifact(path) for path in artifacts],
        "next_step_scoped_runtime_change_authorized": report["precommit_gate_passed"],
        "qp4_authorized": False,
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
