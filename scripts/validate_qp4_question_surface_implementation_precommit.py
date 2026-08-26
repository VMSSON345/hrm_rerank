#!/usr/bin/env python3
"""Validate and lock the QP4 generic question-surface implementation scope."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/finhrm_qp4_question_surface_implementation_precommit_v1.json"
DEFAULT_REPORT = ROOT / "reports/finhrm_qp4_question_surface_implementation_precommit_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports/finhrm_qp4_question_surface_implementation_precommit_v1.md"
DEFAULT_LOCK = ROOT / "reports/finhrm_qp4_question_surface_implementation_precommit_lock_v1.json"
TEST_PATH = ROOT / "tests/test_qp4_question_surface_implementation_precommit.py"
PROTECTED = re.compile(
    r"(?:^|[/_.-])(test|holdout|final|blind|validation)(?:[/_.-]|$)", re.I
)


def _digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    if PROTECTED.search(path.resolve().as_posix()):
        raise ValueError(f"Protected input is forbidden: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _digest(path),
    }


def _check(condition: bool, name: str, failed: list[str]) -> None:
    if not condition:
        failed.append(name)


def _projection_diff_fields(expected: dict[str, Any], observed: dict[str, Any]) -> list[str]:
    return sorted(
        key for key in expected if expected.get(key) != observed.get(key)
    )


def _surface_question(question: str, recipe: dict[str, str]) -> str:
    stripped = question.strip()
    placement = recipe["placement"]
    text = recipe["text"]
    if placement == "prefix":
        return text + stripped[0].lower() + stripped[1:]
    if placement == "suffix_parenthetical":
        return stripped.rstrip("?") + f" ({text})?"
    if placement == "suffix_clause":
        return stripped.rstrip("?") + f", {text}?"
    raise ValueError(f"Unsupported hold-ahead placement: {placement}")


def _hold_ahead_summary(
    config: dict[str, Any], refresh: dict[str, Any]
) -> dict[str, Any]:
    from hrm_finance_qa.question_planner import compile_question_plan
    from scripts.evaluate_qp4_development_baseline import _semantic_projection

    canonical = refresh["compiler"]["canonical"]["details"]
    recipes = config["semantic_diversity_contract"]["orthogonal_hold_ahead"][
        "categories"
    ]
    by_category: Counter[str] = Counter()
    exact_by_category: Counter[str] = Counter()
    operations: Counter[str] = Counter()
    details: list[dict[str, Any]] = []
    for row in canonical:
        family = str(row["family_id"])
        question = str(row["question"])
        base_plan = compile_question_plan(question)
        base_projection = _semantic_projection(base_plan)
        operations[str(base_plan.operation)] += 1
        for category, category_recipes in recipes.items():
            for recipe_index, recipe in enumerate(category_recipes, start=1):
                variant = _surface_question(question, recipe)
                observed = _semantic_projection(compile_question_plan(variant))
                exact = observed == base_projection
                by_category[category] += 1
                exact_by_category[category] += int(exact)
                details.append(
                    {
                        "family_id": family,
                        "category": category,
                        "recipe": recipe_index,
                        "semantic_projection_exact_before_implementation": exact,
                        "changed_fields": _projection_diff_fields(
                            base_projection, observed
                        ),
                    }
                )
    return {
        "questions": len(details),
        "families": len({row["family_id"] for row in details}),
        "categories": dict(sorted(by_category.items())),
        "exact_before_implementation": sum(exact_by_category.values()),
        "exact_by_category_before_implementation": dict(
            sorted(exact_by_category.items())
        ),
        "canonical_root_operation_distribution": dict(sorted(operations.items())),
        "details": details,
    }


def build_report(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _load(config_path)
    failed: list[str] = []
    _check(
        _digest(ROOT / config["claude_md"]["path"])
        == config["claude_md"]["sha256"],
        "claude_hash",
        failed,
    )
    _check(config["claude_md"]["read_before_scope_design"] is True, "claude_read", failed)
    for name, parent in config["parents"].items():
        path = ROOT / parent["path"]
        _check(path.is_file() and _digest(path) == parent["sha256"], f"parent_hash:{name}", failed)
    checkpoint = config["checkpoint"]
    _check(_digest(ROOT / checkpoint["path"]) == checkpoint["sha256"], "checkpoint_hash", failed)
    for relative, expected in config["runtime_hashes_before_implementation"].items():
        _check(_digest(ROOT / relative) == expected, f"runtime_hash:{relative}", failed)

    diagnostic = _load(ROOT / config["parents"]["diagnostic_report"]["path"])
    raw = _load(ROOT / config["parents"]["diagnostic_raw"]["path"])
    refresh = _load(ROOT / config["parents"]["post_unit_scale_refresh"]["path"])
    safe = _load(ROOT / config["parents"]["safe_replay"]["path"])
    evidence = config["generic_failure_evidence"]

    candidate = diagnostic["implementation_precommit_candidate"]
    _check(candidate["nominated"] is True, "planner_nominated", failed)
    _check(candidate["target_file"] == "src/hrm_finance_qa/question_planner.py", "planner_target", failed)
    _check(candidate["runtime_edit_authorized"] is False, "parent_runtime_locked", failed)
    _check(diagnostic["diagnostic_gate_passed"] is True, "diagnostic_passed", failed)
    _check(diagnostic["runtime_changed"] is False, "diagnostic_runtime_unchanged", failed)

    compiler_rows = raw["compiler_results"]
    _check(len(compiler_rows) == evidence["compiler_residuals"] == 18, "compiler_residual_count", failed)
    observed_groups = Counter(str(row["group"]) for row in compiler_rows)
    for group, expected in evidence["groups"].items():
        rows = [row for row in compiler_rows if row["group"] == group]
        _check(observed_groups[group] == expected["failures"], f"failure_count:{group}", failed)
        _check(len({row["family_id"] for row in rows}) == expected["families"], f"family_count:{group}", failed)
        _check(all(row["first_failing_layer"] == "QUESTION_PLAN" for row in rows), f"layer:{group}", failed)
    changed_fields = {
        tuple(
            _projection_diff_fields(
                row["semantic_plan_projection"]["expected"],
                row["semantic_plan_projection"]["observed"],
            )
        )
        for row in compiler_rows
    }
    _check(changed_fields == {("operand_slots",)}, "isolated_projection_field", failed)
    for row in compiler_rows:
        expected_slots = row["semantic_plan_projection"]["expected"]["operand_slots"]
        observed_slots = row["semantic_plan_projection"]["observed"]["operand_slots"]
        _check(len(expected_slots) == len(observed_slots), f"slot_arity:{row['group']}:{row['family_id']}", failed)
        for expected_slot, observed_slot in zip(expected_slots, observed_slots):
            slot_diff = sorted(
                key for key in expected_slot if expected_slot.get(key) != observed_slot.get(key)
            )
            _check(slot_diff == ["metric"], f"slot_diff:{row['group']}:{row['family_id']}", failed)

    surface = refresh["compiler"]["surface"]
    counterfactual = refresh["compiler"]["counterfactual"]
    diversity = config["semantic_diversity_contract"]
    _check(refresh["compiler"]["canonical"]["exact"] == diversity["core_answer_families"] == 13, "canonical_breadth", failed)
    _check(refresh["compiler"]["ambiguity"]["exact"] == diversity["ambiguity_controls"] == 3, "ambiguity_breadth", failed)
    _check(surface["total"] == diversity["frozen_surface_questions"] == 91, "surface_question_count", failed)
    _check(len(surface["by_category"]) == diversity["frozen_surface_categories"] == 7, "surface_category_count", failed)
    _check(all(row["total"] == 13 for row in surface["by_category"].values()), "surface_each_family", failed)
    _check(counterfactual["total"] == counterfactual["detected"] == diversity["frozen_minimal_pairs"] == 57, "minimal_pairs", failed)
    _check(set(counterfactual["by_dimension"]) == set(diversity["minimal_pair_dimensions"]), "minimal_pair_dimensions", failed)

    hold_ahead = _hold_ahead_summary(config, refresh)
    hold_contract = diversity["orthogonal_hold_ahead"]
    _check(hold_ahead["questions"] == hold_contract["total_questions"] == 78, "hold_ahead_questions", failed)
    _check(hold_ahead["families"] == hold_contract["families"] == 13, "hold_ahead_families", failed)
    _check(set(hold_ahead["canonical_root_operation_distribution"]) == set(diversity["root_operation_semantics"]), "root_semantic_breadth", failed)
    _check(all(total == 26 for total in hold_ahead["categories"].values()), "hold_ahead_category_balance", failed)
    _check(hold_contract["runtime_may_match_recipe_literals"] is False, "literal_rules_forbidden", failed)
    _check(hold_contract["counts_as_company_or_layout_confirmation"] is False, "synthetic_no_layout_claim", failed)

    source = refresh["add_source_backed"]
    source_contract = config["source_backed_replay_contract"]
    _check(source["eligible_anchors"] == source_contract["anchors"] == 5, "source_anchors", failed)
    _check(source["companies"] == source_contract["companies"] == 5, "source_companies", failed)
    _check(source["source_documents"] == source_contract["source_documents"] == 5, "source_documents", failed)
    _check(source["real_layout_families"] == source_contract["real_layout_families"] == 3, "source_layouts", failed)
    _check(source["surface"]["total"] == source_contract["surface_queries"] == 35, "source_surface_queries", failed)
    _check(source["surface"]["passed"] == source_contract["currently_passing_surface_queries_to_retain"] == 20, "source_passing_retention", failed)

    scope = config["implementation_scope"]
    _check(scope["runtime_file_whitelist"] == ["src/hrm_finance_qa/question_planner.py"], "one_file_whitelist", failed)
    _check(scope["stage"] == "operand_metric_surface_normalization_only", "narrow_stage", failed)
    _check(scope["exact_symbols"]["new_helper"] == ["_strip_nonsemantic_metric_surface"], "new_helper_scope", failed)
    _check(scope["exact_symbols"]["existing_function_edit"] == ["compile_question_plan"], "existing_symbol_scope", failed)
    _check(not set(scope["runtime_file_whitelist"]) & set(config["frozen_runtime_files"]), "whitelist_frozen_disjoint", failed)
    _check(len(config["frozen_runtime_files"]) == 10, "frozen_runtime_count", failed)
    for non_goal in (
        "operation-intent repair",
        "ADD structural-frame repair",
        "retrieval changes",
        "source loading or extraction changes",
        "candidate binding or program synthesis changes",
        "symbolic verifier or proof changes",
    ):
        _check(non_goal in scope["non_goals"], f"non_goal:{non_goal}", failed)

    safe_results = safe["results"]
    gates = config["post_implementation_gates"]
    _check(safe_results["gold_actions"] == gates["frozen_safe_replay_actions"], "safe_actions", failed)
    _check(safe_results["gold_execution_correct"] == safe_results["gold_total"] == 363, "safe_execution", failed)
    _check(safe_results["permutation_exact"] == safe_results["permutation_total"] == 363, "safe_permutation", failed)
    _check(safe_results["counterfactual_generated"] == gates["frozen_safe_replay_counterfactuals"] == 141, "safe_counterfactual_count", failed)
    _check(safe_results["counterfactual_false_accepts"] == gates["counterfactual_false_accepts"] == 0, "safe_false_accepts", failed)
    _check(all(value is False for value in config["invariants"].values()), "precommit_invariants", failed)

    return {
        "schema": "finhrm-qp4-question-surface-implementation-precommit-report-v1",
        "created_at": config["created_at"],
        "classification": config["classification"],
        "precommit_gate_passed": not failed,
        "failed_gates": failed,
        "generic_failure_evidence": evidence,
        "semantic_diversity": {
            "core_answer_families": diversity["core_answer_families"],
            "ambiguity_controls": diversity["ambiguity_controls"],
            "frozen_surface_questions": surface["total"],
            "frozen_surface_categories": surface["by_category"],
            "frozen_minimal_pairs": counterfactual["total"],
            "minimal_pair_dimensions": counterfactual["by_dimension"],
            "orthogonal_hold_ahead": hold_ahead,
        },
        "source_backed_replay_contract": source_contract,
        "implementation_scope": scope,
        "frozen_runtime_files": config["frozen_runtime_files"],
        "post_implementation_gates": gates,
        "runtime_changed": False,
        "runtime_hashes": config["runtime_hashes_before_implementation"],
        "checkpoint_sha256": checkpoint["sha256"],
        "runtime_change_authorized_in_same_step": False,
        "authorized_next_step": config["authorized_next_step_after_lock"] if not failed else "Resolve precommit validation failures without changing runtime.",
        "qp4_complete": False,
        "qp5_authorized": False,
        "phase_d": "locked",
    }


def _markdown(report: dict[str, Any]) -> str:
    diversity = report["semantic_diversity"]
    hold = diversity["orthogonal_hold_ahead"]
    evidence = report["generic_failure_evidence"]
    lines = [
        "# QP4 question-surface implementation precommit v1",
        "",
        f"- Gate: `{'PASS' if report['precommit_gate_passed'] else 'FAIL'}`.",
        f"- Isolated evidence: `{evidence['compiler_residuals']}` compiler residuals; only `operand_slots[*].metric` differs.",
        f"- Frozen semantic breadth: `{diversity['core_answer_families']}` answer families, `{diversity['ambiguity_controls']}` ambiguity controls, `{diversity['frozen_surface_questions']}` surface questions and `{diversity['frozen_minimal_pairs']}` minimal pairs.",
        f"- Orthogonal hold-ahead: `{hold['questions']}` questions across `{hold['families']}` families and `{len(hold['categories'])}` target surface categories; current exact `{hold['exact_before_implementation']}/{hold['questions']}` is diagnostic only.",
        "- Future runtime scope: only `_strip_nonsemantic_metric_surface` and its wiring inside `compile_question_plan` in `question_planner.py`.",
        "- Operation intent, ADD framing, retrieval, source extraction, synthesis, verifier, renderer, QP5 and Phase D remain locked.",
        "- This precommit changes no runtime and makes no company/layout claim from synthetic question variants.",
    ]
    if report["failed_gates"]:
        lines += ["", "Failed gates: " + ", ".join(report["failed_gates"])]
    return "\n".join(lines) + "\n"


def write_outputs(
    report: dict[str, Any],
    report_path: Path = DEFAULT_REPORT,
    markdown_path: Path = DEFAULT_MARKDOWN,
    lock_path: Path = DEFAULT_LOCK,
) -> None:
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    artifacts = [
        _artifact(DEFAULT_CONFIG),
        _artifact(Path(__file__)),
        _artifact(TEST_PATH),
        _artifact(report_path),
        _artifact(markdown_path),
    ]
    lock = {
        "schema": "finhrm-qp4-question-surface-implementation-precommit-lock-v1",
        "created_at": report["created_at"],
        "classification": report["classification"],
        "status": "complete" if report["precommit_gate_passed"] else "failed",
        "precommit_gate_passed": report["precommit_gate_passed"],
        "runtime_changed": False,
        "runtime_hashes": report["runtime_hashes"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "runtime_file_whitelist_for_next_step": report["implementation_scope"]["runtime_file_whitelist"],
        "exact_symbol_whitelist_for_next_step": report["implementation_scope"]["exact_symbols"],
        "runtime_change_authorized_in_same_step": False,
        "retrieval_source_binding_or_verifier_edit_authorized": False,
        "artifacts": artifacts,
        "authorized_next_step": report["authorized_next_step"],
        "qp4_complete": False,
        "qp5_authorized": False,
        "phase_d": "locked",
    }
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    report = build_report(args.config)
    write_outputs(report, args.report, args.markdown, args.lock)
    print(
        json.dumps(
            {
                "precommit_gate_passed": report["precommit_gate_passed"],
                "failed_gates": report["failed_gates"],
                "runtime_changed": report["runtime_changed"],
                "hold_ahead_questions": report["semantic_diversity"]["orthogonal_hold_ahead"]["questions"],
                "whitelist": report["implementation_scope"]["runtime_file_whitelist"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["precommit_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
