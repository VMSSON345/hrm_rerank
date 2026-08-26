#!/usr/bin/env python3
"""Validate the QP4 question-surface hold-ahead contract correction."""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from scripts.validate_qp4_question_surface_implementation_precommit import (
    _hold_ahead_summary,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/finhrm_qp4_question_surface_implementation_precommit_correction_v1.json"
DEFAULT_REPORT = ROOT / "reports/finhrm_qp4_question_surface_implementation_precommit_correction_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports/finhrm_qp4_question_surface_implementation_precommit_correction_v1.md"
DEFAULT_LOCK = ROOT / "reports/finhrm_qp4_question_surface_implementation_precommit_correction_lock_v1.json"
TEST_PATH = ROOT / "tests/test_qp4_question_surface_implementation_precommit_correction.py"


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _digest(path),
    }


def build_effective_precommit(config: dict[str, Any]) -> dict[str, Any]:
    parent = _load(ROOT / config["parents"]["original_precommit_config"]["path"])
    effective = deepcopy(parent)
    categories = effective["semantic_diversity_contract"]["orthogonal_hold_ahead"][
        "categories"
    ]
    for correction in (config["correction"], config["additional_correction"]):
        recipes = categories[correction["category"]]
        if recipes[correction["recipe_index_zero_based"]] != correction["invalid_recipe"]:
            raise ValueError("Correction target no longer matches the immutable parent")
        recipes[correction["recipe_index_zero_based"]] = correction[
            "replacement_recipe"
        ]
    return effective


def build_report(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _load(config_path)
    failed: list[str] = []

    def check(condition: bool, name: str) -> None:
        if not condition:
            failed.append(name)

    for name, parent in config["parents"].items():
        path = ROOT / parent["path"]
        check(path.is_file() and _digest(path) == parent["sha256"], f"parent_hash:{name}")
    for relative, expected in config["runtime_hashes"].items():
        check(_digest(ROOT / relative) == expected, f"runtime_hash:{relative}")

    parent = _load(ROOT / config["parents"]["original_precommit_config"]["path"])
    correction = config["correction"]
    original_categories = parent["semantic_diversity_contract"][
        "orthogonal_hold_ahead"
    ]["categories"]
    for index, item in enumerate(
        (correction, config["additional_correction"]), start=1
    ):
        original_recipes = original_categories[item["category"]]
        check(
            original_recipes[item["recipe_index_zero_based"]]
            == item["invalid_recipe"],
            f"immutable_recipe_target:{index}",
        )
    check("FORECAST column-role alias" in correction["invalidity"], "typed_conflict_documented")
    check(correction["historical_artifacts_modified"] is False, "history_immutable")
    check(correction["runtime_modified"] is False, "runtime_unchanged")

    effective = build_effective_precommit(config)
    refresh = _load(ROOT / config["parents"]["baseline_refresh"]["path"])
    hold = _hold_ahead_summary(effective, refresh)
    contract = config["effective_hold_ahead_contract"]
    check(hold["questions"] == contract["total_questions"] == 78, "question_count")
    check(hold["families"] == contract["families"] == 13, "family_count")
    check(len(hold["categories"]) == contract["categories"] == 3, "category_count")
    check(hold["exact_before_implementation"] == 42, "isolated_baseline_exact")
    for row in hold["details"]:
        check(
            set(row["changed_fields"]).issubset({"operand_slots"}),
            f"only_operand_metric:{row['category']}:{row['family_id']}:{row['recipe']}",
        )

    scope = config["implementation_scope_unchanged"]
    check(scope["runtime_file_whitelist"] == ["src/hrm_finance_qa/question_planner.py"], "one_file_scope")
    check(
        scope["exact_symbols"]
        == {
            "new_helper": ["_strip_nonsemantic_metric_surface"],
            "existing_function_edit": ["compile_question_plan"],
        },
        "symbol_scope",
    )
    check(scope["operation_intent_repair_authorized"] is False, "operation_stays_locked")
    check(scope["retrieval_source_binding_verifier_edit_authorized"] is False, "downstream_stays_locked")
    check(all(value is False for value in config["invariants"].values()), "invariants")

    return {
        "schema": "finhrm-qp4-question-surface-implementation-precommit-correction-report-v1",
        "created_at": config["created_at"],
        "classification": config["classification"],
        "correction_gate_passed": not failed,
        "failed_gates": failed,
        "correction": correction,
        "additional_correction": config["additional_correction"],
        "effective_hold_ahead": hold,
        "implementation_scope": scope,
        "runtime_changed": False,
        "runtime_hashes": config["runtime_hashes"],
        "authorized_next_step": config["authorized_next_step_after_lock"] if not failed else "Resolve correction validation failures without changing runtime.",
        "qp4_complete": False,
        "qp5_authorized": False,
        "phase_d": "locked",
    }


def write_outputs(
    report: dict[str, Any],
    report_path: Path = DEFAULT_REPORT,
    markdown_path: Path = DEFAULT_MARKDOWN,
    lock_path: Path = DEFAULT_LOCK,
) -> None:
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = (
        "# QP4 question-surface precommit correction v1\n\n"
        f"- Gate: `{'PASS' if report['correction_gate_passed'] else 'FAIL'}`.\n"
        "- Invalid hold-ahead clause contained the typed `FORECAST` role alias and was not eligible as erasable surface.\n"
        "- Replacement excludes qualitative commentary only; effective matrix remains `78` questions and current isolated baseline is `42/78`.\n"
        "- Historical artifacts and runtime remain unchanged; the one-file/two-symbol implementation scope is unchanged.\n"
    )
    markdown_path.write_text(markdown, encoding="utf-8")
    artifacts = [
        _artifact(DEFAULT_CONFIG),
        _artifact(Path(__file__)),
        _artifact(TEST_PATH),
        _artifact(report_path),
        _artifact(markdown_path),
    ]
    lock = {
        "schema": "finhrm-qp4-question-surface-implementation-precommit-correction-lock-v1",
        "created_at": report["created_at"],
        "classification": report["classification"],
        "status": "complete" if report["correction_gate_passed"] else "failed",
        "correction_gate_passed": report["correction_gate_passed"],
        "runtime_changed": False,
        "runtime_hashes": report["runtime_hashes"],
        "implementation_scope": report["implementation_scope"],
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
    print(json.dumps({"correction_gate_passed": report["correction_gate_passed"], "failed_gates": report["failed_gates"], "runtime_changed": False}, ensure_ascii=False))
    return 0 if report["correction_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
