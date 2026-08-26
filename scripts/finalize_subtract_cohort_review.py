#!/usr/bin/env python3
"""Finalize an explicitly reviewed SUBTRACT cohort without silent approvals."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrm_finance_qa.direction_contract import (  # noqa: E402
    assess_program_direction,
    reversed_pair_is_proven_negative,
)
from hrm_finance_qa.evidence_schema import build_evidence_graph  # noqa: E402
from hrm_finance_qa.program_data import (  # noqa: E402
    load_program_examples,
    program_example_to_dict,
)
from hrm_finance_qa.question_planner import compile_question_plan  # noqa: E402
from hrm_finance_qa.v12b2_supervision import supervision_integrity_issues  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--render-report", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.report):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")

    examples = load_program_examples(args.input)
    render = json.loads(args.render_report.read_text(encoding="utf-8"))
    decisions_payload = json.loads(args.decisions.read_text(encoding="utf-8"))
    if render.get("input_sha256") != _sha256(args.input):
        raise ValueError("Render report does not match candidate input")
    if decisions_payload.get("classification") != "development_only":
        raise ValueError("Review decisions must be development_only")
    render_ids = {str(item["example_id"]) for item in render.get("records", [])}
    decisions = {
        str(item["example_id"]): str(item["decision"])
        for item in decisions_payload.get("decisions", [])
    }
    example_ids = {example.example_id for example in examples}
    if render_ids != example_ids or set(decisions) != example_ids:
        raise ValueError("Candidate, render and decision IDs must match exactly")
    if any(value not in {"pass", "reject"} for value in decisions.values()):
        raise ValueError("Review decision must be pass or reject")

    retained = []
    direction_outcomes: Counter[str] = Counter()
    integrity_issues: Counter[str] = Counter()
    unproven_reversed = 0
    for example in examples:
        if decisions[example.example_id] != "pass":
            continue
        graph = build_evidence_graph(example.candidates)
        plan = compile_question_plan(example.question)
        direction = assess_program_direction(plan, example.program, graph)
        direction_outcomes[direction.outcome] += 1
        issues = supervision_integrity_issues(example, graph=graph, plan=plan)
        integrity_issues.update(issues)
        proven = reversed_pair_is_proven_negative(plan, example.program, graph)
        unproven_reversed += int(not proven)
        if issues or direction.outcome != "direction_supported" or not proven:
            continue
        supervision = {
            **dict(example.supervision or {}),
            "review_status": "visually_reviewed_pass",
            "visual_review": {
                "reviewer": decisions_payload.get("reviewer", ""),
                "decision": "pass",
                "render_report_sha256": _sha256(args.render_report),
                "decisions_sha256": _sha256(args.decisions),
            },
        }
        retained.append(replace(example, supervision=supervision))

    report = {
        "schema": "subtract-development-cohort-finalization-v1",
        "classification": "development_only_reviewed",
        "blind_test_used": False,
        "global_validation_loaded": False,
        "model_invoked": False,
        "fine_tuning_performed": False,
        "threshold_changed": False,
        "inputs": {
            "candidates": str(args.input.resolve()),
            "candidates_sha256": _sha256(args.input),
            "render_report": str(args.render_report.resolve()),
            "render_report_sha256": _sha256(args.render_report),
            "decisions": str(args.decisions.resolve()),
            "decisions_sha256": _sha256(args.decisions),
        },
        "metrics": {
            "candidate_examples": len(examples),
            "review_pass": sum(value == "pass" for value in decisions.values()),
            "review_reject": sum(value == "reject" for value in decisions.values()),
            "retained_examples": len(retained),
            "company_groups": len(
                {str(item.supervision.get("company_key", "")) for item in retained}
            ),
            "direction_outcomes": dict(sorted(direction_outcomes.items())),
            "integrity_issue_counts": dict(sorted(integrity_issues.items())),
            "unproven_reversed_pairs": unproven_reversed,
        },
    }
    report["ready_for_phase_c_merge"] = bool(
        retained
        and len(retained) == sum(value == "pass" for value in decisions.values())
        and not integrity_issues
        and not unproven_reversed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(program_example_to_dict(item), ensure_ascii=False) + "\n"
            for item in retained
        ),
        encoding="utf-8",
    )
    report["output_sha256"] = _sha256(args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({**report["metrics"], "ready_for_phase_c_merge": report["ready_for_phase_c_merge"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
