from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from hrm_finance_qa.program_data import load_program_examples
from hrm_finance_qa.question_planner import compile_question_plan
from scripts.build_qp3_core_add_diverse_development_cohort import (
    ROOT,
    DEFAULT_MANIFEST,
    DEFAULT_NATIVE_POOL,
    _explicit_wrong_period_counterfactual,
    _selected,
)
from scripts.evaluate_qp3_core_s4_safe_replay import _proof_decision


OCR_V1 = ROOT / "data/finhrm_qp3_core_add_diversity_ocr_generated_pool_v1.jsonl"
OUTPUT = ROOT / "reports/finhrm_qp3_core_add_relative_period_counterfactual_audit_v1.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report() -> dict[str, object]:
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    by_source = {Path(row["document"]).name: row for row in manifest["documents"]}
    details = []
    for pool in (DEFAULT_NATIVE_POOL, OCR_V1):
        for example in load_program_examples(pool):
            if example.program.operation != "ADD":
                continue
            if not compile_question_plan(example.question).addition_role_hypotheses:
                continue
            gold = _proof_decision(example.question, example.program, example.candidates)
            if gold["action"] != "ANSWER" or gold["proof_contract_state"] != "VALID":
                continue
            counterfactual = _explicit_wrong_period_counterfactual(example)
            if counterfactual is None:
                continue
            program, mutation = counterfactual
            decision = _proof_decision(example.question, program, example.candidates)
            selected = _selected(example)
            metadata = by_source[selected[0].source_name]
            details.append(
                {
                    "example_id": example.example_id,
                    "company_key": metadata["company_key"],
                    "source_key": metadata["source_key"],
                    "layout_family": metadata["layout_family"],
                    "layout_stratum": metadata["layout_stratum"],
                    "question": example.question,
                    "gold_periods": [item.period for item in selected],
                    "mutation": mutation,
                    "counterfactual_action": decision["action"],
                    "counterfactual_proof_contract_state": decision["proof_contract_state"],
                    "counterfactual_reasons": decision["reasons"],
                }
            )
    false_accepts = [row for row in details if row["counterfactual_action"] == "ANSWER"]
    by_layout = Counter(str(row["layout_family"]) for row in false_accepts)
    return {
        "schema": "finhrm-qp3-core-add-relative-period-counterfactual-audit-v1",
        "created_at": "2026-08-20",
        "classification": "development_only_generic_explicit_cell_linked_counterfactual_audit",
        "inputs": {
            "manifest": {"path": DEFAULT_MANIFEST.relative_to(ROOT).as_posix(), "sha256": _digest(DEFAULT_MANIFEST)},
            "native_pool": {"path": DEFAULT_NATIVE_POOL.relative_to(ROOT).as_posix(), "sha256": _digest(DEFAULT_NATIVE_POOL)},
            "ocr_pool": {"path": OCR_V1.relative_to(ROOT).as_posix(), "sha256": _digest(OCR_V1)},
        },
        "metrics": {
            "counterfactuals": len(details),
            "false_accepts": len(false_accepts),
            "false_accept_companies": len({row["company_key"] for row in false_accepts}),
            "false_accept_sources": len({row["source_key"] for row in false_accepts}),
            "false_accept_layout_families": len(by_layout),
            "false_accepts_by_layout": dict(sorted(by_layout.items())),
        },
        "details": details,
        "invariants": {
            "counterfactual_uses_only_explicit_same_source_page_table_row_and_distinct_period_cells": True,
            "candidate_rank_score_or_confidence_used": False,
            "protected_evaluation_loaded": False,
            "abw_loaded": False,
            "runtime_changed_by_audit": False,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = build_report()
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, sort_keys=True))
