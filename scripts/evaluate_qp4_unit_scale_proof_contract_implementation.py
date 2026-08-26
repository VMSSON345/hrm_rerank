#!/usr/bin/env python3
"""Evaluate the QP4 UNIT_SCALE proof-contract implementation and frozen gates."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

from hrm_finance_qa.program_data import currency_unit_scale
from hrm_finance_qa.question_planner import (
    compile_question_plan,
    compile_question_proof_contract,
)
from scripts.evaluate_qp4_development_baseline import (
    _QueryTimeout,
    _answer_signature,
    _parent_signature,
    _source_anchors,
    _source_counterfactual,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "finhrm_qp4_unit_scale_proof_contract_implementation_v1.json"
DEFAULT_SOURCE = ROOT / "reports" / "finhrm_qp4_unit_scale_proof_contract_source_replay_v1.json"
DEFAULT_SAFE = ROOT / "reports" / "finhrm_qp4_unit_scale_proof_contract_safe_replay_v1.json"
DEFAULT_SAFE_MARKDOWN = ROOT / "reports" / "finhrm_qp4_unit_scale_proof_contract_safe_replay_v1.md"
DEFAULT_SAFE_LOCK = ROOT / "reports" / "finhrm_qp4_unit_scale_proof_contract_safe_replay_lock_v1.json"
DEFAULT_REPORT = ROOT / "reports" / "finhrm_qp4_unit_scale_proof_contract_implementation_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "finhrm_qp4_unit_scale_proof_contract_implementation_v1.md"
DEFAULT_LOCK = ROOT / "reports" / "finhrm_qp4_unit_scale_proof_contract_implementation_lock_v1.json"
FUNCTIONAL_TEST = ROOT / "tests" / "test_qp4_unit_scale_proof_contract_implementation.py"
REPORT_TEST = ROOT / "tests" / "test_qp4_unit_scale_proof_contract_implementation_report.py"
PROTECTED = re.compile(r"(?:^|[/_.-])(test|holdout|final|blind|validation)(?:[/_.-]|$)", re.I)


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


def _verify_config(config: dict[str, Any]) -> None:
    for parent in config["parents"].values():
        path = ROOT / parent["path"]
        if _digest(path) != parent["sha256"]:
            raise ValueError(f"Parent hash mismatch: {parent['path']}")
    checkpoint = config["checkpoint"]
    if _digest(ROOT / checkpoint["path"]) != checkpoint["sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    if set(config["runtime_changes"]) != {
        "src/hrm_finance_qa/question_planner.py",
        "src/hrm_finance_qa/symbolic_verifier.py",
    }:
        raise ValueError("Runtime change set is outside the exact whitelist")
    for relative, change in config["runtime_changes"].items():
        if _digest(ROOT / relative) != change["after_sha256"]:
            raise ValueError(f"Postimplementation runtime hash mismatch: {relative}")
    for relative, expected in config["frozen_runtime_hashes"].items():
        if _digest(ROOT / relative) != expected:
            raise ValueError(f"Frozen runtime hash mismatch: {relative}")
    if any(config["invariants"].values()):
        raise ValueError("A release invariant was opened")


def _development_config(config: dict[str, Any]) -> dict[str, Any]:
    precommit = _load(ROOT / config["parents"]["residual_precommit_config"]["path"])
    planner_implementation = _load(
        ROOT / precommit["parents"]["planner_implementation_config"]["path"]
    )
    return _load(
        ROOT / planner_implementation["parents"]["development_cohort"]["path"]
    )


def run_source_replay(config_path: Path, output_path: Path) -> dict[str, Any]:
    from hrm_finance_qa.document_hrm import DocumentHRMBundle
    from hrm_finance_qa.document_reasoning_engine import DocumentHRMReasoningEngine

    config = _load(config_path)
    _verify_config(config)
    development = _development_config(config)
    before = _load(ROOT / config["parents"]["source_trace_before_implementation"]["path"])
    before_by_id = {row["example_id"]: row for row in before["rows"]}
    anchors = _source_anchors(development)
    checkpoint = ROOT / config["checkpoint"]["path"]
    engine = DocumentHRMReasoningEngine(DocumentHRMBundle.load(checkpoint, device="cpu"))
    timeout_seconds = int(
        development["source_backed_cohort"]["per_query_timeout_seconds"]
    )
    rows: list[dict[str, Any]] = []
    for position, (_, anchor) in enumerate(anchors, start=1):
        question = _source_counterfactual(str(anchor["question"]), "UNIT_SCALE")
        plan = compile_question_plan(question)
        parent = _parent_signature(anchor)
        prior = before_by_id[anchor["example_id"]]
        print(f"[{position}/{len(anchors)}] UNIT_SCALE proof replay {anchor['example_id']}", flush=True)
        common = {
            "example_id": anchor["example_id"],
            "company_key": anchor["company_key"],
            "source_key": anchor["source_sha256"],
            "source_path": anchor["source_path"],
            "layout_family": anchor["layout_family"],
            "question": question,
            "requested_scale": plan.requested_scale,
            "prior_action": prior.get("observed_signature", {}).get("action"),
        }
        try:
            with _QueryTimeout(timeout_seconds):
                answer = engine.answer(
                    question,
                    ROOT / anchor["source_path"],
                    mineru_auto_generate=False,
                )
            signature = _answer_signature(answer)
            obligations = [
                {"name": name, "applicability": applicability, "state": state}
                for name, applicability, state in answer.proof_obligations
            ]
            scale_obligation = next(
                (item for item in obligations if item["name"] == "unit_scale_match"),
                None,
            )
            prior_signature = prior.get("observed_signature", {})
            binding_retained = bool(prior_signature) and all(
                json.dumps(signature.get(key), sort_keys=True)
                == json.dumps(prior_signature.get(key), sort_keys=True)
                for key in ("operation", "value", "unit", "evidence")
            )
            same_old_winner = all(
                signature.get(key) == parent.get(key)
                for key in ("action", "operation", "value", "unit", "evidence")
            )
            selected_units = []
            for item in answer.evidence:
                factor = currency_unit_scale(item.candidate.unit)
                selected_units.append(
                    {
                        "unit": item.candidate.unit,
                        "currency_scale": None if factor is None else str(factor),
                    }
                )
            rows.append(
                common
                | {
                    "observed_signature": signature,
                    "selected_units": selected_units,
                    "proof_obligations": obligations,
                    "unit_scale_obligation": scale_obligation,
                    "prior_answer_to_abstain": common["prior_action"] == "ANSWER"
                    and signature.get("action") == "ABSTAIN",
                    "operation_value_unit_evidence_retained": binding_retained,
                    "old_winner_false_accept": same_old_winner,
                }
            )
        except Exception as exc:
            rows.append(
                common
                | {
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "prior_answer_to_abstain": False,
                    "operation_value_unit_evidence_retained": False,
                    "old_winner_false_accept": False,
                }
            )
    prior_answers = [row for row in rows if row["prior_action"] == "ANSWER"]
    raw = {
        "schema": "finhrm-qp4-unit-scale-proof-contract-source-replay-v1",
        "created_at": config["created_at"],
        "classification": "development_only_real_pdf_unit_scale_proof_postimplementation_replay",
        "coverage": {
            "runs": len(rows),
            "companies": len({row["company_key"] for row in rows}),
            "sources": len({row["source_key"] for row in rows}),
            "real_layouts": len({row["layout_family"] for row in rows}),
        },
        "summary": {
            "prior_answer_cases": len(prior_answers),
            "prior_answer_to_abstain": sum(bool(row["prior_answer_to_abstain"]) for row in prior_answers),
            "prior_answer_binding_retained": sum(
                bool(row["operation_value_unit_evidence_retained"])
                for row in prior_answers
            ),
            "prior_answer_scale_required_unknown": sum(
                row.get("unit_scale_obligation")
                == {"name": "unit_scale_match", "applicability": "required", "state": "UNKNOWN"}
                for row in prior_answers
            ),
            "old_winner_false_accepts": sum(bool(row["old_winner_false_accept"]) for row in rows),
            "errors": sum("error" in row for row in rows),
        },
        "rows": rows,
        "runtime_hashes": {
            **{
                relative: change["after_sha256"]
                for relative, change in config["runtime_changes"].items()
            },
            **config["frozen_runtime_hashes"],
        },
        "checkpoint_sha256": config["checkpoint"]["sha256"],
        "abw_records": 0,
        "protected_records": [],
    }
    output_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return raw


def build_report(
    config_path: Path = DEFAULT_CONFIG,
    source_path: Path = DEFAULT_SOURCE,
    safe_path: Path = DEFAULT_SAFE,
) -> dict[str, Any]:
    config = _load(config_path)
    _verify_config(config)
    source = _load(source_path)
    safe = _load(safe_path)
    coverage = source["coverage"]
    summary = source["summary"]
    safe_results = safe["results"]
    gates = config["post_implementation_gates"]

    explicit_plan = compile_question_plan("Doanh thu theo triệu đồng là bao nhiêu?")
    implicit_plan = compile_question_plan("Doanh thu là bao nhiêu?")
    explicit_contract = compile_question_proof_contract(explicit_plan, "SELECT")
    implicit_contract = compile_question_proof_contract(implicit_plan, "SELECT")
    checks = {
        "explicit_scale_obligation_required": "unit_scale_match" in explicit_contract.required,
        "no_scale_obligation_not_applicable": "unit_scale_match" in implicit_contract.not_applicable,
        "source_5_companies_5_sources_3_layouts": coverage
        == {"runs": 5, "companies": 5, "sources": 5, "real_layouts": 3},
        "source_zero_old_winner_false_accepts": summary["old_winner_false_accepts"] == 0,
        "source_four_answers_to_abstain": summary["prior_answer_to_abstain"]
        == summary["prior_answer_cases"]
        == 4,
        "source_four_bindings_retained": summary["prior_answer_binding_retained"] == 4,
        "source_four_required_unknown_scale_proofs": summary[
            "prior_answer_scale_required_unknown"
        ]
        == 4,
        "safe_actions_exact": safe_results["gold_actions"] == gates["frozen_safe_replay_actions"],
        "safe_execution_exact": safe_results["gold_execution_correct"]
        == safe_results["gold_total"]
        == 363,
        "safe_permutation_exact": safe_results["permutation_exact"]
        == safe_results["permutation_total"]
        == 363,
        "safe_counterfactual_exact": safe_results["counterfactual_generated"] == 141
        and safe_results["counterfactual_false_accepts"] == 0,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    source_gate_passed = all(
        checks[name]
        for name in (
            "source_5_companies_5_sources_3_layouts",
            "source_zero_old_winner_false_accepts",
            "source_four_answers_to_abstain",
            "source_four_bindings_retained",
            "source_four_required_unknown_scale_proofs",
        )
    )
    implementation_gate_passed = not failed
    return {
        "schema": "finhrm-qp4-unit-scale-proof-contract-implementation-report-v1",
        "created_at": config["created_at"],
        "classification": config["classification"],
        "status": (
            "complete" if implementation_gate_passed else "implementation_executed_gate_not_closed"
        ),
        "implementation_gate_passed": implementation_gate_passed,
        "failed_gates": failed,
        "runtime_changes": config["runtime_changes"],
        "implementation_contract": config["implementation_contract"],
        "source_replay": {
            "coverage": coverage,
            "summary": summary,
            "gate_passed": source_gate_passed,
        },
        "safe_replay": {
            "actions": safe_results["gold_actions"],
            "expected_actions": gates["frozen_safe_replay_actions"],
            "execution": f"{safe_results['gold_execution_correct']}/{safe_results['gold_total']}",
            "permutation": f"{safe_results['permutation_exact']}/{safe_results['permutation_total']}",
            "counterfactuals": safe_results["counterfactual_generated"],
            "counterfactual_false_accepts": safe_results["counterfactual_false_accepts"],
        },
        "checks": checks,
        "frozen_action_gate_conflict": (
            {
                "observed": safe_results["gold_actions"],
                "expected": gates["frozen_safe_replay_actions"],
                "cause": "Two explicit VND requests have gold candidates with unit_status=missing; the new required proof correctly yields UNKNOWN/ABSTAIN, but the precommit froze their prior ANSWER actions.",
                "examples": ["159", "621"],
                "resolution_requires_new_precommit": True,
            }
            if not checks["safe_actions_exact"]
            else None
        ),
        "further_runtime_change_authorized": False,
        "qp4_complete": False,
        "qp5_authorized": False,
        "phase_d": "locked",
        "authorized_next_step": (
            "QP4-POST-UNIT-SCALE-DEVELOPMENT-BASELINE-REFRESH-PRECOMMIT; no runtime edit is authorized."
            if implementation_gate_passed
            else "QP4-UNIT-SCALE-FROZEN-ACTION-CONTRACT-REVIEW-PRECOMMIT must resolve the two missing-unit proof cases without forcing ANSWER or changing runtime."
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    source = report["source_replay"]
    safe = report["safe_replay"]
    lines = [
        "# QP4 UNIT_SCALE proof-contract implementation v1",
        "",
        f"- Gate: `{'PASS' if report['implementation_gate_passed'] else 'FAIL'}`.",
        f"- Source gate: `{'PASS' if source['gate_passed'] else 'FAIL'}`; old-winner false accepts `{source['summary']['old_winner_false_accepts']}`.",
        f"- Prior ANSWER to ABSTAIN: `{source['summary']['prior_answer_to_abstain']}/4`; operation/value/unit/evidence retained `{source['summary']['prior_answer_binding_retained']}/4`.",
        f"- Frozen replay actions: observed `{safe['actions']}`, expected `{safe['expected_actions']}`.",
        f"- Execution `{safe['execution']}`, permutation `{safe['permutation']}`, counterfactual false accepts `{safe['counterfactual_false_accepts']}`.",
        "- No conversion, rescaling, retrieval, binding, synthesis, engine or renderer change was made.",
        "- QP4 is incomplete; QP5 and Phase D remain locked.",
    ]
    if report["frozen_action_gate_conflict"]:
        lines.append(
            "- Frozen-action conflict: examples `159` and `621` explicitly request VND but their selected evidence has missing unit proof; UNKNOWN/ABSTAIN is required."
        )
    lines.extend(["", f"Next: {report['authorized_next_step']}", ""])
    return "\n".join(lines)


def write_outputs(
    report: dict[str, Any],
    report_path: Path = DEFAULT_REPORT,
    markdown_path: Path = DEFAULT_MARKDOWN,
    lock_path: Path = DEFAULT_LOCK,
) -> None:
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    artifacts = [
        DEFAULT_CONFIG,
        Path(__file__).resolve(),
        FUNCTIONAL_TEST,
        REPORT_TEST,
        DEFAULT_SOURCE,
        DEFAULT_SAFE,
        DEFAULT_SAFE_MARKDOWN,
        DEFAULT_SAFE_LOCK,
        report_path,
        markdown_path,
    ]
    missing = [path for path in artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(str(path) for path in missing))
    lock = {
        "schema": "finhrm-qp4-unit-scale-proof-contract-implementation-lock-v1",
        "created_at": report["created_at"],
        "classification": report["classification"],
        "status": report["status"],
        "implementation_gate_passed": report["implementation_gate_passed"],
        "source_gate_passed": report["source_replay"]["gate_passed"],
        "failed_gates": report["failed_gates"],
        "runtime_changes": report["runtime_changes"],
        "frozen_action_gate_conflict": report["frozen_action_gate_conflict"],
        "further_runtime_change_authorized": False,
        "artifacts": [_artifact(path) for path in artifacts],
        "qp4_complete": False,
        "qp5_authorized": False,
        "phase_d": "locked",
        "authorized_next_step": report["authorized_next_step"],
    }
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--safe", type=Path, default=DEFAULT_SAFE)
    parser.add_argument("--run-source", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.run_source:
        run_source_replay(args.config, args.source)
    report = build_report(args.config, args.source, args.safe)
    if args.write:
        write_outputs(report)
    print(
        json.dumps(
            {
                "implementation_gate_passed": report["implementation_gate_passed"],
                "failed_gates": report["failed_gates"],
                "source_gate_passed": report["source_replay"]["gate_passed"],
                "source_old_winner_false_accepts": report["source_replay"]["summary"][
                    "old_winner_false_accepts"
                ],
                "safe_actions": report["safe_replay"]["actions"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report["implementation_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
