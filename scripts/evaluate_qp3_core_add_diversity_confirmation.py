from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRECOMMIT = ROOT / "config/finhrm_qp3_core_add_diversity_confirmation_precommit_v1.json"
DEFAULT_REPORT = ROOT / "reports/finhrm_qp3_core_add_diversity_confirmation_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports/finhrm_qp3_core_add_diversity_confirmation_v1.md"
DEFAULT_LOCK = ROOT / "reports/finhrm_qp3_core_add_diversity_confirmation_lock_v1.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_supervision(path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            supervision = record.get("supervision", {})
            if not isinstance(supervision, dict):
                supervision = {}
            result[str(record.get("example_id", ""))] = {
                "classification": str(supervision.get("classification", "")),
                "review_status": str(supervision.get("review_status", "")),
            }
    return result


def _layout_coverage(
    positives: list[dict[str, Any]],
    supervision: dict[str, dict[str, str]],
    accepted_review_statuses: set[str],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in positives:
        grouped[str(row["layout_family"])].append(row)
    result: dict[str, dict[str, Any]] = {}
    for layout, rows in sorted(grouped.items()):
        reviewed_pairs = {
            (str(row["company_key"]), str(row["source_key"]))
            for row in rows
            if supervision.get(str(row["example_id"]), {}).get("review_status")
            in accepted_review_statuses
        }
        company_source_pairs = {
            (str(row["company_key"]), str(row["source_key"])) for row in rows
        }
        result[layout] = {
            "positive_examples": len(rows),
            "distinct_companies": len({row["company_key"] for row in rows}),
            "distinct_source_pdfs": len({row["source_key"] for row in rows}),
            "company_source_pairs": len(company_source_pairs),
            "independently_reviewed_company_source_pairs": len(reviewed_pairs),
            "all_company_source_pairs_independently_reviewed": reviewed_pairs
            == company_source_pairs,
        }
    return result


def build_report(precommit_path: Path = DEFAULT_PRECOMMIT) -> dict[str, Any]:
    precommit = json.loads(precommit_path.read_text(encoding="utf-8"))
    parent_paths = {
        name: ROOT / parent["path"] for name, parent in precommit["parents"].items()
    }
    parent_hashes_exact = all(
        "sha256" not in precommit["parents"][name]
        or _digest(path) == precommit["parents"][name]["sha256"]
        for name, path in parent_paths.items()
    )
    confirmation = json.loads(
        parent_paths["candidate_level_confirmation"].read_text(encoding="utf-8")
    )
    source_path = ROOT / confirmation["source"]["path"]
    supervision = _source_supervision(source_path)
    positives = [
        row for row in confirmation["details"] if row["confirmation_role"] == "POSITIVE"
    ]
    accepted_review_statuses = set(
        precommit["diversity_contract"]["accepted_independent_review_statuses"]
    )
    layout = _layout_coverage(positives, supervision, accepted_review_statuses)
    review_statuses = Counter(
        supervision.get(str(row["example_id"]), {}).get("review_status", "")
        for row in positives
    )
    classification_counts = Counter(
        supervision.get(str(row["example_id"]), {}).get("classification", "")
        for row in positives
    )
    stages = {
        "real_pdf_source": all(row["source_pdf_exists"] for row in positives),
        "extraction_or_table_reconstruction": False,
        "typed_evidence_graph": True,
        "retrieval_or_intent_proposal": False,
        "symbolic_binding": False,
        "deterministic_verification_and_execution": all(
            row["gold_execution_correct"] for row in positives
        ),
    }
    contract = precommit["diversity_contract"]
    minima_company = contract["minimum_distinct_companies_per_additional_layout"]
    minima_source = contract["minimum_distinct_source_pdfs_per_additional_layout"]
    checks = {
        "parent_hashes_exact": parent_hashes_exact,
        "candidate_level_confirmation_safety_clean": confirmation[
            "confirmation_gate_passed"
        ]
        and not confirmation["failed_gates"],
        "minimum_two_additional_real_layouts": len(layout)
        >= contract["minimum_additional_real_layout_families"],
        "minimum_three_companies_in_every_additional_layout": bool(layout)
        and all(row["distinct_companies"] >= minima_company for row in layout.values()),
        "minimum_three_source_pdfs_in_every_additional_layout": bool(layout)
        and all(row["distinct_source_pdfs"] >= minima_source for row in layout.values()),
        "independent_review_complete_in_every_company_layout": bool(layout)
        and all(
            row["all_company_source_pairs_independently_reviewed"]
            for row in layout.values()
        ),
        "all_required_end_to_end_stages_exercised": all(
            stages.get(stage, False) for stage in contract["required_stages"]
        ),
        "company_layout_disjoint_confirmation_available": False,
        "candidate_permutation_exact": confirmation["results"]["permutation_exact"]
        == confirmation["results"]["gold_total"],
        "counterfactual_false_accepts_zero": confirmation["results"][
            "counterfactual_false_accepts"
        ]
        == contract["counterfactual_false_accepts"],
        "company_layout_macro_and_worst_reported": all(
            key in confirmation["results"][group]
            for group in ("company", "layout")
            for key in ("macro_answer_rate", "worst_answer_rate")
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "schema": "finhrm-qp3-core-add-diversity-confirmation-v1",
        "created_at": "2026-08-20",
        "classification": "development_only_candidate_evidence_diversity_gap_audit",
        "status": "diversity_gate_closed" if not failed else "diversity_gate_not_closed",
        "diversity_gate_passed": not failed,
        "failed_gates": failed,
        "parents": {
            name: {"path": path.relative_to(ROOT).as_posix(), "sha256": _digest(path)}
            for name, path in parent_paths.items()
        },
        "current_evidence": {
            "role": precommit["current_candidate_evidence_role"],
            "positive_examples": len(positives),
            "distinct_companies": len({row["company_key"] for row in positives}),
            "distinct_source_pdfs": len({row["source_key"] for row in positives}),
            "additional_real_layouts": len(layout),
            "layout_coverage": layout,
            "supervision_classifications": dict(sorted(classification_counts.items())),
            "review_statuses": dict(sorted(review_statuses.items())),
            "independently_reviewed_positives": sum(
                count
                for status, count in review_statuses.items()
                if status in accepted_review_statuses
            ),
            "pipeline_stage_evidence": stages,
            "starts_from_pre_materialized_candidates": True,
        },
        "gate_checks": checks,
        "invariants": {
            "runtime_changed": False,
            "threshold_checkpoint_or_model_changed": False,
            "fine_tuning_performed": False,
            "abw_used_as_confirmation": False,
            "protected_evaluation_loaded": False,
            "synthetic_projection_counted_as_confirmation": False,
            "candidate_rank_score_or_confidence_used_as_proof": False,
            "qp3_answer_presentation_unlocked": False,
            "qp4_authorized": False,
            "phase_d_unlocked": False,
        },
        "interpretation": (
            "The 10 source-backed positives remain valid supporting evidence that the frozen ADD proof "
            "contract behaves safely on materialized candidates. They do not close diverse system operation: "
            "OCR/hybrid has only two companies/sources, all positives remain pending independent review, "
            "and extraction, retrieval/intent proposal and symbolic binding were not replayed end to end."
        ),
        "authorized_next_step": precommit["authorized_next_step"],
    }


def _markdown(report: dict[str, Any]) -> str:
    evidence = report["current_evidence"]
    rows = [
        "# FinHRM QP3-CORE ADD diversity confirmation",
        "",
        f"- Diversity gate: **{'PASS' if report['diversity_gate_passed'] else 'NOT CLOSED'}**.",
        f"- Candidate-level positives: `{evidence['positive_examples']}` across `{evidence['distinct_companies']}` companies / `{evidence['additional_real_layouts']}` layouts; role: `{evidence['role']}`.",
        f"- Independently reviewed positives: `{evidence['independently_reviewed_positives']}/{evidence['positive_examples']}`.",
        f"- Pipeline stages: `{evidence['pipeline_stage_evidence']}`.",
        f"- Failed gates: `{report['failed_gates']}`.",
        "",
        "| Layout | Positives | Companies | PDFs | Reviewed company/PDF pairs |",
        "|---|---:|---:|---:|---:|",
    ]
    for layout, item in evidence["layout_coverage"].items():
        rows.append(
            f"| {layout} | {item['positive_examples']} | {item['distinct_companies']} | "
            f"{item['distinct_source_pdfs']} | {item['independently_reviewed_company_source_pairs']}/{item['company_source_pairs']} |"
        )
    rows.extend(["", report["interpretation"], ""])
    return "\n".join(rows)


def write_outputs(
    report: dict[str, Any], report_path: Path, markdown_path: Path, lock_path: Path
) -> None:
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    artifacts = []
    for path in (
        DEFAULT_PRECOMMIT,
        Path(__file__),
        ROOT / "tests/test_qp3_core_add_diversity_confirmation.py",
        report_path,
        markdown_path,
    ):
        artifacts.append(
            {
                "path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _digest(path),
            }
        )
    lock = {
        "schema": "finhrm-qp3-core-add-diversity-confirmation-lock-v1",
        "created_at": report["created_at"],
        "classification": report["classification"],
        "status": report["status"],
        "diversity_gate_passed": report["diversity_gate_passed"],
        "failed_gates": report["failed_gates"],
        "invariants": report["invariants"],
        "artifacts": artifacts,
        "authorized_next_step": report["authorized_next_step"],
    }
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    report = build_report()
    write_outputs(report, args.report, args.markdown, args.lock)
    print(
        json.dumps(
            {
                "gate": report["diversity_gate_passed"],
                "failed": report["failed_gates"],
                "layout_coverage": report["current_evidence"]["layout_coverage"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report["diversity_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
