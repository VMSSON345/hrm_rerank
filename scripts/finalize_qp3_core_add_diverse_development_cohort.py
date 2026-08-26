from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

from hrm_finance_qa.program_data import load_program_examples, write_program_examples
from scripts.build_qp3_core_add_diverse_development_cohort import (
    ROOT,
    DEFAULT_NATIVE_POOL,
    DEFAULT_OCR_POOL,
    DEFAULT_PRECOMMIT,
    DEFAULT_SOURCE_AUDIT,
    _eligibility,
    _selected,
)


DEFAULT_REVIEW = ROOT / "reports/finhrm_qp3_core_add_diverse_development_visual_review_v1.json"
DEFAULT_OUTPUT = ROOT / "data/finhrm_qp3_core_add_diverse_development_reviewed_cohort_v1.jsonl"
DEFAULT_REPORT = ROOT / "reports/finhrm_qp3_core_add_diverse_development_reviewed_cohort_v1.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_reviewed_cohort() -> tuple[list[Any], dict[str, Any]]:
    precommit = json.loads(DEFAULT_PRECOMMIT.read_text(encoding="utf-8"))
    audit = json.loads(DEFAULT_SOURCE_AUDIT.read_text(encoding="utf-8"))
    review = json.loads(DEFAULT_REVIEW.read_text(encoding="utf-8"))
    strata = precommit["deterministic_selection"]["layout_strata"]
    by_source = {
        row["source_name"]: {
            "source_key": row["source_key"],
            "source_name": row["source_name"],
            "source_path": row["source_path"],
            "sha256": row["source_sha256"],
            "company_key": row["company_key"],
            "layout_family": row["layout_family"],
            "layout_stratum": next(
                name for name, families in strata.items() if row["layout_family"] in families
            ),
        }
        for row in audit["details"]
    }
    reviews = {
        (row["source_name"], row["example_id"]): row for row in review["records"]
    }
    eligible = []
    review_failures: list[dict[str, Any]] = []
    for example in (
        *load_program_examples(DEFAULT_NATIVE_POOL),
        *load_program_examples(DEFAULT_OCR_POOL),
    ):
        selected = _selected(example)
        if not selected:
            continue
        metadata = by_source.get(selected[0].source_name)
        decision = reviews.get((selected[0].source_name, example.example_id))
        if metadata is None or decision is None:
            continue
        if decision["source_sha256"] != metadata["sha256"]:
            raise RuntimeError("review/source hash mismatch")
        if decision["status"] != "independently_reviewed_approved":
            review_failures.append(decision)
            continue
        passed, eligibility = _eligibility(example, metadata)
        observed_values = [str(item.value) for item in selected]
        review_values = [str(value) for value in decision.get("observed_values", [])]
        review_exact = (
            observed_values == review_values
            and str(example.answer_value) == str(decision.get("observed_sum", ""))
        )
        if not passed or not review_exact:
            raise RuntimeError(
                f"approved review does not match eligible evidence: {selected[0].source_name}/{example.example_id}"
            )
        physical = tuple(
            (item.page, item.table_id, item.row_label, item.column_label, item.source_span)
            for item in selected
        )
        eligible.append(
            (
                metadata["layout_stratum"],
                metadata["sha256"],
                physical,
                example,
                metadata,
                eligibility,
                decision,
            )
        )
    eligible.sort(key=lambda item: (item[0], item[1], item[2]))
    target = int(precommit["deterministic_selection"]["target_companies_per_layout"])
    counts: Counter[str] = Counter()
    companies: set[str] = set()
    rows: list[Any] = []
    details: list[dict[str, Any]] = []
    for stratum, _, _, example, metadata, eligibility, decision in eligible:
        company = str(metadata["company_key"])
        if counts[stratum] >= target or company in companies:
            continue
        supervision = dict(example.supervision or {})
        supervision.update(
            {
                "company_key": company,
                "review_status": "independently_reviewed_approved",
                "cohort_layout_stratum": stratum,
                "cohort_layout_family": metadata["layout_family"],
                "source_sha256": metadata["sha256"],
                "classification": "development_only_independently_reviewed_add_cohort",
            }
        )
        reviewed = replace(example, supervision=supervision)
        rows.append(reviewed)
        companies.add(company)
        counts[stratum] += 1
        selected = _selected(reviewed)
        details.append(
            {
                "example_id": reviewed.example_id,
                "company_key": company,
                "source_key": metadata["source_key"],
                "source_name": metadata["source_name"],
                "source_path": metadata["source_path"],
                "source_sha256": metadata["sha256"],
                "layout_stratum": stratum,
                "layout_family": metadata["layout_family"],
                "page": selected[0].page,
                "question": reviewed.question,
                "answer_value": reviewed.answer_value,
                "operands": [
                    {
                        "row": item.row_label,
                        "column": item.column_label,
                        "period": item.period,
                        "value": str(item.value),
                        "source_span": list(item.source_span or ()),
                    }
                    for item in selected
                ],
                "review_status": decision["status"],
                "review_reason": decision["reason"],
                **eligibility,
            }
        )
    checks = {
        "target_companies_per_layout": all(counts[name] == target for name in strata),
        "one_positive_per_company": len(rows) == len(companies),
        "company_disjoint_between_layouts": len(companies)
        == sum(len({row["company_key"] for row in details if row["layout_stratum"] == name}) for name in strata),
        "all_selected_independently_reviewed": all(
            row["review_status"] == "independently_reviewed_approved" for row in details
        ),
        "all_selected_eligibility_checks_pass": all(
            all(row["checks"].values()) for row in details
        ),
        "counterfactual_false_accepts_zero": all(
            row["counterfactual_action"] != "ANSWER" for row in details
        ),
        "visual_rejection_not_selected": all(
            (row["source_name"], row["example_id"])
            not in {(item["source_name"], item["example_id"]) for item in details}
            for row in review_failures
        ),
    }
    report = {
        "schema": "finhrm-qp3-core-add-diverse-development-reviewed-cohort-v1",
        "created_at": "2026-08-20",
        "classification": "development_only_independently_reviewed_company_layout_disjoint_add_cohort",
        "reviewed_cohort_gate_passed": all(checks.values()),
        "failed_gates": [name for name, value in checks.items() if not value],
        "inputs": {
            "precommit": {"path": DEFAULT_PRECOMMIT.relative_to(ROOT).as_posix(), "sha256": _digest(DEFAULT_PRECOMMIT)},
            "source_audit": {"path": DEFAULT_SOURCE_AUDIT.relative_to(ROOT).as_posix(), "sha256": _digest(DEFAULT_SOURCE_AUDIT)},
            "native_pool": {"path": DEFAULT_NATIVE_POOL.relative_to(ROOT).as_posix(), "sha256": _digest(DEFAULT_NATIVE_POOL)},
            "ocr_pool": {"path": DEFAULT_OCR_POOL.relative_to(ROOT).as_posix(), "sha256": _digest(DEFAULT_OCR_POOL)},
            "visual_review": {"path": DEFAULT_REVIEW.relative_to(ROOT).as_posix(), "sha256": _digest(DEFAULT_REVIEW)},
        },
        "metrics": {
            "selected_examples": len(rows),
            "selected_companies": len(companies),
            "selected_by_layout": dict(sorted(counts.items())),
            "reviewed_candidates": len(review["records"]),
            "visual_rejections": len(review_failures),
        },
        "gate_checks": checks,
        "details": details,
        "visual_rejections": review_failures,
        "invariants": {
            "protected_evaluation_loaded": False,
            "abw_loaded": False,
            "synthetic_layout_counted": False,
            "candidate_rank_score_or_confidence_used_as_proof": False,
            "review_decision_used_as_runtime_rule": False,
            "model_checkpoint_or_threshold_changed": False,
        },
    }
    return rows, report


if __name__ == "__main__":
    if DEFAULT_OUTPUT.exists() or DEFAULT_REPORT.exists():
        raise FileExistsError("refusing to overwrite reviewed cohort artifacts")
    examples, report = build_reviewed_cohort()
    write_program_examples(examples, DEFAULT_OUTPUT)
    DEFAULT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gate": report["reviewed_cohort_gate_passed"], **report["metrics"]}, ensure_ascii=False, sort_keys=True))
    raise SystemExit(0 if report["reviewed_cohort_gate_passed"] else 2)
