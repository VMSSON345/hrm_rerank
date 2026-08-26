#!/usr/bin/env python3
"""Extend frozen development folds with reviewed new-company SUBTRACT groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_company_disjoint_folds import (  # noqa: E402
    _company_key,
    _group_metadata,
    _report_keys,
)


PROTECTED_MARKERS = ("blind", "holdout", "final_eval", "test.jsonl", "validation.jsonl")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def _reject_protected(path: Path) -> None:
    normalized = str(path).lower().replace("\\", "/")
    if any(marker in normalized for marker in PROTECTED_MARKERS):
        raise ValueError(f"Protected evaluation input is not allowed: {path}")


def _operation(row: dict[str, object]) -> str:
    program = row.get("program")
    return str(program.get("operation", "")) if isinstance(program, dict) else ""


def _reviewed_subtract(row: dict[str, object]) -> bool:
    supervision = row.get("supervision")
    if not isinstance(supervision, dict):
        return False
    contract = supervision.get("direction_contract")
    negatives = supervision.get("hard_negatives")
    return bool(
        _operation(row) == "SUBTRACT"
        and supervision.get("review_status") == "visually_reviewed_pass"
        and isinstance(contract, dict)
        and contract.get("intent") == "NEW_MINUS_OLD"
        and isinstance(negatives, list)
        and any(
            isinstance(item, dict)
            and item.get("type") == "reversed_operand_order"
            and item.get("proof") == "shared_direction_contract"
            for item in negatives
        )
    )


def assign_new_groups(
    group_ids: list[str], base_calibration_counts: list[int]
) -> tuple[dict[str, int], list[int]]:
    """Greedily equalize final SUBTRACT calibration-group counts."""

    totals = list(base_calibration_counts)
    assignments: dict[str, int] = {}
    for group_id in sorted(group_ids):
        fold = min(range(len(totals)), key=lambda index: (totals[index], index))
        assignments[group_id] = fold
        totals[fold] += 1
    return assignments, totals


def _baseline_counts(phase_c: dict[str, object]) -> tuple[list[int], list[int]]:
    folds = phase_c.get("folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise ValueError("Phase-C manifest must contain five folds")
    train: list[int] = []
    calibration: list[int] = []
    for fold in sorted(folds, key=lambda item: int(item["fold"])):
        eligibility = fold.get("operation_eligibility")
        subtract = eligibility.get("SUBTRACT") if isinstance(eligibility, dict) else None
        if not isinstance(subtract, dict):
            raise ValueError("Phase-C SUBTRACT eligibility is missing")
        train.append(int(subtract["train_groups"]))
        calibration.append(int(subtract["calibration_groups"]))
    return train, calibration


def audit_fold_disjointness(
    rows_by_id: dict[str, dict[str, object]],
    folds: list[dict[str, object]],
) -> list[dict[str, int]]:
    all_ids = set(rows_by_id)
    audits: list[dict[str, int]] = []
    for raw_fold in folds:
        fold = int(raw_fold["fold"])
        train_ids = set(raw_fold["train_example_ids"])
        validation_ids = set(raw_fold["validation_example_ids"])
        if train_ids & validation_ids or train_ids | validation_ids != all_ids:
            raise ValueError(f"Example split is incomplete/overlapping in fold {fold}")
        train_rows = [rows_by_id[example_id] for example_id in train_ids]
        validation_rows = [rows_by_id[example_id] for example_id in validation_ids]
        train_companies = {_company_key(row) for row in train_rows}
        validation_companies = {_company_key(row) for row in validation_rows}
        train_reports = {report for row in train_rows for report in _report_keys(row)}
        validation_reports = {
            report for row in validation_rows for report in _report_keys(row)
        }
        train_groups = {_group_metadata(row)["group_key"] for row in train_rows}
        validation_groups = {
            _group_metadata(row)["group_key"] for row in validation_rows
        }
        overlaps = (
            len(train_companies & validation_companies),
            len(train_reports & validation_reports),
            len(train_groups & validation_groups),
        )
        if any(overlaps):
            raise ValueError(f"Company/report/group overlap in fold {fold}: {overlaps}")
        if train_companies != set(raw_fold["train_companies"]):
            raise ValueError(f"Train company manifest mismatch in fold {fold}")
        if validation_companies != set(raw_fold["validation_companies"]):
            raise ValueError(f"Validation company manifest mismatch in fold {fold}")
        audits.append(
            {
                "fold": fold,
                "company_overlap": overlaps[0],
                "report_overlap": overlaps[1],
                "group_overlap": overlaps[2],
                "train_examples": len(train_ids),
                "validation_examples": len(validation_ids),
            }
        )
    return audits


def seal_development_manifest(
    manifest: dict[str, object], development_path: Path
) -> None:
    """Bind an extended fold manifest to its development-only source."""

    manifest.pop("sealed_validation", None)
    manifest.pop("sealed_validation_sha256", None)
    manifest["sealed_validation_loaded"] = False
    manifest["input"] = str(development_path.resolve())
    manifest["input_sha256"] = _sha256(development_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-fold-manifest", type=Path, required=True)
    parser.add_argument("--base-phase-c-manifest", type=Path, required=True)
    parser.add_argument("--development-jsonl", type=Path, required=True)
    parser.add_argument("--reviewed-cohort", type=Path, action="append", required=True)
    parser.add_argument("--output-development", type=Path, required=True)
    parser.add_argument("--output-fold-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    for path in (
        args.base_fold_manifest,
        args.base_phase_c_manifest,
        args.development_jsonl,
        *args.reviewed_cohort,
    ):
        _reject_protected(path)
    for path in (args.output_development, args.output_fold_manifest, args.report):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")

    base = _read_json(args.base_fold_manifest)
    phase_c = _read_json(args.base_phase_c_manifest)
    if phase_c.get("fold_manifest_sha256") != _sha256(args.base_fold_manifest):
        raise ValueError("Phase-C manifest does not match the frozen fold manifest")
    base_examples = base.get("examples")
    base_operations = base.get("example_operations")
    base_folds = base.get("folds")
    if not isinstance(base_examples, dict) or not isinstance(base_operations, dict):
        raise ValueError("Base fold example metadata is missing")
    if not isinstance(base_folds, list) or len(base_folds) != 5:
        raise ValueError("Base fold manifest must contain five folds")
    base_ids = set(base_examples)
    development_rows = _read_jsonl(args.development_jsonl)
    base_rows = [row for row in development_rows if str(row.get("example_id", "")) in base_ids]
    if {str(row.get("example_id", "")) for row in base_rows} != base_ids:
        raise ValueError("Development JSONL does not contain exactly the frozen base IDs")

    new_rows = [row for path in args.reviewed_cohort for row in _read_jsonl(path)]
    new_ids = [str(row.get("example_id", "")) for row in new_rows]
    if not new_ids or len(set(new_ids)) != len(new_ids) or set(new_ids) & base_ids:
        raise ValueError("New cohort IDs are empty, duplicated, or overlap the base")
    if not all(_reviewed_subtract(row) for row in new_rows):
        raise ValueError("Every new row must be visually reviewed NEW_MINUS_OLD SUBTRACT")

    base_companies = {_company_key(row) for row in base_rows}
    base_reports = {report for row in base_rows for report in _report_keys(row)}
    new_companies = {_company_key(row) for row in new_rows}
    new_reports = {report for row in new_rows for report in _report_keys(row)}
    if base_companies & new_companies or base_reports & new_reports:
        raise ValueError("New cohort overlaps a base company or report")

    new_metadata = {
        str(row["example_id"]): _group_metadata(row) for row in new_rows
    }
    new_group_ids = sorted({item["group_key"] for item in new_metadata.values()})
    if len(new_group_ids) != len(new_companies):
        raise ValueError("Expected one unique reviewed SUBTRACT group per new company")
    group_to_company = {
        metadata["group_key"]: metadata["company"]
        for metadata in new_metadata.values()
    }
    group_to_ids: dict[str, list[str]] = {}
    for example_id, metadata in new_metadata.items():
        group_to_ids.setdefault(metadata["group_key"], []).append(example_id)

    base_train, base_calibration = _baseline_counts(phase_c)
    assignments, final_calibration = assign_new_groups(
        new_group_ids, base_calibration
    )
    new_groups_per_fold = Counter(assignments.values())
    total_new_groups = len(new_group_ids)
    final_train = [
        base_train[index] + total_new_groups - new_groups_per_fold[index]
        for index in range(5)
    ]

    extended_folds = []
    all_new_ids = set(new_ids)
    for raw_fold in sorted(base_folds, key=lambda item: int(item["fold"])):
        fold = int(raw_fold["fold"])
        validation_groups = {
            group for group, assigned_fold in assignments.items() if assigned_fold == fold
        }
        validation_new_ids = sorted(
            example_id
            for group in validation_groups
            for example_id in group_to_ids[group]
        )
        train_new_ids = sorted(all_new_ids - set(validation_new_ids))
        validation_companies = sorted(
            set(raw_fold["validation_companies"])
            | {group_to_company[group] for group in validation_groups}
        )
        train_companies = sorted(
            set(raw_fold["train_companies"])
            | (new_companies - {group_to_company[group] for group in validation_groups})
        )
        if set(train_companies) & set(validation_companies):
            raise ValueError(f"Company overlap in extended fold {fold}")
        train_counts = Counter(raw_fold.get("train_operation_counts", {}))
        calibration_counts = Counter(raw_fold.get("validation_operation_counts", {}))
        train_counts["SUBTRACT"] += len(train_new_ids)
        calibration_counts["SUBTRACT"] += len(validation_new_ids)
        extended_folds.append(
            {
                **raw_fold,
                "train_example_ids": list(raw_fold["train_example_ids"]) + train_new_ids,
                "validation_example_ids": list(raw_fold["validation_example_ids"]) + validation_new_ids,
                "train_companies": train_companies,
                "validation_companies": validation_companies,
                "train_operation_counts": dict(train_counts),
                "validation_operation_counts": dict(calibration_counts),
            }
        )

    extended_examples = {**base_examples, **new_metadata}
    extended_operations = {**base_operations, **{example_id: "SUBTRACT" for example_id in new_ids}}
    output_manifest = {
        **base,
        "schema": "v12b2-reranker-disjoint-folds-v2-subtract-extension",
        "base_fold_manifest": str(args.base_fold_manifest.resolve()),
        "base_fold_manifest_sha256": _sha256(args.base_fold_manifest),
        "development_examples": len(base_ids) + len(new_ids),
        "company_count": int(base.get("company_count", 0)) + len(new_companies),
        "examples": extended_examples,
        "example_operations": extended_operations,
        "folds": extended_folds,
        "blind_test_used": False,
        "sealed_validation_loaded": False,
    }
    combined_rows = base_rows + sorted(new_rows, key=lambda row: str(row["example_id"]))
    combined_by_id = {str(row["example_id"]): row for row in combined_rows}
    fold_disjointness = audit_fold_disjointness(combined_by_id, extended_folds)
    args.output_development.parent.mkdir(parents=True, exist_ok=True)
    args.output_development.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in combined_rows),
        encoding="utf-8",
    )
    seal_development_manifest(output_manifest, args.output_development)
    args.output_fold_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_fold_manifest.write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = {
        "schema": "subtract-reviewed-cohort-fold-extension-audit-v1",
        "classification": "development_only_company_report_disjoint",
        "blind_test_used": False,
        "validation_loaded": False,
        "fine_tuning_performed": False,
        "threshold_changed": False,
        "inputs": {
            "base_fold_manifest_sha256": _sha256(args.base_fold_manifest),
            "base_phase_c_manifest_sha256": _sha256(args.base_phase_c_manifest),
            "reviewed_cohorts": [
                {"path": str(path.resolve()), "sha256": _sha256(path)}
                for path in args.reviewed_cohort
            ],
        },
        "metrics": {
            "base_development_examples": len(base_ids),
            "new_examples": len(new_ids),
            "new_company_groups": len(new_companies),
            "new_composite_groups": total_new_groups,
            "company_overlap": len(base_companies & new_companies),
            "report_overlap": len(base_reports & new_reports),
            "fold_disjointness": fold_disjointness,
            "base_subtract_train_groups": base_train,
            "base_subtract_calibration_groups": base_calibration,
            "new_calibration_groups_by_fold": [new_groups_per_fold[index] for index in range(5)],
            "final_subtract_train_groups": final_train,
            "final_subtract_calibration_groups": final_calibration,
        },
        "gates": {
            "company_report_disjoint": True,
            "subtract_groups_30_train_15_calibration_all_folds": all(
                train >= 30 and calibration >= 15
                for train, calibration in zip(final_train, final_calibration)
            ),
        },
        "outputs": {
            "development": str(args.output_development.resolve()),
            "development_sha256": _sha256(args.output_development),
            "fold_manifest": str(args.output_fold_manifest.resolve()),
            "fold_manifest_sha256": _sha256(args.output_fold_manifest),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({**report["metrics"], **report["gates"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
