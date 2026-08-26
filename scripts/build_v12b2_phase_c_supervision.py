from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrm_finance_qa.program_data import load_program_examples  # noqa: E402
from hrm_finance_qa.v12b2_supervision import (  # noqa: E402
    PHASE_C_SCHEMA,
    build_phase_c_record,
)


TRAIN_GROUP_MINIMUM = 30
CALIBRATION_GROUP_MINIMUM = 15


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def _source_fold_manifest(root: Path, fold: int) -> Path:
    return root / f"fold_{fold}" / "manifest.json"


def _merge_counter(target: Counter[str], raw: object) -> None:
    if isinstance(raw, dict):
        target.update({str(key): int(value) for key, value in raw.items()})


def _build_split(
    source: Path,
    output: Path,
    *,
    examples: dict[str, object],
    group_keys: dict[str, str],
) -> dict[str, object]:
    records = candidates = positive_programs = 0
    multi_positive_records = eligible_records = 0
    multi_positive_example_ids: set[str] = set()
    label_contradictions = execution_only_promoted = 0
    reversed_pair_negatives_withheld = 0
    supervision_integrity_excluded = 0
    supervision_integrity_issues: Counter[str] = Counter()
    positive_classes: Counter[str] = Counter()
    hard_negative_types: Counter[str] = Counter()
    operation_records: Counter[str] = Counter()
    eligible_groups: dict[str, set[str]] = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8") as source_handle, output.open(
        "w", encoding="utf-8"
    ) as output_handle:
        for line_number, line in enumerate(source_handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{source}:{line_number}: expected object")
            example_id = str(record.get("example_id", ""))
            example = examples.get(example_id)
            if example is None:
                raise ValueError(f"No source example for {example_id}")
            supervision, audit = build_phase_c_record(record, example)
            output_handle.write(
                json.dumps(
                    supervision,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            operation = str(supervision["operation"])
            operation_records[operation] += 1
            records += 1
            candidates += int(audit["candidates"])
            positive_programs += int(audit["positive_programs"])
            multi_positive_records += int(
                len(supervision["positive_program_keys"]) > 1
            )
            if len(supervision["positive_program_keys"]) > 1:
                multi_positive_example_ids.add(example_id)
            eligible = bool(audit["eligible_positive_and_hard_negative"])
            eligible_records += int(eligible)
            if eligible:
                eligible_groups.setdefault(operation, set()).add(
                    group_keys[example_id]
                )
            label_contradictions += int(audit["label_contradictions"])
            execution_only_promoted += int(
                audit["execution_only_promoted_to_positive"]
            )
            reversed_pair_negatives_withheld += int(
                audit["reversed_pair_negatives_withheld"]
            )
            integrity_eligible = bool(audit["supervision_integrity_eligible"])
            supervision_integrity_excluded += int(not integrity_eligible)
            supervision_integrity_issues.update(
                str(value)
                for value in audit["supervision_integrity_issues"]
            )
            _merge_counter(positive_classes, audit["positive_classes"])
            _merge_counter(hard_negative_types, audit["hard_negative_types"])
    return {
        "records": records,
        "candidates": candidates,
        "positive_programs": positive_programs,
        "multi_positive_records": multi_positive_records,
        "multi_positive_example_ids": sorted(multi_positive_example_ids),
        "eligible_positive_and_hard_negative_records": eligible_records,
        "positive_classes": dict(positive_classes),
        "hard_negative_types": dict(hard_negative_types),
        "operation_records": dict(operation_records),
        "eligible_group_counts": {
            operation: len(groups)
            for operation, groups in sorted(eligible_groups.items())
        },
        "label_contradictions": label_contradictions,
        "execution_only_promoted_to_positive": execution_only_promoted,
        "reversed_pair_negatives_withheld": reversed_pair_negatives_withheld,
        "supervision_integrity_excluded_records": supervision_integrity_excluded,
        "supervision_integrity_issues": dict(supervision_integrity_issues),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build V12-B.2 Phase-C canonical multi-positive supervision"
    )
    parser.add_argument(
        "--fold-manifest",
        type=Path,
        default=ROOT / "data" / "document_hrm_v12b2_reranker_disjoint_folds_v1.json",
    )
    parser.add_argument(
        "--phase-b-fold-root",
        type=Path,
        default=ROOT / "cache" / "v12b2_folds_candidate_recall",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "cache" / "v12b2_phase_c_v3"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports" / "document_hrm_V12B2_phase_c_v3_audit.json",
    )
    args = parser.parse_args()

    fold_manifest = _read_json(args.fold_manifest)
    if int(fold_manifest.get("fold_count", 0)) != 5:
        raise ValueError("Phase C requires exactly five folds")
    if fold_manifest.get("blind_test_used") is not False:
        raise ValueError("Phase C refuses a fold manifest that touched blind test")
    metadata = fold_manifest.get("examples", {})
    if not isinstance(metadata, dict):
        raise ValueError("Fold metadata is missing")
    group_keys = {
        str(example_id): str(value.get("group_key", ""))
        for example_id, value in metadata.items()
        if isinstance(value, dict)
    }
    source_examples = load_program_examples(Path(str(fold_manifest["input"])))
    allowed_example_ids = set(group_keys)
    examples = {
        example.example_id: example
        for example in source_examples
        if example.example_id in allowed_example_ids
    }
    if set(examples) != allowed_example_ids:
        missing = sorted(allowed_example_ids - set(examples))
        raise ValueError(
            "Fold development examples are missing from the declared input: "
            + ", ".join(missing[:10])
        )
    protected_or_untracked_records_excluded = len(source_examples) - len(examples)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_reports: list[dict[str, object]] = []
    all_operations: set[str] = set()
    total_label_contradictions = total_execution_only_promoted = 0
    total_reversed_pair_negatives_withheld = 0
    total_multi_positive_records = 0
    total_supervision_integrity_excluded = 0
    unique_multi_positive_examples: set[str] = set()
    aggregate_positive_classes: Counter[str] = Counter()
    aggregate_hard_negative_types: Counter[str] = Counter()
    aggregate_supervision_integrity_issues: Counter[str] = Counter()
    for fold in range(5):
        source_manifest_path = _source_fold_manifest(
            args.phase_b_fold_root, fold
        )
        source_manifest = _read_json(source_manifest_path)
        if source_manifest.get("fold_manifest_sha256") != _digest(
            args.fold_manifest
        ):
            raise ValueError(
                f"Phase-B source fold {fold} does not match the requested fold manifest"
            )
        if any(
            int(source_manifest.get(key, -1)) != 0
            for key in ("company_overlap", "report_overlap", "group_overlap")
        ):
            raise ValueError(f"Phase-B source fold {fold} has group leakage")
        fold_dir = args.output_dir / f"fold_{fold}"
        train_source = args.phase_b_fold_root / f"fold_{fold}" / "train_candidates.jsonl"
        calibration_source = (
            args.phase_b_fold_root
            / f"fold_{fold}"
            / "validation_candidates.jsonl"
        )
        if _digest(train_source) != source_manifest["train_cache_sha256"]:
            raise ValueError(f"Fold {fold} train cache SHA mismatch")
        if _digest(calibration_source) != source_manifest["validation_cache_sha256"]:
            raise ValueError(f"Fold {fold} calibration cache SHA mismatch")
        train_output = fold_dir / "train_supervision.jsonl"
        calibration_output = fold_dir / "calibration_supervision.jsonl"
        train_audit = _build_split(
            train_source,
            train_output,
            examples=examples,
            group_keys=group_keys,
        )
        calibration_audit = _build_split(
            calibration_source,
            calibration_output,
            examples=examples,
            group_keys=group_keys,
        )
        operations = set(train_audit["operation_records"]) | set(
            calibration_audit["operation_records"]
        )
        all_operations.update(operations)
        eligibility = {
            operation: {
                "train_groups": int(
                    train_audit["eligible_group_counts"].get(operation, 0)
                ),
                "calibration_groups": int(
                    calibration_audit["eligible_group_counts"].get(operation, 0)
                ),
                "enabled": (
                    int(train_audit["eligible_group_counts"].get(operation, 0))
                    >= TRAIN_GROUP_MINIMUM
                    and int(
                        calibration_audit["eligible_group_counts"].get(
                            operation, 0
                        )
                    )
                    >= CALIBRATION_GROUP_MINIMUM
                ),
            }
            for operation in sorted(operations)
        }
        fold_payload = {
            "schema": "v12b2-phase-c-fold-supervision-v1",
            "phase_c_schema": PHASE_C_SCHEMA,
            "fold": fold,
            "blind_test_used": False,
            "test_jsonl_used": False,
            "company_overlap": 0,
            "report_overlap": 0,
            "group_overlap": 0,
            "train_group_minimum": TRAIN_GROUP_MINIMUM,
            "calibration_group_minimum": CALIBRATION_GROUP_MINIMUM,
            "source_phase_b_manifest": str(source_manifest_path.resolve()),
            "source_phase_b_manifest_sha256": _digest(source_manifest_path),
            "train_supervision": str(train_output.resolve()),
            "train_supervision_sha256": _digest(train_output),
            "calibration_supervision": str(calibration_output.resolve()),
            "calibration_supervision_sha256": _digest(calibration_output),
            "train_audit": train_audit,
            "calibration_audit": calibration_audit,
            "operation_eligibility": eligibility,
        }
        fold_manifest_path = fold_dir / "manifest.json"
        fold_manifest_path.write_text(
            json.dumps(fold_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        fold_reports.append(
            {
                "fold": fold,
                "manifest": str(fold_manifest_path.resolve()),
                "manifest_sha256": _digest(fold_manifest_path),
                "operation_eligibility": eligibility,
            }
        )
        for audit in (train_audit, calibration_audit):
            total_label_contradictions += int(audit["label_contradictions"])
            total_execution_only_promoted += int(
                audit["execution_only_promoted_to_positive"]
            )
            total_reversed_pair_negatives_withheld += int(
                audit["reversed_pair_negatives_withheld"]
            )
            total_multi_positive_records += int(audit["multi_positive_records"])
            total_supervision_integrity_excluded += int(
                audit["supervision_integrity_excluded_records"]
            )
            unique_multi_positive_examples.update(
                str(value) for value in audit["multi_positive_example_ids"]
            )
            _merge_counter(aggregate_positive_classes, audit["positive_classes"])
            _merge_counter(
                aggregate_hard_negative_types, audit["hard_negative_types"]
            )
            _merge_counter(
                aggregate_supervision_integrity_issues,
                audit["supervision_integrity_issues"],
            )

    enabled_operations = sorted(
        operation
        for operation in all_operations
        if all(
            bool(fold["operation_eligibility"][operation]["enabled"])
            for fold in fold_reports
        )
    )
    fallback_operations = sorted(all_operations - set(enabled_operations))
    phase_c_gate = (
        bool(enabled_operations)
        and total_label_contradictions == 0
        and total_execution_only_promoted == 0
    )
    index = {
        "schema": "v12b2-phase-c-supervision-index-v2",
        "phase_c_schema": PHASE_C_SCHEMA,
        "status": (
            "phase_c_supervision_ready_limited_coverage"
            if phase_c_gate and fallback_operations
            else "phase_c_supervision_ready"
            if phase_c_gate
            else "phase_c_gate_failed"
        ),
        "fold_count": 5,
        "fold_manifest": str(args.fold_manifest.resolve()),
        "fold_manifest_sha256": _digest(args.fold_manifest),
        "phase_b_fold_root": str(args.phase_b_fold_root.resolve()),
        "blind_test_used": False,
        "test_jsonl_used": False,
        "sealed_validation_evaluated": False,
        "sealed_validation_loaded": False,
        "development_manifest_examples": len(examples),
        "protected_or_untracked_input_records_excluded": (
            protected_or_untracked_records_excluded
        ),
        "reranker_training_started": False,
        "train_group_minimum": TRAIN_GROUP_MINIMUM,
        "calibration_group_minimum": CALIBRATION_GROUP_MINIMUM,
        "positive_policy": [
            "canonical_gold",
            "commutative_equivalent",
            "grounded_semantic_equivalent",
        ],
        "execution_only_is_positive": False,
        "label_contradictions": total_label_contradictions,
        "execution_only_promoted_to_positive": total_execution_only_promoted,
        "reversed_pair_negatives_withheld": (
            total_reversed_pair_negatives_withheld
        ),
        "supervision_integrity_excluded_records": (
            total_supervision_integrity_excluded
        ),
        "supervision_integrity_issues": dict(
            aggregate_supervision_integrity_issues
        ),
        "multi_positive_records_fold_expanded": total_multi_positive_records,
        "multi_positive_examples_unique": len(unique_multi_positive_examples),
        "multi_positive_example_ids": sorted(unique_multi_positive_examples),
        "positive_classes": dict(aggregate_positive_classes),
        "hard_negative_types": dict(aggregate_hard_negative_types),
        "enabled_operations_all_folds": enabled_operations,
        "fallback_operations": fallback_operations,
        "phase_c_gate": phase_c_gate,
        "folds": fold_reports,
    }
    index_path = args.output_dir / "manifest.json"
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = {
        **index,
        "phase_c_manifest": str(index_path.resolve()),
        "phase_c_manifest_sha256": _digest(index_path),
        "known_limitation": (
            "Operations below 30 train groups or 15 calibration groups in any "
            "fold remain on frozen V12.4 fallback; oversampling is forbidden."
        ),
        "next_allowed_step": (
            "phase_d_enabled_operations_only" if phase_c_gate else "collect_data"
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": index["status"],
                "enabled_operations_all_folds": enabled_operations,
                "fallback_operations": fallback_operations,
                "label_contradictions": total_label_contradictions,
                "execution_only_promoted_to_positive": total_execution_only_promoted,
                "reversed_pair_negatives_withheld": (
                    total_reversed_pair_negatives_withheld
                ),
                "multi_positive_records_fold_expanded": total_multi_positive_records,
                "multi_positive_examples_unique": len(
                    unique_multi_positive_examples
                ),
                "supervision_integrity_excluded_records": (
                    total_supervision_integrity_excluded
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Saved Phase-C manifest: {index_path.resolve()}")
    print(f"Saved Phase-C report: {args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
