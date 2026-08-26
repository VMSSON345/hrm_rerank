from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "config" / "finhrm_qp3_core_generalization_coverage_diagnostic_matrix_v1.json"
)
DEFAULT_REPORT = (
    ROOT / "reports" / "finhrm_qp3_core_generalization_coverage_diagnostic_matrix_v1.json"
)
DEFAULT_MARKDOWN = (
    ROOT / "reports" / "finhrm_qp3_core_generalization_coverage_diagnostic_matrix_v1.md"
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selected_rows(
    details: list[dict[str, Any]],
    operation: str,
    obligations: list[str],
) -> list[dict[str, Any]]:
    markers = {f"required_proof_unknown:{name}" for name in obligations}
    return [
        row
        for row in details
        if row["operation"] == operation
        and markers.intersection(row["gold_reasons"])
    ]


def _layout_cells(
    instances: list[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for instance in instances:
        grouped[instance[1]].append(instance)
    return [
        {
            "layout_family": layout,
            "instances": len(values),
            "distinct_companies": len({value[0] for value in values}),
            "source_kind": sorted({value[2] for value in values}),
        }
        for layout, values in sorted(grouped.items())
    ]


def build_matrix(config: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    parent = config["parent_precommit"]
    parent_path = ROOT / parent["path"]
    if _digest(parent_path) != parent["sha256"]:
        errors.append("parent_precommit_hash_mismatch")
    precommit = json.loads(parent_path.read_text(encoding="utf-8"))

    source = config["source_report"]
    source_path = ROOT / source["path"]
    if _digest(source_path) != source["sha256"]:
        errors.append("source_report_hash_mismatch")
    safe_report = json.loads(source_path.read_text(encoding="utf-8"))
    if source["allowed_split"] != "development_train_only":
        errors.append("non_development_source")
    if safe_report["source"]["protected_records"]:
        errors.append("protected_records_present")
    if safe_report["source"]["abw_records"]:
        errors.append("abw_records_present")
    for key in (
        "abw_used",
        "protected_evaluation_loaded",
        "validation_loaded",
        "blind_loaded",
        "test_data_loaded",
    ):
        if safe_report["invariants"][key]:
            errors.append(f"forbidden_source_invariant:{key}")
    for relative, expected in precommit["runtime_hashes_at_precommit"].items():
        if _digest(ROOT / relative) != expected:
            errors.append(f"runtime_hash_changed:{relative}")

    minimum_companies = int(
        config["diversity_gate"]["minimum_distinct_companies_per_track_and_control"]
    )
    minimum_layouts = int(
        config["diversity_gate"]["minimum_distinct_layouts_per_track_and_control"]
    )
    details = safe_report["details"]
    track_reports: dict[str, Any] = {}
    total_controls = 0
    total_cells = 0

    for track_name, track in config["tracks"].items():
        rows = _selected_rows(
            details,
            track["operation"],
            track["select_when_any_unknown"],
        )
        instances = [
            (row["company_key"], row["layout_family"], "source")
            for row in rows
        ]
        source_layouts = {instance[1] for instance in instances}
        projection = track["layout_projection"]
        projected = 0
        if len(source_layouts) < minimum_layouts and projection.get("enabled"):
            companies = sorted({instance[0] for instance in instances})
            requested = int(projection["source_companies"])
            for company in companies[:requested]:
                instances.append(
                    (company, projection["target_layout"], "synthetic_layout_projection")
                )
                projected += 1

        companies = {instance[0] for instance in instances}
        layouts = {instance[1] for instance in instances}
        layout_cells = _layout_cells(instances)
        controls: dict[str, Any] = {}
        for control_type in config["control_types"]:
            recipe_key = {
                "positive": "positive_recipe",
                "hard_negative": "hard_negative_recipe",
                "counterfactual": "counterfactual_recipe",
                "permutation": "permutation_recipe",
            }[control_type]
            oracle_key = {
                "positive": "positive_oracle",
                "hard_negative": "hard_negative_oracle",
                "counterfactual": "counterfactual_oracle",
                "permutation": "permutation_oracle",
            }[control_type]
            controls[control_type] = {
                "recipe": track[recipe_key],
                "oracle": track[oracle_key],
                "instances": len(instances),
                "distinct_companies": len(companies),
                "distinct_layouts": len(layouts),
                "matrix_cells": layout_cells,
            }
            total_controls += len(instances)
            total_cells += len(layout_cells)
            if len(companies) < minimum_companies:
                errors.append(f"insufficient_company_diversity:{track_name}:{control_type}")
            if len(layouts) < minimum_layouts:
                errors.append(f"insufficient_layout_diversity:{track_name}:{control_type}")

        track_reports[track_name] = {
            "operation": track["operation"],
            "selected_unknown_obligations": track["select_when_any_unknown"],
            "source_examples": len(rows),
            "source_companies": len({row["company_key"] for row in rows}),
            "source_layouts": len(source_layouts),
            "synthetic_layout_projection_instances": projected,
            "matrix_instances": len(instances),
            "matrix_companies": len(companies),
            "matrix_layouts": len(layouts),
            "controls": controls,
        }

    return {
        "schema": "finhrm-qp3-core-generalization-coverage-diagnostic-matrix-report-v1",
        "created_at": "2026-08-14",
        "classification": "development_only_aggregate_diagnostic_controls_not_confirmation",
        "matrix_gate_passed": not errors,
        "errors": errors,
        "source": {
            "path": source["path"],
            "sha256": source["sha256"],
            "split": source["allowed_split"],
            "company_identifiers_persisted": False,
            "example_identifiers_persisted": False,
        },
        "diversity_gate": config["diversity_gate"],
        "tracks": track_reports,
        "totals": {
            "track_control_instances": total_controls,
            "aggregate_matrix_cells": total_cells,
        },
        "invariants": {
            "runtime_changed": False,
            "abw_used_as_confirmation": False,
            "protected_evaluation_loaded": False,
            "threshold_changed": False,
            "checkpoint_changed": False,
            "fine_tuning_performed": False,
            "synthetic_layout_counted_as_unseen_confirmation": False,
            "qp4_authorized": False,
        },
        "interpretation": (
            "This matrix freezes aggregate development-only control recipes. "
            "Synthetic layout projections close diagnostic diversity only and are not unseen evidence."
        ),
        "authorized_next_step": config["authorized_next_step_if_matrix_passes"],
    }


def _markdown(report: dict[str, Any]) -> str:
    rows = [
        "# QP3-CORE-GENERALIZATION-COVERAGE-DIAGNOSTIC-MATRIX",
        "",
        f"Gate: **{'PASS' if report['matrix_gate_passed'] else 'FAIL'}**.",
        "",
        "| Track | Source examples | Companies | Source layouts | Projected | Matrix layouts | Controls |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, track in report["tracks"].items():
        control_count = sum(item["instances"] for item in track["controls"].values())
        rows.append(
            f"| {name} | {track['source_examples']} | {track['source_companies']} | "
            f"{track['source_layouts']} | {track['synthetic_layout_projection_instances']} | "
            f"{track['matrix_layouts']} | {control_count} |"
        )
    rows.extend(
        [
            "",
            "Mỗi track có positive, hard-negative, explicit counterfactual và candidate-permutation controls.",
            "Layout projection của period chỉ là diagnostic synthetic control, không phải unseen confirmation.",
            "Runtime, threshold, checkpoint và QP4 vẫn khóa.",
            "",
        ]
    )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = build_matrix(config)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.markdown.write_text(_markdown(report), encoding="utf-8")
    if not report["matrix_gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
