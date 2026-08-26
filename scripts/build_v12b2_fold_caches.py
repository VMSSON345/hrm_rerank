from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrm_finance_qa.question_planner import compile_question_plan  # noqa: E402
from hrm_finance_qa.evidence_schema import build_evidence_graph  # noqa: E402
from hrm_finance_qa.ontology import normalize_text  # noqa: E402
from hrm_finance_qa.program_data import (  # noqa: E402
    COMMUTATIVE,
    ProgramExample,
    load_program_examples,
)


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


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            output.append(value)
    return output


def _declared_source_paths(fold_manifest: dict[str, object]) -> tuple[Path, ...]:
    """Return only sources the manifest explicitly permits this stage to load."""

    input_path = Path(str(fold_manifest["input"]))
    if fold_manifest.get("sealed_validation_loaded") is False:
        return (input_path,)
    sealed = fold_manifest.get("sealed_validation")
    if not sealed:
        raise ValueError("Legacy fold manifest is missing sealed_validation")
    return (input_path, Path(str(sealed)))


def _counter_total(
    audits: Iterable[dict[str, object]], key: str, name: str
) -> int:
    total = 0
    for audit in audits:
        raw = audit.get(key, {})
        if isinstance(raw, dict):
            total += int(raw.get(name, 0))
    return total


def _failure_ids(
    audits: Iterable[dict[str, object]], prefixes: tuple[str, ...]
) -> set[str]:
    output: set[str] = set()
    for audit in audits:
        failures = audit.get("failure_examples", {})
        if not isinstance(failures, dict):
            continue
        for mode, rows in failures.items():
            if not str(mode).startswith(prefixes) or not isinstance(rows, list):
                continue
            for row in rows:
                if isinstance(row, dict) and row.get("example_id"):
                    output.add(str(row["example_id"]))
    return output


def _audit_filename(raw_path: object) -> str:
    return str(raw_path).replace("\\", "/").rsplit("/", 1)[-1]


def _invalid_gold(
    source_manifest: dict[str, object], source_dir: Path
) -> dict[str, set[str]]:
    output: dict[str, set[str]] = {}

    def merge_failure_obligations(audit: dict[str, object]) -> None:
        failure_examples = audit.get("failure_examples", {})
        if not isinstance(failure_examples, dict):
            return
        rows = failure_examples.get("gold_program_generated_but_ineligible", [])
        if not isinstance(rows, list):
            return
        for row in rows:
            if not isinstance(row, dict):
                continue
            example_id = str(row.get("example_id", ""))
            obligations = row.get("proof_obligations", [])
            if not example_id or not isinstance(obligations, list):
                continue
            for obligation in obligations:
                parts = str(obligation).split(":", 2)
                if (
                    len(parts) == 3
                    and parts[0] == "required"
                    and parts[2] == "INVALID"
                ):
                    output.setdefault(example_id, set()).add(parts[1])

    for key in ("train_shards", "calibration_shards"):
        shards = source_manifest.get(key, [])
        if not isinstance(shards, list):
            continue
        for shard in shards:
            if not isinstance(shard, dict):
                continue
            audit_path = source_dir / "shards" / _audit_filename(
                shard.get("audit_path", "")
            )
            wrapper = _read_json(audit_path)
            audit = wrapper.get("audit", {})
            if not isinstance(audit, dict):
                continue
            merge_failure_obligations(audit)
            contradictions = audit.get("proof_contradiction_examples", {})
            if not isinstance(contradictions, dict):
                continue
            for constraint, rows in contradictions.items():
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    if not isinstance(row, dict) or not row.get("canonical_gold"):
                        continue
                    output.setdefault(str(row.get("example_id", "")), set()).add(
                        str(constraint)
                    )
    # Keep aggregate manifests as a compatibility fallback. Per-shard failure
    # rows above are the complete source when aggregate diagnostics are capped.
    for key in ("train_audit", "calibration_audit"):
        audit = source_manifest.get(key, {})
        if not isinstance(audit, dict):
            continue
        merge_failure_obligations(audit)
    return output


def _load_record_lines(
    paths: Iterable[Path], expected_hashes: Iterable[str]
) -> tuple[dict[str, str], dict[str, dict[str, object]]]:
    lines: dict[str, str] = {}
    records: dict[str, dict[str, object]] = {}
    for path, expected_hash in zip(paths, expected_hashes):
        if _digest(path) != expected_hash:
            raise ValueError(f"Candidate cache SHA-256 mismatch: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                example_id = str(record.get("example_id", ""))
                if not example_id or example_id in records:
                    raise ValueError(
                        f"Missing or duplicate example_id at {path}:{line_number}"
                    )
                records[example_id] = record
                lines[example_id] = line if line.endswith("\n") else line + "\n"
    return lines, records


def _semantic_program_key(
    candidate: dict[str, object],
    example: ProgramExample,
    graph: object | None = None,
    plan: object | None = None,
) -> tuple[object, ...]:
    raw_key = str(candidate.get("program_key", ""))
    operation, separator, raw_operands = raw_key.partition(":")
    if not separator:
        return ("unparsed", raw_key)
    try:
        operands = tuple(
            int(value) for value in raw_operands.split(",") if value != ""
        )
    except ValueError:
        return ("unparsed", raw_key)
    graph = graph or build_evidence_graph(example.candidates)
    plan = plan or compile_question_plan(example.question)
    signatures: list[tuple[object, ...]] = []
    for index in operands:
        if not 0 <= index < len(graph.cells):
            return ("unparsed", raw_key)
        cell = graph.cells[index]
        entity = re.sub(
            r"[^a-z0-9]+",
            "",
            normalize_text(cell.entity_id or cell.entity),
        )
        signatures.append(
            (
                cell.metric_key,
                tuple(sorted(cell.effective_years(plan))),
                cell.period_role,
                entity,
                cell.unit_key,
                str(cell.value),
            )
        )
    if operation in COMMUTATIVE:
        signatures.sort()
    return operation, tuple(signatures)


def _deduplicate_record(
    record: dict[str, object], example: ProgramExample
) -> tuple[dict[str, object], int]:
    candidates = record.get("candidates", [])
    if not isinstance(candidates, list):
        return record, 0
    graph = build_evidence_graph(example.candidates)
    plan = compile_question_plan(example.question)
    selected: dict[
        tuple[object, ...],
        tuple[int, dict[str, object], list[dict[str, object]]],
    ] = {}

    def preference(item: dict[str, object], position: int) -> tuple[int, ...]:
        return (
            int(bool(item.get("exact_gold"))),
            int(item.get("positive_class") == "grounded_equivalent"),
            int(bool(item.get("runtime_eligible"))),
            int(item.get("training_state") == "VALID"),
            -position,
        )

    for position, item in enumerate(candidates):
        if not isinstance(item, dict):
            continue
        key = _semantic_program_key(item, example, graph, plan)
        current = selected.get(key)
        member = {
            "program_key": str(item.get("program_key", "")),
            "positive_class": str(item.get("positive_class", "negative")),
            "exact_gold": bool(item.get("exact_gold")),
            "execution_correct": bool(item.get("execution_correct")),
            "training_state": str(item.get("training_state", "UNKNOWN")),
        }
        if current is None:
            selected[key] = (position, item, [member])
            continue
        members = [*current[2], member]
        if preference(item, position) > preference(current[1], current[0]):
            selected[key] = (position, item, members)
        else:
            selected[key] = (current[0], current[1], members)
    deduplicated: list[dict[str, object]] = []
    for _, item, members in sorted(selected.values(), key=lambda value: value[0]):
        enriched = dict(item)
        enriched["semantic_equivalence_size"] = len(members)
        enriched["semantic_equivalent_members"] = sorted(
            members,
            key=lambda member: (
                str(member["program_key"]),
                str(member["positive_class"]),
            ),
        )
        deduplicated.append(enriched)
    output = dict(record)
    output["candidates"] = deduplicated
    return output, len(candidates) - len(deduplicated)


def _operation(row: dict[str, object]) -> str:
    program = row.get("program")
    return str(program.get("operation", "")) if isinstance(program, dict) else ""


def _operand_available(row: dict[str, object]) -> bool:
    program = row.get("program")
    candidates = row.get("candidates")
    if not isinstance(program, dict) or not isinstance(candidates, list):
        return False
    operands = program.get("operands", [])
    return isinstance(operands, list) and all(
        isinstance(index, int) and 0 <= index < len(candidates)
        for index in operands
    )


def _training_operations(audits: Iterable[dict[str, object]]) -> set[str]:
    totals: dict[str, Counter[str]] = {}
    for audit in audits:
        per_operation = audit.get("per_operation", {})
        if not isinstance(per_operation, dict):
            continue
        for operation, values in per_operation.items():
            if not isinstance(values, dict):
                continue
            totals.setdefault(str(operation), Counter()).update(
                {str(key): int(value) for key, value in values.items()}
            )
    return {
        operation
        for operation, values in totals.items()
        if values["examples"] > 0 and values["excluded_from_reranker"] == 0
    }


def _metrics(
    ids: set[str],
    *,
    examples: dict[str, dict[str, object]],
    records: dict[str, dict[str, object]],
    training_operations: set[str],
    missing_gold: set[str],
    no_trace: set[str],
    invalid_gold: dict[str, set[str]],
    group_examples: dict[str, object],
    semantic_duplicates_dropped: dict[str, int],
) -> dict[str, object]:
    scoped = {
        example_id
        for example_id in ids
        if _operation(examples[example_id]) in training_operations
    }
    generated = scoped - missing_gold
    invalid = scoped & set(invalid_gold)
    retained = generated - invalid
    rerankable = scoped & set(records)
    single_candidate = scoped - rerankable - no_trace
    exact_at_64 = {
        example_id
        for example_id in rerankable
        if any(
            bool(candidate.get("exact_gold"))
            for candidate in records[example_id].get("candidates", [])
            if isinstance(candidate, dict)
        )
    }
    operand_available = sum(_operand_available(examples[item]) for item in scoped)
    false_rejections: dict[str, Counter[str]] = {
        "constraint": Counter(),
        "operation": Counter(),
        "layout_family": Counter(),
        "entity_mode": Counter(),
    }
    for example_id in invalid:
        row = examples[example_id]
        metadata = group_examples.get(example_id, {})
        if not isinstance(metadata, dict):
            metadata = {}
        for constraint in invalid_gold[example_id]:
            false_rejections["constraint"][constraint] += 1
        false_rejections["operation"][_operation(row)] += 1
        false_rejections["layout_family"][
            str(metadata.get("layout_family", "unknown"))
        ] += 1
        false_rejections["entity_mode"][
            compile_question_plan(str(row.get("question", ""))).entity_mode
        ] += 1
    operation_rows: dict[str, dict[str, object]] = {}
    for operation in sorted(training_operations):
        operation_ids = {
            item for item in scoped if _operation(examples[item]) == operation
        }
        operation_generated = operation_ids - missing_gold
        operation_retained = operation_generated - invalid
        operation_rows[operation] = {
            "examples": len(operation_ids),
            "raw_gold_program_generated": len(operation_generated),
            "raw_gold_program_recall": (
                len(operation_generated) / len(operation_ids)
                if operation_ids
                else 0.0
            ),
            "verifier_gold_retained": len(operation_retained),
            "post_verifier_gold_recall": (
                len(operation_retained) / len(operation_ids)
                if operation_ids
                else 0.0
            ),
        }
    scoped_candidates = [
        candidate
        for example_id in rerankable
        for candidate in records[example_id].get("candidates", [])
        if isinstance(candidate, dict)
    ]
    return {
        "examples_all_operations": len(ids),
        "examples_reranker_operations": len(scoped),
        "gold_operands_available": operand_available,
        "raw_gold_operand_recall": operand_available / len(scoped) if scoped else 0.0,
        "raw_gold_program_generated": len(generated),
        "raw_gold_program_recall": len(generated) / len(scoped) if scoped else 0.0,
        "verifier_gold_retained": len(retained),
        "verifier_gold_retention": len(retained) / len(generated) if generated else 0.0,
        "post_verifier_gold_recall": len(retained) / len(scoped) if scoped else 0.0,
        "post_prune_gold_programs_at_64": len(exact_at_64),
        "post_prune_gold_program_recall_at_64": (
            len(exact_at_64) / len(scoped) if scoped else 0.0
        ),
        "no_trace": len(scoped & no_trace),
        "single_candidate": len(single_candidate),
        "rerankable": len(rerankable),
        "semantic_duplicate_programs_dropped": sum(
            semantic_duplicates_dropped.get(item, 0) for item in scoped
        ),
        "rank_positive_candidates": sum(
            candidate.get("positive_class")
            in {"canonical_exact", "grounded_equivalent"}
            for candidate in scoped_candidates
        ),
        "execution_only_candidates": sum(
            candidate.get("positive_class") == "execution_only"
            for candidate in scoped_candidates
        ),
        "execution_only_promoted_to_positive": 0,
        "positive_taxonomy": "canonical-or-semantic-grounded-v1",
        "false_rejection": {
            key: dict(value) for key, value in false_rejections.items()
        },
        "per_operation": operation_rows,
    }


def _write_records(path: Path, ids: set[str], lines: dict[str, str]) -> int:
    selected = sorted(ids & set(lines))
    path.write_text("".join(lines[item] for item in selected), encoding="utf-8")
    return len(selected)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Partition the verified V12-B.2 Top-64 cache into five disjoint folds"
    )
    parser.add_argument(
        "--fold-manifest",
        type=Path,
        default=ROOT / "data" / "document_hrm_v12b2_reranker_disjoint_folds_v1.json",
    )
    parser.add_argument(
        "--source-cache",
        type=Path,
        default=ROOT / "cache" / "v12b2_phase_a" / "manifest.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "cache" / "v12b2_folds"
    )
    args = parser.parse_args()

    fold_manifest = _read_json(args.fold_manifest)
    source_manifest = _read_json(args.source_cache)
    if int(fold_manifest.get("fold_count", 0)) != 5:
        raise ValueError("V12-B.2 requires exactly five folds")
    if fold_manifest.get("blind_test_used") is not False:
        raise ValueError("Fold manifest must prove blind_test_used=false")
    if int(source_manifest.get("top_k", 0)) != 64:
        raise ValueError("Source cache must be Top-64")

    source_dir = args.source_cache.parent
    candidate_paths = (
        source_dir / "train_candidates.jsonl",
        source_dir / "calibration_candidates.jsonl",
    )
    candidate_hashes = (
        str(source_manifest.get("train_cache_sha256", "")),
        str(source_manifest.get("calibration_cache_sha256", "")),
    )
    record_lines, records = _load_record_lines(candidate_paths, candidate_hashes)
    audits = [
        value
        for value in (
            source_manifest.get("train_audit"),
            source_manifest.get("calibration_audit"),
        )
        if isinstance(value, dict)
    ]
    training_operations = _training_operations(audits)
    missing_gold = _failure_ids(
        audits, ("gold_program_not_generated", "no_trace_gold_operands")
    )
    no_trace = _failure_ids(
        audits, ("no_trace_", "trace_without_eligible_program")
    )
    invalid_gold = _invalid_gold(source_manifest, source_dir)

    expected_missing = sum(
        int(audit.get("examples", 0))
        - sum(
            int(values.get("examples", 0))
            for operation, values in audit.get("per_operation", {}).items()
            if operation not in training_operations and isinstance(values, dict)
        )
        - int(audit.get("gold_verifier_generated", 0))
        for audit in audits
    )
    if len(missing_gold) != expected_missing:
        raise ValueError(
            "Source audit does not retain every missing-gold example ID; "
            "regenerate with uncapped per-example diagnostics"
        )
    expected_invalid = sum(
        int(audit.get("gold_verifier_generated", 0))
        - int(audit.get("gold_verifier_retained", 0))
        for audit in audits
    )
    if len(invalid_gold) != expected_invalid:
        raise ValueError("Could not reconstruct every verifier-rejected gold ID")

    source_paths = _declared_source_paths(fold_manifest)
    source_rows = [row for path in source_paths for row in _read_jsonl(path)]
    examples = {
        str(row.get("example_id", "")): row for row in source_rows
    }
    program_examples = {
        example.example_id: example
        for path in source_paths
        for example in load_program_examples(path)
    }
    semantic_duplicates_dropped: dict[str, int] = {}
    for example_id, record in list(records.items()):
        example = program_examples.get(example_id)
        if example is None:
            raise ValueError(f"Candidate record has no source example: {example_id}")
        deduplicated, dropped = _deduplicate_record(record, example)
        records[example_id] = deduplicated
        semantic_duplicates_dropped[example_id] = dropped
        record_lines[example_id] = (
            json.dumps(
                deduplicated,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
    group_examples = fold_manifest.get("examples", {})
    if not isinstance(group_examples, dict):
        raise ValueError("Fold manifest examples must be an object")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_folds: list[dict[str, object]] = []
    raw_recalls: list[float] = []
    retention_values: list[float] = []
    for raw_fold in fold_manifest.get("folds", []):
        if not isinstance(raw_fold, dict):
            continue
        fold_number = int(raw_fold["fold"])
        train_ids = {str(value) for value in raw_fold["train_example_ids"]}
        validation_ids = {
            str(value) for value in raw_fold["validation_example_ids"]
        }
        if train_ids & validation_ids:
            raise ValueError(f"Example overlap in fold {fold_number}")
        if any(
            int(raw_fold.get(key, -1)) != 0
            for key in ("company_overlap", "report_overlap", "group_overlap")
        ):
            raise ValueError(f"Group leakage in fold {fold_number}")
        fold_dir = args.output_dir / f"fold_{fold_number}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_path = fold_dir / "train_candidates.jsonl"
        validation_path = fold_dir / "validation_candidates.jsonl"
        train_records = _write_records(train_path, train_ids, record_lines)
        validation_records = _write_records(
            validation_path, validation_ids, record_lines
        )
        train_metrics = _metrics(
            train_ids,
            examples=examples,
            records=records,
            training_operations=training_operations,
            missing_gold=missing_gold,
            no_trace=no_trace,
            invalid_gold=invalid_gold,
            group_examples=group_examples,
            semantic_duplicates_dropped=semantic_duplicates_dropped,
        )
        validation_metrics = _metrics(
            validation_ids,
            examples=examples,
            records=records,
            training_operations=training_operations,
            missing_gold=missing_gold,
            no_trace=no_trace,
            invalid_gold=invalid_gold,
            group_examples=group_examples,
            semantic_duplicates_dropped=semantic_duplicates_dropped,
        )
        raw_recalls.append(float(validation_metrics["raw_gold_program_recall"]))
        retention_values.append(float(validation_metrics["verifier_gold_retention"]))
        fold_payload = {
            "schema": "v12b2-disjoint-top64-fold-cache-v1",
            "fold": fold_number,
            "disjoint_claim": fold_manifest["disjoint_claim"],
            "top_k": 64,
            "semantic_deduplication": "operation+typed-cell-signatures-v1",
            "training_operations": sorted(training_operations),
            "candidate_pipeline_sha256": source_manifest[
                "candidate_pipeline_sha256"
            ],
            "base_model_sha256": source_manifest["base_model_sha256"],
            "source_cache_identity": source_manifest["cache_identity"],
            "fold_manifest_sha256": _digest(args.fold_manifest),
            "blind_test_used": False,
            "company_overlap": 0,
            "report_overlap": 0,
            "group_overlap": 0,
            "train_cache": str(train_path.resolve()),
            "train_cache_sha256": _digest(train_path),
            "train_records": train_records,
            "validation_cache": str(validation_path.resolve()),
            "validation_cache_sha256": _digest(validation_path),
            "validation_records": validation_records,
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
        }
        fold_manifest_path = fold_dir / "manifest.json"
        fold_manifest_path.write_text(
            json.dumps(fold_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        output_folds.append(
            {
                "fold": fold_number,
                "manifest": str(fold_manifest_path.resolve()),
                "manifest_sha256": _digest(fold_manifest_path),
                "raw_gold_program_recall": validation_metrics[
                    "raw_gold_program_recall"
                ],
                "verifier_gold_retention": validation_metrics[
                    "verifier_gold_retention"
                ],
            }
        )

    if len(output_folds) != 5:
        raise ValueError("Exactly five fold caches must be written")
    summary = {
        "schema": "v12b2-disjoint-top64-fold-cache-index-v1",
        "fold_manifest": str(args.fold_manifest.resolve()),
        "fold_manifest_sha256": _digest(args.fold_manifest),
        "source_cache_manifest": str(args.source_cache.resolve()),
        "source_cache_manifest_sha256": _digest(args.source_cache),
        "source_cache_identity": source_manifest["cache_identity"],
        "blind_test_used": False,
        "test_jsonl_used": False,
        "sealed_validation_loaded": len(source_paths) > 1,
        "fold_count": 5,
        "training_operations": sorted(training_operations),
        "minimum_fold_raw_gold_program_recall": min(raw_recalls),
        "mean_fold_raw_gold_program_recall": sum(raw_recalls) / len(raw_recalls),
        "minimum_fold_verifier_gold_retention": min(retention_values),
        "mean_fold_verifier_gold_retention": (
            sum(retention_values) / len(retention_values)
        ),
        "raw_recall_gate_95": min(raw_recalls) >= 0.95,
        "verifier_retention_gate_95": min(retention_values) >= 0.95,
        "folds": output_folds,
    }
    index_path = args.output_dir / "manifest.json"
    index_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved five-fold cache index: {index_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
