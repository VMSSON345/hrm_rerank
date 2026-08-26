from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrm_finance_qa.cache_schema import (  # noqa: E402
    CANDIDATE_CACHE_SCHEMA,
    candidate_cache_schema_identity,
)
from hrm_finance_qa.document_hrm import DocumentHRMBundle  # noqa: E402
from hrm_finance_qa.program_data import load_program_examples  # noqa: E402
from hrm_finance_qa.program_reranker import (  # noqa: E402
    PROGRAM_RERANKER_FEATURE_SCHEMA,
    train_program_reranker,
)
from hrm_finance_qa.v11_pipeline import (  # noqa: E402
    build_reranker_records,
    checkpoint_sha256,
    read_reranker_records,
    split_reranker_development,
    upgrade_reranker_records_to_supported_tristate,
    write_reranker_records,
)


CACHE_SCHEMA = CANDIDATE_CACHE_SCHEMA
TRAINING_STATE_POLICY = "applicability-aware-proven-hard-negative-v4"
ARCHITECTURE = "v12b2-phase-a-applicability-safe-residual-ablation"


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _source_digest(paths: tuple[Path, ...]) -> str:
    value = sha256()
    for path in paths:
        value.update(str(path.relative_to(ROOT)).encode("utf-8"))
        value.update(path.read_bytes())
    return value.hexdigest()


def _reject_protected(path: Path) -> None:
    normalized = str(path.expanduser()).lower().replace("\\", "/")
    if any(
        marker in normalized
        for marker in ("test.jsonl", "blind", "final_eval", "/final.jsonl")
    ):
        raise ValueError(f"V12-B development refuses protected input: {path}")


def _runtime_bundle(path: Path, *, device: str) -> DocumentHRMBundle:
    bundle = DocumentHRMBundle.load(path, device=device)
    bundle.metadata = dict(bundle.metadata)
    bundle.metadata.update(
        {
            "typed_program_synthesis_enabled": True,
            "typed_program_synthesis_version": "typed-slot-synthesis-v1",
            "financial_identity_catalog_enabled": True,
            "candidate_recall_v12_4_enabled": True,
        }
    )
    if bundle.program_reranker is not None:
        raise RuntimeError("Frozen candidate generator unexpectedly has a reranker")
    return bundle


def _cache_paths(cache_dir: Path) -> tuple[Path, Path, Path]:
    return (
        cache_dir / "train_candidates.jsonl",
        cache_dir / "calibration_candidates.jsonl",
        cache_dir / "manifest.json",
    )


def _expected_manifest(
    *,
    base_model: Path,
    train_path: Path,
    validation_path: Path | None,
    top_k: int,
    shard_size: int,
    split_audit: dict[str, object],
) -> dict[str, object]:
    return {
        **candidate_cache_schema_identity(),
        "feature_schema": PROGRAM_RERANKER_FEATURE_SCHEMA,
        "base_model_sha256": checkpoint_sha256(base_model),
        "train_dataset_sha256": _digest(train_path),
        "validation_dataset_sha256": (
            _digest(validation_path) if validation_path is not None else "not_loaded"
        ),
        "sealed_validation_loaded": validation_path is not None,
        "top_k": top_k,
        "shard_size": shard_size,
        "typed_program_synthesis_enabled": True,
        "financial_identity_catalog_enabled": True,
        "candidate_recall_v12_4_enabled": True,
        "training_state_policy": TRAINING_STATE_POLICY,
        "candidate_pipeline_sha256": _source_digest(
            tuple(
                ROOT / "src" / "hrm_finance_qa" / name
                for name in (
                    "question_planner.py",
                    "evidence_schema.py",
                    "program_synthesis.py",
                    "symbolic_verifier.py",
                    "document_hrm.py",
                    "v11_pipeline.py",
                )
            )
        ),
        "split_audit": split_audit,
        "blind_test_used": False,
        "test_jsonl_used": False,
    }


def _aggregate_audits(audits: list[dict[str, object]]) -> dict[str, object]:
    additive = (
        "examples",
        "records",
        "candidate_sets",
        "no_trace",
        "typed_candidates",
        "typed_gold_candidates",
        "hard_invalid_candidates_dropped",
        "proven_supervision_invalid_candidates_retained",
        "soft_unknown_candidates_retained",
        "pre_prune_candidates",
        "post_prune_candidates",
        "training_trace_sets",
        "pruning_gold_loss_count",
        "gold_verifier_generated",
        "gold_verifier_retained",
    )
    output: dict[str, object] = {
        key: sum(int(audit.get(key, 0)) for audit in audits) for key in additive
    }
    output.update(
        {
            "top_k": max((int(audit.get("top_k", 0)) for audit in audits), default=0),
            "invalid_candidates_retained": sum(
                int(audit.get("invalid_candidates_retained", 0)) for audit in audits
            ),
            "unknown_as_negative": False,
            "training_state_policy": TRAINING_STATE_POLICY,
        }
    )

    def merge_counter(key: str) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for audit in audits:
            raw = audit.get(key, {})
            if isinstance(raw, dict):
                counter.update({str(name): int(value) for name, value in raw.items()})
        return dict(counter)

    for key in (
        "positive_classes",
        "typed_synthesis_statuses",
        "candidate_failure_modes",
        "training_state_counts",
        "retained_training_state_counts",
        "proof_contract_state_counts",
        "proof_obligation_counts",
        "gold_false_rejection_constraints",
    ):
        output[key] = merge_counter(key)

    per_operation: dict[str, Counter[str]] = {}
    failure_examples: dict[str, list[dict[str, object]]] = {}
    proof_contradiction_examples: dict[str, list[dict[str, object]]] = {}
    for audit in audits:
        raw_operations = audit.get("per_operation", {})
        if isinstance(raw_operations, dict):
            for operation, values in raw_operations.items():
                if not isinstance(values, dict):
                    continue
                per_operation.setdefault(str(operation), Counter()).update(
                    {str(name): int(value) for name, value in values.items()}
                )
        raw_failures = audit.get("failure_examples", {})
        if isinstance(raw_failures, dict):
            for mode, values in raw_failures.items():
                if not isinstance(values, list):
                    continue
                rows = failure_examples.setdefault(str(mode), [])
                rows.extend(value for value in values if isinstance(value, dict))
                del rows[40:]
        raw_contradictions = audit.get("proof_contradiction_examples", {})
        if isinstance(raw_contradictions, dict):
            for constraint, values in raw_contradictions.items():
                if not isinstance(values, list):
                    continue
                rows = proof_contradiction_examples.setdefault(
                    str(constraint), []
                )
                rows.extend(value for value in values if isinstance(value, dict))
                del rows[40:]
    output["per_operation"] = {
        operation: dict(values) for operation, values in per_operation.items()
    }
    output["failure_examples"] = failure_examples
    output["proof_contradiction_examples"] = proof_contradiction_examples

    candidate_sets = int(output["candidate_sets"])
    records = int(output["records"])
    trace_sets = int(output["training_trace_sets"])

    def weighted_ratio(key: str, denominator_key: str) -> float:
        numerator = 0.0
        denominator = 0
        for audit in audits:
            weight = int(audit.get(denominator_key, 0))
            numerator += float(audit.get(key, 0.0)) * weight
            denominator += weight
        return numerator / denominator if denominator else 0.0

    output["gold_program_recall_at_k"] = weighted_ratio(
        "gold_program_recall_at_k", "candidate_sets"
    )
    output["execution_recall_at_k"] = weighted_ratio(
        "execution_recall_at_k", "candidate_sets"
    )
    output["rerankable_gold_program_recall_at_k"] = weighted_ratio(
        "rerankable_gold_program_recall_at_k", "records"
    )
    output["rerankable_execution_recall_at_k"] = weighted_ratio(
        "rerankable_execution_recall_at_k", "records"
    )
    output["mean_pre_prune_candidates"] = (
        int(output["pre_prune_candidates"]) / trace_sets if trace_sets else 0.0
    )
    output["mean_post_prune_candidates"] = (
        int(output["post_prune_candidates"]) / trace_sets if trace_sets else 0.0
    )
    output["pre_prune_gold_program_recall"] = weighted_ratio(
        "pre_prune_gold_program_recall", "training_trace_sets"
    )
    output["post_prune_gold_program_recall"] = weighted_ratio(
        "post_prune_gold_program_recall", "training_trace_sets"
    )
    output["gold_verifier_retention"] = (
        int(output["gold_verifier_retained"])
        / int(output["gold_verifier_generated"])
        if int(output["gold_verifier_generated"])
        else 0.0
    )
    for key in (
        "mean_required_valid_fraction",
        "mean_required_unknown_fraction",
        "mean_optional_support_fraction",
    ):
        output[key] = weighted_ratio(key, "post_prune_candidates")
    output["candidate_sets"] = candidate_sets
    output["records"] = records
    return output


def _generate_sharded_records(
    *,
    label: str,
    bundle: DocumentHRMBundle,
    examples: list,
    cache_dir: Path,
    cache_identity: str,
    top_k: int,
    batch_size: int,
    shard_size: int,
    regenerate: bool,
) -> tuple[list, dict[str, object], list[dict[str, object]]]:
    shard_root = cache_dir / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    all_records: list = []
    audits: list[dict[str, object]] = []
    shard_manifests: list[dict[str, object]] = []
    for start in range(0, len(examples), shard_size):
        current = examples[start : start + shard_size]
        end = start + len(current)
        stem = f"{label}_{start:05d}_{end:05d}"
        records_path = shard_root / f"{stem}.jsonl"
        audit_path = shard_root / f"{stem}.json"
        example_fingerprint = sha256(
            "\n".join(item.example_id for item in current).encode("utf-8")
        ).hexdigest()
        reused = False
        if records_path.is_file() and audit_path.is_file() and not regenerate:
            shard_manifest = json.loads(audit_path.read_text(encoding="utf-8"))
            if (
                shard_manifest.get("cache_identity") == cache_identity
                and shard_manifest.get("example_fingerprint") == example_fingerprint
                and shard_manifest.get("records_sha256") == _digest(records_path)
            ):
                records = read_reranker_records(records_path)
                audit = shard_manifest.get("audit")
                if not isinstance(audit, dict):
                    raise RuntimeError(f"Invalid audit in resumable shard {stem}")
                reused = True
            else:
                raise RuntimeError(
                    f"Resumable shard {stem} does not match this run; "
                    "pass --regenerate-cache"
                )
        else:
            records, audit = build_reranker_records(
                bundle,
                current,
                top_k=top_k,
                batch_size=batch_size,
            )
            write_reranker_records(records_path, records)
            shard_manifest = {
                "cache_schema": CACHE_SCHEMA,
                "cache_identity": cache_identity,
                "label": label,
                "start": start,
                "end": end,
                "example_fingerprint": example_fingerprint,
                "records": len(records),
                "records_sha256": _digest(records_path),
                "audit": audit,
            }
            audit_path.write_text(
                json.dumps(shard_manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        print(
            f"[cache {label}] {end}/{len(examples)} "
            f"({'reused' if reused else 'saved'})",
            flush=True,
        )
        all_records.extend(records)
        audits.append(audit)
        shard_manifests.append(
            {
                "path": str(records_path.resolve()),
                "audit_path": str(audit_path.resolve()),
                "records": len(records),
                "reused": reused,
            }
        )
    return all_records, _aggregate_audits(audits), shard_manifests


def _load_or_generate_cache(
    *,
    bundle: DocumentHRMBundle,
    train_examples: list,
    calibration_examples: list,
    expected_manifest: dict[str, object],
    cache_dir: Path,
    top_k: int,
    batch_size: int,
    shard_size: int,
    regenerate: bool,
    max_pruning_gold_loss: int,
) -> tuple[list, list, dict[str, object], bool, float]:
    train_cache, calibration_cache, manifest_path = _cache_paths(cache_dir)
    existing = all(path.is_file() for path in (train_cache, calibration_cache, manifest_path))
    if existing and not regenerate:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mutable_schema_keys = {
            "cache_schema",
            "feature_schema",
            "training_state_policy",
        }
        identity = {
            key: value
            for key, value in expected_manifest.items()
            if key not in mutable_schema_keys | {"split_audit"}
        }
        observed = {key: manifest.get(key) for key in identity}
        current_schema = (
            manifest.get("cache_schema") == CACHE_SCHEMA
            and manifest.get("feature_schema")
            == PROGRAM_RERANKER_FEATURE_SCHEMA
            and manifest.get("training_state_policy")
            == TRAINING_STATE_POLICY
        )
        # Older caches do not contain applicability or proof obligations and
        # cannot be upgraded without running the verifier again.
        migratable_legacy_schema = False
        if (
            observed != identity
            or manifest.get("split_audit") != expected_manifest["split_audit"]
            or not (current_schema or migratable_legacy_schema)
        ):
            raise RuntimeError(
                "Existing V12-B cache does not match the frozen model/data/split; "
                "pass --regenerate-cache to replace it"
            )
        if (
            manifest.get("train_cache_sha256") != _digest(train_cache)
            or manifest.get("calibration_cache_sha256")
            != _digest(calibration_cache)
        ):
            raise RuntimeError("Existing V12-B cache failed its SHA-256 check")
        for name in ("train", "calibration"):
            audit = manifest.get(f"{name}_audit", {})
            if not isinstance(audit, dict):
                raise RuntimeError(f"Existing {name} cache audit is missing")
            if int(audit.get("invalid_candidates_retained", 0)) != 0:
                raise RuntimeError(f"Existing {name} cache retained INVALID rows")
            if int(audit.get("pruning_gold_loss_count", 0)) > max_pruning_gold_loss:
                raise RuntimeError(f"Existing {name} cache exceeds pruning loss gate")
        train_records = read_reranker_records(train_cache)
        calibration_records = read_reranker_records(calibration_cache)
        if (
            len(train_records) != int(manifest.get("train_records", -1))
            or len(calibration_records)
            != int(manifest.get("calibration_records", -1))
        ):
            raise RuntimeError("Existing V12-B cache record count is inconsistent")
        train_records, train_upgrade = (
            upgrade_reranker_records_to_supported_tristate(train_records)
        )
        calibration_records, calibration_upgrade = (
            upgrade_reranker_records_to_supported_tristate(
                calibration_records
            )
        )
        effective_manifest = {
            **manifest,
            "effective_cache_schema": CACHE_SCHEMA,
            "effective_feature_schema": PROGRAM_RERANKER_FEATURE_SCHEMA,
            "effective_training_state_policy": (
                TRAINING_STATE_POLICY
            ),
            "cache_migrated_in_memory": migratable_legacy_schema,
            "train_supported_tristate_upgrade": train_upgrade,
            "calibration_supported_tristate_upgrade": calibration_upgrade,
        }
        return (
            train_records,
            calibration_records,
            effective_manifest,
            True,
            0.0,
        )

    started = perf_counter()
    cache_identity = sha256(
        json.dumps(
            expected_manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    train_records, train_audit, train_shards = _generate_sharded_records(
        label="train",
        bundle=bundle,
        examples=train_examples,
        cache_dir=cache_dir,
        cache_identity=cache_identity,
        top_k=top_k,
        batch_size=batch_size,
        shard_size=shard_size,
        regenerate=regenerate,
    )
    calibration_records, calibration_audit, calibration_shards = (
        _generate_sharded_records(
            label="calibration",
            bundle=bundle,
            examples=calibration_examples,
            cache_dir=cache_dir,
            cache_identity=cache_identity,
            top_k=top_k,
            batch_size=batch_size,
            shard_size=shard_size,
            regenerate=regenerate,
        )
    )
    cache_gate_errors: list[str] = []
    for name, audit in (
        ("train", train_audit),
        ("calibration", calibration_audit),
    ):
        if int(audit.get("invalid_candidates_retained", 0)) != 0:
            cache_gate_errors.append(f"{name} cache retained INVALID candidates")
        losses = int(audit.get("pruning_gold_loss_count", 0))
        if losses > max_pruning_gold_loss:
            cache_gate_errors.append(
                f"{name} pruning lost {losses} gold programs; allowed "
                f"maximum is {max_pruning_gold_loss}"
            )
    if not train_records or not calibration_records:
        raise RuntimeError("V12-B cache did not produce train/calibration records")

    cache_dir.mkdir(parents=True, exist_ok=True)
    write_reranker_records(train_cache, train_records)
    write_reranker_records(calibration_cache, calibration_records)
    manifest = {
        **expected_manifest,
        "train_cache": str(train_cache.resolve()),
        "calibration_cache": str(calibration_cache.resolve()),
        "train_cache_sha256": _digest(train_cache),
        "calibration_cache_sha256": _digest(calibration_cache),
        "train_records": len(train_records),
        "calibration_records": len(calibration_records),
        "cache_identity": cache_identity,
        "train_shards": train_shards,
        "calibration_shards": calibration_shards,
        "train_audit": train_audit,
        "calibration_audit": calibration_audit,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if cache_gate_errors:
        raise RuntimeError(
            "; ".join(cache_gate_errors)
            + f". Audit and cache were saved under {cache_dir}"
        )
    return (
        train_records,
        calibration_records,
        manifest,
        False,
        perf_counter() - started,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the V12-B.2 Phase-A applicability audit/cache; optional "
            "reranker training remains an ablation and keeps V8 frozen"
        )
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=ROOT / "models" / "document_hrm_V8_candidate.pt",
    )
    parser.add_argument(
        "--train", type=Path, default=ROOT / "data" / "train.jsonl"
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=ROOT / "data" / "validation.jsonl",
        help="Used only to seal company/report groups from train/calibration",
    )
    parser.add_argument(
        "--development-only",
        action="store_true",
        help=(
            "Treat --train as the already sealed development corpus and do not "
            "open or hash --validation"
        ),
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=ROOT / "cache" / "v12b2_phase_a"
    )
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=ROOT / "models" / "document_hrm_V12B2_phase_a_ablation.pt",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports" / "document_hrm_V12B2_phase_a_training.json",
    )
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--confidence-epochs", type=int, default=6)
    parser.add_argument("--target-precision", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--regenerate-cache", action="store_true")
    parser.add_argument("--max-pruning-gold-loss", type=int, default=0)
    parser.add_argument(
        "--split-audit-only",
        action="store_true",
        help="Validate the sealed company/report split without loading V8",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Generate/validate the candidate cache and stop before training",
    )
    args = parser.parse_args()

    checked_inputs = (args.train,) if args.development_only else (args.train, args.validation)
    for path in checked_inputs:
        _reject_protected(path)
    required_paths = (args.base_model, *checked_inputs)
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.top_k < 2:
        raise ValueError("--top-k must be at least 2")
    if args.shard_size < args.batch_size:
        raise ValueError("--shard-size must be at least --batch-size")

    base_hash_before = checkpoint_sha256(args.base_model)
    train_input = load_program_examples(args.train)
    validation_examples = (
        [] if args.development_only else load_program_examples(args.validation)
    )
    reranker_train_examples, calibration_examples, split_audit = (
        split_reranker_development(
            [*train_input, *validation_examples],
            validation_examples,
            calibration_fraction=args.calibration_fraction,
            seed=args.seed,
        )
    )
    # ``split_reranker_development`` selects calibration groups with a set.
    # Canonicalize only the cache serialization order so shard fingerprints
    # are reproducible across Python hash seeds and parallel workers.
    calibration_examples.sort(key=lambda example: example.example_id)
    if args.split_audit_only:
        print(json.dumps(split_audit, ensure_ascii=False, indent=2))
        print("Split audit only; V8 was not loaded and no cache was written.")
        return 0
    expected_manifest = _expected_manifest(
        base_model=args.base_model,
        train_path=args.train,
        validation_path=None if args.development_only else args.validation,
        top_k=args.top_k,
        shard_size=args.shard_size,
        split_audit=split_audit,
    )
    generator = _runtime_bundle(args.base_model, device=args.device)
    (
        train_records,
        calibration_records,
        cache_manifest,
        cache_reused,
        cache_seconds,
    ) = _load_or_generate_cache(
        bundle=generator,
        train_examples=reranker_train_examples,
        calibration_examples=calibration_examples,
        expected_manifest=expected_manifest,
        cache_dir=args.cache_dir,
        top_k=args.top_k,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        regenerate=args.regenerate_cache,
        max_pruning_gold_loss=args.max_pruning_gold_loss,
    )

    if args.cache_only:
        print(json.dumps(cache_manifest, ensure_ascii=False, indent=2))
        print("Candidate cache verified; reranker training was not started.")
        print("Blind test was not touched.")
        if args.development_only:
            print("Sealed validation was not opened or hashed.")
        return 0

    started = perf_counter()
    reranker, training_report = train_program_reranker(
        train_records,
        calibration_records,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        confidence_epochs=args.confidence_epochs,
        target_precision=args.target_precision,
        seed=args.seed,
        device=args.device,
        metadata={
            "architecture": ARCHITECTURE,
            "base_model_sha256": base_hash_before,
            "candidate_cache_schema": CACHE_SCHEMA,
            "training_state_policy": TRAINING_STATE_POLICY,
            "top_k": args.top_k,
            "typed_program_synthesis_enabled": True,
            "financial_identity_catalog_enabled": True,
            "candidate_recall_v12_4_enabled": True,
            "validation_labels_used_for_training": False,
            "company_report_disjoint_calibration": True,
        },
    )
    training_seconds = perf_counter() - started
    sidecar_path = reranker.save(args.sidecar)

    base_hash_after = checkpoint_sha256(args.base_model)
    if base_hash_after != base_hash_before:
        raise RuntimeError("Frozen V8 changed during V12-B training")
    report = {
        "architecture": ARCHITECTURE,
        "base_model": str(args.base_model.resolve()),
        "base_model_sha256_before": base_hash_before,
        "base_model_sha256_after": base_hash_after,
        "base_model_unchanged": True,
        "sidecar": str(sidecar_path),
        "sidecar_sha256": _digest(sidecar_path),
        "blind_test_used": False,
        "test_jsonl_used": False,
        "validation_labels_used_for_training": False,
        "split": split_audit,
        "cache_reused": cache_reused,
        "cache_seconds": cache_seconds,
        "training_seconds": training_seconds,
        "cache_manifest": cache_manifest,
        "training": asdict(training_report),
        "reranker_metadata": reranker.metadata,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved V12-B.2 Phase-A ablation sidecar: {sidecar_path}")
    print(f"Saved training report: {args.report}")
    print("Frozen V8 SHA-256 verified unchanged.")
    print("Blind test was not touched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
