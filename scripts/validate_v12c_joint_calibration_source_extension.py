#!/usr/bin/env python3
"""Validate an immutable source-only extension of the V12-C protocol."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

from scripts.validate_v12c_joint_calibration_protocol import validate as validate_base


PROTECTED = re.compile(r"(?:test|holdout|final|blind|validation)", re.IGNORECASE)
FORBIDDEN_RULE_KEYS = {
    "cohort_contract",
    "measurement",
    "precommitted_operating_point_gate",
    "feasibility_stop_rule",
    "interpretation_guard",
    "frozen_ratio_percent_threshold",
}


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _refuse(path: Path) -> None:
    if PROTECTED.search(str(path.resolve()).replace("\\", "/")):
        raise ValueError(f"Protected path is not allowed: {path}")


def _checked_entry(root: Path, entry: dict[str, object], label: str) -> Path:
    relative = entry.get("path")
    expected = entry.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected, str):
        raise ValueError(f"Invalid locked {label} entry")
    path = root / relative
    _refuse(path)
    if _sha256(path) != expected:
        raise ValueError(f"Locked hash mismatch: {label}")
    return path


def validate(extension_path: Path, root: Path) -> dict[str, object]:
    _refuse(extension_path)
    extension = json.loads(extension_path.read_text(encoding="utf-8"))
    if extension.get("protocol") != "v12c-joint-calibration-source-extension-v1":
        raise ValueError("Unsupported source extension protocol")
    if extension.get("status") != "precommitted_before_source_content_discovery":
        raise ValueError("Source extension was not precommitted before discovery")
    if extension.get("classification") != "development_only_source_extension":
        raise ValueError("Source extension classification is invalid")
    if FORBIDDEN_RULE_KEYS & set(extension):
        raise ValueError("Source extension attempts to override a frozen rule")

    immutability = extension.get("immutability")
    if not isinstance(immutability, dict) or any(
        immutability.get(key) is not False
        for key in (
            "base_protocol_change_allowed",
            "quota_change_allowed",
            "operating_point_change_allowed",
            "threshold_change_allowed",
            "model_or_runtime_change_allowed",
        )
    ):
        raise ValueError("Source extension does not preserve frozen rules")

    base_path = _checked_entry(root, extension["base_protocol"], "base_protocol")
    base_report = validate_base(base_path, root)
    catalog_path = _checked_entry(root, extension["source_catalog"], "source_catalog")
    audit_path = _checked_entry(root, extension["intake_audit"], "intake_audit")

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if catalog.get("classification") != "development_only_unseen_company_intake":
        raise ValueError("Source catalog is not an unseen development-only intake")
    documents = catalog.get("documents")
    if not isinstance(documents, dict) or not documents:
        raise ValueError("Source catalog has no documents")

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    metrics = audit.get("metrics", {})
    if audit.get("classification") != "development_only":
        raise ValueError("Intake audit is not development-only")
    if audit.get("questions_answers_or_model_outputs_used") is not False:
        raise ValueError("Intake selection used forbidden content or model outputs")
    if metrics.get("overlap_violations") != 0:
        raise ValueError("Intake audit contains overlap violations")
    if metrics.get("hash_locked_documents_verified") != len(documents):
        raise ValueError("Intake audit did not verify every document hash")

    selection = extension.get("selection_freeze", {})
    expected_groups = len(
        {str(entry.get("company_key", "")) for entry in documents.values()}
    )
    if selection.get("documents") != len(documents):
        raise ValueError("Frozen document count does not match catalog")
    if selection.get("company_groups") != expected_groups:
        raise ValueError("Frozen company count does not match catalog")
    if selection.get("selection_used_financial_content") is not False:
        raise ValueError("Source selection used financial content")

    return {
        "protocol": extension["protocol"],
        "extension_sha256": _sha256(extension_path),
        "base_protocol_sha256": _sha256(base_path),
        "source_catalog_sha256": _sha256(catalog_path),
        "intake_audit_sha256": _sha256(audit_path),
        "documents": len(documents),
        "company_groups": expected_groups,
        "base_precommit_valid": base_report["precommit_valid"],
        "source_extension_valid": True,
        "frozen_rules_changed": False,
        "ready_for_development_only_discovery": True,
        "ready_for_threshold_change": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {args.output}")
    _refuse(args.output)
    report = validate(args.extension, args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
