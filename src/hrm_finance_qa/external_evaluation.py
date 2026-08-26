from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation
from hashlib import blake2b
from pathlib import Path
import re
from typing import Sequence

from .ontology import normalize_text
from .program_data import OPERATIONS, ProgramDataError, ProgramExample, execute_program


PROTOCOL_VERSION = "document-hrm-external-v1"


def audit_external_test_set(
    external_examples: Sequence[ProgramExample],
    reference_examples: Sequence[ProgramExample],
) -> dict[str, object]:
    """Check integrity and train/dev leakage before external evaluation.

    Question overlap is reported but is not a hard failure because finance QA
    legitimately reuses generic question templates across different companies.
    Company, ticker, report/context, and example-ID overlap are hard failures.
    """

    external_ids = [example.example_id for example in external_examples]
    reference_ids = {example.example_id for example in reference_examples}
    duplicate_ids = sorted(
        example_id for example_id, count in Counter(external_ids).items() if count > 1
    )
    synthetic_ids = sorted(
        example.example_id
        for example in external_examples
        if _is_synthetic(example)
    )
    pending_review_ids = sorted(
        example.example_id
        for example in external_examples
        if _requires_independent_review(example)
    )

    invalid_programs: list[dict[str, str]] = []
    execution_mismatches: list[dict[str, str]] = []
    for example in external_examples:
        try:
            example.program.validate(len(example.candidates))
            executed = execute_program(example.program, example.candidates)
            expected = Decimal(example.answer_value)
        except (ProgramDataError, ZeroDivisionError, InvalidOperation) as exc:
            invalid_programs.append(
                {"example_id": example.example_id, "error": str(exc)}
            )
            continue
        if not _close(executed, expected):
            execution_mismatches.append(
                {
                    "example_id": example.example_id,
                    "executed": str(executed),
                    "answer_value": str(expected),
                }
            )

    external_identity = _identity_index(external_examples)
    reference_identity = _identity_index(reference_examples)
    leakage = {
        "example_ids": sorted(set(external_ids) & reference_ids),
        "companies": sorted(
            external_identity["companies"] & reference_identity["companies"]
        ),
        "tickers": sorted(
            external_identity["tickers"] & reference_identity["tickers"]
        ),
        "report_fingerprints": sorted(
            external_identity["report_fingerprints"]
            & reference_identity["report_fingerprints"]
        ),
        "context_fingerprints": sorted(
            external_identity["context_fingerprints"]
            & reference_identity["context_fingerprints"]
        ),
        "questions": sorted(
            external_identity["questions"] & reference_identity["questions"]
        ),
    }

    hard_leakage_keys = (
        "example_ids",
        "companies",
        "tickers",
        "report_fingerprints",
        "context_fingerprints",
    )
    hard_failure_count = (
        int(not external_examples)
        + int(not reference_examples)
        + len(duplicate_ids)
        + len(synthetic_ids)
        + len(pending_review_ids)
        + len(invalid_programs)
        + len(execution_mismatches)
        + sum(len(leakage[key]) for key in hard_leakage_keys)
    )
    operation_counts = Counter(
        example.program.operation for example in external_examples
    )
    return {
        "protocol": PROTOCOL_VERSION,
        "passed": hard_failure_count == 0,
        "external_examples": len(external_examples),
        "reference_examples": len(reference_examples),
        "operation_counts": {
            operation: operation_counts[operation]
            for operation in OPERATIONS
            if operation_counts[operation]
        },
        "missing_operations": [
            operation for operation in OPERATIONS if not operation_counts[operation]
        ],
        "identity_counts": {
            "external_companies": len(external_identity["companies"]),
            "external_tickers": len(external_identity["tickers"]),
            "external_reports": len(external_identity["report_fingerprints"]),
            "reference_companies": len(reference_identity["companies"]),
            "reference_tickers": len(reference_identity["tickers"]),
            "reference_reports": len(reference_identity["report_fingerprints"]),
        },
        "integrity": {
            "empty_test_set": not external_examples,
            "empty_reference_set": not reference_examples,
            "duplicate_example_ids": duplicate_ids,
            "synthetic_example_ids": synthetic_ids,
            "pending_independent_review_example_ids": pending_review_ids,
            "invalid_programs": invalid_programs,
            "execution_mismatches": execution_mismatches,
        },
        "leakage": leakage,
        "warnings": {
            "question_overlap_is_not_a_hard_failure": len(leakage["questions"]),
            "missing_operation_classes": len(
                [operation for operation in OPERATIONS if not operation_counts[operation]]
            ),
        },
    }


def _identity_index(examples: Sequence[ProgramExample]) -> dict[str, set[str]]:
    output = {
        "companies": set(),
        "tickers": set(),
        "report_fingerprints": set(),
        "context_fingerprints": set(),
        "questions": set(),
    }
    for example in examples:
        company, ticker = _company_and_ticker(example.context)
        if company:
            output["companies"].add(company)
        if ticker:
            output["tickers"].add(ticker)
        supervision = example.supervision
        if isinstance(supervision, dict):
            company_key = _normalize_identity_text(
                str(supervision.get("company_key", ""))
            )
            if company_key:
                output["companies"].add(f"key:{company_key}")
        source_names = {
            candidate.source_name
            for candidate in example.candidates
            if candidate.source_name not in {"", "context", "synthetic"}
        }
        for source_name in source_names:
            source_path = Path(source_name)
            normalized_source = _normalize_identity_text(source_path.name)
            output["report_fingerprints"].add(_fingerprint(normalized_source))
            match = re.match(r"([A-Za-z0-9]{3,})_", source_path.stem)
            if match:
                output["tickers"].add(match.group(1).lower())
        if not source_names:
            output["report_fingerprints"].add(_report_fingerprint(example.context))
        output["context_fingerprints"].add(_fingerprint(example.context))
        output["questions"].add(_normalize_identity_text(example.question))
    return output


def _company_and_ticker(context: str) -> tuple[str, str]:
    company = ""
    ticker = ""
    invalid_tickers = {"khong", "none", "null", "na", "n a", "chua"}
    for raw_line in context.splitlines()[:16]:
        line = _normalize_identity_text(raw_line)
        if line.startswith("cong ty ") and not company:
            company = line.removeprefix("cong ty ").strip()
        if line.startswith("ma co phieu ") and not ticker:
            candidate = line.removeprefix("ma co phieu ").split()[0].strip()
            if candidate not in invalid_tickers:
                ticker = candidate
    return company, ticker


def _report_fingerprint(context: str) -> str:
    metadata = "\n".join(context.splitlines()[:8])
    return _fingerprint(metadata)


def _fingerprint(value: str) -> str:
    normalized = _normalize_identity_text(value)
    return blake2b(normalized.encode("utf-8"), digest_size=12).hexdigest()


def _is_synthetic(example: ProgramExample) -> bool:
    difficulty = _normalize_identity_text(example.difficulty)
    return example.example_id.startswith("synthetic-") or difficulty == "synthetic"


def _requires_independent_review(example: ProgramExample) -> bool:
    supervision = example.supervision
    if not isinstance(supervision, dict):
        return False
    status = _normalize_identity_text(str(supervision.get("review_status", "")))
    classification = _normalize_identity_text(
        str(supervision.get("classification", ""))
    )
    return status.startswith("pending") or "pending review" in classification


def _normalize_identity_text(value: str) -> str:
    """Normalize both proper Unicode and the corpus' common UTF-8 mojibake."""

    repaired = value
    if any(marker in value for marker in ("Ã", "Â", "Ä", "áº", "á»")):
        for encoding in ("cp1252", "latin1"):
            try:
                candidate = value.encode(encoding).decode("utf-8")
            except UnicodeError:
                continue
            if candidate.count("Ã") + candidate.count("áº") + candidate.count("á»") < (
                repaired.count("Ã") + repaired.count("áº") + repaired.count("á»")
            ):
                repaired = candidate
                break
    return normalize_text(repaired)


def _close(left: Decimal, right: Decimal) -> bool:
    difference = abs(left - right)
    scale = max(abs(right), Decimal(1))
    tolerance = (
        Decimal("0.035")
        if scale < Decimal(1000)
        else max(Decimal("0.5"), scale * Decimal("0.00000000001"))
    )
    return difference <= tolerance
