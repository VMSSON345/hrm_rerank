from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import signal
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
DEFAULT_CONFIG = ROOT / "config" / "finhrm_qp4_development_cohort_v1.json"
DEFAULT_RAW = ROOT / "reports" / "finhrm_qp4_source_backed_baseline_v1.json"
DEFAULT_REPORT = ROOT / "reports" / "finhrm_qp4_development_baseline_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "finhrm_qp4_development_baseline_v1.md"
DEFAULT_LOCK = ROOT / "reports" / "finhrm_qp4_development_baseline_lock_v1.json"
FUNCTIONAL_TEST = ROOT / "tests" / "test_qp4_development_baseline.py"
REPORT_TEST = ROOT / "tests" / "test_qp4_development_baseline_report.py"
PROTECTED_PARTS = {"validation", "blind", "sealed", "holdout", "final"}


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _digest(path),
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object: {path}:{line_number}")
        output.append(value)
    return output


def _refuse_protected_input(path: Path) -> None:
    parts = {part.lower() for part in path.resolve().parts}
    overlap = sorted(parts & PROTECTED_PARTS)
    if overlap:
        raise ValueError(f"Protected input refused: {path} ({overlap})")


def _verify_inputs(config: dict[str, Any]) -> None:
    for parent in config["parents"].values():
        path = ROOT / parent["path"]
        _refuse_protected_input(path)
        if not path.is_file() or _digest(path) != parent["sha256"]:
            raise ValueError(f"Parent hash mismatch: {path}")
    checkpoint = ROOT / config["checkpoint"]["path"]
    _refuse_protected_input(checkpoint)
    if not checkpoint.is_file() or _digest(checkpoint) != config["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    for relative, expected in config["runtime_hashes"].items():
        path = ROOT / relative
        if not path.is_file() or _digest(path) != expected:
            raise ValueError(f"Runtime hash mismatch: {relative}")


def _value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _semantic_projection(plan: Any) -> dict[str, Any]:
    slots = [
        {
            "id": str(slot.id),
            "metric": str(slot.metric),
            "year": str(slot.year),
            "column_role": str(slot.column_role),
            "entity_scope": str(slot.entity_scope),
            "unit_family": str(slot.unit_family),
        }
        for slot in getattr(plan, "operand_slots", ())
    ]
    dag = [
        {
            "id": str(node.id),
            "operation": str(node.operation),
            "inputs": list(node.inputs),
            "output_unit_family": str(node.output_unit_family),
        }
        for node in getattr(plan, "program_dag", ())
    ]
    return {
        "operation": getattr(plan, "operation", None),
        "operation_candidates": list(getattr(plan, "operation_candidates", ())),
        "answer_mode": str(getattr(plan, "answer_mode", "SINGLE_SCALAR")),
        "requested_output_count": int(getattr(plan, "requested_output_count", 1)),
        "requested_years": list(getattr(plan, "requested_years", ())),
        "from_year": getattr(plan, "from_year", None),
        "to_year": getattr(plan, "to_year", None),
        "entity_mode": str(getattr(plan, "entity_mode", "reporting")),
        "entity_text": str(getattr(plan, "entity_text", "")),
        "requested_unit_family": str(getattr(plan, "requested_unit_family", "")),
        "requested_scale": getattr(plan, "requested_scale", None),
        "direction_intent": _value(getattr(plan, "direction_intent", None)),
        "reference_base": getattr(plan, "reference_base", None),
        "abstention_reason": str(getattr(plan, "abstention_reason", "")),
        "operand_slots": slots,
        "program_dag": dag,
    }


def _dimension_projection(plan: Any, dimension: str) -> Any:
    semantic = _semantic_projection(plan)
    slots = semantic["operand_slots"]
    dag = semantic["program_dag"]
    if dimension == "COLUMN_ROLE":
        roles = [slot["column_role"] for slot in slots]
        return roles or list(getattr(plan, "program_column_roles", ()))
    if dimension == "OPERAND_YEAR":
        years = [slot["year"] for slot in slots]
        return years or semantic["requested_years"]
    if dimension == "ENTITY_SCOPE":
        return {
            "entity_mode": semantic["entity_mode"],
            "entity_text": semantic["entity_text"],
            "slot_scopes": [slot["entity_scope"] for slot in slots],
        }
    if dimension == "UNIT_SCALE":
        return {
            "requested_unit_family": semantic["requested_unit_family"],
            "requested_scale": semantic["requested_scale"],
        }
    if dimension == "DIRECTION":
        return {
            "direction_intent": semantic["direction_intent"],
            "slot_order": [
                (slot["id"], slot["column_role"], slot["year"]) for slot in slots
            ],
            "dag_inputs": [node["inputs"] for node in dag],
        }
    if dimension == "REFERENCE_BASE":
        return {
            "reference_base": semantic["reference_base"],
            "root_inputs": dag[-1]["inputs"] if dag else [],
            "slot_ids": [slot["id"] for slot in slots],
        }
    raise ValueError(f"Unknown dimension: {dimension}")


def _insert_before_suffix(question: str, qualifier: str) -> str:
    stripped = question.strip()
    suffix = " là bao nhiêu?"
    if stripped.lower().endswith(suffix):
        return stripped[: -len(suffix)].rstrip() + f" {qualifier}{suffix}"
    return stripped.rstrip("?") + f", {qualifier}?"


def _generated_surface(question: str, category: str) -> str:
    stripped = question.strip()
    if category == "DISCOURSE_PREFIX_SUFFIX":
        return "Theo số liệu đã công bố trong cùng báo cáo, " + stripped[0].lower() + stripped[1:]
    if category == "PUNCTUATION_VARIANT":
        return stripped.rstrip("?") + " ?"
    if category == "BILINGUAL_PARENTHETICAL":
        return stripped.rstrip("?") + " (đơn vị theo nguồn / source unit)?"
    if category == "IRRELEVANT_DISTRACTOR_CLAUSE":
        return stripped.rstrip("?") + ", không yêu cầu phân tích nguyên nhân?"
    raise ValueError(category)


def _increment_first_year(question: str) -> str:
    def replace(match: re.Match[str]) -> str:
        return str(int(match.group(0)) + 1)

    mutated, count = re.subn(r"\b(?:19|20)\d{2}\b", replace, question, count=1)
    if count != 1:
        raise ValueError(f"No explicit year to mutate: {question}")
    return mutated


def _generated_counterfactual(question: str, dimension: str) -> str:
    if dimension == "OPERAND_YEAR":
        return _increment_first_year(question)
    if dimension == "ENTITY_SCOPE":
        return _insert_before_suffix(question, "của công ty mẹ")
    if dimension == "UNIT_SCALE":
        return _insert_before_suffix(question, "theo đơn vị triệu đồng")
    raise ValueError(dimension)


def _source_period_counterfactual(question: str) -> str:
    if re.search(r"\b(?:19|20)\d{2}\b", question):
        return _increment_first_year(question)
    replacements = (
        ("cuối kỳ", "đầu kỳ"),
        ("cuoi ky", "dau ky"),
        ("đầu năm", "cuối năm"),
        ("dau nam", "cuoi nam"),
        ("cuối năm", "đầu năm"),
        ("cuoi nam", "dau nam"),
        ("năm nay", "năm trước"),
        ("nam nay", "nam truoc"),
    )
    for old, new in replacements:
        mutated, count = re.subn(re.escape(old), new, question, count=1, flags=re.IGNORECASE)
        if count:
            return mutated
    raise ValueError(f"No explicit period to mutate: {question}")


def _source_surface(question: str, category: str) -> str:
    from hrm_finance_qa.ontology import normalize_text

    stripped = question.strip()
    if category == "SHORT_LONG_EQUIVALENT":
        return (
            "Căn cứ đúng bảng số liệu trong báo cáo tài chính và giữ nguyên kỳ, "
            "đơn vị cùng phạm vi, hãy xác định: " + stripped[0].lower() + stripped[1:]
        )
    if category == "DISCOURSE_PREFIX_SUFFIX":
        return "Theo số liệu trong cùng báo cáo, " + stripped[0].lower() + stripped[1:]
    if category == "CLAUSE_PERMUTATION":
        match = re.match(r"^Tổng (.+) tại (.+) là bao nhiêu\?$", stripped, re.IGNORECASE)
        if not match:
            raise ValueError(f"Unsupported ADD clause form: {question}")
        return f"Tại {match.group(2)}, tổng {match.group(1)} là bao nhiêu?"
    if category == "PUNCTUATION_VARIANT":
        return stripped.rstrip("?") + " ?"
    if category == "NO_DIACRITICS":
        return normalize_text(stripped).rstrip("?") + "?"
    if category == "BILINGUAL_PARENTHETICAL":
        return stripped.rstrip("?") + " (đơn vị theo nguồn / source unit)?"
    if category == "IRRELEVANT_DISTRACTOR_CLAUSE":
        return stripped.rstrip("?") + ", không yêu cầu phân tích nguyên nhân?"
    raise ValueError(category)


def _source_counterfactual(question: str, dimension: str) -> str:
    if dimension == "OPERAND_YEAR":
        return _source_period_counterfactual(question)
    if dimension == "ENTITY_SCOPE":
        return _insert_before_suffix(question, "của công ty mẹ")
    if dimension == "UNIT_SCALE":
        return _insert_before_suffix(question, "theo đơn vị triệu đồng")
    raise ValueError(dimension)


def _compiler_baseline(config: dict[str, Any]) -> dict[str, Any]:
    from hrm_finance_qa.question_planner import compile_question_plan
    from scripts.audit_finhrm_thesis_core_compiler import (
        _actual_contract_ir,
        _strict_checks,
    )
    from scripts.audit_question_plan_compiler import _actual_ir, _checks

    contract = _load_json(ROOT / config["parents"]["thesis_core"]["path"])
    catalog = _load_jsonl(ROOT / config["parents"]["question_family_catalog"]["path"])
    compiler = config["compiler_cohort"]
    core = set(compiler["core_answer_families"])
    catalog_core = core - {"IBDS01"}
    by_family_variant = {
        (str(row["family_id"]), str(row["variant"])): row
        for row in catalog
        if row.get("family_id") in catalog_core
    }
    families = {str(row["family_id"]): row for row in contract["families"]}

    canonical_details: list[dict[str, Any]] = []
    canonical_questions: dict[str, str] = {}
    canonical_plans: dict[str, Any] = {}
    for family_id in sorted(core):
        if family_id == "IBDS01":
            row = families[family_id]
            question = str(row["canonical_question"])
            plan = compile_question_plan(question)
            actual = _actual_contract_ir(plan, row["expected"]["root_operation"])
            checks = _strict_checks(row["expected"], actual)
        else:
            row = by_family_variant[(family_id, "canonical")]
            question = str(row["question"])
            plan = compile_question_plan(question)
            actual = _actual_ir(plan)
            checks = _checks(row["expected_ir"], actual)
        canonical_questions[family_id] = question
        canonical_plans[family_id] = plan
        canonical_details.append(
            {
                "family_id": family_id,
                "question": question,
                "exact": all(checks.values()),
                "failed_dimensions": sorted(key for key, value in checks.items() if not value),
            }
        )

    ambiguity_details: list[dict[str, Any]] = []
    for family_id in compiler["ambiguity_controls"]:
        row = families[family_id]
        plan = compile_question_plan(str(row["canonical_question"]))
        actual = _actual_contract_ir(plan, None)
        checks = _strict_checks(row["expected"], actual)
        ambiguity_details.append(
            {
                "family_id": family_id,
                "question": row["canonical_question"],
                "exact": all(checks.values()),
                "expected_reason": row["expected"]["abstention_reason"],
                "actual_reason": actual["abstention_reason"],
                "failed_dimensions": sorted(key for key, value in checks.items() if not value),
            }
        )

    surface_details: list[dict[str, Any]] = []
    catalog_category = {
        "long_clause": "SHORT_LONG_EQUIVALENT",
        "clause_reordered": "CLAUSE_PERMUTATION",
        "no_diacritics": "NO_DIACRITICS",
    }
    for family_id in sorted(catalog_core):
        base_projection = _semantic_projection(canonical_plans[family_id])
        for variant, category in catalog_category.items():
            row = by_family_variant[(family_id, variant)]
            plan = compile_question_plan(str(row["question"]))
            checks = _checks(row["expected_ir"], _actual_ir(plan))
            surface_details.append(
                {
                    "family_id": family_id,
                    "category": category,
                    "source": "frozen_catalog",
                    "expected_ir_exact": all(checks.values()),
                    "semantic_invariant": _semantic_projection(plan) == base_projection,
                    "question": row["question"],
                }
            )
        for category in compiler["generated_meaning_preserving_categories"]:
            question = _generated_surface(canonical_questions[family_id], category)
            plan = compile_question_plan(question)
            surface_details.append(
                {
                    "family_id": family_id,
                    "category": category,
                    "source": "deterministic_generated_question_only",
                    "expected_ir_exact": None,
                    "semantic_invariant": _semantic_projection(plan) == base_projection,
                    "question": question,
                }
            )

    ibds_surfaces = compiler["supplemental_ibds_surface_questions"]
    ibds_base = compile_question_plan(str(ibds_surfaces["canonical"]))
    ibds_projection = _semantic_projection(ibds_base)
    for category, question in ibds_surfaces.items():
        if category == "canonical":
            continue
        plan = compile_question_plan(str(question))
        surface_details.append(
            {
                "family_id": "IBDS01",
                "category": category,
                "source": "reviewed_supplemental_question_only",
                "expected_ir_exact": None,
                "semantic_invariant": _semantic_projection(plan) == ibds_projection,
                "question": question,
            }
        )

    pair_details: list[dict[str, Any]] = []
    for family_id in sorted(catalog_core):
        base_plan = canonical_plans[family_id]
        role_row = by_family_variant[(family_id, "minimal_pair_role_swap")]
        role_plan = compile_question_plan(str(role_row["question"]))
        pair_details.append(
            {
                "pair_id": f"{family_id}-catalog-role-swap",
                "family_id": family_id,
                "dimension": "COLUMN_ROLE",
                "source": "frozen_catalog",
                "detected": _dimension_projection(base_plan, "COLUMN_ROLE")
                != _dimension_projection(role_plan, "COLUMN_ROLE"),
                "base_question": canonical_questions[family_id],
                "mutated_question": role_row["question"],
            }
        )
    for family_id in sorted(core):
        base_question = canonical_questions[family_id]
        base_plan = canonical_plans[family_id]
        for dimension in compiler["generated_counterfactual_dimensions"]:
            question = _generated_counterfactual(base_question, dimension)
            mutated_plan = compile_question_plan(question)
            detected = _dimension_projection(base_plan, dimension) != _dimension_projection(
                mutated_plan, dimension
            )
            if dimension == "UNIT_SCALE":
                detected = detected and getattr(mutated_plan, "requested_scale", None) is not None
            pair_details.append(
                {
                    "pair_id": f"{family_id}-generated-{dimension.lower()}",
                    "family_id": family_id,
                    "dimension": dimension,
                    "source": "deterministic_generated_question_only",
                    "detected": detected,
                    "base_question": base_question,
                    "mutated_question": question,
                }
            )
    for pair in compiler["supplemental_semantic_pairs"]:
        base_plan = compile_question_plan(str(pair["base_question"]))
        mutated_plan = compile_question_plan(str(pair["mutated_question"]))
        dimension = str(pair["dimension"])
        pair_details.append(
            {
                "pair_id": pair["pair_id"],
                "family_id": pair["family_id"],
                "dimension": dimension,
                "source": "reviewed_supplemental_question_only",
                "detected": _dimension_projection(base_plan, dimension)
                != _dimension_projection(mutated_plan, dimension),
                "base_question": pair["base_question"],
                "mutated_question": pair["mutated_question"],
            }
        )

    surface_passed = sum(
        bool(row["semantic_invariant"])
        and row["expected_ir_exact"] is not False
        for row in surface_details
    )
    by_dimension: dict[str, dict[str, Any]] = {}
    dimensions = config["compiler_cohort"]["generated_counterfactual_dimensions"] + [
        "COLUMN_ROLE",
        "DIRECTION",
        "REFERENCE_BASE",
    ]
    for dimension in dimensions:
        rows = [row for row in pair_details if row["dimension"] == dimension]
        by_dimension[dimension] = {
            "pairs": len(rows),
            "detected": sum(bool(row["detected"]) for row in rows),
            "families": len({row["family_id"] for row in rows}),
            "detection_rate": (
                sum(bool(row["detected"]) for row in rows) / len(rows) if rows else 0.0
            ),
        }
    return {
        "canonical": {
            "exact": sum(bool(row["exact"]) for row in canonical_details),
            "total": len(canonical_details),
            "details": canonical_details,
        },
        "ambiguity": {
            "exact": sum(bool(row["exact"]) for row in ambiguity_details),
            "total": len(ambiguity_details),
            "details": ambiguity_details,
        },
        "surface": {
            "exact": surface_passed,
            "total": len(surface_details),
            "exact_rate": surface_passed / len(surface_details) if surface_details else 0.0,
            "by_category": _group_boolean(surface_details, "category", "semantic_invariant"),
            "details": surface_details,
        },
        "counterfactual": {
            "detected": sum(bool(row["detected"]) for row in pair_details),
            "total": len(pair_details),
            "detection_rate": (
                sum(bool(row["detected"]) for row in pair_details) / len(pair_details)
                if pair_details
                else 0.0
            ),
            "by_dimension": dict(sorted(by_dimension.items())),
            "details": pair_details,
        },
    }


def _group_boolean(
    rows: list[dict[str, Any]], group_key: str, boolean_key: str
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)
    return {
        key: {
            "total": len(values),
            "passed": sum(bool(value[boolean_key]) for value in values),
            "pass_rate": sum(bool(value[boolean_key]) for value in values) / len(values),
        }
        for key, values in sorted(grouped.items())
    }


def _physical_signature(item: Any) -> tuple[Any, ...]:
    if isinstance(item, dict):
        get = item.get
    else:
        get = lambda name, default=None: getattr(item, name, default)
    span = tuple(get("source_span", ()) or ())
    return (
        str(get("source_name", "")),
        int(get("page", 0) or 0),
        str(get("row_label", "")),
        str(get("column_label", "")),
        str(get("period", "")),
        span,
        str(get("value", "")),
        str(get("extraction_method", "")),
    )


def _parent_signature(row: dict[str, Any]) -> dict[str, Any]:
    observed = row["observed"]
    evidence = sorted(_physical_signature(item) for item in observed["evidence"])
    return {
        "action": "ANSWER",
        "operation": str(observed["operation"]),
        "value": str(observed["value"]),
        "unit": str(observed["unit"]),
        "evidence": [list(item) for item in evidence],
    }


def _answer_signature(answer: Any) -> dict[str, Any]:
    evidence = sorted(_physical_signature(item.candidate) for item in answer.evidence)
    return {
        "action": str(answer.source_action),
        "operation": str(answer.program.operation),
        "value": None if answer.value is None else str(answer.value),
        "unit": str(answer.unit),
        "evidence": [list(item) for item in evidence],
        "verification_state": str(answer.verification_state),
    }


def _source_anchors(config: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    from scripts.evaluate_qp3_answer_presentation_implementation import _eligible_records

    reviewed = _load_json(ROOT / config["parents"]["reviewed_real_pdf_cohort"]["path"])
    end_to_end = _load_json(ROOT / config["parents"]["frozen_real_pdf_end_to_end"]["path"])
    return _eligible_records(reviewed, end_to_end)


class _QueryTimeout:
    def __init__(self, seconds: int) -> None:
        self.seconds = seconds
        self.previous: Any = None

    def __enter__(self) -> None:
        def timeout_handler(signum: int, frame: Any) -> None:
            raise TimeoutError(f"Source-backed query exceeded {self.seconds} seconds")

        self.previous = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self.previous)


def run_source_backed(config: dict[str, Any], raw_path: Path) -> dict[str, Any]:
    from hrm_finance_qa.document_hrm import DocumentHRMBundle
    from hrm_finance_qa.document_reasoning_engine import DocumentHRMReasoningEngine

    anchors = _source_anchors(config)
    checkpoint = ROOT / config["checkpoint"]["path"]
    bundle = DocumentHRMBundle.load(checkpoint, device="cpu")
    engine = DocumentHRMReasoningEngine(bundle)
    surface_results: list[dict[str, Any]] = []
    counterfactual_results: list[dict[str, Any]] = []
    categories = config["source_backed_cohort"]["meaning_preserving_categories"]
    dimensions = config["source_backed_cohort"]["applicable_counterfactual_dimensions"]
    timeout_seconds = int(config["source_backed_cohort"]["per_query_timeout_seconds"])
    total = len(anchors) * (len(categories) + len(dimensions))
    position = 0
    for _, row in anchors:
        source_path = ROOT / row["source_path"]
        parent = _parent_signature(row)
        common = {
            "example_id": row["example_id"],
            "company_key": row["company_key"],
            "source_key": row["source_sha256"],
            "source_path": row["source_path"],
            "layout_family": row["layout_family"],
            "parent_question": row["question"],
            "parent_signature": parent,
        }
        for category in categories:
            position += 1
            question = _source_surface(str(row["question"]), str(category))
            print(f"[{position}/{total}] surface {category} {row['example_id']}", flush=True)
            try:
                with _QueryTimeout(timeout_seconds):
                    answer = engine.answer(
                        question,
                        source_path,
                        mineru_auto_generate=False,
                    )
                signature = _answer_signature(answer)
                invariant = {
                    key: signature.get(key) == parent.get(key)
                    for key in ("action", "operation", "value", "unit", "evidence")
                }
                surface_results.append(
                    common
                    | {
                        "category": category,
                        "question": question,
                        "observed_signature": signature,
                        "checks": invariant,
                        "invariant": all(invariant.values()),
                    }
                )
            except Exception as exc:
                surface_results.append(
                    common
                    | {
                        "category": category,
                        "question": question,
                        "invariant": False,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    }
                )
        for dimension in dimensions:
            position += 1
            question = _source_counterfactual(str(row["question"]), str(dimension))
            print(f"[{position}/{total}] counterfactual {dimension} {row['example_id']}", flush=True)
            try:
                with _QueryTimeout(timeout_seconds):
                    answer = engine.answer(
                        question,
                        source_path,
                        mineru_auto_generate=False,
                    )
                signature = _answer_signature(answer)
                same_old_winner = all(
                    signature.get(key) == parent.get(key)
                    for key in ("action", "operation", "value", "unit", "evidence")
                )
                counterfactual_results.append(
                    common
                    | {
                        "dimension": dimension,
                        "question": question,
                        "observed_signature": signature,
                        "old_winner_false_accept": same_old_winner,
                    }
                )
            except Exception as exc:
                counterfactual_results.append(
                    common
                    | {
                        "dimension": dimension,
                        "question": question,
                        "old_winner_false_accept": False,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    }
                )
    raw = {
        "schema": "finhrm-qp4-source-backed-baseline-v1",
        "created_at": config["created_at"],
        "classification": "development_only_real_pdf_qp4_baseline_runtime_unchanged",
        "inputs": {
            "cohort_config": DEFAULT_CONFIG.relative_to(ROOT).as_posix(),
            "cohort_config_sha256": _digest(DEFAULT_CONFIG),
            "checkpoint_sha256": _digest(checkpoint),
        },
        "eligible_anchors": len(anchors),
        "companies": len({row["company_key"] for _, row in anchors}),
        "source_documents": len({row["source_sha256"] for _, row in anchors}),
        "real_layout_families": len({row["layout_family"] for _, row in anchors}),
        "surface_results": surface_results,
        "counterfactual_results": counterfactual_results,
        "runtime_hashes": {
            relative: _digest(ROOT / relative) for relative in config["runtime_hashes"]
        },
        "invariants": config["invariants"],
    }
    raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return raw


def _source_summary(config: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    source_contract = config["source_backed_cohort"]
    surface = raw["surface_results"]
    counterfactual = raw["counterfactual_results"]
    surface_pass = sum(bool(row["invariant"]) for row in surface)
    by_dimension: dict[str, dict[str, Any]] = {}
    for dimension in source_contract["applicable_counterfactual_dimensions"]:
        rows = [row for row in counterfactual if row["dimension"] == dimension]
        by_dimension[dimension] = {
            "runs": len(rows),
            "companies": len({row["company_key"] for row in rows}),
            "sources": len({row["source_key"] for row in rows}),
            "real_layouts": len({row["layout_family"] for row in rows}),
            "old_winner_false_accepts": sum(
                bool(row["old_winner_false_accept"]) for row in rows
            ),
            "coverage_gate": (
                len(rows) >= source_contract["minimum_positive_anchors_per_applicable_dimension"]
                and len({row["company_key"] for row in rows})
                >= source_contract["minimum_distinct_companies_per_applicable_dimension"]
                and len({row["source_key"] for row in rows})
                >= source_contract["minimum_distinct_sources_per_applicable_dimension"]
                and len({row["layout_family"] for row in rows})
                >= source_contract["minimum_distinct_real_layouts_per_applicable_dimension"]
            ),
        }
    unconfirmed = {
        dimension: {"status": "NOT_CONFIRMED", "reason": reason}
        for dimension, reason in source_contract[
            "not_applicable_or_unconfirmed_dimensions"
        ].items()
    }
    return {
        "eligible_anchors": raw["eligible_anchors"],
        "companies": raw["companies"],
        "source_documents": raw["source_documents"],
        "real_layout_families": raw["real_layout_families"],
        "surface": {
            "passed": surface_pass,
            "total": len(surface),
            "invariance_rate": surface_pass / len(surface) if surface else 0.0,
            "by_category": _group_boolean(surface, "category", "invariant"),
        },
        "counterfactual": {
            "old_winner_false_accepts": sum(
                bool(row["old_winner_false_accept"]) for row in counterfactual
            ),
            "runs": len(counterfactual),
            "by_dimension": by_dimension | unconfirmed,
        },
    }


def _diagnosis(compiler: dict[str, Any], source: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for category, values in compiler["surface"]["by_category"].items():
        failures = values["total"] - values["passed"]
        if failures:
            families = {
                row["family_id"]
                for row in compiler["surface"]["details"]
                if row["category"] == category and not row["semantic_invariant"]
            }
            output.append(
                {
                    "layer": "question_planner",
                    "failure_group": f"compiler_surface:{category}",
                    "failures": failures,
                    "family_groups": len(families),
                    "generic_evidence": len(families) >= 3,
                }
            )
    for dimension, values in compiler["counterfactual"]["by_dimension"].items():
        failures = values["pairs"] - values["detected"]
        if failures:
            output.append(
                {
                    "layer": "question_planner",
                    "failure_group": f"compiler_counterfactual:{dimension}",
                    "failures": failures,
                    "family_groups": values["families"],
                    "generic_evidence": values["families"] >= 3,
                }
            )
    for category, values in source["surface"]["by_category"].items():
        failures = values["total"] - values["passed"]
        if failures:
            rows = [
                row
                for row in source.get("raw_surface_details", [])
                if row["category"] == category and not row["invariant"]
            ]
            companies = len({row["company_key"] for row in rows})
            sources = len({row["source_key"] for row in rows})
            layouts = len({row["layout_family"] for row in rows})
            compiler_category = compiler["surface"]["by_category"].get(category, {})
            compiler_exact = compiler_category.get("pass_rate") == 1.0
            output.append(
                {
                    "layer": (
                        "retrieval_or_binding_isolation_required"
                        if compiler_exact
                        else "question_planner_then_end_to_end_replay"
                    ),
                    "failure_group": f"source_surface:{category}",
                    "failures": failures,
                    "company_groups": companies,
                    "source_groups": sources,
                    "real_layout_groups": layouts,
                    "generic_evidence": companies >= 3 and sources >= 3 and layouts >= 3,
                    "note": (
                        "Compiler projection passes; retrieval/binding must be isolated and is outside the automatic QP4 whitelist."
                        if compiler_exact
                        else "Repair compiler projection first, then replay end-to-end before attributing any residual."
                    ),
                }
            )
    compiler_dimensions = compiler["counterfactual"]["by_dimension"]
    for dimension, values in source["counterfactual"]["by_dimension"].items():
        if "old_winner_false_accepts" not in values or not values["old_winner_false_accepts"]:
            continue
        rows = [
            row
            for row in source.get("raw_counterfactual_details", [])
            if row["dimension"] == dimension and row["old_winner_false_accept"]
        ]
        companies = len({row["company_key"] for row in rows})
        sources = len({row["source_key"] for row in rows})
        layouts = len({row["layout_family"] for row in rows})
        compiler_detected = compiler_dimensions.get(dimension, {}).get("detection_rate") == 1.0
        output.append(
            {
                "layer": (
                    "symbolic_verifier_or_synthesis_isolation_required"
                    if compiler_detected
                    else "question_planner"
                ),
                "failure_group": f"source_counterfactual:{dimension}",
                "failures": values["old_winner_false_accepts"],
                "company_groups": companies,
                "source_groups": sources,
                "real_layout_groups": layouts,
                "generic_evidence": companies >= 3 and sources >= 3 and layouts >= 3,
                "note": (
                    "Compiler did not represent the mutated semantic dimension; repair the typed intent before testing verifier residuals."
                    if not compiler_detected
                    else "Compiler detected the mutation; isolate binding and proof state before selecting a runtime layer."
                ),
            }
        )
    return output


def build_report(config_path: Path = DEFAULT_CONFIG, raw_path: Path = DEFAULT_RAW) -> dict[str, Any]:
    config = _load_json(config_path)
    _verify_inputs(config)
    raw = _load_json(raw_path)
    if raw.get("schema") != "finhrm-qp4-source-backed-baseline-v1":
        raise ValueError("Unexpected source-backed raw schema")
    if raw.get("inputs", {}).get("cohort_config_sha256") != _digest(config_path):
        raise ValueError("Source-backed baseline does not match cohort config")
    if raw.get("runtime_hashes") != {
        relative: _digest(ROOT / relative) for relative in config["runtime_hashes"]
    }:
        raise ValueError("Source-backed baseline runtime hash mismatch")
    compiler = _compiler_baseline(config)
    source = _source_summary(config, raw)
    source["raw_surface_details"] = raw["surface_results"]
    source["raw_counterfactual_details"] = raw["counterfactual_results"]
    diagnosis = _diagnosis(compiler, source)
    source.pop("raw_surface_details")
    source.pop("raw_counterfactual_details")

    gates = config["baseline_gates"]
    failed: list[str] = []
    if f"{compiler['canonical']['exact']}/{compiler['canonical']['total']}" != gates[
        "compiler_core_family_coverage"
    ]:
        failed.append("compiler_core_family_coverage")
    if f"{compiler['ambiguity']['exact']}/{compiler['ambiguity']['total']}" != gates[
        "compiler_ambiguity_controls"
    ]:
        failed.append("compiler_ambiguity_controls")
    if compiler["surface"]["exact_rate"] != gates[
        "compiler_surface_semantic_invariance_rate"
    ]:
        failed.append("compiler_surface_semantic_invariance_rate")
    if compiler["counterfactual"]["detection_rate"] != gates[
        "compiler_counterfactual_detection_rate"
    ]:
        failed.append("compiler_counterfactual_detection_rate")
    if source["surface"]["invariance_rate"] != gates[
        "source_surface_winner_invariance_rate"
    ]:
        failed.append("source_surface_winner_invariance_rate")
    if source["counterfactual"]["old_winner_false_accepts"] != gates[
        "source_counterfactual_old_winner_false_accepts"
    ]:
        failed.append("source_counterfactual_old_winner_false_accepts")
    for dimension in config["source_backed_cohort"]["applicable_counterfactual_dimensions"]:
        if not source["counterfactual"]["by_dimension"][dimension]["coverage_gate"]:
            failed.append(f"source_coverage:{dimension}")
    unconfirmed = sorted(
        config["source_backed_cohort"]["not_applicable_or_unconfirmed_dimensions"]
    )
    runtime_edit_candidates = [
        row for row in diagnosis if row["generic_evidence"] and row["layer"] == "question_planner"
    ]
    return {
        "schema": "finhrm-qp4-development-baseline-v1",
        "created_at": config["created_at"],
        "classification": config["classification"],
        "status": "BASELINE_COMPLETE_RUNTIME_UNCHANGED",
        "cohort_and_baseline_complete": True,
        "baseline_gates_passed": not failed and not unconfirmed,
        "failed_gates": sorted(set(failed)),
        "source_dimensions_not_confirmed": unconfirmed,
        "runtime_changed": False,
        "compiler": compiler,
        "source_backed": source,
        "diagnosis": diagnosis,
        "generic_runtime_edit_candidate_groups": runtime_edit_candidates,
        "frozen_safe_replay": {
            "actions": gates["frozen_safe_replay_actions"],
            "execution": gates["frozen_safe_replay_execution"],
            "permutation": gates["frozen_safe_replay_permutation"],
            "counterfactual_false_accepts": gates[
                "frozen_safe_replay_counterfactual_false_accepts"
            ],
        },
        "runtime_hashes": {
            relative: _digest(ROOT / relative) for relative in config["runtime_hashes"]
        },
        "checkpoint_sha256": _digest(ROOT / config["checkpoint"]["path"]),
        "invariants": config["invariants"],
        "qp4_runtime_change_authorized": False,
        "qp5_authorized": False,
        "phase_d": "locked",
        "authorized_next_step": (
            "Review the frozen QP4 baseline diagnosis. Build independently reviewed source-backed "
            "ACTUAL/PLAN ratio/direction anchors for COLUMN_ROLE, DIRECTION and REFERENCE_BASE; "
            "then pre-authorize only generic failure groups that meet the exact QP4 whitelist. "
            "No runtime change, QP5 or Phase D is authorized by this baseline."
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    compiler = report["compiler"]
    source = report["source_backed"]
    lines = [
        "# QP4 Development Cohort and Baseline v1",
        "",
        f"- Status: **{report['status']}**",
        "- Runtime changed: **no**",
        f"- Compiler canonical: `{compiler['canonical']['exact']}/{compiler['canonical']['total']}`",
        f"- Ambiguity controls: `{compiler['ambiguity']['exact']}/{compiler['ambiguity']['total']}`",
        f"- Compiler surface invariance: `{compiler['surface']['exact']}/{compiler['surface']['total']}`",
        f"- Compiler counterfactual detection: `{compiler['counterfactual']['detected']}/{compiler['counterfactual']['total']}`",
        f"- Source-backed anchors: `{source['eligible_anchors']}` companies `{source['companies']}` / sources `{source['source_documents']}` / real layouts `{source['real_layout_families']}`",
        f"- Source-backed surface invariance: `{source['surface']['passed']}/{source['surface']['total']}`",
        f"- Source-backed old-winner counterfactual false accepts: `{source['counterfactual']['old_winner_false_accepts']}/{source['counterfactual']['runs']}`",
        f"- Failed baseline gates: `{report['failed_gates']}`",
        f"- Source dimensions not confirmed: `{report['source_dimensions_not_confirmed']}`",
        "- Safe abstention is not counted as answered; synthetic questions are not layout confirmation.",
        "- QP4 runtime, QP5 and Phase D remain locked.",
        "",
        "## Generic diagnostic groups",
        "",
    ]
    if report["diagnosis"]:
        for row in report["diagnosis"]:
            lines.append(
                f"- `{row['failure_group']}`: failures `{row['failures']}`, layer `{row['layer']}`, generic evidence `{row['generic_evidence']}`."
            )
    else:
        lines.append("- None.")
    lines.extend(["", f"Next: {report['authorized_next_step']}", ""])
    return "\n".join(lines)


def write_outputs(report: dict[str, Any]) -> None:
    DEFAULT_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    DEFAULT_MARKDOWN.write_text(_markdown(report), encoding="utf-8")
    artifacts = [
        DEFAULT_CONFIG,
        Path(__file__).resolve(),
        FUNCTIONAL_TEST,
        REPORT_TEST,
        DEFAULT_RAW,
        DEFAULT_REPORT,
        DEFAULT_MARKDOWN,
    ]
    missing = [path for path in artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(str(path) for path in missing))
    lock = {
        "schema": "finhrm-qp4-development-baseline-lock-v1",
        "created_at": report["created_at"],
        "classification": report["classification"],
        "status": "complete",
        "cohort_and_baseline_complete": report["cohort_and_baseline_complete"],
        "baseline_gates_passed": report["baseline_gates_passed"],
        "failed_gates": report["failed_gates"],
        "source_dimensions_not_confirmed": report["source_dimensions_not_confirmed"],
        "runtime_changed": False,
        "runtime_hashes": report["runtime_hashes"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "artifacts": [_artifact(path) for path in artifacts],
        "qp4_runtime_change_authorized": False,
        "qp5_authorized": False,
        "phase_d": "locked",
        "authorized_next_step": report["authorized_next_step"],
    }
    DEFAULT_LOCK.write_text(
        json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--run-source-backed", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    config = _load_json(args.config)
    _verify_inputs(config)
    if args.run_source_backed:
        run_source_backed(config, args.raw)
    report = build_report(args.config, args.raw)
    if args.write:
        write_outputs(report)
    print(
        json.dumps(
            {
                "cohort_and_baseline_complete": report["cohort_and_baseline_complete"],
                "baseline_gates_passed": report["baseline_gates_passed"],
                "failed_gates": report["failed_gates"],
                "source_dimensions_not_confirmed": report[
                    "source_dimensions_not_confirmed"
                ],
                "compiler_surface": (
                    f"{report['compiler']['surface']['exact']}/"
                    f"{report['compiler']['surface']['total']}"
                ),
                "compiler_counterfactual": (
                    f"{report['compiler']['counterfactual']['detected']}/"
                    f"{report['compiler']['counterfactual']['total']}"
                ),
                "source_surface": (
                    f"{report['source_backed']['surface']['passed']}/"
                    f"{report['source_backed']['surface']['total']}"
                ),
                "source_counterfactual_false_accepts": report["source_backed"][
                    "counterfactual"
                ]["old_winner_false_accepts"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
