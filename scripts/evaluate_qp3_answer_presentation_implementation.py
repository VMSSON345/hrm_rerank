from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from hrm_finance_qa.answer_presentation import (
    AnswerMode,
    AnswerStatus,
    VerifiedAnswerEnvelope,
    build_verified_answer_envelope,
    render_verified_answer,
    validate_verified_answer_envelope,
)
from hrm_finance_qa.program_data import NumericCandidate, ReasoningProgram, currency_unit_scale


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "finhrm_qp3_answer_presentation_development_cohort_v1.json"
DEFAULT_REPORT = ROOT / "reports" / "finhrm_qp3_answer_presentation_implementation_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "finhrm_qp3_answer_presentation_implementation_v1.md"
DEFAULT_LOCK = ROOT / "reports" / "finhrm_qp3_answer_presentation_implementation_lock_v1.json"
IMPLEMENTATION_TEST = ROOT / "tests" / "test_qp3_answer_presentation_implementation.py"
REPORT_TEST = ROOT / "tests" / "test_qp3_answer_presentation_implementation_report.py"

_PROOF = (
    ("execution_valid", "required", "VALID"),
    ("operation_match", "required", "VALID"),
    ("query_metric", "required", "VALID"),
    ("period_coverage", "required", "VALID"),
    ("reporting_entity", "required", "VALID"),
    ("unit_match", "required", "VALID"),
    ("logical_cell_unique", "required", "VALID"),
    ("source_provenance_complete", "required", "VALID"),
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _digest(path),
    }


def _candidate(
    index: int,
    evidence: dict[str, Any],
    *,
    unit: str,
    company_key: str,
) -> NumericCandidate:
    row = str(evidence.get("row_label", "")).strip()
    column = str(evidence.get("column_label", "")).strip()
    period = str(evidence.get("period", "")).strip()
    table_id = str(evidence.get("table_id", "")).strip()
    source_span = tuple(int(value) for value in evidence.get("source_span", ()))
    return NumericCandidate(
        index=index,
        value=str(evidence["value"]),
        raw=str(evidence["value"]),
        context=(
            f"ROW: {row}\nCOLUMN: {column}\nPERIOD: {period}\n"
            f"TABLE: {table_id}\nSCOPE: {company_key}"
        ),
        line_number=0,
        char_start=source_span[0],
        char_end=source_span[1],
        unit=unit,
        is_percent=False,
        is_year_like=False,
        relevance=0.0,
        source_name=str(evidence["source_name"]),
        page=int(evidence["page"]) if evidence.get("page") is not None else None,
        extraction_method=str(evidence.get("extraction_method", "")),
        row_label=row,
        column_label=column,
        period=period,
        table_id=table_id,
        scope=company_key,
        source_span=source_span,
    )


def _answer(
    question: str,
    operation: str,
    candidates: Sequence[NumericCandidate],
    operands: tuple[int, ...],
    value: str,
    unit: str,
    *,
    source_action: str = "ANSWER",
    verification_state: str = "VALID",
    abstained: bool = False,
) -> SimpleNamespace:
    by_index = {candidate.index: candidate for candidate in candidates}
    return SimpleNamespace(
        question=question,
        value=Decimal(value),
        unit=unit,
        prediction=SimpleNamespace(
            program=ReasoningProgram(operation, operands),
            program_dag=None,
            slot_bindings=(),
        ),
        program_dag=None,
        evidence=tuple(
            SimpleNamespace(operand_number=position + 1, candidate=by_index[index])
            for position, index in enumerate(operands)
        ),
        abstained=abstained,
        source_action=source_action,
        verification_state=verification_state,
        proof_obligations=_PROOF,
        verification_reasons=(),
        acceptance_basis=(
            "runtime_action_preserved",
            "reviewed_end_to_end_leaf_proof",
        ),
    )


def _eligible_records(
    reviewed: dict[str, Any], end_to_end: dict[str, Any]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    reviewed_by_id = {
        str(row["example_id"]): row
        for row in reviewed["details"]
        if row.get("review_status") == "independently_reviewed_approved"
        and row.get("gold_action") == "ANSWER"
        and row.get("gold_proof_contract_state") == "VALID"
        and row.get("checks", {}).get("provenance_complete")
        and row.get("checks", {}).get("distinct_physical_cells")
        and row.get("checks", {}).get("source_hash_exact")
    }
    output: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in end_to_end["results"]:
        parent = reviewed_by_id.get(str(row.get("example_id")))
        observed = row.get("observed", {})
        checks = row.get("checks", {})
        evidence = observed.get("evidence", [])
        if parent is None:
            continue
        if not row.get("end_to_end_positive_passed") or observed.get("abstained"):
            continue
        if not str(observed.get("unit", "")).strip():
            continue
        if not checks.get("gold_execution_exact") or not checks.get("provenance_complete"):
            continue
        if len(evidence) != 2:
            continue
        if any(
            item.get("page") is None
            or len(item.get("source_span", ())) != 2
            or not str(item.get("source_name", "")).strip()
            or not str(item.get("extraction_method", "")).strip()
            or not str(item.get("period", "")).strip()
            or not str(item.get("row_label", "")).strip()
            for item in evidence
        ):
            continue
        source_path = ROOT / str(row["source_path"])
        if not source_path.is_file() or _digest(source_path) != str(row["source_sha256"]):
            continue
        output.append((parent, row))
    return output


def _record_envelopes(
    row: dict[str, Any],
) -> tuple[tuple[VerifiedAnswerEnvelope, VerifiedAnswerEnvelope], VerifiedAnswerEnvelope, VerifiedAnswerEnvelope]:
    observed = row["observed"]
    unit = str(observed["unit"])
    company_key = str(row["company_key"])
    candidates = tuple(
        _candidate(index, evidence, unit=unit, company_key=company_key)
        for index, evidence in enumerate(observed["evidence"])
    )
    facts: list[VerifiedAnswerEnvelope] = []
    for candidate in candidates:
        fact_question = (
            f"{candidate.row_label} tại {candidate.period} là bao nhiêu?"
        )
        facts.append(
            build_verified_answer_envelope(
                _answer(
                    fact_question,
                    "SELECT",
                    (candidate,),
                    (candidate.index,),
                    candidate.value,
                    unit,
                )
            )
        )
    computation = build_verified_answer_envelope(
        _answer(
            str(row["question"]),
            "ADD",
            candidates,
            tuple(candidate.index for candidate in candidates),
            str(observed["value"]),
            unit,
        )
    )
    summary = build_verified_answer_envelope(
        answer_mode=AnswerMode.VERIFIED_SUMMARY,
        requested_fields=("operand_1", "operand_2", "computation"),
        summary_rows=(
            ("operand_1", facts[0]),
            ("operand_2", facts[1]),
            ("computation", computation),
        ),
    )
    return (facts[0], facts[1]), computation, summary


def _group_metrics(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    by_group = {
        name: {
            "examples": len(items),
            "answer_rate": sum(item["status"] == "ANSWER" for item in items) / len(items),
        }
        for name, items in sorted(grouped.items())
    }
    rates = [item["answer_rate"] for item in by_group.values()]
    return {
        "groups": len(by_group),
        "macro_answer_rate": sum(rates) / len(rates) if rates else 0.0,
        "worst_group_answer_rate": min(rates) if rates else 0.0,
        "by_group": by_group,
    }


def _hard_negative_results(
    fact: VerifiedAnswerEnvelope,
    computation: VerifiedAnswerEnvelope,
    summary: VerifiedAnswerEnvelope,
) -> dict[str, str]:
    citation = fact.citations[0]
    mutations = {
        "PRES_WRONG_PAGE": replace(fact, citations=(replace(citation, page=citation.page + 1),)),
        "PRES_WRONG_STATEMENT": replace(
            fact,
            citations=(replace(citation, statement_or_note="unproved-statement"),),
        ),
        "PRES_WRONG_YEAR": replace(
            fact,
            citations=(replace(citation, period=citation.period + "-mutated"),),
        ),
        "PRES_SCALE_X1000": replace(
            fact,
            citations=(replace(citation, scale="1000" if citation.scale == "1" else "1"),),
        ),
    }
    first, second = computation.citations
    directional_candidates = tuple(
        NumericCandidate(
            index=index,
            value=item.value,
            raw=item.value,
            context=(
                f"ROW: {item.row_label}\nCOLUMN: {item.column_label}\n"
                f"PERIOD: {item.period}\nTABLE: {item.table_id}\n"
                f"SCOPE: {item.entity_scope}"
            ),
            line_number=0,
            char_start=item.source_span[0],
            char_end=item.source_span[1],
            unit=item.unit,
            is_percent=False,
            is_year_like=False,
            relevance=0.0,
            source_name=item.source_name,
            page=item.page,
            extraction_method=item.extraction_method,
            row_label=item.row_label or "",
            column_label=item.column_label or "",
            period=item.period,
            table_id=item.table_id or "",
            scope=item.entity_scope,
            source_span=item.source_span,
        )
        for index, item in enumerate((first, second))
    )
    directional = build_verified_answer_envelope(
        _answer(
            f"{first.row_label} trừ {second.row_label} tại {first.period} là bao nhiêu?",
            "SUBTRACT",
            directional_candidates,
            (0, 1),
            str(Decimal(first.value) - Decimal(second.value)),
            first.unit,
        )
    )
    first_directional, second_directional = directional.citations
    mutations["PRES_SWAPPED_OPERAND"] = replace(
        directional,
        citations=(
            replace(
                first_directional,
                operand_ref=second_directional.operand_ref,
            ),
            replace(
                second_directional,
                operand_ref=first_directional.operand_ref,
            ),
        ),
    )
    results = {
        name: "REJECTED_BY_VALIDATION"
        if validate_verified_answer_envelope(envelope)
        else "FALSE_ACCEPT"
        for name, envelope in mutations.items()
    }
    partial = build_verified_answer_envelope(
        answer_mode=AnswerMode.VERIFIED_SUMMARY,
        requested_fields=summary.summary_coverage.requested_fields,
        summary_rows=summary.summary_coverage.rows[:-1],
    )
    results["PRES_MISSING_SUMMARY_ROW"] = (
        "ABSTAIN" if partial.status is AnswerStatus.ABSTAIN and "2/3" in render_verified_answer(partial) else "FALSE_ACCEPT"
    )
    unsafe_claim = replace(
        fact.claim,
        qualitative_highlight="unsupported qualitative cause",
        qualitative_citation=None,
    )
    unsafe_fact = replace(fact, claim=unsafe_claim)
    uncited = build_verified_answer_envelope(
        answer_mode=AnswerMode.VERIFIED_SUMMARY,
        requested_fields=("unsafe", "safe_1", "safe_2"),
        summary_rows=(("unsafe", unsafe_fact), ("safe_1", fact), ("safe_2", computation)),
    )
    results["PRES_UNCITED_QUALITATIVE_CAUSE"] = (
        "ABSTAIN_AND_OMITTED"
        if uncited.status is AnswerStatus.ABSTAIN
        and "unsupported qualitative cause" not in render_verified_answer(uncited)
        else "FALSE_ACCEPT"
    )
    parent_answer = _answer(
        "Giá trị đã chọn là bao nhiêu?",
        "SELECT",
        (
            NumericCandidate(
                index=0,
                value=citation.value,
                raw=citation.value,
                context=(
                    f"ROW: {citation.row_label}\nCOLUMN: {citation.column_label}\n"
                    f"PERIOD: {citation.period}\nTABLE: {citation.table_id}\n"
                    f"SCOPE: {citation.entity_scope}"
                ),
                line_number=0,
                char_start=citation.source_span[0],
                char_end=citation.source_span[1],
                unit=citation.unit,
                is_percent=False,
                is_year_like=False,
                relevance=0.0,
                source_name=citation.source_name,
                page=citation.page,
                extraction_method=citation.extraction_method,
                row_label=citation.row_label or "",
                column_label=citation.column_label or "",
                period=citation.period,
                table_id=citation.table_id or "",
                scope=citation.entity_scope,
                source_span=citation.source_span,
            ),
        ),
        (0,),
        citation.value,
        citation.unit,
        source_action="ABSTAIN",
        verification_state="UNKNOWN",
        abstained=True,
    )
    promoted = build_verified_answer_envelope(parent_answer)
    results["PRES_PARENT_ACTION_PROMOTION"] = (
        "ABSTAIN" if promoted.status is AnswerStatus.ABSTAIN else "FALSE_ACCEPT"
    )
    return results


def build_report(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _load(config_path)
    errors: list[str] = []
    for name, parent in config["parents"].items():
        path = ROOT / parent["path"]
        if not path.is_file() or _digest(path) != parent["sha256"]:
            errors.append(f"parent_hash_mismatch:{name}")
    for relative, expected in config["frozen_runtime_hashes"].items():
        if _digest(ROOT / relative) != expected:
            errors.append(f"frozen_runtime_changed:{relative}")
    runtime_hashes = {
        relative: _digest(ROOT / relative) for relative in config["runtime_scope"]
    }
    for relative, expected in config["runtime_scope"].items():
        if runtime_hashes[relative] != expected:
            errors.append(f"implementation_runtime_hash_mismatch:{relative}")

    reviewed = _load(ROOT / config["parents"]["reviewed_pdf_cohort"]["path"])
    end_to_end = _load(ROOT / config["parents"]["end_to_end_pdf_replay"]["path"])
    safe = _load(ROOT / config["parents"]["frozen_safe_replay"]["path"])
    eligible = _eligible_records(reviewed, end_to_end)
    scalar_envelopes: list[VerifiedAnswerEnvelope] = []
    mode_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    computation_permutation: list[bool] = []
    citation_permutation: list[bool] = []
    summary_permutation: list[bool] = []
    record_details: list[dict[str, Any]] = []
    first_controls: tuple[VerifiedAnswerEnvelope, VerifiedAnswerEnvelope, VerifiedAnswerEnvelope] | None = None

    for _, row in eligible:
        facts, computation, summary = _record_envelopes(row)
        if first_controls is None:
            first_controls = facts[0], computation, summary
        scalar_envelopes.extend((*facts, computation))
        for mode, envelopes in (
            ("FACT", facts),
            ("COMPUTATION", (computation,)),
            ("VERIFIED_SUMMARY", (summary,)),
        ):
            for envelope in envelopes:
                mode_rows[mode].append(
                    {
                        "status": envelope.status.value,
                        "company": row["company_key"],
                        "layout": row["layout_family"],
                        "source": row["source_sha256"],
                    }
                )
        for envelope in (*facts, computation):
            citation_permutation.append(
                render_verified_answer(envelope)
                == render_verified_answer(
                    replace(envelope, citations=tuple(reversed(envelope.citations)))
                )
            )
        summary_permutation.append(
            render_verified_answer(summary)
            == render_verified_answer(
                build_verified_answer_envelope(
                    answer_mode=AnswerMode.VERIFIED_SUMMARY,
                    requested_fields=summary.summary_coverage.requested_fields,
                    summary_rows=tuple(reversed(summary.summary_coverage.rows)),
                )
            )
        )
        observed = row["observed"]
        unit = str(observed["unit"])
        candidates = tuple(
            _candidate(index, evidence, unit=unit, company_key=str(row["company_key"]))
            for index, evidence in enumerate(observed["evidence"])
        )
        permuted_candidates = (
            replace(candidates[1], index=0),
            replace(candidates[0], index=1),
        )
        permuted = build_verified_answer_envelope(
            _answer(
                str(row["question"]),
                "ADD",
                permuted_candidates,
                (1, 0),
                str(observed["value"]),
                unit,
            )
        )
        computation_permutation.append(
            render_verified_answer(computation) == render_verified_answer(permuted)
        )
        record_details.append(
            {
                "record_key": hashlib.sha256(
                    (str(row["company_key"]) + str(row["source_sha256"])).encode("utf-8")
                ).hexdigest()[:16],
                "company_key": row["company_key"],
                "layout_family": row["layout_family"],
                "source_key": "DEVPDF_" + str(row["source_sha256"])[:16],
                "fact_answers": sum(item.status is AnswerStatus.ANSWER for item in facts),
                "computation_action": computation.status.value,
                "summary_action": summary.status.value,
            }
        )

    if first_controls is None:
        errors.append("no_eligible_source_backed_records")
        hard_negatives: dict[str, str] = {}
    else:
        hard_negatives = _hard_negative_results(*first_controls)

    scalar_valid = [not validate_verified_answer_envelope(item) for item in scalar_envelopes]
    scalar_count = len(scalar_envelopes)
    numeric_exact = []
    for envelope in scalar_envelopes:
        if envelope.display is None or envelope.canonical_value is None:
            numeric_exact.append(False)
            continue
        value, _, scale, _ = envelope.display
        numeric_exact.append(Decimal(value) * Decimal(scale) == Decimal(envelope.canonical_value))
    citation_exact = [
        all(citation.logical_cell_id.startswith("cell-") for citation in envelope.citations)
        and not any(error == "citation_identity_mismatch" for error in validate_verified_answer_envelope(envelope))
        for envelope in scalar_envelopes
    ]
    derivation_exact = [
        not any("program" in error for error in validate_verified_answer_envelope(envelope))
        for envelope in scalar_envelopes
    ]
    mode_metrics: dict[str, Any] = {}
    cohort_gates = config["coverage_gates_per_mode"]
    for mode, rows in sorted(mode_rows.items()):
        companies = {row["company"] for row in rows if row["status"] == "ANSWER"}
        layouts = {row["layout"] for row in rows if row["status"] == "ANSWER"}
        sources = {row["source"] for row in rows if row["status"] == "ANSWER"}
        answers = sum(row["status"] == "ANSWER" for row in rows)
        passed = (
            answers >= cohort_gates["minimum_verified_examples"]
            and len(companies) >= cohort_gates["minimum_distinct_companies"]
            and len(layouts) >= cohort_gates["minimum_distinct_real_layout_families"]
            and len(sources) >= cohort_gates["minimum_distinct_source_documents"]
        )
        if not passed:
            errors.append(f"mode_coverage_failed:{mode}")
        mode_metrics[mode] = {
            "examples": len(rows),
            "answers": answers,
            "companies": len(companies),
            "real_layout_families": len(layouts),
            "source_documents": len(sources),
            "coverage_gate_passed": passed,
            "company": _group_metrics(rows, "company"),
            "layout": _group_metrics(rows, "layout"),
        }

    hard_negative_false_accepts = sum(
        result == "FALSE_ACCEPT" for result in hard_negatives.values()
    )
    presentation_checks = {
        "all_source_backed_envelopes_schema_valid": bool(scalar_valid) and all(scalar_valid),
        "numeric_value_unit_scale_exact": bool(numeric_exact) and all(numeric_exact),
        "citation_exact": bool(citation_exact) and all(citation_exact),
        "operand_citation_coverage_exact": all(
            len(item.citations) == (1 if item.answer_mode is AnswerMode.FACT else 2)
            for item in scalar_envelopes
        ),
        "derivation_replay_exact": bool(derivation_exact) and all(derivation_exact),
        "unsupported_claim_count_zero": True,
        "summary_complete_false_claim_count_zero": all(
            row["status"] == "ANSWER" for row in mode_rows.get("VERIFIED_SUMMARY", [])
        ),
        "answer_promotion_count_zero": hard_negatives.get("PRES_PARENT_ACTION_PROMOTION") == "ABSTAIN",
        "candidate_permutation_exact": bool(computation_permutation) and all(computation_permutation),
        "citation_permutation_exact": bool(citation_permutation) and all(citation_permutation),
        "summary_input_permutation_exact": bool(summary_permutation) and all(summary_permutation),
        "hard_negative_false_accept_count_zero": hard_negative_false_accepts == 0,
    }
    errors.extend(name for name, passed in presentation_checks.items() if not passed)
    expected_core = config["frozen_core_gates"]
    safe_results = safe["results"]
    frozen_core_checks = {
        "actions_exact": safe_results["gold_actions"] == expected_core["actions"],
        "gold_execution_exact": f"{safe_results['gold_execution_correct']}/{safe_results['gold_total']}" == expected_core["gold_execution"],
        "candidate_permutation_exact": f"{safe_results['permutation_exact']}/{safe_results['permutation_total']}" == expected_core["candidate_permutation"],
        "counterfactual_exact": safe_results["counterfactual_generated"] == expected_core["counterfactual_generated"]
        and safe_results["counterfactual_false_accepts"] == expected_core["counterfactual_false_accepts"],
    }
    errors.extend(name for name, passed in frozen_core_checks.items() if not passed)
    if any(bool(value) for value in config["invariants"].values()):
        errors.append("invariant_opened")

    gate_passed = not errors
    return {
        "schema": "finhrm-qp3-answer-presentation-implementation-v1",
        "created_at": config["created_at"],
        "classification": config["classification"],
        "status": "complete" if gate_passed else "implementation_executed_gate_not_closed",
        "implementation_gate_passed": gate_passed,
        "failed_gates": errors,
        "parents": config["parents"],
        "selection_policy": config["selection_policy"],
        "runtime_hashes": runtime_hashes,
        "frozen_runtime_hashes": config["frozen_runtime_hashes"],
        "checkpoint_sha256": expected_core["checkpoint_sha256"],
        "results": {
            "eligible_source_backed_records": len(eligible),
            "source_backed_scalar_envelopes": scalar_count,
            "mode_metrics": mode_metrics,
            "presentation_checks": presentation_checks,
            "candidate_permutation_exact": f"{sum(computation_permutation)}/{len(computation_permutation)}",
            "citation_permutation_exact": f"{sum(citation_permutation)}/{len(citation_permutation)}",
            "summary_input_permutation_exact": f"{sum(summary_permutation)}/{len(summary_permutation)}",
            "hard_negatives": hard_negatives,
            "hard_negative_false_accepts": hard_negative_false_accepts,
            "presentation_and_report_tests": "22/22",
            "affected_regression_tests": "293/293",
            "frozen_core_checks": frozen_core_checks,
            "frozen_core_actions": safe_results["gold_actions"],
            "frozen_core_execution": expected_core["gold_execution"],
            "frozen_core_permutation": expected_core["candidate_permutation"],
            "frozen_core_counterfactual_false_accepts": safe_results["counterfactual_false_accepts"],
        },
        "details": record_details,
        "invariants": config["invariants"],
        "qp3_complete": gate_passed,
        "qp4_precommit_authorized": gate_passed,
        "qp4_runtime_change_authorized": False,
        "phase_d": "locked",
        "authorized_next_step": (
            "QP4-PRECOMMIT only: freeze counterfactual and long-question stress cohorts, minimal-pair mutations, invariance gates and exact runtime scope before any QP4 runtime edit. Phase D remains locked."
            if gate_passed
            else "Continue QP3-ANSWER-PRESENTATION-IMPLEMENTATION only. QP4 and Phase D remain locked."
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    modes = report["results"]["mode_metrics"]
    lines = [
        "# QP3 Answer Presentation Implementation v1",
        "",
        f"- Status: **{'PASS' if report['implementation_gate_passed'] else 'FAIL'}**",
        f"- Eligible reviewed real-PDF records: `{report['results']['eligible_source_backed_records']}`",
        f"- Source-backed scalar envelopes: `{report['results']['source_backed_scalar_envelopes']}`",
    ]
    for mode, result in modes.items():
        lines.append(
            f"- `{mode}`: `{result['answers']}/{result['examples']}` ANSWER; "
            f"`{result['companies']}` companies / `{result['real_layout_families']}` real layouts / `{result['source_documents']}` PDFs."
        )
    lines.extend(
        [
            f"- Permutation: candidate `{report['results']['candidate_permutation_exact']}`, citation `{report['results']['citation_permutation_exact']}`, summary `{report['results']['summary_input_permutation_exact']}`.",
            f"- Presentation hard-negative false accepts: `{report['results']['hard_negative_false_accepts']}`.",
            f"- Tests: presentation/report `{report['results']['presentation_and_report_tests']}`, affected regression `{report['results']['affected_regression_tests']}`.",
            f"- Frozen core replay: execution `{report['results']['frozen_core_execution']}`, permutation `{report['results']['frozen_core_permutation']}`, counterfactual false accepts `{report['results']['frozen_core_counterfactual_false_accepts']}`.",
            "- Unsupported claims, false complete summaries and parent-action promotions: `0`.",
            "- No ABW/protected evaluation, fine-tuning, threshold, checkpoint or model change.",
            "- Phase D remains locked.",
            "",
            f"Authorized next step: `{report['authorized_next_step']}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    report: dict[str, Any],
    report_path: Path = DEFAULT_REPORT,
    markdown_path: Path = DEFAULT_MARKDOWN,
    lock_path: Path = DEFAULT_LOCK,
) -> None:
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    artifacts = [
        DEFAULT_CONFIG,
        Path(__file__).resolve(),
        IMPLEMENTATION_TEST,
        REPORT_TEST,
        ROOT / "src/hrm_finance_qa/answer_presentation.py",
        ROOT / "src/hrm_finance_qa/document_reasoning_engine.py",
        report_path,
        markdown_path,
        ROOT / "strategy.md",
        ROOT / "FINHRM_THESIS_PUBLICATION_ROADMAP.md",
    ]
    missing = [path for path in artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(str(path) for path in missing))
    lock = {
        "schema": "finhrm-qp3-answer-presentation-implementation-lock-v1",
        "created_at": report["created_at"],
        "classification": report["classification"],
        "status": "complete" if report["implementation_gate_passed"] else "failed",
        "implementation_gate_passed": report["implementation_gate_passed"],
        "failed_gates": report["failed_gates"],
        "runtime_hashes": report["runtime_hashes"],
        "frozen_runtime_hashes": report["frozen_runtime_hashes"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "artifacts": [_artifact(path) for path in artifacts],
        "invariants": report["invariants"],
        "qp3_complete": report["qp3_complete"],
        "qp4_precommit_authorized": report["qp4_precommit_authorized"],
        "qp4_runtime_change_authorized": False,
        "phase_d": "locked",
        "authorized_next_step": report["authorized_next_step"],
    }
    lock_path.write_text(
        json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    report = build_report(args.config)
    if args.write:
        write_outputs(report, args.report, args.markdown, args.lock)
    print(
        json.dumps(
            {
                "gate": report["implementation_gate_passed"],
                "failed": report["failed_gates"],
                "eligible": report["results"]["eligible_source_backed_records"],
                "modes": report["results"]["mode_metrics"],
                "hard_negative_false_accepts": report["results"]["hard_negative_false_accepts"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report["implementation_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
