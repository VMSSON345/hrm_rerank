from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hrm_finance_qa.answer_presentation import (
    AnswerMode,
    AnswerStatus,
    VerifiedAnswerEnvelope,
    build_verified_answer_envelope,
    render_verified_answer,
    validate_verified_answer_envelope,
)
from hrm_finance_qa.document_reasoning_engine import (
    DocumentReasoningAnswer,
    DocumentHRMReasoningEngine,
    SelectedEvidence,
)
from hrm_finance_qa.document_page_index import PageIndexedChunks, PageIndexStats
from hrm_finance_qa.document_sources import EvidenceChunk
from hrm_finance_qa.program_data import NumericCandidate, ReasoningProgram


VALID_PROOF = (
    ("execution_valid", "required", "VALID"),
    ("operation_match", "required", "VALID"),
    ("query_metric", "required", "VALID"),
    ("period_coverage", "required", "VALID"),
    ("reporting_entity", "required", "VALID"),
    ("unit_match", "required", "VALID"),
    ("logical_cell_unique", "required", "VALID"),
    ("source_provenance_complete", "required", "VALID"),
)


def _candidate(
    index: int,
    value: str,
    row: str,
    *,
    page: int | None = 7,
    period: str = "2024",
    source: str = "development-report.pdf",
    unit: str = "triệu đồng",
    extraction: str = "native",
) -> NumericCandidate:
    return NumericCandidate(
        index=index,
        value=value,
        raw=value,
        context=(
            f"ROW: {row}\nCOLUMN: Năm {period}\nPERIOD: {period}\n"
            "TABLE: income-statement\nSCOPE: consolidated"
        ),
        line_number=10 + index,
        char_start=100 + index * 20,
        char_end=112 + index * 20,
        unit=unit,
        is_percent=False,
        is_year_like=False,
        relevance=0.8,
        source_name=source,
        page=page,
        extraction_method=extraction,
        row_label=row,
        column_label=f"Năm {period}",
        period=period,
        table_id="income-statement",
        scope="consolidated",
        source_span=(100 + index * 20, 112 + index * 20),
    )


def _answer(
    operation: str,
    candidates: tuple[NumericCandidate, ...],
    operands: tuple[int, ...],
    value: str,
    *,
    question: str,
    source_action: str = "ANSWER",
    verification_state: str = "VALID",
    proof_obligations: tuple[tuple[str, str, str | None], ...] = VALID_PROOF,
    abstained: bool = False,
) -> SimpleNamespace:
    by_index = {candidate.index: candidate for candidate in candidates}
    evidence = tuple(
        SimpleNamespace(operand_number=position + 1, candidate=by_index[index])
        for position, index in enumerate(operands)
    )
    prediction = SimpleNamespace(
        program=ReasoningProgram(operation, operands),
        program_dag=None,
        slot_bindings=(),
    )
    return SimpleNamespace(
        question=question,
        value=Decimal(value),
        unit="triệu đồng" if operation not in {"RATIO_PERCENT", "PERCENT_CHANGE"} else "%",
        prediction=prediction,
        program_dag=None,
        evidence=evidence,
        abstained=abstained,
        source_action=source_action,
        verification_state=verification_state,
        proof_obligations=proof_obligations,
        verification_reasons=(),
        acceptance_basis=("runtime_action_preserved", "required_proof_valid"),
    )


def _fact(row: str, value: str, *, page: int | None = 7, source: str = "development-report.pdf") -> VerifiedAnswerEnvelope:
    candidate = _candidate(0, value, row, page=page, source=source)
    answer = _answer(
        "SELECT",
        (candidate,),
        (0,),
        value,
        question=f"{row} năm 2024 là bao nhiêu?",
    )
    return build_verified_answer_envelope(answer)


class QP3AnswerPresentationImplementationTests(unittest.TestCase):
    def test_fact_exact_value_unit_scale_and_citation(self) -> None:
        envelope = _fact("Doanh thu thuần", "12450")
        self.assertEqual(envelope.status, AnswerStatus.ANSWER)
        self.assertEqual(envelope.answer_mode, AnswerMode.FACT)
        self.assertEqual(envelope.canonical_value, "12450000000")
        self.assertEqual(envelope.display, ("12450", "triệu đồng", "1000000", "vi-VN"))
        self.assertEqual(len(envelope.citations), 1)
        self.assertEqual(validate_verified_answer_envelope(envelope), ())
        rendered = render_verified_answer(envelope)
        self.assertIn("Doanh thu thuần", rendered)
        self.assertIn("12.450", rendered)
        self.assertIn("trang 7", rendered)
        self.assertIn("dòng Doanh thu thuần", rendered)

    def test_fact_without_page_omits_page_without_placeholder(self) -> None:
        rendered = render_verified_answer(_fact("Tổng tài sản", "800", page=None))
        self.assertNotIn("trang", rendered.lower())
        self.assertNotIn("page", rendered.lower())
        self.assertIn("vùng ký tự", rendered)

    def test_computation_has_operand_citations_and_exact_replay(self) -> None:
        left = _candidate(0, "120", "Doanh thu năm nay")
        right = _candidate(1, "100", "Doanh thu năm trước", period="2023")
        answer = _answer(
            "SUBTRACT",
            (left, right),
            (0, 1),
            "20",
            question="Doanh thu năm 2024 tăng bao nhiêu so với năm 2023?",
        )
        envelope = build_verified_answer_envelope(answer)
        self.assertEqual(envelope.status, AnswerStatus.ANSWER)
        self.assertEqual(envelope.answer_mode, AnswerMode.COMPUTATION)
        self.assertEqual(envelope.canonical_value, "20000000")
        self.assertEqual(len(envelope.citations), 2)
        self.assertEqual(validate_verified_answer_envelope(envelope), ())
        rendered = render_verified_answer(envelope)
        self.assertIn("Công thức:", rendered)
        self.assertIn("Toán hạng O1", rendered)
        self.assertIn("Toán hạng O2", rendered)
        self.assertEqual(rendered.count("Nguồn:"), 2)

    def test_parent_abstain_and_unknown_proof_are_never_promoted(self) -> None:
        candidate = _candidate(0, "12450", "Doanh thu thuần")
        unknown = (*VALID_PROOF[:-1], ("source_provenance_complete", "required", "UNKNOWN"))
        answer = _answer(
            "SELECT",
            (candidate,),
            (0,),
            "12450",
            question="Doanh thu thuần năm 2024 là bao nhiêu?",
            source_action="ABSTAIN",
            verification_state="UNKNOWN",
            proof_obligations=unknown,
            abstained=True,
        )
        envelope = build_verified_answer_envelope(answer)
        self.assertEqual(envelope.status, AnswerStatus.ABSTAIN)
        self.assertIsNone(envelope.canonical_value)
        self.assertNotIn("12.450", render_verified_answer(envelope))

    def test_invalid_proof_is_rejected(self) -> None:
        candidate = _candidate(0, "12450", "Doanh thu thuần")
        invalid = (*VALID_PROOF[:-1], ("source_provenance_complete", "required", "INVALID"))
        answer = _answer(
            "SELECT",
            (candidate,),
            (0,),
            "12450",
            question="Doanh thu thuần năm 2024 là bao nhiêu?",
            source_action="REJECT",
            verification_state="INVALID",
            proof_obligations=invalid,
            abstained=True,
        )
        envelope = build_verified_answer_envelope(answer)
        self.assertEqual(envelope.status, AnswerStatus.REJECT)
        self.assertIsNone(envelope.canonical_value)

    def test_wrong_page_statement_year_and_scale_are_detected(self) -> None:
        envelope = _fact("Doanh thu thuần", "12450")
        citation = envelope.citations[0]
        mutations = (
            replace(citation, page=99),
            replace(citation, statement_or_note="Thuyết minh khác"),
            replace(citation, period="2023"),
            replace(citation, scale="1000"),
        )
        for mutation in mutations:
            changed = replace(envelope, citations=(mutation,))
            self.assertTrue(validate_verified_answer_envelope(changed))
            with self.assertRaises(ValueError):
                render_verified_answer(changed)

    def test_swapped_noncommutative_operand_citations_fail_replay(self) -> None:
        left = _candidate(0, "120", "Doanh thu năm nay")
        right = _candidate(1, "100", "Doanh thu năm trước", period="2023")
        envelope = build_verified_answer_envelope(
            _answer(
                "SUBTRACT",
                (left, right),
                (0, 1),
                "20",
                question="Doanh thu năm 2024 tăng bao nhiêu so với năm 2023?",
            )
        )
        first, second = envelope.citations
        changed = replace(
            envelope,
            citations=(
                replace(first, operand_ref=second.operand_ref),
                replace(second, operand_ref=first.operand_ref),
            ),
        )
        self.assertTrue(validate_verified_answer_envelope(changed))

    def test_citation_and_candidate_permutations_render_exactly(self) -> None:
        first = _candidate(0, "40", "Chi phí bán hàng")
        second = _candidate(1, "60", "Chi phí quản lý")
        original = build_verified_answer_envelope(
            _answer(
                "ADD",
                (first, second),
                (0, 1),
                "100",
                question="Tổng chi phí bán hàng và chi phí quản lý năm 2024 là bao nhiêu?",
            )
        )
        first_permuted = replace(first, index=1)
        second_permuted = replace(second, index=0)
        permuted = build_verified_answer_envelope(
            _answer(
                "ADD",
                (second_permuted, first_permuted),
                (1, 0),
                "100",
                question="Tổng chi phí bán hàng và chi phí quản lý năm 2024 là bao nhiêu?",
            )
        )
        self.assertEqual(render_verified_answer(original), render_verified_answer(permuted))
        reversed_citations = replace(original, citations=tuple(reversed(original.citations)))
        self.assertEqual(
            render_verified_answer(original), render_verified_answer(reversed_citations)
        )

    def test_complete_verified_summary_renders_requested_inventory_order(self) -> None:
        rows = (
            ("revenue", _fact("Doanh thu thuần", "12450", source="a.pdf")),
            ("assets", _fact("Tổng tài sản", "8200", source="a.pdf")),
            ("profit", _fact("Lợi nhuận sau thuế", "900", source="a.pdf")),
        )
        summary = build_verified_answer_envelope(
            answer_mode=AnswerMode.VERIFIED_SUMMARY,
            requested_fields=("revenue", "assets", "profit"),
            summary_rows=tuple(reversed(rows)),
        )
        self.assertEqual(summary.status, AnswerStatus.ANSWER)
        self.assertEqual(summary.summary_coverage.verified_count, 3)
        self.assertEqual(validate_verified_answer_envelope(summary), ())
        rendered = render_verified_answer(summary)
        self.assertLess(rendered.index("Doanh thu thuần"), rendered.index("Tổng tài sản"))
        self.assertLess(rendered.index("Tổng tài sản"), rendered.index("Lợi nhuận sau thuế"))
        self.assertIn("Nội dung | Số liệu/so sánh | Điểm nổi bật | Nguồn", rendered)

    def test_partial_summary_abstains_and_labels_n_of_m(self) -> None:
        summary = build_verified_answer_envelope(
            answer_mode=AnswerMode.VERIFIED_SUMMARY,
            requested_fields=("revenue", "assets", "profit"),
            summary_rows=(
                ("revenue", _fact("Doanh thu thuần", "12450")),
                ("assets", _fact("Tổng tài sản", "8200")),
            ),
        )
        self.assertEqual(summary.status, AnswerStatus.ABSTAIN)
        self.assertEqual(summary.summary_coverage.verified_count, 2)
        self.assertEqual(summary.summary_coverage.missing_fields, ("profit",))
        self.assertIn("2/3", render_verified_answer(summary))

    def test_conflicting_summary_rejects_without_choosing_value(self) -> None:
        summary = build_verified_answer_envelope(
            answer_mode=AnswerMode.VERIFIED_SUMMARY,
            requested_fields=("revenue", "assets", "profit"),
            summary_rows=(
                ("revenue", _fact("Doanh thu thuần", "12450")),
                ("assets", _fact("Tổng tài sản", "8200")),
                ("profit", _fact("Lợi nhuận sau thuế", "900")),
            ),
            conflicting_fields=("profit",),
        )
        self.assertEqual(summary.status, AnswerStatus.REJECT)
        rendered = render_verified_answer(summary)
        self.assertIn("mâu thuẫn", rendered.lower())
        self.assertNotIn("900", rendered)

    def test_uncited_qualitative_claim_is_not_rendered(self) -> None:
        fact = _fact("Doanh thu thuần", "12450")
        unsafe_claim = replace(
            fact.claim,
            qualitative_highlight="Tăng mạnh nhờ mở rộng thị trường",
            qualitative_citation=None,
        )
        unsafe_fact = replace(fact, claim=unsafe_claim)
        summary = build_verified_answer_envelope(
            answer_mode=AnswerMode.VERIFIED_SUMMARY,
            requested_fields=("revenue", "assets", "profit"),
            summary_rows=(
                ("revenue", unsafe_fact),
                ("assets", _fact("Tổng tài sản", "8200")),
                ("profit", _fact("Lợi nhuận sau thuế", "900")),
            ),
        )
        self.assertNotEqual(summary.status, AnswerStatus.ANSWER)
        self.assertNotIn("mở rộng thị trường", render_verified_answer(summary))

    def test_same_envelope_renders_same_bytes(self) -> None:
        envelope = _fact("Doanh thu thuần", "12450")
        self.assertEqual(
            render_verified_answer(envelope).encode("utf-8"),
            render_verified_answer(envelope).encode("utf-8"),
        )

    def test_technical_audit_renderer_remains_separate(self) -> None:
        candidate = _candidate(0, "12450", "Doanh thu thuần")
        prediction = SimpleNamespace(
            program=ReasoningProgram("SELECT", (0,)),
            program_dag=None,
            operation_confidence=0.9,
            confidence=0.9,
            segments=1,
            halt_q=0.8,
            continue_q=0.2,
            program_candidates_considered=1,
            symbolic_program_rejections=0,
            symbolic_proof_score=1.0,
            symbolic_unknown_constraints=0,
            grounding_constraints_applied=(),
            neural_operation_confidence=0.9,
        )
        answer = DocumentReasoningAnswer(
            question="Doanh thu thuần năm 2024 là bao nhiêu?",
            value=Decimal("12450"),
            unit="triệu đồng",
            prediction=prediction,
            evidence=(SelectedEvidence(1, candidate),),
            retrieved_chunks=(),
            document_count=1,
            candidate_count=1,
            abstained=False,
            confidence_threshold=0.75,
            warnings=(),
            source_action="ANSWER",
            verification_state="VALID",
            proof_obligations=VALID_PROOF,
            acceptance_basis=("runtime_action_preserved", "required_proof_valid"),
        )
        audit = answer.render()
        product = answer.render_verified()
        self.assertIn("HRM confidence:", audit)
        self.assertNotIn("HRM confidence:", product)
        self.assertIn("Doanh thu thuần", product)

    def test_engine_answer_captures_full_candidate_graph_proof_snapshot(self) -> None:
        candidate = _candidate(0, "12450", "Doanh thu thuần")
        prediction = SimpleNamespace(
            program=ReasoningProgram("SELECT", (0,)),
            program_dag=None,
            slot_bindings=(),
            confidence=0.95,
            operation_confidence=0.95,
            segments=1,
            halt_q=1.0,
            continue_q=0.0,
            grounding_constraints_applied=(),
            neural_operation_confidence=0.95,
        )
        bundle = SimpleNamespace(
            model=SimpleNamespace(
                config=SimpleNamespace(max_candidates=8, hierarchical_input_mode="strict")
            ),
            metadata={"abstention_threshold": 0.80},
            predict=lambda *_args, **_kwargs: prediction,
        )
        chunk = EvidenceChunk(
            chunk_id="development-report:native:7",
            source_path="development-report.pdf",
            source_name="development-report.pdf",
            page_start=7,
            page_end=7,
            line_start=1,
            line_end=2,
            text="Doanh thu thuần | Năm 2024 | 12.450",
            score=1.0,
            extraction_methods=("native",),
        )
        indexed = PageIndexedChunks(
            chunks=(chunk,),
            stats=PageIndexStats(total_pages=8, native_pages=8),
        )
        engine = DocumentHRMReasoningEngine(bundle)
        with patch(
            "hrm_finance_qa.document_reasoning_engine.discover_documents",
            return_value=[Path("development-report.pdf")],
        ), patch(
            "hrm_finance_qa.document_reasoning_engine.build_page_indexed_chunks",
            return_value=indexed,
        ), patch(
            "hrm_finance_qa.document_reasoning_engine.rank_chunks",
            return_value=[chunk],
        ), patch(
            "hrm_finance_qa.document_reasoning_engine.expand_neighbor_chunks",
            return_value=[chunk],
        ), patch(
            "hrm_finance_qa.document_reasoning_engine.merge_candidates",
            return_value=[candidate],
        ), patch(
            "hrm_finance_qa.document_reasoning_engine.execute_program",
            return_value=Decimal("12450"),
        ):
            answer = engine.answer(
                "Doanh thu thuần năm 2024 là bao nhiêu?",
                "development-report.pdf",
                ocr=False,
            )

        self.assertTrue(answer.proof_obligations)
        self.assertIn(answer.verification_state, {"VALID", "UNKNOWN", "INVALID"})
        self.assertIn(answer.source_action, {"ANSWER", "ABSTAIN", "REJECT"})
        self.assertIn(
            "full_candidate_evidence_graph_verification",
            answer.acceptance_basis,
        )


if __name__ == "__main__":
    unittest.main()
