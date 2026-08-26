from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from hrm_finance_qa.candidate_evaluation import (
    aggregate_candidate_availability,
    score_candidate_availability,
)
from hrm_finance_qa.evaluation_metrics import (
    aggregate_document_scores,
    render_document_score_markdown,
    score_document_response,
)
from scripts.score_external_pdf_smoke import (
    parse_report as parse_smoke_report,
    render_markdown as render_smoke_markdown,
)


class DocumentEvaluationMetricTests(unittest.TestCase):
    def test_exact_pages_reject_extra_evidence_page(self) -> None:
        case = {
            "case_id": "case-1",
            "operation": "PERCENT_CHANGE",
            "answer_decimal": "10",
            "operands": [
                {"role": "current", "value": "110", "pdf_page": 7},
                {"role": "previous", "value": "100", "pdf_page": 7},
            ],
        }
        response = {
            "answer": "10",
            "status": "ABSTAIN",
            "program": {"operation": "PERCENT_CHANGE", "operands": [0, 1]},
            "evidence": [
                {"operand_number": 1, "value": "110", "page": 7},
                {"operand_number": 2, "value": "100", "page": 8},
            ],
        }

        scored = score_document_response(case, response)

        self.assertFalse(scored["page_provenance_exact"])
        self.assertTrue(scored["page_provenance_expected_covered"])
        self.assertTrue(scored["page_provenance_any"])
        self.assertFalse(scored["operand_provenance_all"])
        self.assertFalse(scored["grounded_correct"])

    def test_pipeline_failure_is_not_scored_as_model_error(self) -> None:
        case = {
            "case_id": "case-1",
            "operation": "SELECT",
            "answer_decimal": "10",
            "operands": [{"role": "value", "value": "10", "pdf_page": 1}],
        }
        failed = score_document_response(
            case, {}, error="HTTP 409: not ready"
        )
        correct = score_document_response(
            case,
            {
                "answer": "10",
                "status": "ACCEPTED",
                "program": {"operation": "SELECT", "operands": [0]},
                "evidence": [
                    {"operand_number": 1, "value": "10", "page": 1}
                ],
            },
        )

        metrics = aggregate_document_scores([failed, correct])

        self.assertEqual(metrics["cases"], 2)
        self.assertEqual(metrics["predictions"], 1)
        self.assertEqual(metrics["pipeline_failures"], 1)
        self.assertEqual(metrics["operation_correct"], 1)
        self.assertEqual(metrics["accepted_grounded_correct"], 1)
        self.assertEqual(metrics["accepted_grounded_precision"], "1")

    def test_structured_not_ready_is_counted_separately(self) -> None:
        case = {
            "case_id": "case-1",
            "operation": "SELECT",
            "answer_decimal": "10",
            "operands": [{"role": "value", "value": "10", "pdf_page": 1}],
        }
        scored = score_document_response(
            case,
            {},
            error='HTTP 409: {"status": "not_ready"}',
        )

        metrics = aggregate_document_scores([scored])

        self.assertEqual(scored["availability"], "not_ready")
        self.assertEqual(metrics["not_ready"], 1)
        self.assertEqual(metrics["pipeline_failures"], 0)
        self.assertEqual(metrics["predictions"], 0)

    def test_markdown_reports_precision_as_fraction(self) -> None:
        case = {
            "case_id": "case-1",
            "operation": "SELECT",
            "answer_decimal": "10",
            "operands": [{"role": "value", "value": "10", "pdf_page": 1}],
        }
        scored = score_document_response(
            case,
            {
                "answer": "10",
                "status": "ACCEPTED",
                "program": {"operation": "SELECT", "operands": [0]},
                "evidence": [
                    {"operand_number": 1, "value": "10", "page": 1}
                ],
            },
        )
        score = {
            "metrics": aggregate_document_scores([scored]),
            "cases": [scored],
        }

        markdown = render_document_score_markdown(score, title="Test score")

        self.assertIn("Accepted answer precision: 1/1", markdown)
        self.assertIn("Accepted grounded precision: 1/1", markdown)


class CandidateAvailabilityMetricTests(unittest.TestCase):
    def test_distinguishes_extraction_and_quota_loss(self) -> None:
        case = {
            "case_id": "ratio-case",
            "document": "report.pdf",
            "question": "A chiếm bao nhiêu phần trăm B?",
            "operation": "RATIO_PERCENT",
            "operands": [
                {"role": "numerator", "value": "40", "pdf_page": 2},
                {"role": "denominator", "value": "100", "pdf_page": 1},
            ],
        }
        prequota = [
            SimpleNamespace(
                index=0,
                value="40",
                page=2,
                relevance=1.0,
                context="ROW: A",
            ),
            SimpleNamespace(
                index=1,
                value="100",
                page=1,
                relevance=1.0,
                context="ROW: B",
            ),
        ]
        final = [prequota[1]]

        report = score_candidate_availability(
            case,
            retrieved_pages=[1, 2],
            prequota_candidates=prequota,
            final_candidates=final,
        )

        self.assertEqual(report["operands"][0]["loss_stage"], "quota")
        self.assertEqual(report["operands"][1]["loss_stage"], "available")
        self.assertTrue(report["all_operands_prequota"])
        self.assertFalse(report["all_operands_final"])

    def test_aggregate_keeps_pipeline_failures_separate(self) -> None:
        case = {
            "case_id": "select-case",
            "document": "report.pdf",
            "question": "Giá trị là bao nhiêu?",
            "operation": "SELECT",
            "operands": [{"role": "value", "value": "10", "pdf_page": 1}],
        }
        evaluated = score_candidate_availability(
            case,
            retrieved_pages=[1],
            prequota_candidates=[],
            final_candidates=[],
        )
        failed = score_candidate_availability(
            {**case, "case_id": "failed"},
            retrieved_pages=[],
            prequota_candidates=[],
            final_candidates=[],
            error="index unavailable",
        )

        metrics = aggregate_candidate_availability([evaluated, failed])

        self.assertEqual(metrics["evaluated_cases"], 1)
        self.assertEqual(metrics["pipeline_failures"], 1)
        self.assertEqual(metrics["loss_stages"]["extraction"], 1)

    def test_candidate_ceiling_reports_rank_and_quota_cutoffs(self) -> None:
        case = {
            "case_id": "ratio-case",
            "document": "report.pdf",
            "question": "Tỷ lệ A trên B là bao nhiêu?",
            "operation": "RATIO_PERCENT",
            "operands": [
                {"role": "numerator", "value": "10", "pdf_page": 7},
                {"role": "denominator", "value": "20", "pdf_page": 8},
            ],
        }
        numerator = SimpleNamespace(
            index=0, value="10", page=7, relevance=1.0, context="ROW: A"
        )
        denominator = SimpleNamespace(
            index=1, value="20", page=8, relevance=1.0, context="ROW: B"
        )
        report = score_candidate_availability(
            case,
            retrieved_pages=[1, 2, 3, 4, 5, 7, 8],
            prequota_candidates=[numerator, denominator],
            final_candidates=[numerator, denominator],
            candidates_at_32=[numerator],
            candidates_at_64=[numerator, denominator],
        )

        metrics = aggregate_candidate_availability([report])

        self.assertEqual(metrics["gold_page_recall_at_5"]["numerator"], 0)
        self.assertEqual(metrics["gold_page_recall_at_10"]["numerator"], 1)
        self.assertEqual(metrics["operand_recall_at_32"]["numerator"], 1)
        self.assertEqual(metrics["operand_recall_at_64"]["numerator"], 2)
        self.assertEqual(metrics["pair_recall_at_64"]["numerator"], 1)

    def test_separates_page_chunk_representation_and_extraction_loss(self) -> None:
        def candidate_case(case_id: str, value: str, page: int) -> dict[str, object]:
            return {
                "case_id": case_id,
                "document": "report.pdf",
                "question": "Giá trị là bao nhiêu?",
                "operation": "SELECT",
                "operands": [{"role": "value", "value": value, "pdf_page": page}],
            }

        page_chunks = [
            SimpleNamespace(
                page_start=2,
                page_end=2,
                text="Chỉ tiêu | 4.400.000",
            ),
            SimpleNamespace(
                page_start=3,
                page_end=3,
                text="Chỉ tiêu không có ô giá trị",
            ),
            SimpleNamespace(
                page_start=4,
                page_end=4,
                text="Chỉ tiêu | 7.000.000",
            ),
        ]
        ranked_chunks = [
            SimpleNamespace(
                page_start=4,
                page_end=4,
                text="Chỉ tiêu | 7.000.000",
            )
        ]

        chunk_loss = score_candidate_availability(
            candidate_case("chunk", "4400000", 2),
            retrieved_pages=[4],
            prequota_candidates=[],
            final_candidates=[],
            page_chunks=page_chunks,
            retrieved_chunks=ranked_chunks,
        )
        representation_loss = score_candidate_availability(
            candidate_case("representation", "6300000", 3),
            retrieved_pages=[4],
            prequota_candidates=[],
            final_candidates=[],
            page_chunks=page_chunks,
            retrieved_chunks=ranked_chunks,
        )
        extraction_loss = score_candidate_availability(
            candidate_case("extraction", "7000000", 4),
            retrieved_pages=[4],
            prequota_candidates=[],
            final_candidates=[],
            page_chunks=page_chunks,
            retrieved_chunks=ranked_chunks,
        )
        page_loss = score_candidate_availability(
            candidate_case("page", "100", 5),
            retrieved_pages=[4],
            prequota_candidates=[],
            final_candidates=[],
            page_chunks=page_chunks,
            retrieved_chunks=ranked_chunks,
        )

        self.assertEqual(
            chunk_loss["operands"][0]["loss_stage"], "chunk_retrieval"
        )
        self.assertEqual(
            representation_loss["operands"][0]["loss_stage"],
            "source_representation",
        )
        self.assertEqual(
            extraction_loss["operands"][0]["loss_stage"], "extraction"
        )
        self.assertEqual(
            page_loss["operands"][0]["loss_stage"], "page_retrieval"
        )

    def test_candidate_pool_can_be_wider_than_display_retrieval(self) -> None:
        case = {
            "case_id": "selected-page-sibling",
            "document": "report.pdf",
            "question": "Giá trị là bao nhiêu?",
            "operation": "SELECT",
            "operands": [{"role": "value", "value": "4400000", "pdf_page": 2}],
        }
        selected_page_chunks = [
            SimpleNamespace(
                page_start=2,
                page_end=2,
                text="Chỉ tiêu | 4.400.000",
            ),
            SimpleNamespace(
                page_start=2,
                page_end=2,
                text="Dòng khớp truy vấn",
            ),
        ]
        display_chunks = [selected_page_chunks[1]]
        candidate = SimpleNamespace(
            index=0,
            value="4400000",
            page=2,
            relevance=1.0,
            context="ROW: Chỉ tiêu",
        )

        report = score_candidate_availability(
            case,
            retrieved_pages=[2],
            prequota_candidates=[candidate],
            final_candidates=[candidate],
            page_chunks=selected_page_chunks,
            retrieved_chunks=display_chunks,
            candidate_chunks=selected_page_chunks,
        )

        operand = report["operands"][0]
        self.assertFalse(operand["raw_value_in_retrieved_chunks"])
        self.assertTrue(operand["raw_value_in_candidate_chunks"])
        self.assertEqual(operand["loss_stage"], "available")


class SmokeScoreMetricTests(unittest.TestCase):
    def test_smoke_score_requires_exact_page_set(self) -> None:
        report = """\
================================================================================
CASE: sample
DOCUMENT: report.pdf
QUESTION: Giá trị là bao nhiêu?
GOLD_OPERATION: SELECT
GOLD_VALUE: 10
GOLD_PDF_PAGE: 1
--------------------------------------------------------------------------------
Answer: 10
Status: ACCEPTED
Program: SELECT(0,)
Evidence:
- operand[1]=10 | report.pdf, page 1
- operand[2]=20 | report.pdf, page 2
Retrieved source chunks:
COMMAND_EXIT: 0
"""
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "report.txt"
            path.write_text(report, encoding="utf-8")

            cases = parse_smoke_report(path)
            markdown = render_smoke_markdown(path, cases)

        self.assertEqual(len(cases), 1)
        self.assertFalse(cases[0]["page_exact"])
        self.assertTrue(cases[0]["page_expected_covered"])
        self.assertTrue(cases[0]["page_any"])
        self.assertIn("Exact page provenance: 0/1", markdown)
        self.assertIn("Accepted answer precision: 1/1", markdown)
        self.assertIn("Accepted grounded precision: 0/1", markdown)


if __name__ == "__main__":
    unittest.main()
