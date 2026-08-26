from __future__ import annotations

import unittest
from dataclasses import replace

from hrm_finance_qa.external_evaluation import audit_external_test_set
from hrm_finance_qa.program_data import induce_program_example


class ExternalEvaluationAuditTests(unittest.TestCase):
    def _example(self, example_id: str, company: str, ticker: str, value: int):
        row = {
            "id": example_id,
            "task": "QA",
            "difficulty": "Easy_QA",
            "question": "Doanh thu thuần là bao nhiêu?",
            "context": (
                f"Công ty: {company}\n"
                f"Mã cổ phiếu: {ticker}\n"
                "Báo cáo: Báo cáo tài chính 2024\n"
                "| Chỉ tiêu | Giá trị |\n"
                "|---|---:|\n"
                f"| Doanh thu thuần | {value} VND |"
            ),
            "answer": f"Doanh thu thuần là {value} VND.",
        }
        example = induce_program_example(row)
        self.assertIsNotNone(example)
        return example

    def test_passes_company_disjoint_real_examples(self) -> None:
        reference = self._example("train-1", "Công ty Alpha", "AAA", 100)
        external = self._example("test-1", "Công ty Beta", "BBB", 200)

        audit = audit_external_test_set([external], [reference])

        self.assertTrue(audit["passed"])
        self.assertEqual(audit["leakage"]["companies"], [])
        self.assertEqual(
            audit["warnings"]["question_overlap_is_not_a_hard_failure"], 1
        )

    def test_rejects_company_or_ticker_leakage(self) -> None:
        reference = self._example("train-1", "Công ty Alpha", "AAA", 100)
        external = self._example("test-1", "Công ty Alpha", "AAA", 200)

        audit = audit_external_test_set([external], [reference])

        self.assertFalse(audit["passed"])
        self.assertTrue(audit["leakage"]["companies"])
        self.assertEqual(audit["leakage"]["tickers"], ["aaa"])

    def test_detects_leakage_across_mojibake_and_unicode(self) -> None:
        reference = self._example("train-1", "Công ty Alpha", "AAA", 100)
        reference = replace(
            reference,
            context=reference.context.encode("utf-8").decode("latin1"),
        )
        external = self._example("test-1", "Công ty Alpha", "AAA", 200)

        audit = audit_external_test_set([external], [reference])

        self.assertFalse(audit["passed"])
        self.assertTrue(audit["leakage"]["companies"])
        self.assertEqual(audit["leakage"]["tickers"], ["aaa"])

    def test_rejects_synthetic_and_bad_execution_label(self) -> None:
        reference = self._example("train-1", "Công ty Alpha", "AAA", 100)
        external = self._example("test-1", "Công ty Beta", "BBB", 200)
        external = replace(
            external,
            example_id="synthetic-select-test",
            answer_value="201",
        )

        audit = audit_external_test_set([external], [reference])

        self.assertFalse(audit["passed"])
        self.assertEqual(
            audit["integrity"]["synthetic_example_ids"],
            ["synthetic-select-test"],
        )
        self.assertEqual(len(audit["integrity"]["execution_mismatches"]), 1)

    def test_detects_candidate_source_and_company_key_leakage(self) -> None:
        reference = self._example("train-1", "", "", 100)
        external = self._example("test-1", "", "", 200)
        reference = replace(
            reference,
            candidates=(
                replace(reference.candidates[0], source_name="ABC_24CN_BCTN.pdf"),
            ),
            supervision={"company_key": "company-abc"},
        )
        external = replace(
            external,
            candidates=(
                replace(external.candidates[0], source_name="ABC_24CN_BCTN.pdf"),
            ),
            supervision={"company_key": "company-abc"},
        )

        audit = audit_external_test_set([external], [reference])

        self.assertFalse(audit["passed"])
        self.assertEqual(audit["leakage"]["tickers"], ["abc"])
        self.assertEqual(audit["leakage"]["companies"], ["key:company abc"])
        self.assertTrue(audit["leakage"]["report_fingerprints"])

    def test_rejects_mechanical_candidate_until_independent_review(self) -> None:
        reference = self._example("train-1", "CÃ´ng ty Alpha", "AAA", 100)
        external = self._example("test-1", "CÃ´ng ty Beta", "BBB", 200)
        external = replace(
            external,
            supervision={
                "classification": "external_candidate_pending_review",
                "review_status": "pending_independent_review",
            },
        )

        audit = audit_external_test_set([external], [reference])

        self.assertFalse(audit["passed"])
        self.assertEqual(
            audit["integrity"]["pending_independent_review_example_ids"],
            ["test-1"],
        )

if __name__ == "__main__":
    unittest.main()
