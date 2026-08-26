from __future__ import annotations

import unittest

from scripts.evaluate_v12c_interest_bearing_debt_share_frozen import (
    build_runtime_candidates,
    summarize,
)
from hrm_finance_qa.document_reasoning_engine import (
    _augment_financial_identity_candidates,
)


class EvaluateFrozenV12CTests(unittest.TestCase):
    def test_adapter_preserves_exact_period_and_runtime_currency_unit(self) -> None:
        record = self._record("31/12/2024", "million_vnd")
        candidates = build_runtime_candidates(record)
        self.assertEqual(candidates[0].period, "31/12/2024")
        self.assertEqual(candidates[0].unit, "triệu VND")
        augmented = _augment_financial_identity_candidates(
            record["question"], candidates, max_candidates=8
        )
        self.assertEqual(len(augmented), 4)
        self.assertEqual(augmented[-1].derivation_id, "interest_bearing_debt")

    def test_opening_date_materializes_without_relabelling_endpoint(self) -> None:
        record = self._record("01/01/2024", "VND")
        candidates = build_runtime_candidates(record)
        augmented = _augment_financial_identity_candidates(
            record["question"], candidates, max_candidates=8
        )
        self.assertEqual(len(augmented), 4)
        self.assertEqual(augmented[-1].derivation_id, "interest_bearing_debt")
        self.assertTrue(
            all(item.period == "01/01/2024" for item in augmented[:3])
        )

    def test_summary_is_group_disjoint_and_acceptance_safe(self) -> None:
        rows = [
            {
                "company_key": "a",
                "period": "31/12/2024",
                "accepted": True,
                "program_correct": True,
                "execution_correct": True,
                "gold_derivation_materialized": True,
                "abstain_reasons": [],
            },
            {
                "company_key": "b",
                "period": "01/01/2024",
                "accepted": False,
                "program_correct": False,
                "execution_correct": False,
                "gold_derivation_materialized": False,
                "abstain_reasons": ["below_frozen_threshold"],
            },
        ]
        report = summarize(rows)
        self.assertEqual(report["overall"]["accepted"], 1)
        self.assertEqual(report["overall"]["accepted_execution_precision"], 1.0)
        self.assertEqual(set(report["company_groups"]), {"a", "b"})

    def test_zero_acceptance_precision_is_unobserved(self) -> None:
        rows = [{
            "company_key": "a",
            "period": "01/01/2024",
            "accepted": False,
            "program_correct": False,
            "execution_correct": False,
            "gold_derivation_materialized": False,
            "abstain_reasons": ["below_frozen_threshold"],
        }]
        report = summarize(rows)
        self.assertIsNone(report["overall"]["accepted_program_precision"])
        self.assertIsNone(report["overall"]["accepted_execution_precision"])
        self.assertIn("opening_endpoint", report["period_semantics"])

    @staticmethod
    def _record(period: str, unit: str) -> dict[str, object]:
        values = ("80", "20", "1000")
        roles = (
            "short_term_borrowings",
            "long_term_borrowings",
            "total_capital",
        )
        return {
            "company_key": "development-company-a",
            "question": (
                "Tổng nợ vay tài chính chịu lãi gồm vay ngắn hạn và dài hạn "
                f"chiếm bao nhiêu phần trăm Tổng nguồn vốn tại {period}?"
            ),
            "evidence": [
                {
                    "role": role,
                    "value": value,
                    "raw_value": value,
                    "period": period,
                    "unit": unit,
                    "source_name": "DEV.pdf",
                    "page": 2,
                    "table_id": "DEV:page-2:balance-sheet",
                    "line_number": index + 1,
                    "source_line": role,
                }
                for index, (role, value) in enumerate(zip(roles, values))
            ],
        }


if __name__ == "__main__":
    unittest.main()
