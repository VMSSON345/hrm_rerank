from __future__ import annotations

from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.freeze_v12c_interest_bearing_debt_share_cohort import freeze


class FreezeV12CCohortTests(unittest.TestCase):
    def test_freeze_requires_pinned_complete_visual_approval(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "development-candidates.jsonl"
            evidence = [
                self._leaf("short_term_borrowings", "80"),
                self._leaf("long_term_borrowings", "20"),
                self._leaf("total_capital", "1000"),
            ]
            with localcontext() as context:
                context.prec = 34
                answer = (Decimal(80) + Decimal(20)) / Decimal(1000) * Decimal(100)
            record = {
                "schema": "candidate",
                "classification": "development_only_pending_visual_qa",
                "example_id": "v12c-1",
                "company_key": "development-company-a",
                "report_id": "DEV.pdf",
                "source_sha256": "source-hash",
                "page": 2,
                "table_id": "DEV:page-2:balance-sheet",
                "period": "31/12/2025",
                "answer_value": format(answer, "f"),
                "evidence": evidence,
                "review_status": "pending_visual_qa",
                "eligible_for_positive": False,
            }
            candidates.write_text(json.dumps(record) + "\n", encoding="utf-8")
            candidate_hash = hashlib.sha256(candidates.read_bytes()).hexdigest()
            checks = {
                "balance_sheet_page": True,
                "roles_unique": True,
                "values_match": True,
                "period_headers_match": True,
                "unit_match": True,
                "entity_scope_match": True,
                "no_cross_panel_leakage": True,
            }
            decisions = root / "development-decisions.json"
            decisions.write_text(json.dumps({
                "classification": "development_only_visual_qa",
                "review_id": "review-1",
                "candidate_sha256": candidate_hash,
                "records": [{
                    "source_name": "DEV.pdf",
                    "page": 2,
                    "render_sha256": "render-hash",
                    "decision": "approved",
                    "checks": checks,
                    "observed_values": {"31/12/2025": {
                        "short_term_borrowings": "80",
                        "long_term_borrowings": "20",
                        "total_capital": "1000",
                    }},
                }],
            }), encoding="utf-8")
            renders = root / "development-renders.json"
            renders.write_text(json.dumps({
                "candidate_sha256": candidate_hash,
                "records": [{
                    "source_name": "DEV.pdf",
                    "page": 2,
                    "render_sha256": "render-hash",
                }],
            }), encoding="utf-8")
            frozen, audit = freeze(
                candidates,
                decisions,
                renders,
                minimum_company_groups=1,
            )
            self.assertEqual(len(frozen), 1)
            self.assertTrue(frozen[0]["eligible_for_positive"])
            self.assertTrue(audit["summary"]["ready_for_frozen_evaluation"])
            self.assertFalse(audit["summary"]["ready_for_calibration"])

    @staticmethod
    def _leaf(role: str, value: str) -> dict[str, object]:
        return {
            "role": role,
            "value": value,
            "period": "31/12/2025",
            "unit": "VND",
            "source_name": "DEV.pdf",
            "source_sha256": "source-hash",
            "page": 2,
            "table_id": "DEV:page-2:balance-sheet",
        }


if __name__ == "__main__":
    unittest.main()
