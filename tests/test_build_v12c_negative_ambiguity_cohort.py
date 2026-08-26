from __future__ import annotations

from pathlib import Path
import unittest

from scripts.build_v12c_negative_ambiguity_cohort import build


ROOT = Path(__file__).resolve().parents[1]


class V12CNegativeAmbiguityCohortTests(unittest.TestCase):
    def test_locked_visual_decisions_freeze_independent_negative_controls(self) -> None:
        records, audit = build(
            ROOT / "reports/v12c_interest_bearing_debt_share_negative_ambiguity_page_discovery_v1.json",
            ROOT / "reports/v12c_interest_bearing_debt_share_negative_ambiguity_render_queue_v1.json",
            ROOT / "reports/v12c_interest_bearing_debt_share_negative_ambiguity_visual_decisions_v1.json",
            [
                ROOT / "data/v12c_interest_bearing_debt_share_development_frozen_v1.jsonl",
                ROOT / "data/v12c_interest_bearing_debt_share_independent_calibration_frozen_v1.jsonl",
            ],
            minimum_company_groups=10,
        )
        self.assertEqual(len(records), 26)
        self.assertEqual(audit["summary"]["independent_company_groups"], 13)
        self.assertEqual(audit["summary"]["approved_pages"], 13)
        self.assertEqual(audit["summary"]["company_overlap_with_repair_or_positive_cohorts"], 0)
        self.assertTrue(all(row["expected_outcome"] == "ABSTAIN" for row in records))
        self.assertTrue(all(row["eligible_for_positive"] is False for row in records))
        self.assertFalse(audit["summary"]["ready_for_threshold_change"])


if __name__ == "__main__":
    unittest.main()
