from __future__ import annotations

from pathlib import Path
import unittest

from scripts.build_v12c_independent_calibration_cohort import build


ROOT = Path(__file__).resolve().parents[1]


class BuildIndependentV12CCalibrationCohortTests(unittest.TestCase):
    def test_locked_visual_decisions_build_group_disjoint_cohort(self) -> None:
        records, audit = build(
            ROOT / "reports/v12c_interest_bearing_debt_share_independent_calibration_page_discovery_v1.json",
            ROOT / "reports/v12c_interest_bearing_debt_share_independent_calibration_render_queue_v1.json",
            ROOT / "reports/v12c_interest_bearing_debt_share_independent_calibration_visual_decisions_v1.json",
            ROOT / "data/v12c_interest_bearing_debt_share_development_frozen_v1.jsonl",
            minimum_company_groups=8,
        )

        self.assertEqual(len(records), 20)
        self.assertEqual(audit["summary"]["independent_company_groups"], 9)
        self.assertEqual(audit["summary"]["company_overlap_with_repair_cohort"], 0)
        self.assertEqual(audit["summary"]["rejected_pages"], 2)
        self.assertEqual(audit["summary"]["excluded_non_numeric_periods"], 2)
        self.assertFalse(audit["summary"]["ready_for_threshold_calibration"])


if __name__ == "__main__":
    unittest.main()
