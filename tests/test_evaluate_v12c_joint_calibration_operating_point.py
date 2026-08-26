import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.evaluate_v12c_joint_calibration_operating_point import (
    _cp_one_sided_lower,
    evaluate,
)


class JointOperatingPointTests(unittest.TestCase):
    def test_zero_acceptance_has_no_precision_lower_bound_evidence(self) -> None:
        self.assertIsNone(_cp_one_sided_lower(0, 0))

    def test_zero_successes_has_zero_lower_bound(self) -> None:
        self.assertEqual(_cp_one_sided_lower(0, 30), 0.0)

    def test_all_successes_exact_lower_bound(self) -> None:
        self.assertAlmostEqual(_cp_one_sided_lower(30, 30), 0.05 ** (1 / 30), places=12)

    def test_passing_repair_replay_does_not_unlock_phase_d(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_path = root / "protocol.json"
            positive_path = root / "positive.json"
            negative_path = root / "negative.json"
            threshold = 0.99
            checkpoint = "checkpoint"
            protocol_path.write_text(json.dumps({
                "classification": "development_only_precommitted_calibration_protocol",
                "locked_inputs": {
                    "frozen_ratio_percent_threshold": threshold,
                    "checkpoint": {"sha256": checkpoint},
                },
                "precommitted_operating_point_gate": {
                    "minimum_accepted_precision_point_estimate": 0.9,
                    "minimum_accepted_precision_one_sided_95_lower_bound": 0.9,
                    "minimum_accepted_answers": 30,
                    "minimum_accepted_company_groups": 15,
                    "minimum_positive_accepted_coverage": 0.5,
                    "maximum_negative_false_accept_rate": 0.0,
                },
            }), encoding="utf-8")
            positive_rows = [
                {
                    "company_key": f"positive-{index // 2}",
                    "accepted": True,
                    "program_correct": True,
                    "execution_correct": True,
                    "confidence_threshold": threshold,
                }
                for index in range(36)
            ]
            negative_rows = [
                {
                    "company_key": f"negative-{index}",
                    "false_accept": False,
                    "proof_failure_type": f"stratum-{index % 3}",
                    "confidence_threshold": threshold,
                }
                for index in range(31)
            ]
            common = {
                "model_sha256_before": checkpoint,
                "model_sha256_after": checkpoint,
                "threshold_changed": False,
            }
            positive_path.write_text(json.dumps({
                **common,
                "classification": "development_only_frozen_evaluation",
                "records": positive_rows,
            }), encoding="utf-8")
            negative_path.write_text(json.dumps({
                **common,
                "classification": "development_only_negative_control_frozen_evaluation",
                "records": negative_rows,
            }), encoding="utf-8")

            report = evaluate(protocol_path, positive_path, negative_path)

        self.assertTrue(report["gate"]["all_conditions_pass"])
        self.assertFalse(report["gate"]["threshold_change_authorized"])
        self.assertFalse(report["gate"]["phase_d_unlocked"])
        self.assertIn("repair evidence", report["conclusion"])


if __name__ == "__main__":
    unittest.main()
