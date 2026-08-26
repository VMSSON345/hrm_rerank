from __future__ import annotations

import unittest

from scripts.evaluate_v12c_negative_ambiguity_frozen import summarize


class V12CNegativeAmbiguityEvaluatorTests(unittest.TestCase):
    def test_false_accept_is_counted_as_failure(self) -> None:
        rows = [
            {
                "false_accept": True, "grounding_rejections": [], "abstain_reasons": [],
                "confidence": 0.995, "confidence_threshold": 0.99,
                "proof_failure_type": "missing", "required_proof_status": "INCOMPLETE",
                "company_key": "a", "predicted_operation": "RATIO_PERCENT",
            },
            {
                "false_accept": False, "grounding_rejections": ["missing_role"],
                "abstain_reasons": ["below_frozen_threshold", "missing_role"],
                "confidence": 0.8, "confidence_threshold": 0.99,
                "proof_failure_type": "missing", "required_proof_status": "INCOMPLETE",
                "company_key": "b", "predicted_operation": "RATIO_PERCENT",
            },
        ]
        summary = summarize(rows)
        self.assertEqual(summary["overall"]["controls"], 2)
        self.assertEqual(summary["overall"]["false_accepts"], 1)
        self.assertEqual(summary["overall"]["false_accept_rate"], 0.5)
        self.assertEqual(summary["overall"]["safe_abstentions"], 1)


if __name__ == "__main__":
    unittest.main()
