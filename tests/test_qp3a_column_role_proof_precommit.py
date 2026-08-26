from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_qp3a_column_role_proof_cohort import build_records
from scripts.validate_qp3a_column_role_proof_precommit import (
    DEFAULT_COHORT,
    build_report,
)


class QP3AColumnRoleProofPrecommitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = build_records()
        cls.report = build_report()

    def test_builder_is_balanced_and_deterministic(self) -> None:
        self.assertEqual(self.records, build_records())
        self.assertEqual(len(self.records), 128)
        scenarios = Counter(record["scenario_id"] for record in self.records)
        self.assertEqual(set(scenarios.values()), {16})
        groups = Counter(record["group_id"] for record in self.records)
        self.assertEqual(len(groups), 16)
        self.assertEqual(set(groups.values()), {8})

    def test_precommit_gate_and_oracle_are_exact(self) -> None:
        self.assertTrue(self.report["precommit_gate_passed"])
        self.assertEqual(self.report["errors"], [])
        self.assertEqual(self.report["oracle"]["state_exact"], 128)
        self.assertEqual(self.report["oracle"]["action_exact"], 128)

    def test_same_value_permutation_and_unknown_inventory_are_locked(self) -> None:
        counterfactuals = self.report["counterfactuals"]
        self.assertEqual(counterfactuals["same_value_conflict_exact"], 16)
        self.assertEqual(counterfactuals["permutation_exact"], 16)
        self.assertEqual(counterfactuals["unknown_inventory_exact"], 16)

    def test_no_numeric_or_score_signal_is_present(self) -> None:
        for record in self.records:
            self.assertFalse(record["numeric_value_used"])
            self.assertFalse(record["numeric_answer_used"])
            self.assertFalse(record["candidate_score_used"])
            self.assertFalse(record["candidate_order_is_signal"])
            for cell in record["evidence_cells"]:
                self.assertTrue(cell["extraction_value_token"].startswith("TOKEN_"))
                self.assertNotIn("numeric_value", cell)
                self.assertNotIn("candidate_score", cell)

    def test_tampered_expected_state_fails_gate(self) -> None:
        records = [dict(record) for record in self.records]
        records[0]["expected_column_role_proof_state"] = "UNKNOWN"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cohort.jsonl"
            path.write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                encoding="utf-8",
            )
            report = build_report(cohort_path=path)
        self.assertFalse(report["precommit_gate_passed"])
        self.assertEqual(report["oracle"]["state_exact"], 127)

    def test_checked_in_cohort_equals_builder_output(self) -> None:
        observed = [
            json.loads(line)
            for line in DEFAULT_COHORT.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(observed, self.records)


if __name__ == "__main__":
    unittest.main()
