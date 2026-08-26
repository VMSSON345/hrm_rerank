from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_qp3c_role_inventory_producer_cohort import (
    DEFAULT_OUTPUT,
    DEFAULT_PRECOMMIT,
    build_records,
)
from scripts.validate_qp3c_role_inventory_producer_precommit import (
    _refuse_protected,
    audit_inventory,
    build_report,
)


class QP3CRoleInventoryProducerPrecommitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.precommit = json.loads(DEFAULT_PRECOMMIT.read_text(encoding="utf-8"))
        cls.records = [
            json.loads(line)
            for line in DEFAULT_OUTPUT.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _record(self, layout: str, scenario: str) -> dict[str, object]:
        return deepcopy(
            next(
                record
                for record in self.records
                if record["layout_id"] == layout
                and record["scenario_id"] == scenario
            )
        )

    def test_builder_reproduces_frozen_cohort_exactly(self) -> None:
        rebuilt = build_records(self.precommit)
        self.assertEqual(rebuilt, self.records)
        self.assertEqual(len(rebuilt), 96)

    def test_precommit_gate_and_all_quotas_pass(self) -> None:
        report = build_report()
        self.assertTrue(report["precommit_gate_passed"])
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["oracle"]["state_exact"], 96)
        self.assertEqual(report["oracle"]["reason_set_exact"], 96)
        self.assertEqual(report["counterfactuals"]["permutation_exact"], 8)
        self.assertEqual(report["counterfactuals"]["inherited_anchor_exact"], 8)
        self.assertEqual(report["counterfactuals"]["conflict_negative_exact"], 24)

    def test_complete_table_requires_every_proof(self) -> None:
        record = self._record("L01_FLAT_PLAN_ACTUAL", "complete_explicit")
        self.assertEqual(
            audit_inventory(record, self.precommit),
            ("EXPLICIT_COMPLETE", ()),
        )
        mutations = (
            ("table_boundary_state", "PARTIAL", "incomplete_table_boundary"),
            ("header_alignment_state", "UNALIGNED", "header_alignment_unproven"),
            ("extractor_alignment_state", "CONFLICT", "cross_extractor_role_conflict"),
        )
        for field, value, reason in mutations:
            with self.subTest(field=field):
                mutated = deepcopy(record)
                mutated[field] = value
                state, reasons = audit_inventory(mutated, self.precommit)
                self.assertEqual(state, "UNKNOWN")
                self.assertIn(reason, reasons)

    def test_inherited_role_without_explicit_anchor_is_unknown(self) -> None:
        record = self._record(
            "L02_HIERARCHICAL_MERGED", "complete_inherited_with_anchor"
        )
        self.assertEqual(audit_inventory(record, self.precommit)[0], "EXPLICIT_COMPLETE")
        record["inheritance_anchors"] = []
        state, reasons = audit_inventory(record, self.precommit)
        self.assertEqual(state, "UNKNOWN")
        self.assertIn("missing_inheritance_anchor", reasons)

    def test_top_k_subset_cannot_claim_complete_cell_coverage(self) -> None:
        record = self._record("L04_PARALLEL_ENTITY_PANELS", "complete_explicit")
        record["observed_cell_coordinates"] = record["observed_cell_coordinates"][:2]
        record["physical_observations"] = record["physical_observations"][:2]
        state, reasons = audit_inventory(record, self.precommit)
        self.assertEqual(state, "UNKNOWN")
        self.assertIn("cell_coverage_mismatch", reasons)

    def test_numeric_values_and_question_signals_are_absent(self) -> None:
        serialized = "\n".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True)
            for record in self.records
        )
        for forbidden in (
            '"numeric_value":',
            '"numeric_answer":',
            '"candidate_score":',
            '"question":',
            '"question_text":',
        ):
            self.assertNotIn(forbidden, serialized)

    def test_runtime_and_checkpoint_hashes_are_frozen(self) -> None:
        report = build_report()
        self.assertFalse(report["runtime_baseline"]["runtime_source_changed"])
        self.assertEqual(report["runtime_baseline"]["mismatches"], [])
        self.assertFalse(report["invariants"]["threshold_changed"])
        self.assertFalse(report["invariants"]["checkpoint_changed"])

    def test_validator_refuses_protected_input_path(self) -> None:
        protected = Path(tempfile.gettempdir()) / "validation" / "cohort.jsonl"
        with self.assertRaises(ValueError):
            _refuse_protected(protected)


if __name__ == "__main__":
    unittest.main()
