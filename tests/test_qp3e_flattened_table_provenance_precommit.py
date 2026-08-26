from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_qp3e_flattened_table_provenance_cohort import (
    DEFAULT_OUTPUT,
    DEFAULT_PRECOMMIT,
    build_records,
)
from scripts.validate_qp3e_flattened_table_provenance_precommit import (
    _refuse_protected,
    audit_reconstruction,
    build_report,
)


class QP3EFlattenedTableProvenancePrecommitTests(unittest.TestCase):
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
        self.assertEqual(len(rebuilt), 128)

    def test_precommit_gate_and_all_quotas_pass(self) -> None:
        report = build_report()
        self.assertTrue(report["precommit_gate_passed"])
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["oracle"]["state_exact"], 128)
        self.assertEqual(report["oracle"]["reason_set_exact"], 128)
        self.assertEqual(report["oracle"]["complete"], 24)
        self.assertEqual(report["oracle"]["unknown"], 104)
        self.assertEqual(report["counterfactuals"]["permutation_exact"], 8)
        self.assertEqual(
            report["counterfactuals"]["repeated_header_positive_exact"], 8
        )
        self.assertEqual(
            report["counterfactuals"]["boundary_hard_negative_exact"], 16
        )
        self.assertEqual(
            report["counterfactuals"]["conflict_hard_negative_exact"], 32
        )

    def test_complete_record_requires_every_independent_proof(self) -> None:
        record = self._record("F01_NATIVE_SINGLE_PAGE", "complete_explicit")
        self.assertEqual(
            audit_reconstruction(record, self.precommit),
            ("RECONSTRUCTABLE_COMPLETE", ()),
        )
        mutations = (
            ("source_boundary_state", "PARTIAL", "incomplete_source_boundary"),
            ("table_continuation_state", "OPEN", "incomplete_table_continuation"),
            (
                "extractor_alignment_state",
                "CONFLICT",
                "cross_extractor_layout_conflict",
            ),
        )
        for field, value, reason in mutations:
            with self.subTest(field=field):
                mutated = deepcopy(record)
                mutated[field] = value
                state, reasons = audit_reconstruction(mutated, self.precommit)
                self.assertEqual(state, "UNKNOWN")
                self.assertIn(reason, reasons)

    def test_role_cardinality_and_row_cell_coverage_are_separate(self) -> None:
        header_case = self._record(
            "F02_NATIVE_MULTI_PAGE", "header_role_cardinality_mismatch"
        )
        state, reasons = audit_reconstruction(header_case, self.precommit)
        self.assertEqual(state, "UNKNOWN")
        self.assertEqual(reasons, ("header_role_cardinality_mismatch",))

        cell_case = self._record("F02_NATIVE_MULTI_PAGE", "missing_row_cell")
        state, reasons = audit_reconstruction(cell_case, self.precommit)
        self.assertEqual(state, "UNKNOWN")
        self.assertEqual(reasons, ("cell_coverage_mismatch",))

    def test_physical_conflicts_cannot_be_repaired_by_same_logical_coverage(self) -> None:
        span_case = self._record("F03_OCR_BBOX", "overlapping_cell_span")
        state, reasons = audit_reconstruction(span_case, self.precommit)
        self.assertEqual(state, "UNKNOWN")
        self.assertEqual(reasons, ("physical_span_conflict",))

        row_case = self._record("F03_OCR_BBOX", "ambiguous_row_membership")
        state, reasons = audit_reconstruction(row_case, self.precommit)
        self.assertEqual(state, "UNKNOWN")
        self.assertEqual(reasons, ("row_membership_conflict",))

    def test_serialization_permutation_is_not_a_signal(self) -> None:
        base = self._record("F07_REVISED_PLAN", "complete_explicit")
        permuted = self._record("F07_REVISED_PLAN", "complete_permuted")
        self.assertEqual(
            audit_reconstruction(base, self.precommit),
            audit_reconstruction(permuted, self.precommit),
        )
        self.assertEqual(
            audit_reconstruction(permuted, self.precommit)[0],
            "RECONSTRUCTABLE_COMPLETE",
        )

    def test_numeric_question_candidate_and_top_k_signals_are_absent(self) -> None:
        serialized = "\n".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True)
            for record in self.records
        )
        for forbidden in (
            '"numeric_value":',
            '"numeric_answer":',
            '"candidate_score":',
            '"candidate_rank":',
            '"question":',
            '"question_text":',
            '"top_k_candidates":',
        ):
            self.assertNotIn(forbidden, serialized)

    def test_runtime_checkpoint_and_acceptance_remain_frozen(self) -> None:
        report = build_report()
        self.assertFalse(report["runtime_baseline"]["runtime_source_changed"])
        self.assertEqual(report["runtime_baseline"]["mismatches"], [])
        self.assertFalse(report["invariants"]["runtime_implementation_performed"])
        self.assertFalse(report["invariants"]["threshold_changed"])
        self.assertFalse(report["invariants"]["checkpoint_changed"])

    def test_validator_refuses_protected_input_path(self) -> None:
        protected = Path(tempfile.gettempdir()) / "validation" / "cohort.jsonl"
        with self.assertRaises(ValueError):
            _refuse_protected(protected)


if __name__ == "__main__":
    unittest.main()
