from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from hrm_finance_qa.evidence_schema import build_evidence_graph
from hrm_finance_qa.flattened_table_reconstruction import (
    RECONSTRUCTABLE_COMPLETE,
    assess_flattened_table_reconstruction,
    qp3d_manifest_from_flattened_record,
)
from hrm_finance_qa.program_data import extract_numeric_candidates
from scripts.evaluate_qp3f_flattened_table_reconstruction import (
    DEFAULT_COHORT,
    _refuse_protected,
    build_report,
    flattened_context,
)


def _candidates(context: str, question: str = "Chỉ tiêu 1 thực hiện năm 2023 là bao nhiêu?"):
    return extract_numeric_candidates(
        question,
        context,
        source_name="qp3f-test-development.txt",
        page=1,
        source_line_start=100,
        extraction_method="native",
    )


def _manifests(candidates) -> list[dict[str, object]]:
    marker = "ROLE_INVENTORY_MANIFEST: "
    return [
        json.loads(line[len(marker) :])
        for candidate in candidates
        for line in candidate.context.splitlines()
        if line.startswith(marker)
    ]


class QP3FFlattenedTableReconstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = [
            json.loads(line)
            for line in DEFAULT_COHORT.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_frozen_qp3e_state_reason_and_manifest_policy_are_exact(self) -> None:
        state_exact = 0
        reason_exact = 0
        manifest_exact = 0
        for record in self.records:
            assessment = assess_flattened_table_reconstruction(record)
            expected_state = record["expected_reconstruction_state"]
            state_exact += assessment.state == expected_state
            reason_exact += assessment.reasons == tuple(
                sorted(record["expected_reason_set"])
            )
            manifest_exact += (
                qp3d_manifest_from_flattened_record(record) is not None
            ) == (expected_state == RECONSTRUCTABLE_COMPLETE)
        self.assertEqual((state_exact, reason_exact, manifest_exact), (128, 128, 128))

    def test_complete_native_flattened_table_reaches_evidence_graph(self) -> None:
        candidates = _candidates(flattened_context())
        self.assertTrue(candidates)
        self.assertTrue(_manifests(candidates))
        graph = build_evidence_graph(candidates)
        self.assertEqual(graph.column_role_inventory_state, "EXPLICIT_COMPLETE")
        self.assertTrue(graph.table_role_inventories)
        self.assertTrue(all(item.state == "EXPLICIT_COMPLETE" for item in graph.table_role_inventories))

    def test_incomplete_or_conflicting_flattened_inputs_emit_no_manifest(self) -> None:
        controls = {
            "partial": flattened_context(bounded=False),
            "role_cardinality": flattened_context(role_cardinality_mismatch=True),
            "geometry": flattened_context(geometry_conflict=True),
        }
        for name, context in controls.items():
            with self.subTest(name=name):
                candidates = _candidates(context)
                self.assertFalse(_manifests(candidates))
                self.assertEqual(build_evidence_graph(candidates).column_role_inventory_state, "")

    def test_full_structure_is_captured_before_top_k(self) -> None:
        candidates = extract_numeric_candidates(
            "Chỉ tiêu 1 thực hiện năm 2023 là bao nhiêu?",
            flattened_context(rows=20),
            max_candidates=64,
            source_name="qp3f-large-development.txt",
            page=1,
            source_line_start=100,
            extraction_method="native",
        )
        manifests = _manifests(candidates)
        self.assertEqual(len(candidates), 64)
        self.assertTrue(manifests)
        self.assertEqual(
            max(len(item["observed_cell_coordinates"]) for item in manifests),
            100,
        )

    def test_question_and_numeric_identity_do_not_change_structural_manifest(self) -> None:
        context = flattened_context()
        changed_values = context.replace("1000000000", "7000000000").replace(
            "1100000000", "7100000000"
        )
        first = _manifests(_candidates(context, "Thực hiện năm 2023 là bao nhiêu?"))
        second = _manifests(_candidates(changed_values, "Kế hoạch năm 2023 là bao nhiêu?"))
        self.assertTrue(first and second)
        self.assertEqual(first[0], second[0])
        serialized = json.dumps(first[0], ensure_ascii=False, sort_keys=True)
        self.assertNotIn("1000000000", serialized)
        self.assertNotIn("7000000000", serialized)

    def test_qp3f_evaluator_passes_without_threshold_or_checkpoint_change(self) -> None:
        report = build_report()
        self.assertTrue(report["implementation_gate_passed"])
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["end_to_end"]["pass"], 9)
        self.assertFalse(report["invariants"]["threshold_changed"])
        self.assertFalse(report["invariants"]["checkpoint_changed"])

    def test_evaluator_refuses_protected_input_path(self) -> None:
        protected = Path(tempfile.gettempdir()) / "blind" / "cohort.jsonl"
        with self.assertRaises(ValueError):
            _refuse_protected(protected)


if __name__ == "__main__":
    unittest.main()
