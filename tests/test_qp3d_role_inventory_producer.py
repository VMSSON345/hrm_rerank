from __future__ import annotations

import json
import unittest

from hrm_finance_qa.evidence_schema import build_evidence_graph
from hrm_finance_qa.program_data import ReasoningProgram, extract_numeric_candidates
from hrm_finance_qa.question_planner import compile_question_plan
from hrm_finance_qa.symbolic_verifier import verify_program
from scripts.evaluate_qp3d_role_inventory_producer import (
    _end_to_end_cases,
    build_report,
)


def _extract(question: str, context: str, *, max_candidates: int = 64):
    return extract_numeric_candidates(
        question,
        context,
        max_candidates=max_candidates,
        source_name="synthetic-development.md",
        page=1,
        extraction_method="native",
    )


class QP3DRoleInventoryProducerTests(unittest.TestCase):
    def test_frozen_qp3c_cohort_and_end_to_end_gate_pass(self) -> None:
        report = build_report()
        self.assertTrue(report["implementation_gate_passed"])
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["frozen_cohort"]["state_exact"], 96)
        self.assertEqual(report["frozen_cohort"]["reason_set_exact"], 96)
        self.assertEqual(report["end_to_end"]["pass"], 8)

    def test_every_end_to_end_control_passes_individually(self) -> None:
        cases = _end_to_end_cases()
        self.assertEqual(len(cases), 8)
        self.assertEqual(
            [item["id"] for item in cases if not item["passed"]], []
        )

    def test_manifest_is_created_before_top_k_pruning(self) -> None:
        rows = [
            f"| Chi tieu {index} | {100000 + index} | {200000 + index} |"
            for index in range(40)
        ]
        context = "\n".join(
            (
                "Bang ke hoach",
                "| Chi tieu | Ke hoach 2023 | Thuc hien 2023 |",
                "|---|---:|---:|",
                *rows,
                "Ket thuc bang",
            )
        )
        candidates = _extract(
            "Chi tieu 20 nam 2023 la bao nhieu?", context, max_candidates=64
        )
        self.assertEqual(len(candidates), 64)
        manifest_line = next(
            line
            for line in candidates[0].context.splitlines()
            if line.startswith("ROLE_INVENTORY_MANIFEST:")
        )
        manifest = json.loads(manifest_line.split(":", 1)[1].strip())
        declared = set(manifest["declared_data_column_ids"])
        declared_coordinates = [
            coordinate
            for coordinate in manifest["observed_cell_coordinates"]
            if coordinate.split("|", 1)[1] in declared
        ]
        self.assertEqual(len(declared_coordinates), 80)
        graph = build_evidence_graph(candidates)
        self.assertEqual(graph.column_role_inventory_state, "EXPLICIT_COMPLETE")

    def test_plaintext_and_ordinary_period_tables_remain_not_applicable(self) -> None:
        plaintext = build_evidence_graph(
            _extract(
                "Doanh thu nam 2023 la bao nhieu?",
                "Doanh thu nam 2023 dat 100000 dong.",
            )
        )
        period_table = build_evidence_graph(
            _extract(
                "Tai san nam 2023 la bao nhieu?",
                "\n".join(
                    (
                        "Bang can doi",
                        "| Chi tieu | 2022 | 2023 |",
                        "|---|---:|---:|",
                        "| Tai san | 90000 | 100000 |",
                        "Ket thuc bang",
                    )
                ),
            )
        )
        self.assertEqual(plaintext.column_role_inventory_state, "")
        self.assertEqual(period_table.column_role_inventory_state, "")
        self.assertEqual(period_table.table_role_inventories, ())

    def test_table_scoped_inventory_does_not_leak_to_other_tables(self) -> None:
        context = "\n".join(
            (
                "Bang ke hoach",
                "| Chi tieu | Ke hoach 2023 | Thuc hien 2023 |",
                "|---|---:|---:|",
                "| Doanh thu | 110000 | 100000 |",
                "Ket thuc bang 1",
                "",
                "Bang can doi",
                "| Chi tieu | 2022 | 2023 |",
                "|---|---:|---:|",
                "| Tai san | 90000 | 120000 |",
                "Ket thuc bang 2",
            )
        )
        candidates = _extract("Tai san nam 2023 la bao nhieu?", context)
        graph = build_evidence_graph(candidates)
        asset = next(
            cell
            for cell in graph.cells
            if cell.metric_key == "taisan" and "2023" in cell.explicit_years
        )
        verification = verify_program(
            compile_question_plan("Tai san nam 2023 la bao nhieu?"),
            ReasoningProgram("SELECT", (asset.index,)),
            graph,
        )
        obligation = next(
            item
            for item in verification.proof_obligations
            if item.name == "column_role_unique"
        )
        self.assertEqual(obligation.applicability, "not_applicable")

    def test_explicit_empty_override_preserves_diagnostic_control(self) -> None:
        context = "\n".join(
            (
                "Bang ket qua",
                "| Chi tieu | Thuc hien 2023 |",
                "|---|---:|",
                "| Doanh thu | 100000 |",
                "Ket thuc bang",
            )
        )
        candidates = _extract("Doanh thu nam 2023 la bao nhieu?", context)
        automatic = build_evidence_graph(candidates)
        overridden = build_evidence_graph(
            candidates, column_role_inventory_state=""
        )
        self.assertEqual(automatic.column_role_inventory_state, "EXPLICIT_COMPLETE")
        self.assertEqual(overridden.column_role_inventory_state, "")
        self.assertEqual(overridden.table_role_inventories, ())


if __name__ == "__main__":
    unittest.main()
