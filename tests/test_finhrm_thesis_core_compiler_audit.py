from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_finhrm_thesis_core_compiler import (
    DEFAULT_CATALOG,
    DEFAULT_CONTRACT,
    _canonical_details,
    _read_catalog,
    _strict_checks,
    build_report,
)


class FinHRMThesisCoreCompilerAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads(DEFAULT_CONTRACT.read_text(encoding="utf-8"))

    def test_canonical_contract_has_exactly_sixteen_families(self) -> None:
        details = _canonical_details(self.contract)
        self.assertEqual(len(details), 16)
        self.assertEqual(len({row["family_id"] for row in details}), 16)

    def test_legacy_catalog_scope_is_135_rows_and_excludes_ibds(self) -> None:
        core_ids = set(self.contract["scope"]["core_family_ids"])
        rows = [row for row in _read_catalog(DEFAULT_CATALOG) if row["family_id"] in core_ids]
        self.assertEqual(len(rows), 135)
        self.assertNotIn("IBDS01", {row["family_id"] for row in rows})

    def test_strict_check_does_not_invent_operand_slots(self) -> None:
        expected = self.contract["families"][0]["expected"]
        actual = {
            "behavior": "ANSWER",
            "abstention_reason": "",
            "answer_mode": "SINGLE_SCALAR",
            "requested_unit_family": "",
            "operation": "SELECT",
            "operation_candidates": ["SELECT"],
            "requested_years": ["2023"],
            "column_roles": ["ACTUAL"],
            "operand_slots": [],
            "direction_intent": None,
            "reference_base": None,
            "program_dag_operations": [],
        }
        checks = _strict_checks(expected, actual)
        self.assertTrue(checks["operation"])
        self.assertTrue(checks["column_roles"])
        self.assertFalse(checks["operand_schema_present"])
        self.assertFalse(checks["operand_metric_binding"])

    def test_abstention_requires_the_precommitted_reason(self) -> None:
        expected = next(
            row["expected"] for row in self.contract["families"] if row["family_id"] == "C14"
        )
        checks = _strict_checks(
            expected,
            {
                "behavior": "ABSTAIN",
                "abstention_reason": "",
                "answer_mode": "SINGLE_SCALAR",
                "requested_unit_family": "percent",
            },
        )
        self.assertTrue(checks["behavior"])
        self.assertFalse(checks["abstention_reason"])

    def test_report_is_compiler_only_and_keeps_runtime_locked(self) -> None:
        report = build_report()
        self.assertEqual(report["strict_canonical_contract"]["examples"], 16)
        self.assertEqual(report["legacy_catalog_robustness"]["examples"], 135)
        self.assertFalse(report["release_gate"]["runtime_change_authorized"])
        self.assertFalse(report["invariants"]["model_loaded"])
        self.assertFalse(report["invariants"]["threshold_changed"])
        self.assertFalse(report["invariants"]["phase_d_unlocked"])

    def test_tampered_catalog_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            path.write_text(DEFAULT_CATALOG.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                build_report(DEFAULT_CONTRACT, path)


if __name__ == "__main__":
    unittest.main()
