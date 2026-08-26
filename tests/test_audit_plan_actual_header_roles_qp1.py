from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from scripts.audit_plan_actual_header_roles_qp1 import (
    _guard_development_path,
    _requested_role,
    _swap_role_text,
)


class AuditPlanActualHeaderRolesQP1Tests(unittest.TestCase):
    def test_role_swap_handles_vietnamese_ascii_and_abbreviations(self) -> None:
        swapped = _swap_role_text(
            "Kế hoạch | Thực hiện | Ke hoach | Thuc hien | KH | TH"
        )
        self.assertEqual(
            swapped,
            "Thực hiện | Kế hoạch | Thực hiện | Kế hoạch | Thực hiện | Kế hoạch",
        )

    def test_requested_role_does_not_force_mixed_question(self) -> None:
        self.assertEqual(
            _requested_role("Doanh thu thực hiện năm 2023 là bao nhiêu?"),
            "actual",
        )
        self.assertEqual(
            _requested_role("Doanh thu kế hoạch năm 2023 là bao nhiêu?"),
            "plan",
        )
        self.assertEqual(
            _requested_role("So sánh thực hiện với kế hoạch năm 2023"), ""
        )

    def test_guard_refuses_non_development_and_protected_input(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            development = Path(root) / "cohort_development.jsonl"
            _guard_development_path(development)
            with self.assertRaises(ValueError):
                _guard_development_path(Path(root) / "cohort.jsonl")
            with self.assertRaises(ValueError):
                _guard_development_path(
                    Path(root) / "validation" / "cohort_development.jsonl"
                )


if __name__ == "__main__":
    unittest.main()
