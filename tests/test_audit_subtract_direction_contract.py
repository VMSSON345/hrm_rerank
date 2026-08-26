from __future__ import annotations

from pathlib import Path
import unittest

from scripts.audit_subtract_direction_contract import (
    _protected_record_reason,
    refuse_protected_source,
)


class SubtractDirectionAuditTests(unittest.TestCase):
    def test_refuses_protected_paths(self) -> None:
        for name in (
            "blind.jsonl",
            "company_test.jsonl",
            "final-eval.jsonl",
            "frozen_holdout.jsonl",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                refuse_protected_source(Path(name))

    def test_excludes_embedded_validation_record(self) -> None:
        reason = _protected_record_reason(
            "synthetic-validation-subtract-0001", None
        )

        self.assertEqual(reason, "example_id:validation")

    def test_development_record_remains_eligible(self) -> None:
        reason = _protected_record_reason(
            "synthetic-train-subtract-0001", {"split": "development"}
        )

        self.assertEqual(reason, "")


if __name__ == "__main__":
    unittest.main()
