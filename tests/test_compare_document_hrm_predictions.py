import unittest

from scripts.compare_document_hrm_predictions import compare_records, render_markdown


def record(
    example_id: str,
    operation: str,
    *,
    operation_correct: bool,
    pointer_correct: bool,
    execution_correct: bool,
    predicted_operation: str | None = None,
    operands: list[int] | None = None,
) -> dict[str, object]:
    return {
        "example_id": example_id,
        "question": f"Question {example_id}",
        "checks": {
            "operation_correct": operation_correct,
            "pointer_pair_correct": pointer_correct,
            "execution_correct": execution_correct,
        },
        "expected": {"operation": operation},
        "predicted": {
            "operation": predicted_operation or operation,
            "operands": operands or [0, 1],
        },
    }


class PredictionComparisonTests(unittest.TestCase):
    def test_reports_complementary_execution_and_oracle(self):
        baseline = [
            record("1", "SELECT", operation_correct=True, pointer_correct=True, execution_correct=True),
            record("2", "SUBTRACT", operation_correct=True, pointer_correct=True, execution_correct=True),
            record("3", "SUBTRACT", operation_correct=True, pointer_correct=False, execution_correct=False),
            record("4", "RATIO_PERCENT", operation_correct=True, pointer_correct=False, execution_correct=False),
        ]
        candidate = [
            record("1", "SELECT", operation_correct=True, pointer_correct=True, execution_correct=True),
            record("2", "SUBTRACT", operation_correct=True, pointer_correct=False, execution_correct=False, operands=[2, 3]),
            record("3", "SUBTRACT", operation_correct=True, pointer_correct=True, execution_correct=True, operands=[4, 5]),
            record("4", "RATIO_PERCENT", operation_correct=True, pointer_correct=False, execution_correct=False, operands=[6, 7]),
        ]

        result = compare_records(baseline, candidate)
        execution = result["overall"]["execution_correct"]
        self.assertEqual(execution["both_correct"], 1)
        self.assertEqual(execution["baseline_only"], 1)
        self.assertEqual(execution["candidate_only"], 1)
        self.assertEqual(execution["neither"], 1)
        self.assertEqual(execution["oracle_accuracy"], 0.75)
        self.assertEqual(
            result["per_operation"]["SUBTRACT"]["execution_correct"]["oracle_accuracy"],
            1.0,
        )
        markdown = render_markdown(
            result, baseline_label="v7", candidate_label="v8b"
        )
        self.assertIn("75.00%", markdown)
        self.assertIn("v7", markdown)
        self.assertIn("v8b", markdown)

    def test_requires_same_full_example_set(self):
        baseline = [
            record("1", "SELECT", operation_correct=True, pointer_correct=True, execution_correct=True)
        ]
        candidate = [
            record("2", "SELECT", operation_correct=True, pointer_correct=True, execution_correct=True)
        ]
        with self.assertRaisesRegex(ValueError, "--include-correct"):
            compare_records(baseline, candidate)


if __name__ == "__main__":
    unittest.main()
