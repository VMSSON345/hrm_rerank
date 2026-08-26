from pathlib import Path
import unittest

from hrm_finance_qa.program_data import load_program_examples
from scripts.build_qp3_core_add_diverse_development_cohort import (
    _cohort_counterfactual,
)
from scripts.evaluate_qp3_core_s4_safe_replay import _proof_decision


ROOT = Path(__file__).resolve().parents[1]
OCR_V1 = ROOT / "data/finhrm_qp3_core_add_diversity_ocr_generated_pool_v1.jsonl"
OCR_V2 = ROOT / "data/finhrm_qp3_core_add_diversity_ocr_generated_pool_v2.jsonl"


class Qp3CoreAddDiverseDevelopmentCohortTests(unittest.TestCase):
    def test_explicit_wrong_period_counterfactual_is_deterministic_and_safe(self) -> None:
        example = next(
            item
            for item in load_program_examples(OCR_V1)
            if item.example_id == "synthetic-document-000023"
        )

        first = _cohort_counterfactual(example)
        second = _cohort_counterfactual(example)

        self.assertIsNotNone(first)
        self.assertEqual(first, second)
        assert first is not None
        question, program, mutation = first
        self.assertEqual(mutation, "EXPLICIT_WRONG_PERIOD")
        selected = {candidate.index: candidate for candidate in example.candidates}
        changed = [
            (gold, mutated)
            for gold, mutated in zip(example.program.operands, program.operands)
            if gold != mutated
        ]
        self.assertEqual(len(changed), 1)
        gold, mutated = changed[0]
        self.assertEqual(selected[gold].row_label, selected[mutated].row_label)
        self.assertNotEqual(selected[gold].period, selected[mutated].period)
        decision = _proof_decision(question, program, example.candidates)
        self.assertNotEqual(decision["action"], "ANSWER")

    def test_wrong_operation_fallback_covers_pool_without_alternative_cell(self) -> None:
        example = next(
            item
            for item in load_program_examples(OCR_V2)
            if item.example_id == "synthetic-document-000054"
        )

        counterfactual = _cohort_counterfactual(example)

        self.assertIsNotNone(counterfactual)
        assert counterfactual is not None
        question, program, mutation = counterfactual
        self.assertEqual(mutation, "WRONG_OPERATION")
        self.assertEqual(program.operation, "SUBTRACT")
        self.assertEqual(program.operands, example.program.operands)
        decision = _proof_decision(question, program, example.candidates)
        self.assertNotEqual(decision["action"], "ANSWER")


if __name__ == "__main__":
    unittest.main()
