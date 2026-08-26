from __future__ import annotations

import unittest

from hrm_finance_qa.program_data import (
    NumericCandidate,
    ProgramExample,
    ReasoningProgram,
)
from hrm_finance_qa.v12b2_supervision import build_phase_c_record


def _candidate(index: int, value: str, metric: str, year: str = "2025") -> NumericCandidate:
    return NumericCandidate(
        index=index,
        value=value,
        raw=value,
        context=(
            f"ROW: {metric}\nCOLUMN: {year}\nPERIOD: {year}\nTABLE: table-1"
        ),
        line_number=index + 1,
        char_start=index * 4,
        char_end=index * 4 + len(value),
        unit="VND",
        is_percent=False,
        is_year_like=False,
        relevance=1.0,
        source_name="unseen-company.pdf",
        extraction_method="native",
    )


class V12B2PhaseCSupervisionTests(unittest.TestCase):
    @staticmethod
    def _pair_record(example: ProgramExample) -> dict[str, object]:
        operation = example.program.operation
        left, right = example.program.operands
        return {
            "example_id": example.example_id,
            "operation": operation,
            "candidates": [
                {
                    "program_key": f"{operation}:{left},{right}",
                    "positive_class": "canonical_exact",
                    "exact_gold": True,
                    "execution_correct": True,
                    "training_state": "VALID",
                    "proof_contract_state": "VALID",
                },
                {
                    "program_key": f"{operation}:{right},{left}",
                    "positive_class": "negative",
                    "exact_gold": False,
                    "execution_correct": False,
                    "training_state": "INVALID",
                    "proof_contract_state": "INVALID",
                },
            ],
        }

    def test_par_value_multiply_with_mismatched_cell_is_excluded(self) -> None:
        share_count = _candidate(0, "1000000", "Số cổ phiếu đang lưu hành")
        wrong_neighbour = NumericCandidate(
            **{
                **_candidate(
                    1, "120", "Mệnh giá cổ phiếu: 10.000 VND/cổ phiếu"
                ).__dict__,
                "context": (
                    "ROW: Mệnh giá cổ phiếu: 10.000 VND/cổ phiếu\n"
                    "COLUMN: 2025\nPERIOD: 2025\nTABLE: table-1\n"
                    "CELL: | Mệnh giá cổ phiếu: 10.000 VND/cổ phiếu | "
                    "Trong nước | <VALUE> 120 </VALUE> | 10.000.000 |"
                ),
            }
        )
        example = ProgramExample(
            example_id="phase-c-invalid-multiply",
            question=(
                "Giá trị theo mệnh giá của số cổ phiếu đang lưu hành "
                "năm 2025 là bao nhiêu?"
            ),
            context="",
            answer_text="120000000",
            candidates=(share_count, wrong_neighbour),
            program=ReasoningProgram("MULTIPLY", (0, 1)),
            answer_value="120000000",
        )
        record = {
            "example_id": example.example_id,
            "operation": "MULTIPLY",
            "candidates": [
                {
                    "program_key": "MULTIPLY:0,1",
                    "positive_class": "canonical_exact",
                    "exact_gold": True,
                    "execution_correct": True,
                    "training_state": "VALID",
                    "proof_contract_state": "VALID",
                },
                {
                    "program_key": "MULTIPLY:1,0",
                    "positive_class": "canonical_exact",
                    "exact_gold": True,
                    "execution_correct": True,
                    "training_state": "VALID",
                    "proof_contract_state": "VALID",
                },
            ],
        }

        supervision, audit = build_phase_c_record(record, example)

        self.assertFalse(supervision["has_rank_positive"])
        self.assertFalse(audit["supervision_integrity_eligible"])
        self.assertIn(
            "multiply_par_value_unit_algebra_unproven",
            audit["supervision_integrity_issues"],
        )

    def test_semantic_grounded_member_is_positive_but_execution_only_is_not(self) -> None:
        example = ProgramExample(
            example_id="phase-c-select",
            question="Doanh thu nam 2025 la bao nhieu?",
            context="",
            answer_text="10",
            candidates=(
                _candidate(0, "10", "Doanh thu"),
                _candidate(1, "10", "Doanh thu"),
                _candidate(2, "10", "Loi nhuan"),
                _candidate(3, "8", "Doanh thu", "2024"),
            ),
            program=ReasoningProgram("SELECT", (0,)),
            answer_value="10",
        )
        record = {
            "example_id": example.example_id,
            "operation": "SELECT",
            "candidates": [
                {
                    "program_key": "SELECT:0",
                    "positive_class": "canonical_exact",
                    "exact_gold": True,
                    "execution_correct": True,
                    "training_state": "VALID",
                    "proof_contract_state": "VALID",
                    "required_unknown_fraction": 0.0,
                    "semantic_equivalent_members": [
                        {
                            "program_key": "SELECT:0",
                            "positive_class": "canonical_exact",
                            "exact_gold": True,
                            "execution_correct": True,
                            "training_state": "VALID",
                        },
                        {
                            "program_key": "SELECT:1",
                            "positive_class": "grounded_equivalent",
                            "exact_gold": False,
                            "execution_correct": True,
                            "training_state": "VALID",
                        },
                    ],
                },
                {
                    "program_key": "SELECT:2",
                    "positive_class": "execution_only",
                    "exact_gold": False,
                    "execution_correct": True,
                    "training_state": "VALID",
                    "proof_contract_state": "VALID",
                    "required_unknown_fraction": 0.0,
                },
                {
                    "program_key": "SELECT:3",
                    "positive_class": "negative",
                    "exact_gold": False,
                    "execution_correct": False,
                    "training_state": "UNKNOWN",
                    "proof_contract_state": "UNKNOWN",
                    "required_unknown_fraction": 0.5,
                },
            ],
        }

        supervision, audit = build_phase_c_record(record, example)

        self.assertEqual(
            supervision["positive_program_keys"], ["SELECT:0", "SELECT:1"]
        )
        self.assertIn(
            "execution_only_same_result",
            supervision["candidates"][1]["hard_negative_types"],
        )
        self.assertIn(
            "wrong_metric_same_table",
            supervision["candidates"][1]["hard_negative_types"],
        )
        self.assertFalse(supervision["candidates"][1]["rank_positive"])
        self.assertEqual(audit["execution_only_promoted_to_positive"], 0)
        self.assertEqual(audit["label_contradictions"], 0)

    def test_commutative_permutation_is_a_canonical_positive(self) -> None:
        example = ProgramExample(
            example_id="phase-c-add",
            question="Tong Doanh thu va Loi nhuan nam 2025 la bao nhieu?",
            context="",
            answer_text="15",
            candidates=(
                _candidate(0, "10", "Doanh thu"),
                _candidate(1, "5", "Loi nhuan"),
                _candidate(2, "7", "No phai tra"),
            ),
            program=ReasoningProgram("ADD", (0, 1)),
            answer_value="15",
        )
        record = {
            "example_id": example.example_id,
            "operation": "ADD",
            "candidates": [
                {
                    "program_key": "ADD:0,1",
                    "positive_class": "canonical_exact",
                    "exact_gold": True,
                    "execution_correct": True,
                    "training_state": "VALID",
                    "proof_contract_state": "VALID",
                    "semantic_equivalent_members": [
                        {
                            "program_key": "ADD:0,1",
                            "positive_class": "canonical_exact",
                            "exact_gold": True,
                            "execution_correct": True,
                            "training_state": "VALID",
                        },
                        {
                            "program_key": "ADD:1,0",
                            "positive_class": "canonical_exact",
                            "exact_gold": True,
                            "execution_correct": True,
                            "training_state": "VALID",
                        },
                    ],
                },
                {
                    "program_key": "ADD:0,2",
                    "positive_class": "negative",
                    "exact_gold": False,
                    "execution_correct": False,
                    "training_state": "VALID",
                    "proof_contract_state": "VALID",
                },
            ],
        }

        supervision, audit = build_phase_c_record(record, example)

        self.assertEqual(
            supervision["positive_program_keys"], ["ADD:0,1", "ADD:1,0"]
        )
        self.assertEqual(audit["positive_classes"]["canonical_exact"], 2)
        self.assertTrue(audit["eligible_positive_and_hard_negative"])

    def test_ambiguous_subtract_is_excluded_from_positive_supervision(self) -> None:
        example = ProgramExample(
            example_id="phase-c-ambiguous-subtract",
            question="Chênh lệch giữa Doanh thu và Chi phí là bao nhiêu?",
            context="",
            answer_text="5",
            candidates=(
                _candidate(0, "10", "Doanh thu"),
                _candidate(1, "5", "Chi phi"),
            ),
            program=ReasoningProgram("SUBTRACT", (0, 1)),
            answer_value="5",
        )

        supervision, audit = build_phase_c_record(
            self._pair_record(example), example
        )

        self.assertFalse(supervision["has_rank_positive"])
        self.assertIn(
            "gold_direction_ambiguous",
            audit["supervision_integrity_issues"],
        )

    def test_explicit_subtract_creates_proven_reversed_pair_negative(self) -> None:
        example = ProgramExample(
            example_id="phase-c-directed-subtract",
            question="Tính Doanh thu trừ Chi phí năm 2025 là bao nhiêu?",
            context="",
            answer_text="5",
            candidates=(
                _candidate(0, "10", "Doanh thu"),
                _candidate(1, "5", "Chi phi"),
            ),
            program=ReasoningProgram("SUBTRACT", (0, 1)),
            answer_value="5",
        )

        supervision, audit = build_phase_c_record(
            self._pair_record(example), example
        )

        self.assertTrue(supervision["has_rank_positive"])
        self.assertIn(
            "reversed_direction_or_roles",
            supervision["candidates"][1]["hard_negative_types"],
        )
        self.assertEqual(audit["reversed_pair_negatives_withheld"], 0)

    def test_unproven_identity_reversal_is_withheld_not_generic_negative(self) -> None:
        example = ProgramExample(
            example_id="phase-c-unproven-identity-direction",
            question="Vốn lưu động thuần năm 2025 là bao nhiêu?",
            context="",
            answer_text="5",
            candidates=(
                _candidate(0, "10", "Tai san ngan han"),
                _candidate(1, "5", "No ngan han"),
            ),
            program=ReasoningProgram("SUBTRACT", (0, 1)),
            answer_value="5",
        )

        supervision, audit = build_phase_c_record(
            self._pair_record(example), example
        )

        self.assertTrue(supervision["has_rank_positive"])
        self.assertEqual(
            supervision["candidates"][1]["hard_negative_types"], []
        )
        self.assertEqual(
            supervision["candidates"][1]["hard_negative_withheld"],
            "unproven_direction",
        )
        self.assertEqual(audit["reversed_pair_negatives_withheld"], 1)

    def test_header_endpoint_is_excluded_from_positive_supervision(self) -> None:
        header = NumericCandidate(
            **{
                **_candidate(0, "10", "Chi tieu").__dict__,
                "row_type": "header",
            }
        )
        example = ProgramExample(
            example_id="phase-c-header-endpoint",
            question="Chi tiêu năm 2025 là bao nhiêu?",
            context="",
            answer_text="10",
            candidates=(header,),
            program=ReasoningProgram("SELECT", (0,)),
            answer_value="10",
        )
        record = {
            "example_id": example.example_id,
            "operation": "SELECT",
            "candidates": [
                {
                    "program_key": "SELECT:0",
                    "positive_class": "canonical_exact",
                    "exact_gold": True,
                    "execution_correct": True,
                    "training_state": "VALID",
                    "proof_contract_state": "VALID",
                }
            ],
        }

        supervision, audit = build_phase_c_record(record, example)

        self.assertFalse(supervision["has_rank_positive"])
        self.assertIn(
            "gold_endpoint_non_value_row:header",
            audit["supervision_integrity_issues"],
        )


if __name__ == "__main__":
    unittest.main()
