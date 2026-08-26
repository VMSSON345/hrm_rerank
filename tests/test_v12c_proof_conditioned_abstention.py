from types import SimpleNamespace
import unittest

from hrm_finance_qa.document_hrm import (
    _proof_conditioned_acceptance_eligible,
)
from hrm_finance_qa.program_data import ReasoningProgram


def _prediction(**overrides):
    program = ReasoningProgram("RATIO_PERCENT", (3, 2))
    trace = SimpleNamespace(
        program=program,
        generation_sources=("typed_slot_synthesis",),
        eligible=True,
        rejection_reasons=(),
        proof_contract_state="VALID",
        required_valid_fraction=1.0,
        required_unknown_fraction=0.0,
        proof_obligations=(
            "required:execution_valid:VALID",
            "required:operand_roles:VALID",
            "optional:same_row:UNKNOWN",
        ),
    )
    values = {
        "program": program,
        "program_dag": SimpleNamespace(nodes=(object(), object())),
        "program_search_trace": (trace,),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class ProofConditionedAbstentionTests(unittest.TestCase):
    def test_complete_required_proof_accepts_executable_derived_program(self):
        self.assertTrue(_proof_conditioned_acceptance_eligible(_prediction()))

    def test_optional_unknown_does_not_block_complete_required_proof(self):
        self.assertTrue(_proof_conditioned_acceptance_eligible(_prediction()))

    def test_direct_or_flat_program_is_not_eligible(self):
        self.assertFalse(
            _proof_conditioned_acceptance_eligible(
                _prediction(program_dag=None)
            )
        )

    def test_select_and_add_remain_on_calibrated_threshold(self):
        for operation in ("SELECT", "ADD"):
            with self.subTest(operation=operation):
                prediction = _prediction()
                program = ReasoningProgram(
                    operation,
                    (3,) if operation == "SELECT" else (0, 1),
                )
                trace = SimpleNamespace(
                    **{
                        **prediction.program_search_trace[0].__dict__,
                        "program": program,
                    }
                )
                self.assertFalse(
                    _proof_conditioned_acceptance_eligible(
                        _prediction(
                            program=program,
                            program_search_trace=(trace,),
                        )
                    )
                )

    def test_required_unknown_is_not_eligible(self):
        prediction = _prediction()
        trace = SimpleNamespace(
            **{
                **prediction.program_search_trace[0].__dict__,
                "proof_contract_state": "UNKNOWN",
                "required_valid_fraction": 0.5,
                "required_unknown_fraction": 0.5,
                "proof_obligations": (
                    "required:execution_valid:VALID",
                    "required:operand_roles:UNKNOWN",
                ),
            }
        )
        self.assertFalse(
            _proof_conditioned_acceptance_eligible(
                _prediction(program_search_trace=(trace,))
            )
        )

    def test_search_rejection_is_not_eligible(self):
        prediction = _prediction()
        trace = SimpleNamespace(
            **{
                **prediction.program_search_trace[0].__dict__,
                "eligible": False,
                "rejection_reasons": ("missing_required_role",),
            }
        )
        self.assertFalse(
            _proof_conditioned_acceptance_eligible(
                _prediction(program_search_trace=(trace,))
            )
        )

    def test_typed_synthesis_source_is_required(self):
        prediction = _prediction()
        trace = SimpleNamespace(
            **{
                **prediction.program_search_trace[0].__dict__,
                "generation_sources": ("neural_baseline",),
            }
        )
        self.assertFalse(
            _proof_conditioned_acceptance_eligible(
                _prediction(program_search_trace=(trace,))
            )
        )


if __name__ == "__main__":
    unittest.main()
