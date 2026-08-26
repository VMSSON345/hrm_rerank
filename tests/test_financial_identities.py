from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from hrm_finance_qa.evidence_schema import build_evidence_graph
from hrm_finance_qa.financial_identities import (
    FINANCIAL_IDENTITIES,
    FINANCIAL_IDENTITY_CATALOG_PATH,
    FinancialIdentityCatalogError,
    get_financial_identity,
    load_financial_identity_catalog,
    match_financial_identities,
)
from hrm_finance_qa.program_data import NumericCandidate, OPERATION_ARITY, ReasoningProgram
from hrm_finance_qa.program_synthesis import synthesize_typed_programs
from hrm_finance_qa.question_planner import compile_question_plan
from hrm_finance_qa.symbolic_verifier import VerificationState, verify_program


def _candidate(
    index: int,
    value: str,
    metric: str,
    year: str = "2024",
    *,
    unit: str = "VND",
) -> NumericCandidate:
    return NumericCandidate(
        index=index,
        value=value,
        raw=value,
        context=(
            f"ROW: {metric}\nROW_TYPE: detail\nCOLUMN: {year}\n"
            f"PERIOD: {year}\nTABLE: balance-sheet\nSCOPE: reporting"
        ),
        line_number=index + 1,
        char_start=index * 10,
        char_end=index * 10 + len(value),
        unit=unit,
        is_percent=False,
        is_year_like=False,
        relevance=1.0 - index * 0.05,
        source_name="unseen-company-report.pdf",
        extraction_method="native",
    )


class FinancialIdentityCatalogTests(unittest.TestCase):
    def test_catalog_is_loaded_from_versioned_yaml(self) -> None:
        self.assertTrue(FINANCIAL_IDENTITY_CATALOG_PATH.is_file())
        self.assertGreater(len(load_financial_identity_catalog()), 10)

    def test_loader_rejects_operation_arity_mismatch(self) -> None:
        content = """
schema_version: 1
identities:
  - id: invalid.add
    target: {key: total, aliases: [total]}
    operation: ADD
    operands:
      - {key: only_one, aliases: [one], unit_family: currency}
    result_unit_family: currency
"""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "identities.yaml"
            path.write_text(content, encoding="utf-8")

            with self.assertRaises(FinancialIdentityCatalogError):
                load_financial_identity_catalog(path)

    def test_catalog_ids_and_arities_are_valid(self) -> None:
        ids = [identity.identity_id for identity in FINANCIAL_IDENTITIES]

        self.assertEqual(len(ids), len(set(ids)))
        for identity in FINANCIAL_IDENTITIES:
            self.assertIn(identity.operation, OPERATION_ARITY)
            self.assertEqual(
                len(identity.operands), OPERATION_ARITY[identity.operation]
            )

    def test_total_assets_matches_both_safe_derivations(self) -> None:
        identities = match_financial_identities("Tong tai san")

        self.assertEqual(
            {identity.identity_id for identity in identities},
            {
                "balance.total_assets.by_term",
                "balance.accounting_equation",
            },
        )

    def test_specific_ratio_does_not_activate_total_assets_identity(self) -> None:
        identities = match_financial_identities(
            "He so no tren tong tai san"
        )

        self.assertEqual(
            tuple(identity.identity_id for identity in identities),
            ("ratio.debt_to_assets",),
        )

    def test_narrow_liability_metric_does_not_activate_total_liabilities(self) -> None:
        self.assertEqual(
            match_financial_identities("no phai tra dai han"), ()
        )

    def test_financial_debt_is_a_safe_active_one_step_identity(self) -> None:
        matches = match_financial_identities("tong no vay chiu lai")

        self.assertEqual(
            tuple(identity.identity_id for identity in matches),
            ("balance.financial_debt",),
        )
        self.assertEqual(matches[0].operation, "ADD")
        self.assertEqual(matches[0].status, "active")

    def test_conditional_identity_is_not_active_by_default(self) -> None:
        active = match_financial_identities("ROA")
        audited = match_financial_identities(
            "ROA", include_conditional=True
        )

        self.assertEqual(active, ())
        self.assertEqual(
            tuple(identity.identity_id for identity in audited),
            ("conditional.return_on_assets",),
        )

    def test_planner_attaches_identity_without_forcing_arithmetic(self) -> None:
        plan = compile_question_plan(
            "Tong no phai tra nam 2024 la bao nhieu?"
        )

        self.assertIsNone(plan.operation)
        self.assertEqual(plan.operation_candidates, ())
        self.assertEqual(
            plan.financial_identity_ids,
            ("balance.total_liabilities",),
        )

    def test_identity_synthesis_uses_typed_roles(self) -> None:
        candidates = (
            _candidate(0, "60", "No ngan han"),
            _candidate(1, "40", "No dai han"),
            _candidate(2, "100", "Doanh thu thuan"),
        )
        graph = build_evidence_graph(candidates)
        plan = compile_question_plan(
            "Tong no phai tra nam 2024 la bao nhieu?"
        )
        identity = get_financial_identity("balance.total_liabilities")

        result = synthesize_typed_programs(
            plan,
            identity.operation,
            graph,
            focus_text=plan.metric_text,
            operand_role_texts=identity.operand_texts,
            operand_unit_families=tuple(
                role.unit_family for role in identity.operands
            ),
        )

        self.assertIn(ReasoningProgram("ADD", (0, 1)), result.programs)
        self.assertNotIn(ReasoningProgram("ADD", (1, 2)), result.programs)

    def test_component_check_followup_still_activates_total_liabilities_identity(self) -> None:
        plan = compile_question_plan(
            "Tong no phai tra nam 2026 la bao nhieu? "
            "Hay liet ke no ngan han va no dai han de kiem tra."
        )

        self.assertEqual(
            plan.financial_identity_ids, ("balance.total_liabilities",)
        )

    def test_verifier_accepts_identity_and_rejects_arbitrary_pair(self) -> None:
        candidates = (
            _candidate(0, "60", "No ngan han"),
            _candidate(1, "40", "No dai han"),
            _candidate(2, "100", "Doanh thu thuan"),
        )
        graph = build_evidence_graph(candidates)
        plan = compile_question_plan(
            "Tong no phai tra nam 2024 la bao nhieu?"
        )

        valid = verify_program(
            plan,
            ReasoningProgram("ADD", (0, 1)),
            graph,
            financial_identity_catalog_enabled=True,
        )
        invalid = verify_program(
            plan,
            ReasoningProgram("ADD", (1, 2)),
            graph,
            financial_identity_catalog_enabled=True,
        )

        self.assertNotEqual(valid.state, VerificationState.INVALID)
        self.assertEqual(
            valid.feature_vector["financial_identity"], 1.0
        )
        self.assertEqual(invalid.state, VerificationState.INVALID)
        self.assertIn("financial_identity_mismatch", invalid.reasons)

    def test_direct_disclosed_target_is_valid_identity_evidence(self) -> None:
        graph = build_evidence_graph(
            (_candidate(0, "100", "Tong no phai tra"),)
        )
        plan = compile_question_plan(
            "Tong no phai tra nam 2024 la bao nhieu?"
        )

        result = verify_program(
            plan,
            ReasoningProgram("SELECT", (0,)),
            graph,
            financial_identity_catalog_enabled=True,
        )

        self.assertTrue(result.accepted)
        self.assertEqual(
            result.feature_vector["financial_identity"], 1.0
        )

    def test_par_value_division_has_typed_unit_and_role_proof(self) -> None:
        graph = build_evidence_graph(
            (
                _candidate(0, "100000000", "Von dieu le", unit="VND"),
                _candidate(
                    1,
                    "10000",
                    "So luong co phieu da phat hanh",
                    unit="shares",
                ),
            )
        )
        plan = compile_question_plan(
            "Menh gia co phieu nam 2024 la bao nhieu dong?"
        )

        result = verify_program(
            plan,
            ReasoningProgram("DIVIDE", (0, 1)),
            graph,
            financial_identity_catalog_enabled=True,
        )

        self.assertEqual(
            result.training_support_state, VerificationState.VALID
        )
        obligation_states = {
            item.name: item.state for item in result.proof_obligations
        }
        self.assertEqual(
            obligation_states["unit_compatibility"], VerificationState.VALID
        )
        self.assertEqual(
            obligation_states["operand_roles"], VerificationState.VALID
        )

    def test_catalog_constraints_are_disabled_by_default(self) -> None:
        graph = build_evidence_graph(
            (
                _candidate(0, "40", "No dai han"),
                _candidate(1, "100", "Doanh thu thuan"),
            )
        )
        plan = compile_question_plan(
            "Tong no phai tra nam 2024 la bao nhieu?"
        )

        result = verify_program(
            plan, ReasoningProgram("ADD", (0, 1)), graph
        )

        self.assertNotIn("financial_identity", result.feature_vector)
        self.assertNotIn("financial_identity_mismatch", result.reasons)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
