"""Numeric program proposals must come from checkpoint heads, not planner fallback."""
from types import SimpleNamespace
from decimal import Decimal
from unittest import TestCase

from hrm_finance_qa.post_hrm_numeric_program import _proposals
from hrm_finance_qa.program_data import OPERATIONS


class NumericProgramProposalTests(TestCase):
    def test_operation_and_ordered_operands_follow_neural_heads(self):
        scores = [-10.0] * len(OPERATIONS)
        scores[OPERATIONS.index('SELECT')] = 1.0
        scores[OPERATIONS.index('PERCENT_CHANGE')] = 4.0
        ranking = {'operation_scores': scores, 'select': [5.0, 1.0],
                   'pairs': {'PERCENT_CHANGE': [[-100.0, 1.0], [9.0, -100.0]]}}
        facts = [SimpleNamespace(), SimpleNamespace()]
        allowed = ('SELECT', 'PERCENT_CHANGE')
        self.assertEqual(_proposals('hrm', allowed, ranking, facts, [(facts[0], [])])[0],
                         ('PERCENT_CHANGE', (1, 0)))
        self.assertEqual(_proposals('baseline', allowed, {}, facts, [(facts[0], [])])[0],
                         ('SELECT', (0,)))

    def test_missing_operation_head_cannot_be_replaced_by_planner(self):
        facts = [SimpleNamespace()]
        self.assertEqual(_proposals('hrm', ('SELECT',), {'select': [1.0]}, facts, []), [])


class SourceComparatorTests(TestCase):
    def test_verified_finer_source_precedes_document_order(self):
        coarse = SimpleNamespace(unit_resolved=True, quantum=Decimal(1000),
                                 unit_assumption='', reporting_entity=True,
                                 source_name='a.md', line_number=1, fact_id='coarse')
        fine = SimpleNamespace(unit_resolved=True, quantum=Decimal(1),
                               unit_assumption='', reporting_entity=True,
                               source_name='b.md', line_number=2, fact_id='fine')
        self.assertEqual(_proposals('baseline', ('SELECT',), {},
                                    [coarse, fine], [])[0], ('SELECT', (1,)))
