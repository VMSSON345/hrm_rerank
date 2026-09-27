"""Raw financial disclosure must never become a canonical money answer."""
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase

from hrm_finance_qa.post_hrm_contract import SourceFact, compute_records
from hrm_finance_qa.post_hrm_engine import ContractDocumentEngine

class FlatRanking:
    model=SimpleNamespace(config=SimpleNamespace(max_candidates=64))
    def rank_raw(self,question,candidates):
        return {'pointers':[[0.0]*len(candidates)]*4,'select':[0.0]*len(candidates),'pairs':{},'segments':0}

class UnitStateRoutingTests(TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'UNSEEN.md'
    def answer(self,text,question):
        self.path.write_text(text,encoding='utf8')
        return ContractDocumentEngine(FlatRanking()).answer(question,self.path)
    def balance(self):
        return ('## Page 1\nIII. Báo cáo và đánh giá của Ban Giám đốc\n'
                '2. Tình hình tài chính\n'
                'Chỉ tiêu | 01/01/2032 | 31/12/2032\n--- | --- | ---\n'
                'Tài sản ngắn hạn | 1000 | 800\n')
    def test_neutral_lookup_discloses_raw_text_without_money_value(self):
        result=self.answer(self.balance(),'Tài sản ngắn hạn cuối năm 2032 là bao nhiêu?')
        self.assertEqual(result['status'],'RAW_DISCLOSURE',result['rendered'])
        self.assertEqual(result['answer_type'],'raw_disclosure')
        self.assertEqual(result['raw_value'],'800')
        self.assertIsNone(result['canonical_value']);self.assertIsNone(result['unit'])
        self.assertFalse(result['numeric_answer'])
        self.assertEqual(result['citations'][0]['unit_state'],'UNIT_UNRESOLVED_RAW')
        self.assertIn('chưa ghi đơn vị',result['rendered'])
    def test_explicit_vnd_and_money_subtraction_halt_with_raw_evidence(self):
        amount=self.answer(self.balance(),'Tài sản ngắn hạn cuối năm 2032 là bao nhiêu đồng?')
        self.assertEqual(amount['status'],'ABSTAIN');self.assertIsNone(amount['canonical_value'])
        self.assertIn('`800`',amount['rendered']);self.assertIn('đơn vị',amount['rendered'])
        delta=self.answer(self.balance(),'Tài sản ngắn hạn cuối năm 2032 giảm bao nhiêu đồng so với đầu năm 2032?')
        self.assertEqual(delta['status'],'ABSTAIN',delta['rendered'])
        self.assertIsNone(delta['canonical_value'])
        self.assertIn('`800`',delta['rendered']);self.assertIn('`1000`',delta['rendered'])
    def test_distinct_revenue_labels_remain_independent(self):
        source=('## Page 1\nAnnual Report 2032\n1. Financial Performance\n'
                'Unit: VND\nIndicator | Actual 2032 | Actual 2031\n--- | --- | ---\n'
                'Total Revenue | 1200 | 900\n\n2. Financial Position\n'
                'Indicator | Actual 2032 | Actual 2031\n--- | --- | ---\n'
                'Net Revenue | 1200 | 900\n')
        result=self.answer(source,'Doanh thu thuần năm 2032 là bao nhiêu?')
        self.assertEqual(result['status'],'RAW_DISCLOSURE',result['rendered'])
        self.assertEqual(result['metric_link_state'],'NAME_COLLISION_AMBIGUOUS')
        self.assertIsNone(result['canonical_value'])
        self.assertEqual({d['metric_label'] for d in result['raw_disclosures']},{'Net Revenue','Total Revenue'})
        self.assertEqual({c['unit_state'] for c in result['citations']},
                         {'UNIT_UNRESOLVED_RAW','EXPLICIT_BOUND'})

    def test_unresolved_competing_amount_blocks_publication(self):
        source = ("## Page 1\n"
                  "Chỉ tiêu | Thực hiện 2032\n--- | ---\n"
                  "Lợi nhuận sau thuế | Thực hiện 2032: 11.994.503.019\n"
                  "## Page 2\nĐơn vị: Đồng\n"
                  "Chỉ tiêu | Thực hiện 2032\n--- | ---\n"
                  "Lợi nhuận sau thuế | Thực hiện 2032: 11.944.503.019\n")
        result = self.answer(source, "Lợi nhuận sau thuế năm 2032 là bao nhiêu đồng?")
        self.assertEqual(result["status"], "ABSTAIN", result["rendered"])
        self.assertIn("unresolved_competing_disclosure",
                      [check["code"] for check in result["validation"]["checks"]])
        self.assertEqual({citation["raw_value"] for citation in result["citations"]},
                         {"11.994.503.019", "11.944.503.019"})

    def test_short_currency_followup_keeps_metric_and_period_only(self):
        from hrm_finance_qa.conversation_context import resolve_followup
        result=resolve_followup('Vậy quy ra đồng là bao nhiêu?',
                                'Tài sản ngắn hạn cuối năm 2032 là bao nhiêu?')
        self.assertEqual(result['resolved_question'],
                         'Tài sản ngắn hạn cuối năm 2032 là bao nhiêu đồng?')
        self.assertNotIn('value',result['inherited'])
    def test_unit_states_are_separate_from_metric_link(self):
        base=dict(fact_id='f',metric='net_revenue',metric_label='Net Revenue',kind='money',
                  value=Decimal(12),raw_value='12',unit='VND',currency='VND',
                  entity='issuer',source_name='source.md',document_sha256='0'*64,
                  source_line_sha256='1'*64,line_number=1,source_span=(0,2),source_verified=True)
        direct=SourceFact(**base,unit_evidence=({'basis':'row_or_cell'},))
        inherited=SourceFact(**base,unit_evidence=({'basis':'source_declaration'},))
        unresolved=SourceFact(**{**base,'unit':'','unit_resolved':False,
                                 'unit_assumption':'source_unit_unresolved'})
        self.assertEqual(direct.unit_state,'EXPLICIT_BOUND')
        self.assertEqual(inherited.unit_state,'INHERITED_PROVEN')
        self.assertEqual(unresolved.unit_state,'UNIT_UNRESOLVED_RAW')
        self.assertIsNone(unresolved.canonical_value)
        with self.assertRaises(ValueError):
            compute_records([('answer','SELECT',('f',))],{'f':unresolved},output_unit='VND')

if __name__=='__main__':
    import unittest
    unittest.main()
