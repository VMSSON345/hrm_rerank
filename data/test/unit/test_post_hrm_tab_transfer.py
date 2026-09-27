"""Transfer checks against source-bound TAB disclosures and safe missing-unit output."""
import hashlib
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase

from hrm_finance_qa.post_hrm_engine import ContractDocumentEngine
from hrm_finance_qa.post_hrm_published import answer_published_ratio


class FlatRanking:
    model=SimpleNamespace(config=SimpleNamespace(max_candidates=64))
    def rank_raw(self,question,candidates):
        return {'pointers':[[0.0]*len(candidates)]*4,
                'select':[0.0]*len(candidates),'pairs':{},'segments':0}


class TabTransferTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=(Path(__file__).resolve().parents[3]/
                    'data/test/suites/independent_reports_v2/sources/TAB.md')
        cls.lines=cls.source.read_text(encoding='utf-8-sig').splitlines()
        cls.digest=hashlib.sha256(cls.source.read_bytes()).hexdigest()
        cls.engine=ContractDocumentEngine(FlatRanking())

    def ask(self,question):
        result=self.engine.answer(question,self.source)
        for cite in result.get('citations') or []:
            self.assertEqual(cite['document_sha256'],self.digest)
            line=self.lines[cite['line_number']-1]
            variants=(line,re.sub(r'\s+',' ',line).strip())
            matched=next((v for v in variants if hashlib.sha256(v.encode()).hexdigest()==cite['source_line_sha256']),None)
            self.assertIsNotNone(matched)
            lo,hi=cite['source_span']
            self.assertTrue(0<=lo<hi<=len(matched))
            if cite.get('raw_value'):self.assertIn(cite['raw_value'],matched[lo:hi])
        return result

    def test_published_growth_and_plan_discrepancy(self):
        growth=self.ask('Doanh thu thuần năm 2024 tăng hay giảm bao nhiêu phần trăm so với năm 2023?')
        self.assertEqual(growth['status'],'ACCEPTED')
        self.assertEqual(growth['canonical_value'],'-14.45')
        self.assertIn('giảm 14,45%',growth['rendered'])
        self.assertIsNone(answer_published_ratio(
            'Doanh thu thuần năm 2023 tăng hay giảm bao nhiêu phần trăm so với năm 2024?',
            list(self.engine._catalog_cache.values())[-1][2]))
        plan=self.ask('Doanh thu thuần thực hiện năm 2024 đạt bao nhiêu phần trăm kế hoạch năm 2024?')
        self.assertEqual(plan['status'],'ACCEPTED')
        self.assertIn('68,40%',plan['rendered'])
        self.assertIn('68,41%',plan['rendered'])

    def test_published_leverage_and_efficiency_metrics(self):
        for question,value in (
            ('Hệ số nợ trên tổng tài sản năm 2024 là bao nhiêu?','0.2'),
            ('Hệ số nợ trên vốn chủ sở hữu năm 2024 là bao nhiêu?','0.3'),
            ('ROE năm 2024 là bao nhiêu phần trăm?','5.6'),
            ('ROA năm 2024 là bao nhiêu phần trăm?','4.2'),
            ('Vòng quay tổng tài sản năm 2024 là bao nhiêu lần?','0.8'),
            ('Các khoản phải thu ngắn hạn năm 2024 biến động thế nào so với năm 2023?','-64'),
        ):
            with self.subTest(question=question):
                result=self.ask(question)
                self.assertEqual(result['status'],'ACCEPTED',result['rendered'])
                self.assertEqual(result['canonical_value'],value)

    def test_unparsed_money_stays_raw_and_date_guard_remains(self):
        debt=self.ask('Tổng nợ phải trả cuối năm 2024 là bao nhiêu đồng?')
        self.assertEqual(debt['status'],'ABSTAIN')
        self.assertEqual(debt['answer_type'],'explained_halt')
        self.assertIsNone(debt['canonical_value'])
        self.assertIn('`19.765`',debt['rendered'])
        loans=self.ask('Vay ngắn hạn và vay dài hạn năm 2024 là bao nhiêu?')
        self.assertEqual(loans['status'],'RAW_DISCLOSURE')
        self.assertIsNone(loans['canonical_value'])
        self.assertIn('`2.500`',loans['rendered'])
        self.assertIn('`7.500`',loans['rendered'])
        assets=self.ask('Tổng tài sản cuối năm 2024 là bao nhiêu đồng?')
        self.assertEqual(assets['status'],'ABSTAIN')

    def test_index_percent_is_not_a_growth_percent(self):
        sgd=(Path(__file__).resolve().parents[3]/
             'data/test/suites/source_binding_v2/fresh_sgd/SGD_23CN_BCTN.md')
        result=ContractDocumentEngine(FlatRanking()).answer(
            'Doanh thu thuần năm 2023 tăng hay giảm bao nhiêu phần trăm so với năm 2022?',sgd)
        self.assertEqual(result['status'],'ACCEPTED',result['rendered'])
        self.assertIn('giảm khoảng 3,43%',result['rendered'])
        self.assertNotIn('tăng 96,54%',result['rendered'])

    def test_two_ratio_metrics_keep_two_outputs(self):
        alt=(Path(__file__).resolve().parents[3]/
             'data/test/suites/alt_independent_v1/ALT_25CN_BCTN.md')
        result=ContractDocumentEngine(FlatRanking()).answer(
            'ROA và ROE năm 2025 lần lượt là bao nhiêu phần trăm?',alt)
        self.assertEqual(result['status'],'ACCEPTED')
        self.assertEqual([part['canonical_value'] for part in result['outputs']],['3.17','5.16'])

    def test_year_only_column_in_formal_dated_balance_sheet(self):
        source=('## Page 1\nBẢNG CÂN ĐỐI KẾ TOÁN\nTại ngày 31/12/2024\n'
                'Đơn vị tính: đồng\nChỉ tiêu | Năm 2023 | Năm 2024\n'
                '--- | --- | ---\nTổng cộng tài sản | 100 | 120\n')
        with TemporaryDirectory() as directory:
            path=Path(directory)/'formal.md'
            path.write_text(source,encoding='utf8')
            accepted=ContractDocumentEngine(FlatRanking()).answer(
                'Tổng tài sản cuối năm 2024 là bao nhiêu đồng?',path)
            self.assertEqual(accepted['status'],'ACCEPTED',accepted['rendered'])
            self.assertEqual(accepted['canonical_value'],'120')
            self.assertEqual(accepted['citations'][0]['as_of_date'],'2024-12-31')
            path.write_text(source.replace('31/12/2024','30/06/2024'),encoding='utf8')
            guarded=ContractDocumentEngine(FlatRanking()).answer(
                'Tổng tài sản cuối năm 2024 là bao nhiêu đồng?',path)
            self.assertEqual(guarded['status'],'ABSTAIN')

    def test_narrative_auditor_and_liquidity_risk(self):
        for question,fragment in (
            ('Biên lợi nhuận gộp năm 2024 thay đổi thế nào so với năm 2023?','5,7% năm 2023 lên 6,4% năm 2024'),
            ('Doanh thu năm 2024 chủ yếu đến từ mảng kinh doanh nào?','nguyên vật liệu thép'),
            ('Công ty kiểm toán báo cáo tài chính năm 2024 là ai và đưa ra ý kiến gì?','An Việt; ý kiến: Chấp nhận toàn phần'),
        ):
            with self.subTest(question=question):
                result=self.ask(question)
                self.assertEqual(result['status'],'ACCEPTED',result['rendered'])
                self.assertIn(fragment,result['rendered'])
        risk=self.ask('Dựa trên các hệ số thanh toán, có thể kết luận công ty mất khả năng thanh toán không?')
        self.assertEqual(risk['status'],'ABSTAIN')
        self.assertIn('Chưa đủ căn cứ',risk['rendered'])


if __name__=='__main__':
    import unittest
    unittest.main()
