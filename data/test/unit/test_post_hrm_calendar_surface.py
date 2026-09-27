"""Calendar prepositions frame a metric but are never part of its row label."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from hrm_finance_qa.question_planner import compile_question_plan, resolve_public_question
from hrm_finance_qa.post_hrm_engine import ContractDocumentEngine
from test_post_hrm_semantic_transfer import FlatRanking


class CalendarSurfaceTests(TestCase):
    def test_question_variants_keep_metric_year_and_distinct_qualifier(self):
        for question in (
            'Doanh thu hoạt động là bao nhiêu vào năm 2031?',
            'Doanh thu hoạt động vào năm 2031 là bao nhiêu?',
            'Doanh thu hoạt động trong năm 2031 là bao nhiêu?',
            'Doanh thu hoạt động năm 2031 là bao nhiêu?',
        ):
            plan = compile_question_plan(resolve_public_question(question))
            self.assertEqual(plan.operation, 'SELECT', question)
            self.assertEqual(plan.metric_text, 'doanh thu hoat dong', question)
            self.assertEqual(plan.requested_years, ('2031',), question)
        for qualifier in ('trong nước', 'khác'):
            question=f'Doanh thu hoạt động {qualifier} trong năm 2031 là bao nhiêu?'
            plan=compile_question_plan(resolve_public_question(question))
            self.assertEqual(plan.metric_text,
                             'doanh thu hoat dong '+('trong nuoc' if qualifier=='trong nước' else 'khac'))
            self.assertEqual(plan.requested_years,('2031',))

    def test_markdown_selection_uses_source_unit_and_correct_row(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'sample.md'
            path.write_text(
                '## Page 1\n'
                'Tình hình thực hiện kết quả hoạt động kinh doanh\n'
                'Đơn vị: VND\n'
                'Chỉ tiêu | Kết quả năm 2031 | Kết quả năm 2030\n'
                '--- | --- | ---\n'
                'Doanh thu hoạt động | 180.000.000 | 120.000.000\n'
                'Doanh thu hoạt động khác | 50.000.000 | 40.000.000\n',
                encoding='utf-8')
            engine=ContractDocumentEngine(FlatRanking())
            for question in (
                'Doanh thu hoạt động là bao nhiêu vào năm 2031?',
                'Doanh thu hoạt động trong năm 2031 là bao nhiêu?',
            ):
                result=engine.answer(question,path)
                self.assertEqual(result['status'],'ACCEPTED',result['rendered'])
                self.assertEqual(result['canonical_value'],'180000000')
                self.assertEqual(result['unit'],'VND')
                self.assertTrue(result['citations'])
            other=engine.answer('Doanh thu hoạt động khác trong năm 2031 là bao nhiêu?',path)
            self.assertEqual(other['status'],'ACCEPTED',other['rendered'])
            self.assertEqual(other['canonical_value'],'50000000')
