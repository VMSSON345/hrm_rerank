"""Dated period clarifications must keep intent and revalidate the source."""
import unittest
from copy import deepcopy

from finhrm_web.clarification import pending_period_choice, complete_pending_period, unmatched_period_reply, begin_clarification, resume_clarification
from hrm_finance_qa.conversation_context import resolve_followup


QUESTION = 'Vốn chủ sở hữu của công ty là bao nhiêu'
DIGEST = 'source-sha'


def ambiguous():
    return {
        'status': 'ABSTAIN',
        'validation': {'checks': [{
            'code': 'source_period_ambiguous',
            'status': 'INSUFFICIENT_EVIDENCE',
            'blocking': True,
            'fact_ids': ['opening', 'closing'],
        }]},
        'citations': [
            {'operand_ref': 'opening', 'row_label': 'Vốn chủ sở hữu',
             'period_role': 'opening', 'as_of_date': '2025-01-01',
             'entity_scope': 'company', 'document_sha256': DIGEST},
            {'operand_ref': 'closing', 'row_label': 'Vốn chủ sở hữu',
             'period_role': 'closing', 'as_of_date': '2025-12-31',
             'entity_scope': 'company', 'document_sha256': DIGEST},
        ],
    }


class PeriodChoiceTests(unittest.TestCase):
    def test_screenshot_choices_rewrite_original_metric(self):
        pending = pending_period_choice(ambiguous(), QUESTION, DIGEST)
        self.assertIsNotNone(pending)
        self.assertEqual(
            complete_pending_period('cuối kỳ 31/12/2025', pending, DIGEST),
            'Vốn chủ sở hữu cuối kỳ ngày 31/12/2025 của công ty là bao nhiêu')
        self.assertEqual(
            complete_pending_period('đầu kỳ ngày 01/01/2025.', pending, DIGEST),
            'Vốn chủ sở hữu đầu kỳ ngày 01/01/2025 của công ty là bao nhiêu')

    def test_unoffered_or_ambiguous_choice_is_not_completed(self):
        pending = pending_period_choice(ambiguous(), QUESTION, DIGEST)
        for reply in ['năm 2025', '31/12/2024', 'Lợi nhuận sau thuế năm 2025',
                      'đầu kỳ 31/12/2025', 'cuối kỳ 31/12/2024']:
            self.assertIsNone(complete_pending_period(reply, pending, DIGEST), reply)
        self.assertIsNone(complete_pending_period('cuối kỳ 31/12/2025', pending, 'changed'))

    def test_options_require_same_source_metric_scope_and_dated_fact(self):
        for change in [
            lambda x: x['citations'][1].update(document_sha256='changed'),
            lambda x: x['citations'][1].update(row_label='Lợi nhuận sau thuế'),
            lambda x: x['citations'][1].update(entity_scope='other-company'),
            lambda x: x['citations'][1].update(as_of_date=''),
            lambda x: x['validation']['checks'][0].update(code='source_values_conflict'),
        ]:
            result = ambiguous()
            change(result)
            self.assertIsNone(pending_period_choice(result, QUESTION, DIGEST))

    def test_year_only_options_require_a_bound_physical_column(self):
        result=deepcopy(ambiguous())
        for citation,role in zip(result['citations'],('opening','closing')):
            citation['as_of_date']=''
            citation['period']='2024'
            citation['period_binding']='audited_statement_period'
            citation['column_label']='Số đầu năm' if role=='opening' else 'Số cuối năm'
        pending=pending_period_choice(result,QUESTION,DIGEST)
        self.assertIsNotNone(pending)
        self.assertEqual(complete_pending_period('cuối năm 2024',pending,DIGEST),
                         'Vốn chủ sở hữu cuối năm 2024 của công ty là bao nhiêu')
        self.assertEqual(complete_pending_period('đầu kỳ năm 2024',pending,DIGEST),
                         'Vốn chủ sở hữu đầu năm 2024 của công ty là bao nhiêu')
        self.assertIsNone(complete_pending_period('năm 2024',pending,DIGEST))
        self.assertIsNone(complete_pending_period('cuối năm 2025',pending,DIGEST))
        result['citations'][0]['period_binding']='profile_report_title'
        self.assertIsNone(pending_period_choice(result,QUESTION,DIGEST))
        result['citations'][0]['period_binding']='audited_statement_period'
        result['citations'][0]['column_label']='Năm 2024'
        self.assertIsNone(pending_period_choice(result,QUESTION,DIGEST))

    def test_year_only_clarification_uses_original_question_and_prompt(self):
        result=deepcopy(ambiguous())
        result['rendered']='Nguồn có nhiều thời điểm: năm 2023; năm 2024. Bạn muốn dùng kỳ nào?'
        for citation,year in zip(result['citations'],('2023','2024')):
            citation.update(period_role='',as_of_date='',period=year,
                            period_binding='source_column',column_label='Năm '+year)
        original='Vốn chủ sở hữu của công ty là'
        pending=pending_period_choice(result,original,DIGEST)
        self.assertIsNotNone(pending)
        self.assertEqual(complete_pending_period('2023',pending,DIGEST),
                         'Vốn chủ sở hữu năm 2023 của công ty là')
        self.assertEqual(complete_pending_period('năm 2024',pending,DIGEST),
                         'Vốn chủ sở hữu năm 2024 của công ty là')
        self.assertIsNone(complete_pending_period('đầu năm 2024',pending,DIGEST))
        self.assertIn('năm 2023; năm 2024',
                      unmatched_period_reply('đầu năm 2024',pending,DIGEST))
        self.assertIsNone(unmatched_period_reply('Lợi nhuận sau thuế năm 2024',
                                                   pending,DIGEST))
        result['citations'][0]['column_label']='Năm báo cáo'
        self.assertIsNone(pending_period_choice(result,original,DIGEST))

    def test_one_resume_flow_keeps_other_clarifications(self):
        metric_question='Lợi nhuận năm 2024 là bao nhiêu?'
        metric_result={'status':'ABSTAIN','rendered':'Bạn muốn lợi nhuận trước hay sau thuế?',
                       'question_resolution':{'basis':'ambiguous_source_metrics',
                                              'document_sha256':DIGEST,
                                              'original_question':metric_question,
                                              'bare_metric':'lợi nhuận',
                                              'alternatives':['Lợi nhuận trước thuế',
                                                              'Lợi nhuận sau thuế']}}
        pending=begin_clarification(metric_result,metric_question,'',DIGEST)
        self.assertIsNotNone(pending)
        self.assertIn('Bạn muốn lợi nhuận',resume_clarification('lợi nhuận gộp',
                                                                  pending,DIGEST)['message'])
        self.assertIsNone(resume_clarification('Vốn điều lệ năm 2024 là bao nhiêu?',
                                                pending,DIGEST))
        self.assertIsNone(resume_clarification('lợi nhuận gộp',pending,'changed'))

        compare='So với năm 2024 giảm hay tăng bao nhiêu phần trăm?'
        result={'status':'ABSTAIN','rendered':'Bạn muốn so với năm nào?',
                'intent_clarification_required':True}
        pending=begin_clarification(result,compare,'Doanh thu thuần năm 2024 là bao nhiêu?',DIGEST)
        self.assertIsNotNone(pending)
        self.assertIn('Bạn muốn so với năm nào',
                      resume_clarification('năm 2024',pending,DIGEST)['message'])
        completed=resume_clarification('năm 2025',pending,DIGEST)
        self.assertIn('2025',completed['question'])


    def test_named_ratio_year_reply_uses_offered_source_year(self):
        question='Chỉ số ROE'
        result={'status':'ABSTAIN',
                'rendered':'Nguồn có ROE năm 2023, 2024. Bạn muốn dùng năm nào?',
                'named_ratio_year_choices':{
                    'metric':'return_on_equity','years':['2023','2024'],
                    'question':question},
                'citations':[{'document_sha256':DIGEST,'period':'2023'},
                             {'document_sha256':DIGEST,'period':'2024'}]}
        pending=begin_clarification(result,question,'',DIGEST)
        self.assertIsNotNone(pending)
        self.assertEqual(resume_clarification('năm 2024',pending,DIGEST)['question'],
                         'ROE năm 2024 là bao nhiêu?')
        self.assertIn('Bạn muốn dùng năm nào?',
                      resume_clarification('năm 2025',pending,DIGEST)['message'])
        self.assertIsNone(resume_clarification('năm 2024',pending,'changed'))

    def test_second_period_reply_requeries_with_new_date(self):
        prior='Vốn chủ sở hữu cuối kỳ ngày 31/12/2025 của công ty là bao nhiêu'
        result=resolve_followup('đầu kỳ ngày 01/01/2025.',prior)
        self.assertIsNotNone(result)
        self.assertEqual(result['resolved_question'],
                         'Vốn chủ sở hữu đầu kỳ ngày 01/01/2025 của công ty là bao nhiêu')
        self.assertIn('đầu kỳ ngày 01/01/2024',
                      resolve_followup('đầu kỳ ngày 01/01/2024.',prior)['resolved_question'])
        self.assertIsNone(resolve_followup('đầu kỳ ngày 01/01/2025.',
                                           'Vốn chủ sở hữu năm 2025 trừ năm 2024 là bao nhiêu?'))


if __name__ == '__main__':
    unittest.main()
