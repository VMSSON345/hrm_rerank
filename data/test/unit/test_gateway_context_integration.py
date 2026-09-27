"""Gateway context tests with an isolated SQLite DB and fake QA transport.
Never uses the demo's account, database, passwords, or network endpoint.
"""
from unittest import TestCase
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path
from contextlib import contextmanager,ExitStack
import sqlite3
from fastapi.testclient import TestClient
from finhrm_web import app as web
from finhrm_web.deps import get_current_user

class GatewayContextTests(TestCase):
 def setUp(self):
  self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup);root=Path(self.tmp.name);self.paths={}
  for name in ['one','two']:
   p=root/(name+'.md');p.write_text('## Page 1\nSource '+name,encoding='utf8');self.paths[name]=p
  dbpath=root/'routing.sqlite3'
  @contextmanager
  def connection():
   conn=sqlite3.connect(dbpath);conn.row_factory=sqlite3.Row
   try:
    with conn:yield conn
   finally:conn.close()
  with connection() as c:c.execute('CREATE TABLE conversations(id TEXT PRIMARY KEY)')
  self.stack=ExitStack();self.addCleanup(self.stack.close);self.calls=[]
  self.stack.enter_context(patch.object(web.db,'DB_PATH',str(dbpath)))
  self.stack.enter_context(patch.object(web.db,'init_db',lambda:None))
  self.stack.enter_context(patch.object(web.db,'get_conn',connection))
  self.stack.enter_context(patch.object(web.db,'get_conversation',lambda *args:{'id':'conversation'}))
  self.stack.enter_context(patch.object(web.db,'get_document',lambda i:{'id':i,'status':'indexed','filename':i+'.pdf','file_path':str(self.paths[i])}))
  self.stack.enter_context(patch.object(web,'refresh_document',lambda d:d))
  self.stack.enter_context(patch.object(web,'prepared_artifact',lambda d:self.paths[d['id']]))
  def transport(path,payload=None,**kw):
   self.calls.append(dict(payload));return {'status':'ACCEPTED','rendered':'Fixture response','citations':[]}
  self.stack.enter_context(patch.object(web,'hrm',transport))
  api=web.create_app();api.dependency_overrides[get_current_user]=lambda:{'id':'test-user','role':'admin'}
  self.client=TestClient(api);self.addCleanup(self.client.close)
 def ask(self,q,ids=None,reset=False):
  r=self.client.post('/api/rag/ask',json={'question':q,'document_ids':ids or ['one'],'conversation_id':'conversation','reset_context':reset});self.assertEqual(r.status_code,200,r.text)
 def test_metric_followup_and_third_turn_keep_canonical_context(self):
  self.ask('Doanh thu thuần năm 2032 tăng bao nhiêu phần trăm so với 2031?');self.ask('Còn lợi nhuận sau thuế?');self.assertIn('phần trăm',self.calls[-1]['question']);self.assertIn('2031',self.calls[-1]['question']);self.assertIn('lợi nhuận sau thuế',self.calls[-1]['question'])
  self.ask('Còn lợi nhuận trước thuế?');self.assertIn('2032',self.calls[-1]['question']);self.assertIn('phần trăm',self.calls[-1]['question']);self.assertIn('lợi nhuận trước thuế',self.calls[-1]['question'])
 def test_document_switch_and_reset_do_not_inherit(self):
  for ids,reset in [(['two'],False),(['one'],True),(['one','two'],False)]:
   self.ask('Doanh thu thuần năm 2032 tăng bao nhiêu phần trăm so với 2031?');self.ask('Còn lợi nhuận sau thuế?',ids,reset);self.assertEqual(self.calls[-1]['question'],'Còn lợi nhuận sau thuế?');self.assertNotIn('previous_question',self.calls[-1])
 def test_changed_artifact_does_not_inherit(self):
  self.ask('Doanh thu thuần năm 2032 tăng bao nhiêu phần trăm so với 2031?');self.paths['one'].write_text('Changed source',encoding='utf8');self.ask('Còn lợi nhuận sau thuế?');self.assertNotIn('previous_question',self.calls[-1])

 def test_temporal_followup_requeries_markdown(self):
  self.ask('Tài sản ngắn hạn năm 2023 là bao nhiêu đồng?')
  self.ask('Còn đầu năm?')
  self.assertEqual(len(self.calls),2)
  self.assertEqual(self.calls[-1]['document'],str(self.paths['one']))
  self.assertIn('đầu năm 2023',self.calls[-1]['question'])
  self.assertNotIn('Fixture response',str(self.calls[-1]))
 def test_endpoint_does_not_cross_document_reset_or_artifact(self):
  for ids,reset in [(['two'],False),(['one'],True),(['one','two'],False)]:
   self.ask('Tài sản ngắn hạn năm 2023 là bao nhiêu đồng?')
   self.ask('Còn đầu năm?',ids,reset)
   self.assertEqual(self.calls[-1]['question'],'Còn đầu năm?')
   self.assertNotIn('previous_question',self.calls[-1])
  self.ask('Tài sản ngắn hạn năm 2023 là bao nhiêu đồng?')
  self.paths['one'].write_text('Source changed',encoding='utf8')
  self.ask('Còn đầu năm?')
  self.assertNotIn('previous_question',self.calls[-1])
 def test_percentage_followup_and_explicit_year(self):
  self.ask('Doanh thu thuần năm 2024 là bao nhiêu đồng?')
  self.ask('So với năm 2022 tăng bao nhiêu phần trăm?')
  self.assertIn('2024',self.calls[-1]['question'])
  self.assertIn('2022',self.calls[-1]['question'])
  self.ask('Lợi nhuận sau thuế năm 2035 là bao nhiêu?')
  self.assertEqual(self.calls[-1]['question'],'Lợi nhuận sau thuế năm 2035 là bao nhiêu?')

 def test_screenshot_comparison_chain(self):
  self.ask('Doanh thu thuần năm 2025 là bao nhiêu?')
  for order in ['tăng hay giảm','giảm hay tăng']:
   self.ask('So với năm 2024 '+order+' bao nhiêu phần trăm?')
   self.assertIn('doanh thu thuan',self.calls[-1]['question'])
   self.assertIn('2025',self.calls[-1]['question']);self.assertIn('2024',self.calls[-1]['question'])
 def test_pending_year_completion_and_source_guards(self):
  question='So với năm 2024 giảm hay tăng bao nhiêu phần trăm?'
  prior='Doanh thu thuần năm 2024 là bao nhiêu?'
  def transport(path,payload=None,**kw):
   self.calls.append(dict(payload))
   return {'status':'ABSTAIN','rendered':'Xin bổ sung năm','intent_clarification_required':True,'citations':[]}
  with patch.object(web,'hrm',transport):
   for mode in ['same','switch','reset','hash']:
    self.ask(prior,reset=True);self.ask(question)
    if mode=='hash':self.paths['one'].write_text('Changed source again',encoding='utf8')
    self.ask('năm 2025',ids=['two'] if mode=='switch' else ['one'],reset=mode=='reset')
    if mode=='same':
     self.assertIn('doanh thu thuan',self.calls[-1]['question']);self.assertIn('2024',self.calls[-1]['question']);self.assertIn('2025',self.calls[-1]['question'])
    else:self.assertEqual(self.calls[-1]['question'],'năm 2025')

 def test_dated_period_choice_keeps_metric_and_requeries(self):
  from hashlib import sha256
  original='Vốn chủ sở hữu của công ty là bao nhiêu'
  digest=sha256(self.paths['one'].read_bytes()).hexdigest()
  citations=[{'operand_ref':'opening','row_label':'Vốn chủ sở hữu','period_role':'opening',
              'as_of_date':'2025-01-01','entity_scope':'company','document_sha256':digest},
             {'operand_ref':'closing','row_label':'Vốn chủ sở hữu','period_role':'closing',
              'as_of_date':'2025-12-31','entity_scope':'company','document_sha256':digest}]
  def transport(path,payload=None,**kw):
   self.calls.append(dict(payload))
   q=payload['question']
   if q==original:
    return {'status':'ABSTAIN','rendered':'Nguồn có nhiều thời điểm.',
            'validation':{'checks':[{'code':'source_period_ambiguous',
                                      'status':'INSUFFICIENT_EVIDENCE','blocking':True,
                                      'fact_ids':['opening','closing']}]},
            'citations':citations}
   if 'cuối kỳ ngày 31/12/2025' in q:
    return {'status':'ACCEPTED','rendered':'222.727.022.972 đồng, trang 4.','citations':[]}
   if 'đầu kỳ ngày 01/01/2025' in q:
    return {'status':'ABSTAIN','rendered':'Chưa xác định được đơn vị/thang đo.','citations':[]}
   return {'status':'ABSTAIN','rendered':'Câu hỏi chưa đủ thông tin.','citations':[]}
  with patch.object(web,'hrm',transport):
   self.ask(original)
   self.ask('cuối kỳ 31/12/2025')
   self.assertIn('Vốn chủ sở hữu cuối kỳ ngày 31/12/2025 của công ty',self.calls[-1]['question'])
   self.assertEqual(self.calls[-1]['document'],str(self.paths['one']))
   self.ask('đầu kỳ ngày 01/01/2025.')
   self.assertIn('Vốn chủ sở hữu đầu kỳ ngày 01/01/2025 của công ty',self.calls[-1]['question'])
   self.assertEqual(self.calls[-1]['document'],str(self.paths['one']))
   self.ask(original,reset=True)
   self.ask('cuối kỳ 31/12/2025',ids=['two'])
   self.assertEqual(self.calls[-1]['question'],'cuối kỳ 31/12/2025')

 def test_year_clarification_uses_source_options_and_keeps_prompt(self):
  from hashlib import sha256
  base='Vốn chủ sở hữu của công ty là'
  digest=sha256(self.paths['one'].read_bytes()).hexdigest()
  citations=[{'operand_ref':year,'row_label':'Vốn chủ sở hữu','period_role':'',
              'period':year,'as_of_date':'','period_binding':'source_column',
              'column_label':'Năm '+year,'entity_scope':'company','document_sha256':digest}
             for year in ('2023','2024')]
  def transport(path,payload=None,**kw):
   self.calls.append(dict(payload))
   q=payload['question']
   if q==base:
    return {'status':'ABSTAIN','rendered':'Nguồn có nhiều thời điểm: năm 2023; năm 2024. Bạn muốn dùng kỳ nào?',
            'validation':{'checks':[{'code':'source_period_ambiguous',
                                      'status':'INSUFFICIENT_EVIDENCE','blocking':True,
                                      'fact_ids':['2023','2024']}]},
            'citations':citations}
   if 'năm 2023 của công ty' in q:
    return {'status':'ACCEPTED','rendered':'1.179.225.599.547 đồng, trang 22.','citations':[]}
   if 'đầu năm 2024 của công ty' in q:
    return {'status':'ABSTAIN','rendered':'Chưa xác nhận được số đầu năm 2024.','citations':[]}
   return {'status':'ABSTAIN','rendered':'Không có số liệu.','citations':[]}
  with patch.object(web,'hrm',transport):
   self.ask(base)
   self.ask('đầu năm 2024')
   self.assertEqual(len(self.calls),1)
   self.ask('2023')
   self.assertIn('Vốn chủ sở hữu năm 2023 của công ty',self.calls[-1]['question'])
   self.ask('đầu năm 2024')
   self.assertIn('Vốn chủ sở hữu đầu năm 2024 của công ty',self.calls[-1]['question'])

 def test_timings_belong_to_single_backend_call(self):
  timing={'understanding':0.1,'retrieval':0.2,'hrm':0.3,'total':0.6}
  with patch.object(web,'hrm',return_value={'status':'ACCEPTED','rendered':'Fixture','citations':[],'timings_seconds':timing}) as call:
   response=self.client.post('/api/rag/ask',json={'question':'Doanh thu thuần năm 2024 là bao nhiêu?','document_ids':['one'],'conversation_id':'conversation'})
   self.assertEqual(response.status_code,200);self.assertEqual(call.call_count,1)
   self.assertEqual(response.json()['request_timings'],[{'document_id':'one','timings_seconds':timing}])

 def test_scope_reply_uses_previous_question_and_assistant_source_choices(self):
  from hashlib import sha256
  original='Doanh thu thuần năm 2033 là bao nhiêu đồng?'
  digest=sha256(self.paths['one'].read_bytes()).hexdigest()
  citations=[
   {'operand_ref':'summary','row_label':'Doanh thu thuần','entity_scope':'entity:source:sample',
    'document_sha256':digest,'period':'2033'},
   {'operand_ref':'consolidated','row_label':'Doanh thu thuần','entity_scope':'consolidated',
    'document_sha256':digest,'period':'2033','page':2},
  ]
  def transport(path,payload=None,**kw):
   self.calls.append(dict(payload));question=payload['question']
   if question==original:
    return {'status':'ABSTAIN','rendered':'Nguồn có số chưa rõ phạm vi và số hợp nhất. Bạn chọn phạm vi nào?',
            'validation':{'checks':[{'code':'source_scope_ambiguous','status':'INSUFFICIENT_EVIDENCE',
                                     'blocking':True,'fact_ids':['summary','consolidated']}]},
            'citations':citations}
   if 'hợp nhất' in question:
    return {'status':'ACCEPTED','rendered':'123 đồng từ báo cáo hợp nhất.','citations':[]}
   return {'status':'ABSTAIN','rendered':'Chưa đủ câu hỏi.','citations':[]}
  with patch.object(web,'hrm',transport):
   self.ask(original,reset=True)
   self.ask('hợp nhất')
   self.assertEqual(len(self.calls),2)
   self.assertIn('Doanh thu thuần',self.calls[-1]['question'])
   self.assertIn('2033',self.calls[-1]['question'])
   self.assertIn('hợp nhất',self.calls[-1]['question'])
   self.assertEqual(self.calls[-1]['document'],str(self.paths['one']))
   self.ask(original,reset=True)
   before=len(self.calls)
   response=self.client.post('/api/rag/ask',json={'question':'riêng lẻ','document_ids':['one'],
       'conversation_id':'conversation'})
   self.assertEqual(response.status_code,200,response.text)
   self.assertEqual(len(self.calls),before)
   self.assertIn('chưa chứng minh số liệu riêng lẻ',response.json()['answer'])
   self.ask('hợp nhất')
   self.assertIn('hợp nhất',self.calls[-1]['question'])
   self.ask(original,reset=True)
   self.ask('hợp nhất',ids=['two'])
   self.assertEqual(self.calls[-1]['question'],'hợp nhất')
