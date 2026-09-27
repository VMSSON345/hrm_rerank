import copy, importlib.util, json, unittest
from pathlib import Path
SUITE = Path(__file__).resolve().parents[3] / 'data/test/suites/source_period_v1'
spec = importlib.util.spec_from_file_location('grading_contract', SUITE / 'grading_contract.py')
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)

class AcceptanceGradingTests(unittest.TestCase):
    def setUp(self):
        self.ref = json.loads((SUITE/'replay_reference.json').read_text(encoding='utf8'))
        self.run = json.loads((SUITE/'live_qualified_http.json').read_text(encoding='utf8'))
    def test_original(self):
        g.validate_run(self.ref, self.run)
        rows = {r['case_id']: r for r in self.run['rows']}
        self.assertTrue(all(g.grade_response(c, rows[c['case_id']]) for c in self.ref['cases']))
    def test_missing_duplicate_empty_unexpected(self):
        for rows in (self.run['rows'][:-1], [], self.run['rows']+[self.run['rows'][0]], self.run['rows'][:-1]+[{'case_id':'unknown'}]):
            with self.subTest(n=len(rows)), self.assertRaises(ValueError):
                g.validate_run(self.ref, {**self.run, 'rows': rows})
    def test_currency(self):
        c = next(c for c in self.ref['cases'] if c['gold']['kind']=='answerable')
        r = copy.deepcopy(next(r for r in self.run['rows'] if r['case_id']==c['case_id']))
        r['response']['unit']='USD'
        self.assertFalse(g.grade_response(c,r))
    def test_error_not_abstention(self):
        c = next(c for c in self.ref['cases'] if c['gold']['kind']!='answerable')
        self.assertFalse(g.grade_response(c, {'http_status':200,'response':{'status':'ERROR'}}))
    def test_computed_recovery(self):
        self.assertEqual(g.recovery_summary(self.ref,[])['original8']['recovered'],0)
    def test_reordered(self):
        g.validate_run(self.ref,{**self.run,'rows':list(reversed(self.run['rows']))})


class FullGraderMutationTests(unittest.TestCase):
    setUp = AcceptanceGradingTests.setUp
    def grade(self, result):
        import sys
        sys.modules['grading_contract'] = g
        spec = importlib.util.spec_from_file_location('grade_live', SUITE/'grade_live.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.grade_results(self.ref, result)
    def test_full_original(self):
        self.assertTrue(self.grade(self.run)['all_checks_pass'])
    def test_full_delete_result(self):
        run=copy.deepcopy(self.run);run['rows'].pop()
        with self.assertRaises(ValueError):self.grade(run)
    def test_full_currency_and_value_mutations(self):
        for key,value in [('unit','USD'),('canonical_value','123'),('canonical_value','NaN')]:
            run=copy.deepcopy(self.run)
            row=next(r for r in run['rows'] if r['case_id']=='ccm-assets')
            row['response'][key]=value
            result=self.grade(run)
            self.assertFalse(result['all_checks_pass'])
            self.assertEqual(result['recovery']['original8']['recovered'],3)
    def test_full_error_is_counted_separately(self):
        run=copy.deepcopy(self.run)
        run['rows'][0]['response']={'status':'ERROR'}
        result=self.grade(run)
        self.assertFalse(result['all_checks_pass'])
        self.assertEqual(result['system_errors'],1)
    def test_full_http_error_not_refusal(self):
        run=copy.deepcopy(self.run);run['rows'][0]['http_status']=500
        self.assertFalse(self.grade(run)['all_checks_pass'])
