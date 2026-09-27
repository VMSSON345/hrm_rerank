"""Evaluation labels, safety counts, and status matrices stay distinct."""
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from hrm_finance_qa.numeric_value_eval import (
    fact_signature, score_response, summarize, validate_manifest,
    verify_holdout_lock,
)


class NumericValueEvaluationTests(TestCase):
    def _manifest(self, path, *, scope='development_replay'):
        source = path.read_bytes()
        start = source.decode().splitlines()[0].index('250')
        fact = {'source_name': path.name, 'line_number': 1,
                'source_span': [start, start + 3], 'raw_value': '250', 'source_line_sha256':
                    sha256(source.decode().splitlines()[0].encode()).hexdigest(),
                'metric': 'profit_after_tax', 'period': '2031',
                'role': 'actual', 'scope': '',
                'unit_state': 'EXPLICIT_BOUND'}
        return {
            'schema_version': 'numeric-program-value-eval-v1',
            'evaluation_scope': scope, 'locked_before_inference': False,
            'sources': [{'id': 'source', 'artifact': str(path),
                         'sha256': sha256(source).hexdigest(),
                         'exposure': 'previously_used', 'layout_tags': ['horizontal_table']}],
            'cases': [{'id': 'case', 'source_id': 'source',
                       'question': 'Lợi nhuận sau thuế năm 2031 là bao nhiêu?',
                       'answer_gold': {'value': '250', 'tolerance': '0', 'unit': 'VND'},
                       'gold_programs': [{'operation': 'SELECT', 'facts': [fact]}],
                       'intent_allowed_operations': ['SELECT'],
                       'choice_class_gold': 'single',
                       'source_sufficiency_gold': True}],
        }

    def test_development_replay_scores_program_and_answer_separately(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'report.md'
            path.write_text('Lợi nhuận: 250\n', encoding='utf8')
            manifest = self._manifest(path)
            validate_manifest(manifest)
            gold = manifest['cases'][0]
            fact = gold['gold_programs'][0]['facts'][0]
            pool_item = {**fact, 'citation': {key: fact[key] for key in
                ('source_name', 'line_number', 'source_span')}}
            response = {'status': 'ACCEPTED', 'canonical_value': '250',
                        'unit': 'VND',
                        'program': {'operation': 'SELECT', 'operands': [0]},
                        'citations': [{'operand_ref': 'operand_0',
                                       'source_name': path.name, 'line_number': 1,
                                       'source_span': fact['source_span']}],
                        'program_trace': {'fact_pool': [pool_item],
                                          'proposals': [{'operation': 'SELECT',
                                                         'operands': [0],
                                                         'reason': 'accepted'}],
                                          'choice_class': 'single_valid_program'}}
            score = score_response(gold, response)
            self.assertTrue(score['correct'])
            self.assertTrue(score['candidate_recall'])
            self.assertTrue(score['operation_correct'])
            self.assertTrue(score['operands_correct'])
            self.assertTrue(score['first_proposal_correct'])
            self.assertTrue(score['intent_alignment'])
            response['unit'] = '%'
            self.assertFalse(score_response(gold, response)['correct'])
            response['unit'] = 'VND'
            gold['answer_gold']['scope'] = 'consolidated'
            response['citations'][0]['entity_scope'] = 'consolidated'
            self.assertTrue(score_response(gold, response)['correct'])
            response['citations'][0]['entity_scope'] = 'separate'
            self.assertFalse(score_response(gold, response)['correct'])
            response['citations'][0]['entity_scope'] = 'consolidated'
            response['canonical_value'] = '251'
            wrong = score_response(gold, response)
            self.assertTrue(wrong['wrong_accepted'])
            self.assertTrue(wrong['operands_correct'])

    def test_no_pool_does_not_measure_candidate_recall(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'report.md'
            path.write_text('Lợi nhuận: 250\n', encoding='utf8')
            case = self._manifest(path)['cases'][0]
            stopped = score_response(case, {
                'status': 'ABSTAIN', 'canonical_value': None,
                'program_gate_trace': {'stage': 'typed_fact_filter',
                                       'stop_reason': 'typed_fact_missing'},
            }, selector='hrm')
            self.assertIsNone(stopped['candidate_recall'])
            self.assertIsNone(stopped['operation_correct'])
            self.assertIsNone(stopped['operands_correct'])
            self.assertEqual(stopped['diagnostic_stage'], 'pool_not_observed')
            self.assertFalse(stopped['pool_observed'])
            empty = score_response(case, {'status': 'ABSTAIN',
                'program_trace': {'fact_pool': []}}, selector='baseline')
            self.assertIsNone(empty['candidate_recall'])
            self.assertIsNone(empty['pool_sha256'])

    def test_pool_token_span_excludes_witnessed_cell_whitespace(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'report.md'
            path.write_text('Lợi nhuận: 250 \n', encoding='utf8')
            case = self._manifest(path)['cases'][0]
            gold = case['gold_programs'][0]['facts'][0]
            padded = [gold['source_span'][0], gold['source_span'][1] + 1]
            item = {**gold, 'source_token_span': gold['source_span'],
                    'citation': {'source_name': path.name, 'line_number': 1,
                                 'source_span': padded}}
            score = score_response(case, {'status': 'ABSTAIN',
                'program_trace': {'fact_pool': [item]}}, selector='baseline')
            self.assertTrue(score['candidate_recall'])

    def test_paired_multiple_valid_denominator_requires_same_pool(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'report.md'
            path.write_text('Lợi nhuận: 250\n', encoding='utf8')
            case = self._manifest(path)['cases'][0]
            fact = case['gold_programs'][0]['facts'][0]
            pool_item = {**fact, 'citation': {key: fact[key] for key in
                         ('source_name', 'line_number', 'source_span')}}
            response = {'status': 'ABSTAIN', 'program_trace': {
                'fact_pool': [pool_item],
                'choice_class': 'multiple_valid_programs'}}
            baseline = score_response(case, response, selector='baseline')
            hrm = score_response(case, response, selector='hrm')
            off = score_response(case, {'status': 'ABSTAIN'}, selector='off')
            row = {'id': 'case', 'choice_class_gold': 'single',
                   'source_sufficiency_gold': True,
                   'modes': {'off': off, 'baseline': baseline, 'hrm': hrm},
                   'seconds': {'off': 0, 'baseline': 0, 'hrm': 0}}
            paired = summarize([row])['paired_multiple_valid_programs']
            self.assertEqual(paired['cases'], 1)
            self.assertEqual(paired['case_ids'], ['case'])
            row['modes']['hrm']['pool_sha256'] = 'different'
            self.assertEqual(summarize([row])['paired_multiple_valid_programs']['cases'], 0)

    def test_semantic_recall_uses_only_reviewed_identity_and_same_physical_fact(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'report.md'
            path.write_text('Tài sản ngắn hạn: 250\n', encoding='utf8')
            case = self._manifest(path)
            gold = case['cases'][0]
            fact = gold['gold_programs'][0]['facts'][0]
            fact['metric'] = 'current_assets'
            pool = {**fact, 'metric': 'literal:tai san ngan han',
                    'citation': {key: fact[key] for key in
                                 ('source_name', 'line_number', 'source_span')}}
            response = {'status': 'ABSTAIN',
                        'program_trace': {'fact_pool': [pool]}}
            score = score_response(gold, response, selector='baseline')
            self.assertFalse(score['candidate_recall'])
            self.assertTrue(score['semantic_candidate_recall'])
            for changed in ({'metric': 'literal:tai san ngan han khac'},
                            {'period': '2030'},
                            {'unit_state': 'UNIT_UNRESOLVED_RAW'}):
                score = score_response(gold, {'status': 'ABSTAIN',
                    'program_trace': {'fact_pool': [{**pool, **changed}]}},
                    selector='baseline')
                self.assertFalse(score['semantic_candidate_recall'])

    def test_holdout_requires_new_source_and_reviewed_gold(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'report.md'
            path.write_text('Lợi nhuận: 250\n', encoding='utf8')
            manifest = self._manifest(path, scope='independent_holdout')
            with self.assertRaises(ValueError):
                validate_manifest(manifest)
            manifest['locked_before_inference'] = True
            manifest['sources'][0]['exposure'] = 'new_unseen'
            with self.assertRaises(ValueError):
                validate_manifest(manifest)
            pdf = path.with_suffix('.pdf')
            pdf.write_bytes(b'%PDF-1.4\nsource fixture')
            manifest['sources'][0]['pdf_path'] = str(pdf)
            manifest['sources'][0]['pdf_sha256'] = sha256(pdf.read_bytes()).hexdigest()
            manifest['sources'][0]['pdf_pages'] = 1
            freeze = Path(directory) / 'runtime.json'
            freeze.write_text(json.dumps({
                'model_inference_runs_on_selected_sources': 0,
                'files': [{'path': str(path),
                           'sha256': sha256(path.read_bytes()).hexdigest()}],
            }), encoding='utf8')
            manifest['runtime_freeze'] = {
                'path': str(freeze),
                'sha256': sha256(freeze.read_bytes()).hexdigest()}
            with self.assertRaises(ValueError):
                validate_manifest(manifest)
            manifest['cases'][0]['gold_review'] = {
                'reviewer_id': 'reviewer-example',
                'reviewed_before_inference': True}
            self.assertEqual(len(validate_manifest(manifest)), 1)
            path.write_text('Lợi nhuận: 251\n', encoding='utf8')
            with self.assertRaises(ValueError):
                validate_manifest(manifest)

    def test_pdf_visual_gold_exposes_ocr_candidate_miss(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'report.md'
            path.write_text('OCR omitted the value\n', encoding='utf8')
            pdf = Path(directory) / 'report.pdf'
            pdf.write_bytes(b'%PDF-1.4\nvisual-only fixture')
            freeze = Path(directory) / 'runtime.json'
            freeze.write_text(json.dumps({
                'model_inference_runs_on_selected_sources': 0,
                'files': [{'path': str(path),
                           'sha256': sha256(path.read_bytes()).hexdigest()}],
            }), encoding='utf8')
            manifest = {
                'schema_version': 'numeric-program-value-eval-v1',
                'evaluation_scope': 'independent_holdout',
                'locked_before_inference': True,
                'runtime_freeze': {
                    'path': str(freeze),
                    'sha256': sha256(freeze.read_bytes()).hexdigest()},
                'sources': [{
                    'id': 'source', 'artifact': str(path),
                    'sha256': sha256(path.read_bytes()).hexdigest(),
                    'pdf_path': str(pdf),
                    'pdf_sha256': sha256(pdf.read_bytes()).hexdigest(),
                    'pdf_pages': 1, 'exposure': 'new_unseen',
                    'layout_tags': ['scanned_table']}],
                'cases': [{
                    'id': 'case', 'source_id': 'source',
                    'question': 'Lợi nhuận năm 2031 là bao nhiêu?',
                    'answer_gold': {'value': '250', 'tolerance': '0', 'unit': 'VND'},
                    'choice_class_gold': 'single',
                    'source_sufficiency_gold': True,
                    'intent_allowed_operations': ['SELECT'],
                    'gold_review': {
                        'reviewer_id': 'source-reviewer',
                        'reviewed_before_inference': True},
                    'gold_programs': [{
                        'operation': 'SELECT',
                        'facts': [{
                            'evidence_kind': 'pdf_visual',
                            'source_name': pdf.name,
                            'pdf_sha256': sha256(pdf.read_bytes()).hexdigest(),
                            'page_number': 1, 'raw_value': '250',
                            'metric': 'profit_after_tax', 'period': '2031',
                            'role': 'actual', 'scope': '',
                            'unit_state': 'EXPLICIT_BOUND',
                            'visual_review': {'reviewed_pdf_page': True},
                        }]}],
                }],
            }
            validate_manifest(manifest)
            score = score_response(
                manifest['cases'][0],
                {'status': 'ABSTAIN',
                 'program_trace': {'fact_pool': []}}, selector='hrm')
            self.assertIsNone(score['candidate_recall'])
            self.assertEqual(score['diagnostic_stage'],
                             'pool_not_observed')

    def test_equivalent_program_answer_and_intent_are_distinct(self):
        ratio = {'source_name': 'report.md', 'line_number': 1,
                 'source_span': [0, 3], 'metric': 'net_margin',
                 'period': '2031', 'role': 'actual', 'scope': ''}
        profit = {**ratio, 'line_number': 2, 'metric': 'profit_after_tax'}
        revenue = {**ratio, 'line_number': 3, 'metric': 'net_revenue'}
        case = {'answer_gold': {'value': '25', 'tolerance': '0'},
                'gold_programs': [
                    {'operation': 'SELECT', 'facts': [ratio]},
                    {'operation': 'RATIO_PERCENT', 'facts': [profit, revenue]}],
                'intent_allowed_operations': ['SELECT']}
        response = {'status': 'ACCEPTED', 'canonical_value': '25',
                    'program': {'operation': 'RATIO_PERCENT', 'operands': [0, 1]},
                    'program_trace': {'fact_pool': [profit, revenue]}}
        result = score_response(case, response)
        self.assertTrue(result['correct'])
        self.assertTrue(result['operands_correct'])
        self.assertFalse(result['intent_alignment'])
        case['intent_allowed_operations'] = ['SELECT', 'RATIO_PERCENT']
        self.assertTrue(score_response(case, response)['intent_alignment'])

    def test_negative_gold_acceptance_is_counted_as_wrong_publication(self):
        case = {'choice_class_gold': 'no_valid_program',
                'source_sufficiency_gold': False,
                'answer_gold': None, 'gold_programs': [],
                'intent_allowed_operations': ['SELECT']}
        stopped = score_response(case, {'status': 'ABSTAIN'})
        accepted = score_response(case, {'status': 'ACCEPTED',
                                         'canonical_value': '123'})
        self.assertTrue(stopped['negative_correct'])
        self.assertFalse(stopped['wrong_accepted'])
        self.assertFalse(accepted['negative_correct'])
        self.assertTrue(accepted['wrong_accepted'])
        row = {'id': 'negative', 'source_id': 'source-1',
               'choice_class_gold': 'no_valid_program',
               'source_sufficiency_gold': False,
               'modes': {'off': stopped, 'baseline': stopped, 'hrm': accepted},
               'seconds': {'off': 0.1, 'baseline': 0.1, 'hrm': 0.1}}
        summary = summarize([row])
        self.assertEqual(summary['modes']['hrm']['negative_accepted'], 1)
        self.assertEqual(summary['modes']['hrm']['wrong_accepted'], 1)
        self.assertEqual(
            summary['by_gold_choice_class']['no_valid_program']['hrm_wrong_accepted'], 1)

    def test_paired_effect_reports_discordant_cases_and_source_uncertainty(self):
        def outcome(correct):
            return {'status': 'ACCEPTED' if correct else 'ABSTAIN',
                    'correct': correct, 'wrong_accepted': False,
                    'abstain_despite_sufficient_source': not correct,
                    'halt_despite_sufficient_source': not correct,
                    'operands_correct': None, 'candidate_recall': None,
                    'operation_correct': None, 'first_proposal_correct': None,
                    'intent_alignment': None, 'physical_program': None}
        rows = [
            {'id': f'case-{i}', 'source_id': 'source-1' if i < 2 else 'source-2',
             'choice_class_gold': 'multiple_competing',
             'source_sufficiency_gold': True,
             'modes': {'off': outcome(base), 'baseline': outcome(base),
                       'hrm': outcome(hrm)},
             'seconds': {'off': 0.1, 'baseline': 0.1, 'hrm': 0.1}}
            for i, (base, hrm) in enumerate(
                [(True, True), (True, False), (False, True), (False, True)])
        ]
        effect = summarize(rows)['paired_answer_effects']['baseline_vs_hrm']
        self.assertEqual(effect['positive_gold_cases'], 4)
        self.assertEqual(effect['baseline_only_correct'], 1)
        self.assertEqual(effect['hrm_only_correct'], 2)
        self.assertEqual(effect['delta_accuracy'], 0.25)
        self.assertEqual(len(effect['source_cluster_bootstrap_95_descriptive']), 2)

    def test_fact_signature_distinguishes_unit_interpretations(self):
        common = {'source_name': 'x.md', 'line_number': 1,
                  'source_span': [0, 3], 'metric': 'revenue',
                  'period': '2031', 'role': 'actual', 'scope': ''}
        self.assertNotEqual(
            fact_signature({**common, 'unit_state': 'EXPLICIT_BOUND'}),
            fact_signature({**common, 'unit_state': 'DIMENSIONLESS_RAW'}))

    def test_holdout_lock_rejects_changed_manifest(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'protocol.json'
            artifact = Path(directory) / 'runtime.py'
            artifact.write_text('FROZEN = True\n', encoding='utf8')
            runtime = Path(directory) / 'runtime.json'
            runtime.write_text(json.dumps({
                'model_inference_runs_on_selected_sources': 0,
                'files': [{'path': str(artifact),
                           'sha256': sha256(artifact.read_bytes()).hexdigest()}],
            }), encoding='utf8')
            runtime_hash = sha256(runtime.read_bytes()).hexdigest()
            original = json.dumps({
                'schema_version': 'numeric-program-value-eval-v1',
                'runtime_freeze': {'path': str(runtime),
                                   'sha256': runtime_hash},
            }).encode()
            path.write_bytes(original)
            lock = path.with_suffix('.lock.json')
            lock.write_text(json.dumps({
                'manifest_sha256': sha256(original).hexdigest(),
                'checkpoint_sha256': 'checkpoint-hash',
                'runtime_freeze_sha256': runtime_hash,
                'locked_at_utc': '2026-09-25T00:00:00+00:00',
            }), encoding='utf8')
            verify_holdout_lock(path, original, 'checkpoint-hash')
            with self.assertRaises(ValueError):
                verify_holdout_lock(path, original + b' ', 'checkpoint-hash')
            with self.assertRaises(ValueError):
                verify_holdout_lock(path, original, 'other-checkpoint')
            artifact.write_text('FROZEN = False\n', encoding='utf8')
            with self.assertRaises(ValueError):
                verify_holdout_lock(path, original, 'checkpoint-hash')

    def test_status_matrix_does_not_demand_identical_final_answers(self):
        base = {'status': 'ACCEPTED', 'correct': True,
                'wrong_accepted': False, 'abstain_despite_sufficient_source': False,
                'operands_correct': None, 'candidate_recall': None,
                'operation_correct': None, 'first_proposal_correct': None,
                'intent_alignment': None,
                'physical_program': ['SELECT', [['report.md', 1, [13, 16]]]]}
        stopped = {**base, 'status': 'ABSTAIN', 'correct': False,
                   'physical_program': None}
        row = {'id': 'case', 'choice_class_gold': None,
               'modes': {'off': base, 'baseline': base, 'hrm': stopped},
               'seconds': {'off': 0.1, 'baseline': 0.1, 'hrm': 0.2}}
        summary = summarize([row])
        self.assertEqual(
            summary['decision_matrices']['baseline_vs_hrm']['ACCEPTED/NOT_ACCEPTED'], 1)
        self.assertEqual(summary['observed_program_overlap']['baseline_vs_hrm']['count'], 0)
        self.assertEqual(summary['modes']['hrm']['program_gold_scored'], 0)
        self.assertIsNone(summary['modes']['hrm']['candidate_recall'])
        self.assertIsNone(summary['modes']['hrm']['intent_aligned'])
        self.assertEqual(summary['by_gold_choice_class']['unlabelled']['cases'], 1)
