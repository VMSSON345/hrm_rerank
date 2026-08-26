from __future__ import annotations

import argparse
import dataclasses
import difflib
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GOVERNING = 'bb191858f92731a3521296017c77acc21ac8967c'
TRUST_ROOT = '5da2ff00bf58496a36be7c08d5bbdb1959a9542e'
PLANNER_REL = 'src/hrm_finance_qa/question_planner.py'
COHORT_REL = 'reports/finhrm_qp4_post_question_surface_post_cache_residual_coverage_cohort_build_v1.json'
RAW_REL = 'reports/finhrm_qp4_full_source_semantic_replay_refresh_v1.json'
REVIEW_REL = 'reports/finhrm_qp4_positive_target_question_plan_diagnosis_instrumentation_trust_baseline_review_v1.json'
CLOSURE_REL = 'reports/finhrm_qp4_question_plan_clean_diagnosis_closure_v1.json'
INVARIANT_REL = 'config/finhrm_report_generalization_invariant_v1.json'
INVARIANT_LOCK_REL = 'reports/finhrm_report_generalization_invariant_lock_v1.json'
BASELINE = ROOT / 'reports/finhrm_qp4_question_plan_generic_runtime_repair_baseline_v1.json'
PRECOMMIT = ROOT / 'config/finhrm_qp4_question_plan_generic_runtime_repair_precommit_v1.json'
PRECOMMIT_MD = ROOT / 'reports/finhrm_qp4_question_plan_generic_runtime_repair_precommit_v1.md'
LOCK = ROOT / 'reports/finhrm_qp4_question_plan_generic_runtime_repair_precommit_lock_v1.json'
EVALUATION = ROOT / 'reports/finhrm_qp4_question_plan_generic_runtime_repair_evaluation_v1.json'
EVALUATION_MD = ROOT / 'reports/finhrm_qp4_question_plan_generic_runtime_repair_evaluation_v1.md'
SURFACES = {'canonical': 0, 'paraphrase_1': 1, 'long_clause': 2, 'clause_reordered': 3, 'no_diacritics': 4}
COLLECTIONS = {'base_target': 'base_surfaces', 'reference_base_mutation_target': 'reference_base_mutation_surfaces'}
GENERIC_SIGNATURES = (
    '4c68f8ce8afc17484f216813ac2abd803df2b517a1888368f9c0c3093d734614',
    '86db219d02d5a40dc50ebb2e3d6eaac6df8ea1c315978ddf54e4e6e3abffea5d',
    '285c10f624322f00ba6e72e3bf1ddf28a087074b9019a0a1447a1a67f2f98cd7',
)
LOCAL_SIGNATURE = '2557d375f56992b2235dd00882eb8ae21bf144668baad0dab3d6b68b6cf635a6'
CONTROL_CATEGORIES = ('minimal_pair_role_swap', 'minimal_pair_year_shift', 'hard_negative_missing_role', 'ambiguity_control', 'PA07_operation_contrast')
ALLOWED_CHANGED_FIELDS = {'operation', 'operation_candidates', 'operand_slots', 'reference_base', 'program_dag', 'program_dag_operations', 'program_column_roles'}
FAILED_PREFLIGHT_COMPILER_CALLS = 91


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_value(value: Any) -> str:
    return sha_bytes(canonical(value).encode('utf-8'))


def file_artifact(path: Path) -> dict[str, Any]:
    return {'path': path.relative_to(ROOT).as_posix(), 'size_bytes': path.stat().st_size, 'sha256': sha_bytes(path.read_bytes())}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def git(*args: str, text: bool = True) -> Any:
    command = ['git', '-c', f'safe.directory={ROOT.as_posix()}', *args]
    return subprocess.check_output(command, cwd=ROOT, text=text, encoding='utf-8' if text else None)


def committed_bytes(path: str) -> bytes:
    return git('show', f'{GOVERNING}:{path}', text=False)


def committed_json(path: str) -> dict[str, Any]:
    value = json.loads(committed_bytes(path).decode('utf-8'))
    if not isinstance(value, dict):
        raise RuntimeError(f'HOLD: committed input is not an object: {path}')
    return value


def committed_artifact(path: str) -> dict[str, Any]:
    payload = committed_bytes(path)
    return {'path': path, 'commit': GOVERNING, 'size_bytes': len(payload), 'sha256': sha_bytes(payload)}


def check(condition: bool, name: str, failures: list[str]) -> None:
    if not condition:
        failures.append(name)


def safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value):
        return {field.name: safe(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, (tuple, list)):
        return [safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    return value


def plan_projection(plan: Any) -> dict[str, Any]:
    fields = safe(plan)
    question = fields.pop('question')
    fields['question_sha256'] = sha_bytes(question.encode('utf-8'))
    fields['program_dag_operations'] = list(plan.program_dag_operations)
    fields['program_column_roles'] = list(plan.program_column_roles)
    fields['program_dag_validation_errors'] = list(plan.program_dag_validation_errors)
    fields['required_operand_years'] = list(plan.required_operand_years)
    fields['interest_bearing_debt_share'] = bool(plan.interest_bearing_debt_share)
    return fields


def question_for(anchor: dict[str, Any], category: str, surface: str) -> tuple[str, dict[str, Any]]:
    if category in COLLECTIONS:
        selector = {'collection': COLLECTIONS[category], 'index': SURFACES[surface]}
        return str(anchor['questions'][selector['collection']][selector['index']]), selector
    selector = {'collection': 'controls', 'key': category}
    return str(anchor['questions']['controls'][category]), selector


def derive_rows(cohort: dict[str, Any], raw: dict[str, Any]) -> list[dict[str, Any]]:
    anchors = {row['anchor_id']: row for row in cohort['anchors']}
    rows: list[dict[str, Any]] = []
    for ordinal, raw_row in enumerate(raw['rows'], 1):
        anchor = anchors[raw_row['anchor_id']]
        category, surface = raw_row['category'], raw_row['surface_label']
        question, selector = question_for(anchor, category, surface)
        rows.append({
            'ordinal': ordinal, 'oracle_row_id': raw_row['oracle_row_id'],
            'anchor_id': anchor['anchor_id'], 'company_key': anchor['company_key'],
            'source_sha256': anchor['source_sha256'],
            'document_layout_family': anchor['document_layout_family'],
            'category': category, 'surface_label': surface, 'question': question,
            'question_selector': selector, 'question_sha256': sha_bytes(question.encode('utf-8')),
        })
    return rows


def integrity(failures: list[str]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    head = git('rev-parse', 'HEAD').strip()
    parents = git('rev-list', '--parents', '-n', '1', GOVERNING).strip().split()
    check(head == GOVERNING, 'head_is_governing_commit', failures)
    check(len(parents) == 2 and parents[1] == TRUST_ROOT, 'governing_directly_descends_from_trust_root', failures)
    check(not git('diff', '--name-only', GOVERNING, '--', PLANNER_REL).strip(), 'runtime_unchanged_from_governing', failures)
    closure, review = committed_json(CLOSURE_REL), committed_json(REVIEW_REL)
    check(closure.get('status') == 'PASS', 'clean_diagnosis_closure_pass', failures)
    check(review.get('status') == 'PASS', 'clean_diagnosis_review_pass', failures)
    check(closure.get('protected_content_audit', {}).get('forbidden_payload_key_hits') == {}, 'closure_no_protected_payload', failures)
    cohort, raw = committed_json(COHORT_REL), committed_json(RAW_REL)
    check(str(cohort.get('classification', '')).startswith('development_only_'), 'cohort_development_only', failures)
    check(str(raw.get('classification', '')).startswith('development_only_'), 'raw_development_only', failures)
    for source in (cohort, raw):
        inv = source.get('invariants', {})
        for key in ('protected_evaluation_loaded', 'abw_used', 'runtime_changed', 'model_changed', 'threshold_changed', 'checkpoint_changed'):
            if key in inv:
                check(inv[key] is False, f'input_invariant_{key}_false', failures)
    return cohort, raw, review


def cluster_contract(review: dict[str, Any], failures: list[str]) -> dict[str, Any]:
    groups = {row['normalized_causal_signature_sha256']: row for row in review['normalized_causal_clusters']}
    check(set(groups) == set(GENERIC_SIGNATURES) | {LOCAL_SIGNATURE}, 'exact_four_closed_clusters', failures)
    for signature in GENERIC_SIGNATURES:
        group = groups[signature]
        check(group['conclusion'] == 'GENERIC_ROOT_CAUSE', f'generic:{signature}', failures)
        check(group['diversity'] == {'rows': 6, 'companies': 6, 'sources': 6, 'real_document_layouts': 3}, f'diversity_6_6_6_3:{signature}', failures)
    local = groups[LOCAL_SIGNATURE]
    check(local['conclusion'] == 'LOCAL_INSUFFICIENT_FOR_GENERICITY', 'singleton_local', failures)
    check(local['diversity']['rows'] == 1, 'singleton_one_row', failures)
    first_575 = set(groups[GENERIC_SIGNATURES[1]]['case_ids'])
    second_575 = set(groups[GENERIC_SIGNATURES[2]]['case_ids'])
    all_sets = [set(groups[s]['case_ids']) for s in GENERIC_SIGNATURES] + [set(local['case_ids'])]
    check(first_575.isdisjoint(second_575), 'line_575_clusters_disjoint', failures)
    check(sum(map(len, all_sets)) == 19 and len(set().union(*all_sets)) == 19, 'four_group_union_19_unique', failures)
    return {
        'operand_slots_1144': groups[GENERIC_SIGNATURES[0]]['case_ids'],
        'select_to_ratio_575': groups[GENERIC_SIGNATURES[1]]['case_ids'],
        'select_to_ratio_plus_year_order_575': groups[GENERIC_SIGNATURES[2]]['case_ids'],
        'excluded_singleton_521': local['case_ids'],
        'line_575_clusters_disjoint': first_575.isdisjoint(second_575),
        'union_unique_count': len(set().union(*all_sets)),
    }


def ratio_node(slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reference = slots[1]['id']
    return [{
        'id': 'answer', 'operation': 'RATIO_PERCENT',
        'inputs': [slot['id'] for slot in slots], 'output_unit_family': 'percent',
        'intent_sources': ['operation:question_compiler', 'inputs:operand_slot_ids', f'reference_base:{reference}'],
    }]


def expected_after(base: dict[str, Any], row: dict[str, Any], anchor: dict[str, Any]) -> dict[str, Any]:
    expected = json.loads(json.dumps(base))
    category, surface = row['category'], row['surface_label']
    if category == 'base_target' and surface == 'clause_reordered':
        slots = sorted(expected['operand_slots'], key=lambda slot: {'ACTUAL': 0, 'PLAN': 1}.get(slot['column_role'], 9))
        expected['operand_slots'] = slots
        expected['reference_base'] = slots[1]['id']
        expected['program_dag'] = ratio_node(slots)
        expected['program_dag_operations'] = ['RATIO_PERCENT']
        expected['program_column_roles'] = [slot['column_role'] for slot in slots]
        expected['program_dag_validation_errors'] = []
    elif category == 'reference_base_mutation_target' and surface in {'paraphrase_1', 'clause_reordered'}:
        evidence = anchor['evidence']
        operands = anchor['programs']['reference_base_mutation']['operands']
        desired_years = [str(evidence[key]['period']) for key in operands]
        by_year = {str(slot['year']): slot for slot in expected['operand_slots']}
        slots = [by_year[year] for year in desired_years]
        for slot in slots:
            slot['unit_family'] = 'currency'
        expected['operation'] = 'RATIO_PERCENT'
        expected['operation_candidates'] = ['RATIO_PERCENT']
        expected['column_roles'] = [slot['column_role'] for slot in slots]
        expected['operand_slots'] = slots
        expected['reference_base'] = slots[1]['id']
        expected['program_dag'] = ratio_node(slots)
        expected['program_dag_operations'] = ['RATIO_PERCENT']
        expected['program_column_roles'] = ['ACTUAL', 'ACTUAL']
        expected['program_dag_validation_errors'] = []
    return expected


def build_baseline(rows: list[dict[str, Any]], failures: list[str]) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / 'src'))
    from hrm_finance_qa.question_planner import compile_question_plan

    records = []
    for row in rows:
        row_id = row['oracle_row_id']
        projection = plan_projection(compile_question_plan(row['question']))
        check(projection['question_sha256'] == row['question_sha256'], f'question_hash:{row_id}', failures)
        metadata = {key: value for key, value in row.items() if key != 'question'}
        records.append(metadata | {'projection': projection, 'projection_sha256': sha_value(projection)})
    result = {
        'schema': 'finhrm-qp4-question-plan-generic-runtime-repair-baseline-v1',
        'created_at': now(), 'classification': 'development_only_question_plan_full_projection_baseline',
        'status': 'PASS' if not failures else 'HOLD', 'governing_commit': GOVERNING,
        'compiler_calls': len(records), 'candidate_or_binding_calls': 0, 'model_calls': 0,
        'source_pdf_reads': 0, 'page_index_reads': 0, 'records': records,
    }
    write_json(BASELINE, result)
    return result


def runtime_diff_contract() -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    old = committed_bytes(PLANNER_REL).decode('utf-8').splitlines()
    current = (ROOT / PLANNER_REL).read_text(encoding='utf-8').splitlines()
    additions: list[str] = []
    changes = []
    for group in difflib.SequenceMatcher(a=old, b=current).get_grouped_opcodes(0):
        for tag, i1, i2, j1, j2 in group:
            if tag == 'equal':
                continue
            changes.append({'tag': tag, 'baseline_lines': [i1 + 1, i2], 'current_lines': [j1 + 1, j2]})
            additions.extend(current[j1:j2])
    diff = git('diff', '--unified=0', GOVERNING, '--', PLANNER_REL)
    check(bool(changes), 'runtime_change_present', failures)
    check(len(changes) <= 6, 'runtime_change_blocks_at_most_6', failures)
    check(sum(marker in diff for marker in ('compile_question_plan', '_compile_operand_slots', 'operation = SELECT', 'column_roles=lexical_column_roles', 'elif len(roles) == len(requested_years)')) >= 3, 'authorized_contexts_present', failures)
    joined = '\n'.join(additions)
    check(not any(token in joined for token in ('qp4-rcc-', 'development-company', 'source_sha256', 'layout_family', 'checkpoint', 'threshold', 'model')), 'no_case_provenance_model_gate_literals', failures)
    import re
    check(re.search(r'\b(?:19|20)\d{2}\b', joined) is None, 'no_year_literal', failures)
    return {'changes': changes, 'added_lines': len(additions), 'diff_sha256': sha_bytes(diff.encode('utf-8'))}, failures


def edit_surface() -> dict[str, Any]:
    return {
        'runtime_files': [PLANNER_REL], 'symbols': ['compile_question_plan', '_compile_operand_slots'],
        'hunks': [
            {'order': 1, 'symbol': 'compile_question_plan', 'branch': 'immediately before unary/same-role SELECT fallback', 'edit': 'classify explicit percent + exactly two same typed roles + two distinct explicit periods as RATIO_PERCENT'},
            {'order': 2, 'symbol': 'compile_question_plan', 'branch': '_compile_operand_slots call', 'edit': 'pass aligned/canonical column_roles instead of lexical_column_roles'},
            {'order': 3, 'symbol': '_compile_operand_slots', 'branch': 'two roles / two explicit periods', 'edit': 'for RATIO_PERCENT with one repeated typed role only, pair later/current period before earlier/prior period'},
        ],
        'implementation_order_locked': True,
        'interaction': 'Hunk 1 establishes RATIO_PERCENT before hunk 3 applies ratio-only period order; hunk 2 fixes completion ACTUAL/PLAN order while the role-swap control remains lexical because it is not canonicalized by that completion branch.',
        'genericity_constraints': ['no company', 'no source', 'no layout', 'no literal year', 'no numeric value', 'no full question literal', 'no case ID'],
        'forbidden': ['compile_question_plan:521 branch', '_ratio_roles', '_compile_column_roles', '_align_column_roles', 'runtime files other than planner', 'model', 'threshold', 'checkpoint'],
    }


def evaluation_contract(expected: list[dict[str, Any]], counts: dict[str, int]) -> dict[str, Any]:
    return {
        'command': 'py -3.13 scripts/run_qp4_question_plan_generic_runtime_repair_precommit.py evaluate',
        'frozen_case_count': 90, 'new_cases_after_precommit_allowed': False,
        'canonical_compiler_calls': 90, 'permuted_compiler_calls': 90, 'total_compiler_calls': 180,
        'permutation': 'same 90 questions ordered by SHA-256(oracle_row_id + fixed salt qp4-runtime-repair-v1)',
        'frozen_19_replay': {'generic_repaired': 18, 'excluded_singleton_exact_regression': 1},
        'exact_regression_cases': 72, 'counterfactual_control_categories': counts,
        'target_expected_projections': expected,
        'conditions': ['18/18 exact repaired projections', '72/72 full-plan exact regressions', '19/19 closure cases disposed without double-count', '90/90 canonical equals permuted', '30/30 named controls exact', 'zero DAG validation errors introduced', 'no runtime diff outside exact edit surface'],
        'candidate_or_binding_calls': 0, 'model_calls': 0, 'source_pdf_reads': 0, 'page_index_reads': 0,
    }


def run_precommit() -> int:
    failures: list[str] = []
    cohort, raw, review = integrity(failures)
    accounting = cluster_contract(review, failures)
    rows = derive_rows(cohort, raw)
    check(len(rows) == 90 and len({row['oracle_row_id'] for row in rows}) == 90, 'frozen_cohort_90_unique', failures)
    counts = {category: sum(row['category'] == category for row in rows) for category in CONTROL_CATEGORIES}
    check(all(value == 6 for value in counts.values()), 'counterfactual_control_categories_6_each', failures)
    baseline = build_baseline(rows, failures)
    anchors = {row['anchor_id']: row for row in cohort['anchors']}
    generic_ids = set(accounting['operand_slots_1144'] + accounting['select_to_ratio_575'] + accounting['select_to_ratio_plus_year_order_575'])
    all_closed_ids = generic_ids | set(accounting['excluded_singleton_521'])
    expected = []
    rows_by_id = {row['oracle_row_id']: row for row in rows}
    for record in baseline['records']:
        row = rows_by_id[record['oracle_row_id']]
        row_id = row['oracle_row_id']
        target = expected_after(record['projection'], row, anchors[row['anchor_id']])
        changed = sorted(key for key in target if target[key] != record['projection'].get(key))
        check(set(changed).issubset(ALLOWED_CHANGED_FIELDS), f'expected_delta_whitelist:{row_id}', failures)
        check(bool(changed) == (row_id in generic_ids), f'exact_18_changed:{row_id}', failures)
        expected.append({'oracle_row_id': row['oracle_row_id'], 'disposition': 'REPAIR' if row['oracle_row_id'] in generic_ids else 'EXACT_REGRESSION', 'changed_fields': changed, 'expected_projection': target, 'expected_projection_sha256': sha_value(target)})
    check(len(generic_ids) == 18 and len(all_closed_ids) == 19, 'repair_18_replay_19', failures)
    check(sum(bool(item['changed_fields']) for item in expected) == 18, 'exact_expected_changes_18', failures)
    source = committed_bytes(PLANNER_REL).decode('utf-8')
    quote = chr(34)
    source_markers = ('elif asks_percent and change_wording:', f'operation = {quote}PERCENT_CHANGE{quote}', f'operation = {quote}SELECT{quote}', 'column_roles=lexical_column_roles', 'def _compile_operand_slots(')
    for marker in source_markers:
        check(marker in source, f'authorized_anchor:{marker}', failures)
    input_paths = {
        'closure': CLOSURE_REL, 'diagnosis_review': REVIEW_REL,
        'development_cohort': COHORT_REL, 'raw_development_row_index': RAW_REL,
        'generalization_invariant': INVARIANT_REL,
        'generalization_invariant_lock': INVARIANT_LOCK_REL,
        'question_planner': PLANNER_REL,
    }
    inputs = {name: committed_artifact(path) for name, path in input_paths.items()}
    result = {
        'schema': 'finhrm-qp4-question-plan-generic-runtime-repair-precommit-v1',
        'created_at': now(),
        'classification': 'development_only_question_plan_generic_runtime_repair_precommit_no_runtime_change',
        'status': 'PASS' if not failures else 'HOLD', 'precommit_gate_passed': not failures,
        'governing_commit': GOVERNING, 'direct_parent': TRUST_ROOT, 'inputs': inputs,
        'protected_scope': {'protected_evaluation_content_loaded': False, 'source_pdf_content_read': False, 'page_index_read': False},
        'frozen_diagnosis': accounting,
        'frozen_open_states': {'candidate': '21/24 OPEN', 'binding_provenance': '15/17 OPEN', 'compile_question_plan_521': 'LOCAL_INSUFFICIENT_FOR_GENERICITY'},
        'authorized_edit_surface': edit_surface(),
        'locked_post_repair_evaluation': evaluation_contract(expected, counts),
        'cross_layer_risk': {
            'rating': 'MATERIAL_UPSTREAM_BUT_CONTAINED_BY_FULL_PLAN_PROJECTION_GATE',
            'candidate_status_after_pass': '21/24 OPEN', 'binding_status_after_pass': '15/17 OPEN',
            'additional_cohort_locked': False,
            'reason': 'The committed 90-case development cohort already supplies six companies, six sources, three layouts, 30 explicit controls, and all 19 closure cases; adding or silently reusing a downstream cohort would exceed the frozen scope.',
        },
        'execution': {'failed_preflight_compiler_calls': FAILED_PREFLIGHT_COMPILER_CALLS, 'final_baseline_materialization_compiler_calls': 90, 'total_precommit_construction_compiler_calls': 181, 'candidate_or_binding_calls': 0, 'runtime_changed': False},
        'failures': failures,
        'authorized_next_action': 'Implement only the three ordered hunks, then run the locked evaluator' if not failures else 'HOLD',
    }
    write_json(PRECOMMIT, result)
    status = result['status']
    next_action = result['authorized_next_action']
    report = '\n'.join([
        '# QP4 Question Plan generic runtime-repair precommit', '',
        f'Status: **{status}**.', '',
        f'Governing commit: `{GOVERNING}`; direct parent: `{TRUST_ROOT}`.', '',
        'Authorized runtime surface: `src/hrm_finance_qa/question_planner.py` only, within `compile_question_plan` and `_compile_operand_slots`, limited to the three ordered hunks. The `compile_question_plan:521` singleton is frozen and excluded.', '',
        'The repair predicate is typed question-side semantics only: explicit percent request, exactly two occurrences of the same typed role, and two distinct explicit periods. Company, source, layout, literal year, value, full question, and case ID are forbidden.', '',
        'Post-repair: 90 canonical plus 90 deterministic order-permuted compiler calls. Exact gates are 18 repaired, 72 unchanged, 19/19 closure replay, 30/30 controls, 90/90 order invariance, and no introduced DAG error.', '',
        'Candidate 21/24 and Binding/Provenance 15/17 remain OPEN. No downstream call or correctness claim is made.', '',
        f'Next action: {next_action}.', '',
    ])
    PRECOMMIT_MD.write_text(report, encoding='utf-8')
    lock = {
        'schema': 'finhrm-qp4-question-plan-generic-runtime-repair-precommit-lock-v1',
        'created_at': result['created_at'], 'status': status,
        'governing_commit': GOVERNING,
        'artifacts': {
            'baseline': file_artifact(BASELINE),
            'precommit': file_artifact(PRECOMMIT),
            'markdown': file_artifact(PRECOMMIT_MD),
            'harness': file_artifact(Path(__file__)),
        },
        'compiler_calls': {'failed_preflight': FAILED_PREFLIGHT_COMPILER_CALLS, 'final_baseline_materialization': 90, 'total': 181}, 'runtime_changed': False,
        'authorized_next_action': next_action,
    }
    write_json(LOCK, lock)
    print(f'status={status} compiler_calls=90 cases=90 repair=18 regression=72')
    if failures:
        print('failures=' + ','.join(failures))
    return 0 if not failures else 2


def run_evaluate() -> int:
    precommit = json.loads(PRECOMMIT.read_text(encoding='utf-8'))
    lock = json.loads(LOCK.read_text(encoding='utf-8'))
    failures: list[str] = []
    check(precommit.get('status') == 'PASS' and lock.get('status') == 'PASS', 'precommit_pass', failures)
    locked_hash = lock.get('artifacts', {}).get('precommit', {}).get('sha256')
    check(locked_hash == sha_bytes(PRECOMMIT.read_bytes()), 'precommit_lock', failures)
    diff_contract, diff_failures = runtime_diff_contract()
    failures.extend(diff_failures)
    cohort, raw, _ = integrity([])
    rows = derive_rows(cohort, raw)
    entries = precommit['locked_post_repair_evaluation']['target_expected_projections']
    expected = {row['oracle_row_id']: row for row in entries}
    sys.path.insert(0, str(ROOT / 'src'))
    from hrm_finance_qa.question_planner import compile_question_plan

    canonical_results = {}
    for row in rows:
        canonical_results[row['oracle_row_id']] = plan_projection(compile_question_plan(row['question']))
    permuted_results = {}
    order = sorted(rows, key=lambda row: sha_bytes((row['oracle_row_id'] + 'qp4-runtime-repair-v1').encode('utf-8')))
    for row in order:
        permuted_results[row['oracle_row_id']] = plan_projection(compile_question_plan(row['question']))
    generic_ids = {key for key, item in expected.items() if item['disposition'] == 'REPAIR'}
    regression_ids = set(expected) - generic_ids
    exact = sum(canonical_results[key] == item['expected_projection'] for key, item in expected.items())
    order_exact = sum(canonical_results[key] == permuted_results[key] for key in canonical_results)
    repaired = sum(canonical_results[key] == expected[key]['expected_projection'] for key in generic_ids)
    regressions = sum(canonical_results[key] == expected[key]['expected_projection'] for key in regression_ids)
    control_ids = {row['oracle_row_id'] for row in rows if row['category'] in CONTROL_CATEGORIES}
    controls = sum(canonical_results[key] == expected[key]['expected_projection'] for key in control_ids)
    singleton = precommit['frozen_diagnosis']['excluded_singleton_521'][0]
    singleton_exact = canonical_results[singleton] == expected[singleton]['expected_projection']
    check(exact == 90, 'expected_exact_90', failures)
    check(repaired == 18, 'repaired_exact_18', failures)
    check(regressions == 72, 'regressions_exact_72', failures)
    check(order_exact == 90, 'execution_order_permutation_90', failures)
    check(controls == 30, 'counterfactual_controls_exact_30', failures)
    check(singleton_exact, 'excluded_singleton_exact', failures)
    status = 'PASS' if not failures else 'HOLD'
    measurements = {
        'compiler_calls': 180, 'exact': f'{exact}/90',
        'generic_repaired': f'{repaired}/18', 'exact_regressions': f'{regressions}/72',
        'execution_order_permutation': f'{order_exact}/90',
        'counterfactual_controls': f'{controls}/30',
        'excluded_singleton_exact': singleton_exact, 'candidate_or_binding_calls': 0,
    }
    result = {
        'schema': 'finhrm-qp4-question-plan-generic-runtime-repair-evaluation-v1',
        'created_at': now(), 'status': status, 'governing_commit': GOVERNING,
        'runtime_diff_contract': diff_contract, 'measurements': measurements,
        'failures': failures,
        'authorized_next_action': 'Runtime-repair closure review only' if status == 'PASS' else 'HOLD',
    }
    write_json(EVALUATION, result)
    summary = (
        '# QP4 Question Plan runtime-repair evaluation\n\n'
        f'Status: **{status}**.\n\n'
        f'Exact: {exact}/90; repaired: {repaired}/18; regressions: {regressions}/72; '
        f'controls: {controls}/30; execution-order permutation: {order_exact}/90.\n'
    )
    EVALUATION_MD.write_text(summary, encoding='utf-8')
    print(f'status={status} compiler_calls=180 repaired={repaired}/18 regressions={regressions}/72 permutation={order_exact}/90')
    return 0 if status == 'PASS' else 3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('precommit', 'evaluate'))
    args = parser.parse_args()
    return run_precommit() if args.mode == 'precommit' else run_evaluate()


if __name__ == '__main__':
    raise SystemExit(main())
