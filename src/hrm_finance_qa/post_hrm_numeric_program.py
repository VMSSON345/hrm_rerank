"""Experimental source-bound numeric programs selected by HRM or a fixed baseline.

Both selectors consume the same fact pool and pass through the same verifier.
No predicted number, planner operation, or pre-rendered answer is executed.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from decimal import Decimal
import re
import time

from .ontology import normalize_text
from .post_hrm_contract import (ValidationReport, accounting_checks,
                                cross_statement_checks, compute_records,
                                same_amount, same_row_unresolved_percent,
                                same_table_unresolved_ratio)
from .post_hrm_published import answer_published_ratio
from .post_hrm_proposals import _valid_citations
from .post_hrm_source import metric_key
from .program_data import OPERATIONS
from .question_planner import compile_question_plan, resolve_public_question


def _intent(question, plan, ratio_options):
    """Only words in the question restrict the neural operation head."""
    q = normalize_text(question)
    if tuple(slot.id for slot in plan.operand_slots) == ('profit', 'revenue'):
        return (('SELECT', 'RATIO_PERCENT') if ratio_options and not re.search(r'\btinh\b', q)
                else ('RATIO_PERCENT',))
    if ('phan tram' in q or '%' in question) and ('so voi' in q or 'bien dong' in q):
        return ('SELECT', 'PERCENT_CHANGE') if ratio_options else ('PERCENT_CHANGE',)
    if ('phan tram' in q or '%' in question) and 'ke hoach' in q and 'thuc hien' in q:
        return ('SELECT', 'RATIO_PERCENT') if ratio_options else ('RATIO_PERCENT',)
    if 'chiem bao nhieu phan tram' in q:
        return ('RATIO_PERCENT',)
    if (len(plan.operand_slots) == 1 or ratio_options) and ('bao nhieu' in q or 'la bao nhieu' in q):
        return ('SELECT',)
    return ()


def _source_token_span(fact):
    """Trim only whitespace proven by the exact witnessed source line."""
    start, end = fact.source_span
    for witness in fact.source_context:
        if (witness.get('source_name') == fact.source_name
                and witness.get('line_number') == fact.line_number):
            line = witness.get('text', '')
            if end <= len(line) and line[start:end].strip() == fact.raw_value:
                piece = line[start:end]
                return [start + len(piece) - len(piece.lstrip()),
                        end - len(piece) + len(piece.rstrip())]
    return [start, end]


def _fact_identity(fact):
    return (fact.source_name, fact.line_number, tuple(fact.source_span),
            fact.metric, fact.period, fact.period_role, fact.role, fact.scope,
            fact.value, fact.unit, fact.unit_resolved)

def _pool(question, plan, all_candidates, all_facts, ratio_options, limit, retrieved_indices=None, slot_matches=None, *, include_all_verified=False):
    keys = {metric_key(slot.metric) for slot in plan.operand_slots}
    keys.update(fact.metric for fact, _ in ratio_options)
    years = set(plan.requested_years)
    fiscal_query='nam tai chinh' in normalize_text(plan.question)
    eligible = [fact for fact in all_facts if fact.source_verified
                and fact.kind != 'text'
                and (include_all_verified or (
                    fact.metric in keys
                    and (not years or not fact.period or fact.period in years
                         or (fiscal_query and fact.fiscal_year in years)
                         or (fiscal_query and fact.fiscal_end_date[:4] in years))))
                and 0 <= fact.candidate_index < len(all_candidates)]
    available = len(eligible)
    # A verified pool that fits the checkpoint is complete already. Retrieval
    # may rank it, but must not erase a valid interpretation merely because
    # the lexical/dense shortlist missed its physical cell.
    if retrieved_indices is not None and available > limit:
        retained = set(retrieved_indices)
        eligible = [fact for fact in eligible if fact.candidate_index in retained]
    ratio_ids = {_fact_identity(fact) for fact, _ in ratio_options}
    slot_keys = {metric_key(slot.metric) for slot in plan.operand_slots}
    ordered = sorted(eligible, key=lambda fact: (
        0 if _fact_identity(fact) in ratio_ids else 1 if fact.metric in slot_keys else 2,
        all_candidates[fact.candidate_index].source_name,
        all_candidates[fact.candidate_index].line_number,
        all_candidates[fact.candidate_index].index,
        fact.metric, fact.period, fact.role, fact.scope, fact.fact_id))
    # A crowded source must retain at least one candidate for each explicit
    # operand role, even when many printed ratios precede the operands.
    reserved = []
    if len(ordered) > limit:
        if ratio_ids:
            printed = next((fact for fact in ordered
                            if _fact_identity(fact) in ratio_ids), None)
            if printed is not None:
                reserved.append(printed)
        for slot in plan.operand_slots:
            match = next((
                fact for fact in ordered
                if fact not in reserved and (
                    slot_matches(fact, slot) if slot_matches is not None else
                    (fact.metric == metric_key(slot.metric) and
                     (not slot.year or fact.period == slot.year) and
                     (not slot.column_role or
                      fact.role == slot.column_role.lower())))), None)
            if match is not None:
                reserved.append(match)
    selected = (reserved + [fact for fact in ordered if fact not in reserved])[:limit]
    candidates = []
    facts = []
    for index, fact in enumerate(selected):
        physical = all_candidates[fact.candidate_index]
        # One neural pointer represents one typed fact interpretation. A source
        # cell with multiple interpretations therefore occupies multiple slots
        # rather than silently retaining the first catalog entry.
        candidates.append(replace(
            physical, index=index, value=str(fact.value), unit=fact.unit,
            row_label=fact.metric_label, period=fact.period, column_role=fact.role,
            scope=fact.scope, is_percent=fact.unit == '%',
            unit_assumption=fact.unit_assumption,
            context=(physical.context + '\nFACT_METRIC: ' + fact.metric +
                     '\nFACT_PERIOD: ' + fact.period + '\nFACT_ROLE: ' + fact.role +
                     '\nFACT_SCOPE: ' + fact.scope +
                     '\nFACT_FISCAL_YEAR: ' + fact.fiscal_year +
                     '\nFACT_FISCAL_END_DATE: ' + fact.fiscal_end_date +
                     '\nFACT_UNIT_STATE: ' + fact.unit_state)))
        facts.append(fact.model_copy(update={'candidate_index': index}))
    return candidates, facts, available


def _baseline_quality(fact):
    """Fixed source-only comparator; never uses an answer or neural score."""
    return (not getattr(fact, 'unit_resolved', True),
            getattr(fact, 'quantum', Decimal(1)),
            bool(getattr(fact, 'unit_assumption', '')),
            not getattr(fact, 'reporting_entity', True),
            getattr(fact, 'source_name', ''),
            getattr(fact, 'line_number', 0),
            getattr(fact, 'fact_id', ''))

def _rank(bundle, question, candidates):
    from .post_hrm_engine import _rank_with_timing
    if hasattr(bundle, 'rank_raw'):
        result = _rank_with_timing(bundle, question, candidates)
        return result
    final, segments = bundle._predict_single_output(
        question, candidates, bundle.model.config.max_segments,
        max(1, bundle.model.config.minimum_halt_segments))
    return {
        'operation_scores': final.operation_logits[0].detach().cpu().tolist(),
        'select': final.select_pointer_logits[0].detach().cpu().tolist(),
        'pointers': final.pointer_logits[0].detach().cpu().tolist(),
        'pairs': {op: final.pair_logits[0, i].detach().cpu().tolist()
                  for i, op in enumerate(OPERATIONS)} if final.pair_logits is not None else {},
        'segments': segments,
    }


def _proposals(selector, allowed, ranking, facts, ratio_options):
    count = len(facts)
    if selector == 'baseline':
        operations = sorted(allowed, key=lambda op: (op != 'SELECT' if ratio_options else
                             op != 'PERCENT_CHANGE', op))
    else:
        scores = ranking.get('operation_scores') or []
        if len(scores) != len(OPERATIONS):
            return []
        operations = sorted(allowed, key=lambda op: -scores[OPERATIONS.index(op)])
    programs = []
    for op in operations:
        if op == 'SELECT':
            scores = ranking.get('select') or []
            indices = sorted(range(count), key=lambda i: (-scores[i], i)) if selector == 'hrm' and len(scores) >= count else sorted(range(count), key=lambda i: _baseline_quality(facts[i]))
            programs.extend((op, (i,)) for i in indices[:8])
        else:
            matrix = (ranking.get('pairs') or {}).get(op) or []
            indices = [(i, j) for i in range(count) for j in range(count) if i != j]
            if selector == 'hrm':
                if len(matrix) < count or any(len(row) < count for row in matrix[:count]):
                    continue
                indices.sort(key=lambda pair: (-matrix[pair[0]][pair[1]], pair))
            else:
                indices.sort(key=lambda pair: (_baseline_quality(facts[pair[0]]),
                                               _baseline_quality(facts[pair[1]])))
            programs.extend((op, pair) for pair in indices[:16])
    return programs


def _printed_margin_issue(engine, plan, printed, all_facts):
    """Return any conflict and the independently verified supporting pair."""
    if tuple(slot.id for slot in plan.operand_slots) != ('profit', 'revenue'):
        return '', ()
    match_plan = replace(plan, operation='RATIO_PERCENT')
    choices = []
    for slot in plan.operand_slots:
        choices.append([fact for fact in all_facts if fact.source_verified
                        and fact.kind == 'money'
                        and fact.entity == printed.entity and fact.scope == printed.scope
                        and fact.role == printed.role
                        and engine._matches(fact, slot, match_plan)])
    if not all(choices):
        return (('reconstructed_ratio_operands_missing'
                 if printed.period_binding == 'aligned_compound_ratio_column'
                 else ''), ())
    tolerance = max(printed.printed_quantum / Decimal(2), Decimal('0.005'))
    supporting = ()
    for profit in choices[0]:
        for revenue in choices[1]:
            if profit.currency != revenue.currency or not (
                    (profit.unit_resolved and revenue.unit_resolved)
                    or same_table_unresolved_ratio(
                        [profit, revenue], engine.snapshot)):
                continue
            if not _valid_citations(engine.snapshot,
                                    [profit.citation(), revenue.citation()]):
                continue
            try:
                result = compute_records(
                    [('answer', 'RATIO_PERCENT', ('profit', 'revenue'))],
                    {'profit': profit, 'revenue': revenue}, output_unit='%',
                    source_snapshot=engine.snapshot)
            except (ValueError, ArithmeticError):
                continue
            if abs(result.decimal_value - printed.value) > tolerance:
                return 'calculation_disagrees_with_printed_ratio', ()
            if not supporting:
                supporting = (profit, revenue)
    return (('reconstructed_ratio_operands_missing'
             if not supporting and
             printed.period_binding == 'aligned_compound_ratio_column'
             else ''), supporting)


def _verify(engine, question, plan, operation, indices, pool_facts, all_facts,
            ratio_options):
    chosen = [pool_facts[i] for i in indices]
    match_plan = replace(plan, operation=operation)
    report = ValidationReport()
    crosscheck = ()
    if not chosen or any(not f.source_verified or f.kind == 'text' for f in chosen):
        return None, 'source_fact_unverified'
    if not engine.snapshot.unchanged() or not _valid_citations(engine.snapshot,
            [f.citation() for f in chosen]):
        return None, 'source_hash_or_span_invalid'
    if operation == 'SELECT':
        fact = chosen[0]
        if ratio_options:
            valid = {_fact_identity(f) for f, _ in ratio_options}
            if _fact_identity(fact) not in valid:
                return None, 'published_ratio_role_or_period_mismatch'
            if len({(f.value, f.scope) for f, _ in ratio_options}) != 1:
                return None, 'published_ratio_conflict'
            issue, crosscheck = _printed_margin_issue(engine, plan, fact, all_facts)
            if issue:
                return None, issue
            if fact.period_binding == 'aligned_compound_ratio_column':
                report.add('data_contract', 'aligned_compound_ratio', 'PASS',
                           'Tỷ lệ có span riêng, gắn nhãn theo thứ tự một-một của bảng và khớp hai toán hạng độc lập.',
                           facts=(fact, *crosscheck))
        elif len(plan.operand_slots) != 1 or not engine._matches(fact, plan.operand_slots[0], match_plan):
            return None, 'requested_fact_mismatch'
    else:
        if len(plan.operand_slots) != 2 or len(chosen) != 2:
            return None, 'operand_arity_mismatch'
        if any(not engine._matches(f, slot, match_plan) for f, slot in zip(chosen, plan.operand_slots)):
            return None, 'operand_metric_period_role_mismatch'
        if chosen[0].fact_id == chosen[1].fact_id or (
                chosen[0].entity, chosen[0].scope) != (chosen[1].entity, chosen[1].scope):
            return None, 'operand_identity_or_scope_mismatch'
        if operation == 'PERCENT_CHANGE' and (chosen[0].period, chosen[0].period_role) == (chosen[1].period, chosen[1].period_role):
            return None, 'same_period_operands'
        if operation == 'RATIO_PERCENT' and (chosen[0].period != chosen[1].period or
                {chosen[0].role, chosen[1].role} not in ({'actual', 'plan'}, {'actual'})):
            return None, 'ratio_period_or_role_mismatch'
    from .post_hrm_engine import _logical_key
    for slot_index, fact in enumerate(chosen):
        key = _logical_key(fact)
        # Compare declarations of the same identity, including physical date
        # and period basis. A different closing date is a temporal alternative,
        # not a contradictory value for this one date.
        companions = [other for other in all_facts if other.source_verified
                      and _logical_key(other) == key]
        explicit = [other for other in companions if other.unit_resolved]
        unresolved_companions = [other for other in companions if not other.unit_resolved]
        if explicit and unresolved_companions:
            report.add('data_contract', 'optional_disclosure_unit_missing',
                       'NOT_APPLICABLE', 'Công bố phụ thiếu đơn vị; dùng fact có đơn vị đã chứng minh.',
                       facts=tuple(unresolved_companions))
        if fact.unit_resolved and any(not same_amount(fact, peer) for peer in explicit):
            return None, 'source_values_conflict'
        if not fact.unit_resolved and not explicit and len({f.value for f in companions}) > 1:
            return None, 'source_values_conflict'
        if not plan.requested_accounting_scope:
            scope_key = key[:7] + (fact.source_name,) + key[9:]
            variants = [other for other in all_facts if other.source_verified
                        and (_logical_key(other)[:7] + (other.source_name,) +
                             _logical_key(other)[9:]) == scope_key]
            if len({other.scope for other in variants}) > 1 and any(
                    left.unit_resolved and right.unit_resolved
                    and not same_amount(left, right)
                    for left in variants for right in variants):
                return None, 'source_scope_ambiguous'
        if operation == 'SELECT' and ratio_options:
            continue
        slot = plan.operand_slots[slot_index]
        temporal = [other for other in all_facts if other.source_verified
                    and engine._matches(other, slot, match_plan)
                    and (other.metric, other.entity, other.scope, other.role,
                         other.currency) == (fact.metric, fact.entity, fact.scope,
                                             fact.role, fact.currency)]
        if any(f.period_role == 'closing' for f in temporal) and not any(
                f.period_role == 'opening' for f in temporal):
            temporal = [f for f in temporal if f.period_role != 'opening']
        def point(item):
            return item.as_of_date or (item.period_start if item.period_role == 'opening'
                                       else item.period_end if item.period_role == 'closing' else '')
        dated = [f for f in temporal if point(f)]
        identities = [f for f in temporal if point(f) or not any(
            f.period == d.period and f.period_basis == d.period_basis
            and (not f.period_role or f.period_role == d.period_role)
            and not f.period_start and not f.period_end and same_amount(f, d)
            for d in dated)]
        if len({_logical_key(f)[1:6] for f in identities}) > 1:
            return None, 'source_period_ambiguous'
    common_scale = (operation == 'RATIO_PERCENT'
                    and same_table_unresolved_ratio(chosen, engine.snapshot))
    if any(not f.unit_resolved for f in chosen) and not (
            (operation == 'PERCENT_CHANGE'
             and same_row_unresolved_percent(chosen)) or common_scale):
        return None, 'source_unit_scale_missing'
    if common_scale:
        report.add('data_contract', 'same_table_common_scale', 'PASS',
                   'Hai toán hạng cùng bảng và cột kỳ có chung thang đo chưa biết; chỉ triệt tiêu trong tỷ lệ.',
                   facts=tuple(chosen))
    refs = {f'operand_{i}': f for i, f in enumerate(chosen)}
    output_unit = '%' if operation in {'PERCENT_CHANGE', 'RATIO_PERCENT'} or (operation == 'SELECT' and chosen[0].unit == '%') else 'VND' if all(f.kind == 'money' and f.currency == 'VND' for f in chosen) else None
    record = [('answer', operation, tuple(refs))]
    try:
        result = compute_records(record, refs, output_unit=output_unit,
                                 source_snapshot=engine.snapshot)
    except (ValueError, ArithmeticError) as exc:
        return None, 'executor:' + str(exc)
    if operation == 'RATIO_PERCENT' and ratio_options and any(
            abs(result.decimal_value - printed.value) > max(printed.printed_quantum / Decimal(2), Decimal('0.005'))
            for printed, _ in ratio_options):
        return None, 'calculation_disagrees_with_printed_ratio'
    report.add('data_contract', 'source_and_intent', 'PASS',
               'Fact và vai trò khớp ràng buộc tường minh của câu hỏi.', facts=tuple(chosen))
    report.add('computation', 'independent_recompute', 'PASS',
               'Đã thực thi và kiểm tra số học chính xác.', facts=tuple(chosen))
    related = [f for f in all_facts if any((f.entity, f.scope, f.period, f.role) ==
               (s.entity, s.scope, s.period, s.role) for s in chosen)]
    report.checks.extend(accounting_checks(related))
    report.checks.extend(cross_statement_checks(all_facts))
    if not report.can_publish:
        return None, 'blocking_validation_check'
    from .post_hrm_engine import ENGINE_ID, _number, _source_display_name
    value = result.decimal_value
    unit = result.output.unit.symbol
    display_scale = ({'million_vnd': Decimal(1000000),
                      'billion_vnd': Decimal(1000000000),
                      'thousand_vnd': Decimal(1000)}.get(plan.requested_scale, Decimal(1))
                     if unit == 'VND' else Decimal(1))
    display_unit = {Decimal(1000000): 'triệu đồng',
                    Decimal(1000000000): 'tỷ đồng',
                    Decimal(1000): 'nghìn đồng'}.get(display_scale, unit)
    if operation == 'SELECT' and ratio_options:
        rendered = f'{plan.metric_text}: tỷ lệ công bố {_number(value)}%.' if unit == '%' else f'{plan.metric_text}: tỷ số công bố {_number(value)}.'
    elif (operation == 'RATIO_PERCENT' and
          tuple(slot.id for slot in plan.operand_slots) == ('profit', 'revenue')):
        from .metric_semantics import display_label
        rendered = (f'{display_label(plan.metric_text)} năm {chosen[0].period} là '
                    f'{_number(value, True)}% (tính từ '
                    f'{display_label(chosen[0].metric_label)} / '
                    f'{display_label(chosen[1].metric_label)}).')
    else:
        rendered = engine._render(replace(plan, operation=operation), value / display_scale, display_unit, chosen)
    rendered += ' Nguồn: ' + '; '.join(sorted({f'{_source_display_name(f.source_name)}, trang {f.page}' for f in chosen})) + '.'
    selected_ids = {f.fact_id for f in chosen}
    warnings = [check.message for check in report.checks if check.status == 'WARNING'
                and selected_ids.intersection(check.fact_ids)]
    if warnings:
        rendered += '\nLưu ý: ' + ' '.join(dict.fromkeys(warnings))
    if any(check.code == 'optional_disclosure_unit_missing' for check in report.checks):
        rendered += '\nCó công bố khác chưa xác định được đơn vị/thang đo; chưa thể đối chiếu với số có đơn vị rõ được dùng ở đây.'
    citations = [f.citation(ref) for ref, f in refs.items()]
    citations.extend(f.citation(ref) for ref, f in zip(
        ('crosscheck_profit', 'crosscheck_revenue'), crosscheck))
    return {'status': 'ACCEPTED', 'final_action': 'ANSWER', 'answer': str(value),
            'canonical_value': str(value), 'unit': unit, 'display_value': str(value / display_scale),
            'display_unit': display_unit, 'rendered': rendered, 'citations': citations,
            'validation_engine': ENGINE_ID, 'validation': report.model_dump(mode='json'),
            'execution_route': 'numeric_program',
            'program': {'operation': operation, 'operands': list(indices)},
            'verified_program': [('answer', operation, list(refs))],
            'proof': {'verification_engine': ENGINE_ID, 'source_action': 'ANSWER',
                      'verification_state': 'VALID',
                      'execution_certificate': result.certificate.to_dict()}}, ''


def _valid_choice_class(engine, question, plan, allowed, facts, all_facts, ratio_options):
    signatures = set()
    for op in allowed:
        if op == 'SELECT':
            valid_ids = {_fact_identity(f) for f, _ in ratio_options} if ratio_options else None
            choices = [(i,) for i, f in enumerate(facts) if
                       (_fact_identity(f) in valid_ids if valid_ids is not None else
                        len(plan.operand_slots) == 1 and engine._matches(f, plan.operand_slots[0], replace(plan, operation=op)))]
        elif len(plan.operand_slots) == 2:
            left = [i for i, f in enumerate(facts) if engine._matches(f, plan.operand_slots[0], replace(plan, operation=op))]
            right = [i for i, f in enumerate(facts) if engine._matches(f, plan.operand_slots[1], replace(plan, operation=op))]
            choices = [(i, j) for i in left for j in right if i != j]
        else:
            choices = []
        for indices in choices:
            answer, _ = _verify(engine, question, plan, op, indices, facts, all_facts, ratio_options)
            if answer is None:
                continue
            signature = (op, tuple((facts[i].metric, facts[i].period, facts[i].role,
                                    facts[i].scope, str(facts[i].canonical_value)) for i in indices))
            signatures.add(signature)
            if len(signatures) >= 2:
                return 'multiple_valid_programs'
    return 'single_valid_program' if signatures else 'no_valid_program'


def evaluate_numeric_program(engine, question, chunks, all_candidates, all_facts, *, selector,
                             injected_pool_facts=None):
    """Return None outside the initial numeric scope; otherwise a verified result."""
    started = time.perf_counter()
    from .post_hrm_engine import _PROGRAM_GATE_TRACE
    gate = _PROGRAM_GATE_TRACE.get()
    plan = compile_question_plan(resolve_public_question(question))
    from .post_hrm_formulas import margin_operand_plan
    margin = margin_operand_plan(question)
    if margin is not None:
        plan = margin
    if (not plan.operand_slots and plan.operation == 'SELECT'
            and len(plan.requested_years) <= 1 and
            any(f.metric == metric_key(plan.metric_text) for f in all_facts)):
        from .question_planner import OperandSlot
        slot = OperandSlot('lookup', plan.metric_text,
                           plan.requested_years[0] if plan.requested_years else '',
                           plan.column_roles[0] if plan.column_roles else 'ACTUAL',
                           plan.entity_mode, plan.requested_unit_family or 'currency_or_source',
                           ('explicit_source_metric', 'actual_default'))
        plan = replace(plan, operand_slots=(slot,))
    selectable_facts = all_facts if injected_pool_facts is None else injected_pool_facts
    ratio_options = answer_published_ratio(question, selectable_facts, fact_options_only=True) or []
    if margin is not None:
        from .question_planner import OperandSlot
        printed_slot = OperandSlot(
            'printed_margin', plan.metric_text,
            plan.requested_years[0] if plan.requested_years else '',
            'ACTUAL', plan.entity_mode, 'percent',
            ('published_named_margin',))
        printed_plan = replace(plan, operation='SELECT')
        ratio_options = [(f, []) for f in selectable_facts if f.source_verified
                         and f.unit == '%' and f.reporting_entity
                         and engine._matches(f, printed_slot, printed_plan)]
    if not ratio_options and not plan.operand_slots and 'bao nhieu' in normalize_text(question):
        years = set(plan.requested_years)
        key = metric_key(plan.metric_text)
        ratio_options = [(f, []) for f in selectable_facts if f.source_verified
                         and f.metric == key and f.unit in {'%', 'x', 'ratio'}
                         and f.reporting_entity and f.role == 'actual'
                         and (not years or f.period in years)]
    allowed = _intent(question, plan, ratio_options)
    if gate is not None:
        gate.update({"stage": "question_planning", "planner_operation": plan.operation,
                     "planner_slot_count": len(plan.operand_slots),
                     "planner_slots": [{"id": slot.id, "metric": metric_key(slot.metric),
                                        "year": slot.year} for slot in plan.operand_slots],
                     "allowed_operations": list(allowed),
                     "printed_ratio_options": len(ratio_options),
                     "planner_abstention_reason": plan.abstention_reason})
    if not allowed or (not plan.operand_slots and not ratio_options) or plan.abstention_reason:
        if gate is not None:
            gate["stop_reason"] = ("planner_abstention" if plan.abstention_reason else
                                   "planner_slots_missing" if not plan.operand_slots and not ratio_options else
                                   "operation_intent_unsupported")
        return None
    # An exact, source-verified fact pool that fits the checkpoint must stay
    # complete. Invoke lexical/dense/graph retrieval only to allocate a
    # limited candidate budget when that pool is too large.
    limit = engine.bundle.model.config.max_candidates
    candidates, facts, available = _pool(
        question, plan, all_candidates, selectable_facts, ratio_options, limit,
        include_all_verified=injected_pool_facts is not None)
    if gate is not None:
        keys = {metric_key(slot.metric) for slot in plan.operand_slots}
        keys.update(fact.metric for fact, _ in ratio_options)
        years = set(plan.requested_years)
        gate.update({"stage": "typed_fact_filter",
                     "requested_metric_keys": sorted(keys),
                     "source_cells_with_requested_label": sum(
                         metric_key(c.row_label) in keys for c in all_candidates),
                     "typed_facts_with_requested_metric": sum(
                         f.metric in keys for f in all_facts),
                     "typed_facts_with_requested_period": sum(
                         f.metric in keys and (not years or f.period in years)
                         for f in all_facts),
                     "eligible_fact_count": available})
    if injected_pool_facts is not None and available > limit:
        raise ValueError('Injected pool exceeds checkpoint candidate budget')
    retrieved = []
    if available <= limit:
        retrieval_trace = {
            'mode': 'verified_catalog_complete',
            'dense': 'skipped_complete_pool',
            'pool_selection': 'verified_pool_fits_budget',
            'budget': limit,
            'catalog_candidates': len(all_candidates),
            'selected_candidates': len(candidates),
            'graph_edges': 0,
            'graph_added_selected': [],
        }
    else:
        # Retrieval chooses physical cells. Each cell may still contribute
        # several typed fact interpretations to the neural pool.
        ratio_ids = {_fact_identity(fact) for fact, _ in ratio_options}
        retrieval_slots = list(plan.operand_slots)
        if ratio_ids:
            retrieval_slots.append(SimpleNamespace(id='printed_ratio'))

        def retrieval_match(fact, slot):
            if not fact.source_verified or fact.kind == 'text':
                return False
            if slot.id == 'printed_ratio':
                return _fact_identity(fact) in ratio_ids
            return any(engine._matches(fact, slot, replace(plan, operation=op))
                       for op in allowed)

        retrieved, retrieval_trace = engine.retriever.retrieve(
            resolve_public_question(question), chunks, all_candidates, all_facts,
            retrieval_slots, retrieval_match, limit,
            use_graph=engine.retrieval_mode in {'adaptive_graph', 'hybrid_graph'})
        retained_indices = [
            item['catalog_index'] for item in retrieval_trace['candidates']
            if item['catalog_index'] is not None]
        candidates, facts, available = _pool(
            question, plan, all_candidates, all_facts, ratio_options,
            limit, retained_indices, retrieval_match)
        retrieval_trace['pool_selection'] = 'hybrid_retrieval_budgeted'
    retrieval_trace['eligible_fact_count'] = available
    retrieval_trace['retained_typed_fact_count'] = len(facts)
    if not candidates:
        if gate is not None:
            gate.update({"stage": "fact_pool", "stop_reason":
                         "typed_fact_missing" if available == 0 else "retrieval_pool_empty"})
        if available == 0:
            return None
        report = ValidationReport()
        report.add(
            'data_contract', 'numeric_retrieval_pool_empty',
            'INSUFFICIENT_EVIDENCE',
            'Có fact số đã xác minh nhưng truy hồi không giữ được ứng viên trong ngân sách.',
            blocking=True)
        stopped = engine._empty(
            question,
            'Đã thấy fact số trong nguồn nhưng chưa chọn được pool đủ chứng cứ để trả lời.',
            report)
        stopped['program_selector'] = selector
        stopped['program_trace'] = {
            'mode': selector, 'candidate_count': 0,
            'candidate_available': available, 'retrieval': retrieval_trace,
            'proposals': [], 'fact_pool': []}
        return stopped
    if gate is not None:
        gate.update({"stage": "program_verifier", "fact_pool_count": len(facts),
                     "stop_reason": ""})
    ranking = _rank(engine.bundle, question, candidates) if selector == 'hrm' else {}
    tried = []
    programs = _proposals(selector, allowed, ranking, facts, ratio_options)
    answer = None
    for op, indices in programs:
        candidate, reason = _verify(engine, question, plan, op, indices, facts,
                                    all_facts, ratio_options)
        tried.append({'operation': op, 'operands': list(indices),
                      'fact_ids': [facts[i].fact_id for i in indices],
                      'reason': reason or 'accepted'})
        if candidate is not None:
            answer = candidate
            break
    choice_class = _valid_choice_class(engine, question, plan, allowed, facts,
                                       all_facts, ratio_options)
    trace = {'mode': selector, 'candidate_count': len(facts),
             'candidate_available': available,
             'retrieval': retrieval_trace,
             'retrieved_physical_candidates': len(retrieved),
             'injected_pool': injected_pool_facts is not None,
             'choice_class': choice_class,
             'allowed_operations': list(allowed),
             'neural_operation': OPERATIONS[max(range(len(ranking['operation_scores'])),
                                   key=lambda i: ranking['operation_scores'][i])] if ranking.get('operation_scores') else None,
             'proposals': tried, 'seconds': time.perf_counter() - started,
             'fact_pool': [{'fact_id': f.fact_id, 'metric': f.metric, 'period': f.period,
                            'fiscal_year': f.fiscal_year, 'fiscal_end_date': f.fiscal_end_date,
                            'role': f.role, 'scope': f.scope, 'unit_state': f.unit_state,
                            'source_token_span': _source_token_span(f),
                            'citation': f.citation()} for f in facts]}
    if answer is None:
        unresolved = next((item for item in tried if item['reason'] == 'source_unit_scale_missing'), None)
        if unresolved and len(unresolved['operands']) == 1:
            fact = facts[unresolved['operands'][0]]
            companions = [f for f in all_facts if f.source_verified and
                          (f.metric, f.period, f.period_role, f.role, f.entity, f.scope) ==
                          (fact.metric, fact.period, fact.period_role, fact.role, fact.entity, fact.scope)]
            answer = engine._unit_response(question, plan, ValidationReport(), fact,
                                           all_facts, companions)
        elif unresolved:
            selected = [facts[i] for i in unresolved['operands']]
            report = ValidationReport()
            report.add('data_contract', 'source_unit_scale_missing', 'INSUFFICIENT_EVIDENCE',
                       'Các toán hạng chưa chứng minh đơn vị và thang đo.',
                       blocking=True, facts=tuple(selected))
            answer = engine._empty(question,
                'Đã thấy các số gốc nhưng chưa chứng minh đơn vị để thực hiện phép tính.',
                report, [f.citation() for f in selected])
            answer['answer_type'] = 'explained_halt'
            answer['raw_disclosures'] = [{'metric': f.metric, 'raw_value': f.raw_value,
                                           'unit_state': f.unit_state, 'citations': [f.citation()]}
                                          for f in selected]
    if answer is None:
        messages = {
            'source_period_ambiguous': 'Nguồn có nhiều kỳ hoặc ngày chốt phù hợp; cần làm rõ thời điểm.',
            'source_values_conflict': 'Các công bố cùng chỉ tiêu, kỳ và phạm vi chưa thống nhất.',
            'source_scope_ambiguous': 'Nguồn có nhiều phạm vi công bố khác nhau; cần xác nhận phạm vi.',
            'published_ratio_conflict': 'Các tỷ lệ công bố phù hợp câu hỏi chưa thống nhất.',
        }
        blocked = next((item for reason in messages for item in tried
                        if item['reason'] == reason), None)
        reason = blocked['reason'] if blocked else (
            'hrm_program_unverified' if selector == 'hrm' else 'baseline_program_unverified')
        message = messages.get(reason, 'Chưa xác nhận được chương trình và toán hạng phù hợp.')
        evidence = [facts[i] for i in blocked['operands']] if blocked else []
        report = ValidationReport()
        report.add('data_contract', reason, 'INSUFFICIENT_EVIDENCE', message,
                   blocking=True, facts=tuple(evidence))
        answer = engine._empty(question, message, report, [f.citation() for f in evidence])
    if gate is not None:
        gate.update({"stage": "program_verifier", "verifier_choice_class": choice_class,
                     "tried_proposal_count": len(tried),
                     "accepted_program": answer.get("status") == "ACCEPTED",
                     "verifier_reasons": list(dict.fromkeys(item["reason"] for item in tried))})
    answer['program_selector'] = selector
    answer['program_trace'] = trace
    return answer
