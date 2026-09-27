"""Bounded multi-part answers for an open comparison, with no evidence guessing."""
import re
from .ontology import normalize_text
from .question_planner import compile_question_plan, normalize_question_framing, resolve_public_question, public_question_assumptions

STRATEGY = 'COMPARISON_OVERVIEW'
_UNITS = {'vnd':'đồng','thousand_vnd':'nghìn đồng','million_vnd':'triệu đồng','billion_vnd':'tỷ đồng'}
_PREFIX = r'^(?:so sánh|so sanh)\s+'
_JOIN = r'(?:và|va|với|voi|so với|so voi)'
_YEAR = r'(?:năm\s+|nam\s+)?((?:19|20)\d{2})'


def build_comparison_request(question):
    from .question_request_planner import QuestionRequestPlan, RequestedOutput, MULTI_OUTPUT
    original=normalize_question_framing(question).strip()
    if not re.match(_PREFIX,original,re.I):return None
    base=compile_question_plan(question)
    if base.abstention_reason!='ambiguous_comparison_measure' or base.requested_scale=='conflict':return None
    unit=_UNITS.get(base.requested_scale,'')
    temporal=re.fullmatch(_PREFIX+r'(.+?)\s+(?:(?:giữa|giua)\s+)?'+_YEAR+r'\s+'+_JOIN+r'\s+'+_YEAR+r'(.*)',original,re.I)
    role_pair=re.fullmatch(_PREFIX+r'(.+?)\s+(kế hoạch|ke hoach|thực hiện|thuc hien)\s+'+_JOIN+r'\s+(kế hoạch|ke hoach|thực hiện|thuc hien)\s+'+_YEAR+r'(.*)',original,re.I)
    if temporal and len(set(base.column_roles))<=1:
        subject,first,second,tail=temporal.groups()
        if first==second:return None
        earlier,later=sorted([first,second],key=int)
        questions=[f'{subject} năm {earlier} là bao nhiêu {unit}?',
                   f'{subject} năm {later} là bao nhiêu {unit}?',
                   f'{subject} năm {later} tăng bao nhiêu {unit} so với năm {earlier}?']
    elif role_pair:
        subject,left,right,year,tail=role_pair.groups()
        if {normalize_text(left),normalize_text(right)}!={'ke hoach','thuc hien'}:return None
        questions=[f'{subject} kế hoạch năm {year} là bao nhiêu {unit}?',
                   f'{subject} thực hiện năm {year} là bao nhiêu {unit}?',
                   f'{subject} thực hiện năm {year} trừ {subject} kế hoạch năm {year} là bao nhiêu {unit}?']
    else:return None
    # Extra qualifiers are never discarded just because the prefix matched.
    tail=normalize_text(tail)
    if tail and tail not in {normalize_text(unit),*(normalize_text(p+' '+unit) for p in ['tính bằng','quy đổi sang','quy đổi ra','biểu thị bằng','theo đơn vị'])}:return None
    outputs=[]
    for i,q in enumerate(questions,1):
        q=re.sub(r'\s+',' ',q).replace(' ?','?')
        resolved=resolve_public_question(q);plan=compile_question_plan(resolved)
        if plan.abstention_reason or plan.requested_output_count!=1:return None
        outputs.append(RequestedOutput(output_id=f'output_{i:02d}',ordinal=i,source_span=(0,len(question)),
            source_text=question,question=resolved,plan=plan,assumptions=public_question_assumptions(q)))
    return QuestionRequestPlan(question=question,mode=MULTI_OUTPUT,outputs=tuple(outputs),split_strategy=STRATEGY,abstention_reason='')


def render_comparison(answer):
    from .answer_presentation import render_verified_answer, AnswerStatus
    complete=answer.status is AnswerStatus.ANSWER
    lines=['Đối chiếu từng kỳ/loại số liệu và chênh lệch.' if complete else
           'Đã xác minh một phần thông tin so sánh; những mục chưa đủ căn cứ được ghi rõ dưới đây.'
           if answer.verified_output_ids else
           'Chưa xác minh được số liệu để so sánh; lý do của từng mục được ghi rõ dưới đây.']
    if any('column_role:default_actual' in item.request.assumptions for item in answer.outputs):
        lines.append('Đang dùng số thực hiện theo cách hiểu mặc định.')
    labels=['Số liệu thứ nhất','Số liệu thứ hai','Chênh lệch']
    for label,item in zip(labels,answer.outputs):
        lines.extend(['',label+':',render_verified_answer(item.envelope)])
    return '\n'.join(lines)


def verified_comparison_inputs(answers):
    """Retain original source cells; validate parents before deriving a value."""
    from dataclasses import replace
    from .answer_presentation import AnswerStatus, validate_verified_answer_envelope
    from .evidence_schema import build_evidence_graph
    from .disclosure_equivalence import metric_identity, _scope_key

    candidates = tuple(replace(a.evidence[0].candidate, index=i)
                       for i, a in enumerate(answers))
    for answer in answers:
        envelope = answer.verified_envelope()
        if (answer.program.operation != 'SELECT' or len(answer.evidence) != 1
                or answer.evidence_resolution != 'UNIQUE'
                or envelope.status is not AnswerStatus.ANSWER
                or validate_verified_answer_envelope(envelope)):
            return candidates, 'comparison_input_unverified'
    left, right = build_evidence_graph(candidates).cells
    # Individually valid facts need a common perimeter before subtraction.
    # Matching numbers alone are never a compatibility condition.
    scopes = (_scope_key(left), _scope_key(right))
    if (left.source_name != right.source_name or left.entity_id != right.entity_id
            or left.entity != right.entity
            or metric_identity(left.metric_label) != metric_identity(right.metric_label)
            or left.unit_key != right.unit_key
            or scopes[0] is None or scopes[0] != scopes[1]):
        return candidates, 'comparison_inputs_not_comparable'
    return candidates, ''
