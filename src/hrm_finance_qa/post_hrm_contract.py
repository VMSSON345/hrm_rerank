"""Independent post-HRM data, accounting and computation contracts.

No legacy symbolic verifier or publication policy is called here.
"""
from __future__ import annotations
from collections import defaultdict
from decimal import Decimal
from fractions import Fraction
import re
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator
from .exact_executor import ExactOperand, parse_unit
from .exact_dag_executor import execute_program_records, verify_dag_proof_certificate

SCHEMA = "post-hrm-data-accounting-contract-v1"

class SourceFact(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    fact_id: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    metric_label: str = Field(min_length=1)
    kind: Literal["money", "ratio", "quantity", "text"]
    value: Decimal | None = None
    text_value: str = ""
    raw_value: str = Field(min_length=1)
    unit: str = ""
    currency: str = ""
    unit_resolved: bool = True
    printed_quantum: Decimal = Decimal(1)
    scale: Decimal = Decimal(1)
    quantum: Decimal = Decimal(1)
    period: str = ""
    fiscal_year: str = ""
    fiscal_end_date: str = ""
    fiscal_evidence: tuple[dict, ...] = ()
    period_role: str = ""
    comparison_role: str = ""
    period_binding: str = ""
    period_basis: str = ""
    as_of_date: str = ""
    period_start: str = ""
    period_end: str = ""
    role: str = ""
    entity: str = Field(min_length=1)
    scope: str = ""
    source_name: str = Field(min_length=1)
    document_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_line_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    line_number: int = Field(gt=0)
    source_span: tuple[int, int]
    page: int | None = Field(default=None, gt=0)
    source_context: tuple[dict, ...] = ()
    unit_evidence: tuple[dict, ...] = ()
    column_label: str = ""
    table_key: str = ""
    section: str = ""
    statement: str = ""
    cashflow_method: str = ""
    cashflow_start: bool = False
    restated: bool = False
    unit_assumption: str = ""
    number_format_basis: str = ""
    source_kind: str = ""
    is_profile: bool = False
    is_historical: bool = False
    reporting_entity: bool = True
    report_year: str = ""
    source_years: tuple[str, ...] = ()
    source_verified: bool = False
    candidate_index: int = Field(default=-1, ge=-1)

    @model_validator(mode="after")
    def finite_values(self):
        if self.kind == "text":
            if not self.text_value or self.value is not None:
                raise ValueError("A text fact requires text, not a numeric placeholder")
        elif self.value is None or not self.value.is_finite():
            raise ValueError("A numeric fact requires a finite Decimal")
        if not self.scale.is_finite() or self.scale <= 0 or not self.quantum.is_finite() or self.quantum <= 0:
            raise ValueError("Scale and printed quantum must be finite and positive")
        if self.kind == "money" and self.unit_resolved and (not self.currency or not self.unit):
            raise ValueError("Monetary facts require a resolved currency and unit")
        if self.source_span[0] < 0 or self.source_span[1] <= self.source_span[0]:
            raise ValueError("Invalid source span")
        if self.kind != 'text' and self.unit and Decimal(parse_unit(self.unit).scale_to_canonical.numerator) / Decimal(parse_unit(self.unit).scale_to_canonical.denominator) != self.scale:
            raise ValueError('Declared unit and numeric scale disagree')
        return self

    @computed_field(return_type=Literal['EXPLICIT_BOUND','INHERITED_PROVEN','UNIT_UNRESOLVED_RAW','NOT_APPLICABLE'])
    @property
    def unit_state(self):
        if self.kind != 'money':
            return 'NOT_APPLICABLE'
        if not self.unit_resolved:
            return 'UNIT_UNRESOLVED_RAW'
        bases={w.get('basis') for w in self.unit_evidence}
        if bases and bases <= {'source_declaration'}:
            return 'INHERITED_PROVEN'
        return 'EXPLICIT_BOUND'

    @property
    def canonical_value(self):
        return None if self.value is None or not self.unit_resolved else self.value * self.scale

    def citation(self, operand_ref=""):
        return {"operand_ref":operand_ref or self.fact_id,"value":str(self.value) if self.value is not None else None,
            "raw_value":self.raw_value,"printed_quantum":str(self.printed_quantum),"quantum":str(self.quantum) if self.unit_resolved else None,
            "unit_resolved":self.unit_resolved,"unit_state":self.unit_state,"unit":self.unit,"scale":str(self.scale) if self.unit_resolved else None,"period":self.period,"entity_scope":self.scope or self.entity,
            "source_name":self.source_name,"page":self.page,"line_number":self.line_number,
            "source_span":list(self.source_span),"source_span_scope":"line_v2","source_line_sha256":self.source_line_sha256,
            "row_label":self.metric_label,"column_label":self.column_label,"unit_assumption":self.unit_assumption,
            "as_of_date":self.as_of_date,"period_start":self.period_start,"period_end":self.period_end,"period_role":self.period_role,"comparison_role":self.comparison_role,"period_binding":self.period_binding,
            "fiscal_year":self.fiscal_year,"fiscal_end_date":self.fiscal_end_date,"fiscal_evidence":list(self.fiscal_evidence),
            "extraction_method":"mineru_markdown","document_sha256":self.document_sha256,"source_context":list(self.source_context),"unit_evidence":list(self.unit_evidence),"number_format_basis":self.number_format_basis}

class CheckResult(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    layer: Literal["data_contract","accounting","computation","cross_statement"]
    code: str
    status: Literal["PASS","FAIL","WARNING","NOT_APPLICABLE","INSUFFICIENT_EVIDENCE"]
    blocking: bool = False
    message: str
    fact_ids: tuple[str, ...] = ()
    observed: str = ""
    expected: str = ""
    tolerance: str = ""

class ValidationReport(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    schema_version: str = SCHEMA
    checks: list[CheckResult] = Field(default_factory=list)

    @property
    def can_publish(self):
        return not any(x.blocking and x.status in {"FAIL", "INSUFFICIENT_EVIDENCE"} for x in self.checks)

    def add(self, layer, code, status, message, *, blocking=False, facts=(), **detail):
        self.checks.append(CheckResult(layer=layer,code=code,status=status,blocking=blocking,
            message=message,fact_ids=tuple(f.fact_id for f in facts),**detail))


def same_amount(left: SourceFact, right: SourceFact):
    if not left.unit_resolved or not right.unit_resolved:
        return False
    if left.kind != right.kind or left.currency != right.currency:
        return False
    if left.kind == "text":
        return left.text_value == right.text_value
    # Equal-precision conflicting values are not merged. A coarser disclosure
    # can represent a finer one only within its printed rounding interval.
    a,b = left.canonical_value,right.canonical_value
    if left.quantum == right.quantum:
        return a == b
    return abs(a-b) <= max(left.quantum,right.quantum)/2


def _unique_metric(facts, metric):
    matches=sorted((f for f in facts if f.metric==metric and f.source_verified),key=lambda f:(f.quantum,f.fact_id))
    if not matches or any(not same_amount(matches[0],f) for f in matches[1:]):
        return None
    return matches[0]


def accounting_checks(facts: list[SourceFact]):
    report=ValidationReport();groups=defaultdict(list)
    for f in facts:
        if f.source_verified and f.unit_resolved and f.kind=="money":
            groups[(f.source_name,f.entity,f.scope,f.period,f.period_role,f.as_of_date,f.period_start,f.period_end,f.role,f.currency)].append(f)
    equations = [
        ("balance_sheet",(("total_assets",1),("total_liabilities",-1),("owners_equity",-1))),
        ("net_revenue",(("net_revenue",1),("gross_revenue",-1),("revenue_deductions",1))),
        ("profit_after_tax",(("profit_after_tax",1),("profit_before_tax",-1),("current_income_tax",1),("deferred_income_tax",1))),
    ]
    for key,group in groups.items():
        for code,terms in equations:
            selected=[_unique_metric(group,metric) for metric,_ in terms]
            if any(f is None for f in selected):
                relevant = any(f.metric in {metric for metric,_ in terms} for f in group)
                report.add("accounting",code,"INSUFFICIENT_EVIDENCE" if relevant else "NOT_APPLICABLE",
                    "Thiếu số đối chiếu cùng kỳ/phạm vi; không chặn câu trả lời." if relevant else "Không có chỉ tiêu liên quan tới phép đối chiếu này.")
                continue
            residual=sum((f.canonical_value*coefficient for f,(_,coefficient) in zip(selected,terms)),Decimal(0))
            tolerance=sum((f.quantum*abs(coefficient)/2 for f,(_,coefficient) in zip(selected,terms)),Decimal(0))
            report.add("accounting",code,"PASS" if abs(residual)<=tolerance else "WARNING",
                "Đẳng thức kế toán khớp theo độ chính xác nguồn." if abs(residual)<=tolerance else "Các số công bố chưa khớp đẳng thức kế toán; cần đối chiếu nguồn.",
                facts=selected,observed=str(residual),expected="0",tolerance=str(tolerance))
    if not groups:
        report.add("accounting","accounting_inputs","NOT_APPLICABLE","Câu hỏi không có đủ số liệu kế toán để đối chiếu.")
    return report.checks


def cross_statement_checks(facts: list[SourceFact]):
    facts=[f for f in facts if f.unit_resolved]
    report=ValidationReport();tested=set()
    for opening in facts:
        if not opening.source_verified or opening.period_role != "opening" or not opening.period_start:
            continue
        from datetime import date,timedelta
        try:prior_day=str(date.fromisoformat(opening.period_start)-timedelta(days=1))
        except ValueError:continue
        for closing in facts:
            key=(opening.fact_id,closing.fact_id)
            if key in tested or not closing.source_verified or closing.period_role!="closing" or closing.period_end!=prior_day:
                continue
            if (opening.metric,opening.entity,opening.scope,opening.currency,opening.role)!=(closing.metric,closing.entity,closing.scope,closing.currency,closing.role):
                continue
            tested.add(key)
            status="PASS" if same_amount(opening,closing) else "WARNING"
            if opening.restated!=closing.restated:
                status="NOT_APPLICABLE"
            report.add("cross_statement","opening_closing",status,
                "Đối chiếu số dư cùng chỉ tiêu, phạm vi và hai ngày liền kề; trình bày lại được tách riêng.",facts=(opening,closing),observed=str(opening.canonical_value),expected=str(closing.canonical_value))
    for start in facts:
        if not(start.source_verified and start.statement=="cash_flow" and start.cashflow_method=="indirect" and start.cashflow_start):
            continue
        corresponding=[f for f in facts if f.source_verified and f.statement=="income_statement" and
            (f.metric,f.entity,f.scope,f.period,f.role,f.currency)==(start.metric,start.entity,start.scope,start.period,start.role,start.currency)]
        other=_unique_metric(corresponding,start.metric)
        if other:
            report.add("cross_statement","indirect_cashflow_start","PASS" if same_amount(start,other) else "WARNING",
                "Đối chiếu đúng chỉ tiêu mở đầu LCTT gián tiếp, không mặc định là LNST.",facts=(start,other))
    if not report.checks:
        relevant = any(f.source_verified and (f.period_role in {"opening","closing"} or (f.cashflow_method=="indirect" and f.cashflow_start)) for f in facts)
        report.add("cross_statement","cross_statement_inputs","INSUFFICIENT_EVIDENCE" if relevant else "NOT_APPLICABLE","Không có cặp số phù hợp để đối chiếu liên kỳ/liên báo cáo.")
    return report.checks


def same_row_unresolved_percent(facts):
    """A percentage change cancels an unknown common scale in one physical row."""
    if len(facts) != 2:
        return False
    a, b = facts
    if not all(f.source_verified and f.kind == 'money' and f.source_kind == 'table'
               and not f.unit_resolved and not f.unit
               and f.unit_assumption == 'source_unit_unresolved'
               and f.value is not None and f.period and f.role == 'actual'
               and f.reporting_entity and f.table_key.endswith('|amount') for f in facts):
        return False
    if not a.table_key or a.value == 0 or b.value == 0:
        return False
    if (a.period, a.period_role) == (b.period, b.period_role):
        return False
    return (a.source_name, a.document_sha256, a.source_line_sha256,
            a.line_number, a.table_key, a.metric, a.entity, a.scope,
            a.currency, a.role, a.period_basis, a.number_format_basis,
            a.printed_quantum) == (b.source_name, b.document_sha256,
            b.source_line_sha256, b.line_number, b.table_key, b.metric,
            b.entity, b.scope, b.currency, b.role, b.period_basis,
            b.number_format_basis, b.printed_quantum)


def same_table_unresolved_ratio(facts, snapshot):
    """A same-column, same-table money pair has one unknown cancelling scale.

    This is evidence only for a dimensionless ratio. It never assigns VND or
    makes either operand eligible for an absolute monetary answer.
    """
    if snapshot is None or len(facts) != 2 or not snapshot.unchanged():
        return False
    a, b = facts
    if not all(f.source_verified and f.kind == 'money' and f.source_kind == 'table'
               and not f.unit_resolved and not f.unit
               and f.unit_assumption == 'source_unit_unresolved'
               and f.value is not None and f.period and f.role == 'actual'
               and f.reporting_entity and f.table_key.endswith('|amount')
               and f.number_format_basis == 'intrinsic_token'
               and f.period_binding == 'source_column' for f in facts):
        return False
    if not b.value or a.metric == b.metric or not a.table_key:
        return False
    if (a.source_name, a.document_sha256, a.table_key, a.page,
        a.column_label, a.period, a.period_role, a.period_basis,
        a.entity, a.scope, a.currency, a.role) != (
        b.source_name, b.document_sha256, b.table_key, b.page,
        b.column_label, b.period, b.period_role, b.period_basis,
        b.entity, b.scope, b.currency, b.role):
        return False
    if not 0 < abs(a.line_number - b.line_number) <= 16:
        return False
    from hashlib import sha256
    from .ontology import normalize_text
    from .program_data import _iter_number_matches
    lines = snapshot.lines.get(a.source_name, ())
    if not lines or snapshot.hashes.get(a.source_name) != a.document_sha256:
        return False
    for fact in facts:
        if not 1 <= fact.line_number <= len(lines):
            return False
        raw = lines[fact.line_number - 1]
        variants = (raw, re.sub(r'\s+', ' ', raw).strip())
        if not any(sha256(line.encode()).hexdigest() == fact.source_line_sha256
                   and 0 <= fact.source_span[0] < fact.source_span[1] <= len(line)
                   and line[fact.source_span[0]:fact.source_span[1]].strip() == fact.raw_value
                   and any(m.span() == fact.source_span
                           for m in _iter_number_matches(line))
                   for line in variants):
            return False
    lo, hi = sorted((a.line_number, b.line_number))
    for line in lines[lo - 1:hi]:
        if line.strip() and ('|' not in line or line.startswith('## Page')):
            return False
        normal = normalize_text(line)
        if re.search(
                r'\b(?:don vi|trieu dong|nghin dong|ty dong|vnd|usd|eur|jpy|gbp|cny)\b',
                normal):
            return False
    return True


def compute_records(records, facts_by_ref, *, output_node="answer", output_unit=None, source_snapshot=None):
    """Replay fixed operations over source-bound operands, independent of old gates."""
    dimensionless = (len(records) == 1 and output_unit == '%'
        and len(records[0][2]) == 2
        and set(records[0][2]) == set(facts_by_ref)
        and ((records[0][1] == 'PERCENT_CHANGE'
              and same_row_unresolved_percent(list(facts_by_ref.values())))
             or (records[0][1] == 'RATIO_PERCENT'
                 and same_table_unresolved_ratio(
                     list(facts_by_ref.values()), source_snapshot))))
    for f in facts_by_ref.values():
        if not f.source_verified or (not f.unit_resolved and not dimensionless) or f.value is None:
            raise ValueError("Computation requires verified numeric source facts")
    currencies={f.currency for f in facts_by_ref.values() if f.kind=="money"}
    if len(currencies)>1:
        raise ValueError("Currency conversion requires an explicit dated FX input")
    operands={ref:ExactOperand(f.value,"ratio" if dimensionless else f.unit or "ratio",f.citation(ref),ref) for ref,f in facts_by_ref.items()}
    result=execute_program_records(records,operands,output_node=output_node,output_unit=output_unit)
    if not verify_dag_proof_certificate(result.certificate):
        raise ValueError("Independent arithmetic replay failed")
    return result


def compute_named(name: str, inputs: dict[str, SourceFact], *, price_date=""):
    """Explicit inputs for standard metrics; no raw percentages from a model."""
    definitions={
        "growth":("current","base","PERCENT_CHANGE","%"),
        "margin":("profit","revenue","RATIO_PERCENT","%"),
        "eps":("earnings_attributable_to_common","weighted_average_common_shares","DIVIDE",None),
        "pe":("price_per_share","eps","DIVIDE",None),
    }
    if name not in definitions:raise ValueError("Unregistered financial calculation")
    left,right,operation,unit=definitions[name]
    if left not in inputs or right not in inputs:raise ValueError("Missing required calculation inputs")
    if name=="pe" and not price_date:raise ValueError("P/E requires a source price date")
    if inputs[left].entity != inputs[right].entity or inputs[left].scope != inputs[right].scope:
        raise ValueError("Calculation inputs must refer to the same entity and scope")
    if name in {"eps","margin"} and (inputs[left].period,inputs[left].role)!=(inputs[right].period,inputs[right].role):
        raise ValueError("Calculation inputs must use the same period and actual/plan basis")
    if name=="eps" and inputs[left].metric!="earnings_attributable_to_common":
        raise ValueError("EPS requires earnings attributable to common shareholders")
    if name=="pe" and inputs[right].metric!="earnings_per_share":
        raise ValueError("P/E requires an identified EPS input")
    if name=="eps" and inputs[right].metric!="weighted_average_common_shares":
        raise ValueError("EPS denominator must be weighted-average common shares")
    if name=="pe" and (inputs[left].period!=price_date or inputs[left].metric!="price_per_share"):
        raise ValueError("Price date/identity does not match its source fact")
    return compute_records([("answer",operation,(left,right))],{left:inputs[left],right:inputs[right]},output_unit=unit)

