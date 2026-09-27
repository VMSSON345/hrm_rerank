import unittest
from decimal import Decimal
from pydantic import ValidationError
from hrm_finance_qa.post_hrm_contract import SourceFact,ValidationReport,accounting_checks,cross_statement_checks,compute_records,compute_named

def fact(metric,amount,**changes):
    data=dict(fact_id=metric,metric=metric,metric_label=metric,kind='money',value=Decimal(str(amount)),raw_value=str(amount),unit='VND',currency='VND',entity='company',scope='consolidated',period='2024',role='actual',source_name='test.md',document_sha256='0'*64,source_line_sha256='1'*64,line_number=1,source_span=(0,1),source_verified=True)
    data.update(changes);return SourceFact(**data)

class PostHRMContractTests(unittest.TestCase):
    def test_schema_rejects_float_nan_and_missing_currency(self):
        for kwargs in [dict(value=1.5),dict(value=Decimal('NaN')),dict(currency=''),dict(source_span=(3,2))]:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValidationError):fact('profit',1,**kwargs)
    def test_optional_accounting_inputs_do_not_block_lookup(self):
        f=fact('charter_capital',100);r=ValidationReport(checks=accounting_checks([f])+cross_statement_checks([f]));self.assertTrue(r.can_publish);self.assertTrue(all(c.status=='NOT_APPLICABLE' for c in r.checks))
    def test_balance_and_current_deferred_tax_checks(self):
        fs=[fact('total_assets',100),fact('total_liabilities',40),fact('owners_equity',60),fact('profit_before_tax',30),fact('current_income_tax',7),fact('deferred_income_tax',-2),fact('profit_after_tax',25)]
        checks=accounting_checks(fs);self.assertEqual(next(c.status for c in checks if c.code=='balance_sheet'),'PASS');self.assertEqual(next(c.status for c in checks if c.code=='profit_after_tax'),'PASS')
    def test_wrong_scope_not_mixed_and_imbalance_is_diagnostic(self):
        fs=[fact('total_assets',100),fact('total_liabilities',40),fact('owners_equity',20)]
        r=ValidationReport(checks=accounting_checks(fs));self.assertTrue(r.can_publish);self.assertEqual(next(c.status for c in r.checks if c.code=='balance_sheet'),'WARNING')
        fs[-1]=fact('owners_equity',60,scope='separate');self.assertTrue(all(c.status in {'NOT_APPLICABLE','INSUFFICIENT_EVIDENCE'} for c in accounting_checks(fs)))
    def test_growth_is_recomputed_and_zero_base_rejected(self):
        r=compute_records([('answer','PERCENT_CHANGE',('current','base'))],{'current':fact('profit',80),'base':fact('profit',100,period='2023')},output_unit='%')
        self.assertEqual(r.decimal_value,Decimal('-20'))
        with self.assertRaises(ZeroDivisionError):compute_records([('answer','PERCENT_CHANGE',('a','b'))],{'a':fact('profit',80),'b':fact('profit',0)})
    def test_same_row_unknown_scale_cancels_only_for_percent_change(self):
        shared=dict(unit='',unit_resolved=False,unit_assumption='source_unit_unresolved',
                    currency='VND',source_kind='table',table_key='test-table|amount',
                    number_format_basis='intrinsic_token')
        current=fact('profit',80,fact_id='current',source_span=(2,3),**shared)
        base=fact('profit',100,fact_id='base',period='2023',source_span=(4,5),**shared)
        records=[('answer','PERCENT_CHANGE',('current','base'))]
        result=compute_records(records,{'current':current,'base':base},output_unit='%')
        self.assertEqual(result.decimal_value,Decimal('-20'))
        self.assertIsNone(current.canonical_value)
        for changed in (dict(line_number=2),dict(table_key='other|amount'),dict(metric='revenue'),
                        dict(unit='VND',unit_resolved=True,unit_assumption='')):
            with self.subTest(changed=changed),self.assertRaises(ValueError):
                compute_records(records,{'current':current,'base':base.model_copy(update=changed)},output_unit='%')
        with self.assertRaises(ValueError):
            compute_records([('answer','SELECT',('current',))],{'current':current},output_unit='VND')
    def test_reported_percentage_is_a_valid_lookup(self):
        r=compute_records([('answer','SELECT',('reported',))],{'reported':fact('reported_comparison',87,kind='ratio',unit='%',currency='',scale=Decimal('.01'))},output_unit='%');self.assertEqual(r.decimal_value,87)
    def test_unverified_source_cannot_be_computed(self):
        with self.assertRaises(ValueError):compute_records([('answer','SELECT',('x',))],{'x':fact('profit',80,source_verified=False)})
    def test_opening_closing_requires_dates_and_preserves_restatement(self):
        a=fact('cash',100,period='2024',period_role='opening',period_start='2024-01-01');b=fact('cash',100,period='2023',period_role='closing',period_end='2023-12-31',fact_id='prior')
        self.assertEqual(cross_statement_checks([a,b])[0].status,'PASS')
        b=fact('cash',90,period='2023',period_role='closing',period_end='2023-12-31',fact_id='prior',restated=True)
        self.assertEqual(cross_statement_checks([a,b])[0].status,'NOT_APPLICABLE')
    def test_cashflow_start_uses_actual_metric_and_skips_direct_method(self):
        a=fact('profit_before_tax',100,statement='cash_flow',cashflow_method='indirect',cashflow_start=True);b=fact('profit_before_tax',100,statement='income_statement',fact_id='income')
        self.assertEqual(cross_statement_checks([a,b])[0].status,'PASS')
        a=fact('profit_before_tax',100,statement='cash_flow',cashflow_method='direct',cashflow_start=True);self.assertEqual(cross_statement_checks([a,b])[0].status,'NOT_APPLICABLE')
    def test_eps_requires_weighted_shares_and_pe_dated_price(self):
        with self.assertRaises(ValueError):compute_named('eps',{'earnings_attributable_to_common':fact('earnings_attributable_to_common',100),'weighted_average_common_shares':fact('closing_shares',10,kind='quantity',unit='shares',currency='')})
        with self.assertRaises(ValueError):compute_named('pe',{'price_per_share':fact('price_per_share',100),'eps':fact('earnings_per_share',10)})

    def test_unit_scale_disagreement_is_rejected(self):
        with self.assertRaises(ValueError):fact('profit',100,unit='VND',scale=Decimal(1000000))

if __name__=='__main__':unittest.main()
