from decimal import Decimal
from unittest import TestCase,main
from hrm_finance_qa.post_hrm_formulas import financial_calculation_plan
from hrm_finance_qa.post_hrm_contract import compute_named
from test_post_hrm_contract import fact
class CalculationContractTests(TestCase):

 def test_current_ratio_uses_two_explicit_same_year_operands(self):
  question = "Hệ số thanh toán hiện hành cuối năm 2034 bằng tài sản ngắn hạn chia nợ ngắn hạn là bao nhiêu?"
  plan = financial_calculation_plan(question)[0]
  self.assertEqual(plan.operation, "DIVIDE")
  self.assertEqual([(s.id, s.year, s.period_role) for s in plan.operand_slots],
                   [("current_assets", "2034", "closing"),
                    ("current_liabilities", "2034", "closing")])
  self.assertTrue(financial_calculation_plan(
      "Hệ số thanh toán hiện hành bằng tài sản ngắn hạn chia nợ ngắn hạn là bao nhiêu?")[0].abstention_reason)
 def test_eps_lookup_and_calculation_differ(self):
  lookup=financial_calculation_plan('EPS năm 2032 là bao nhiêu')[0];calc=financial_calculation_plan('Tính EPS năm 2032')[0]
  self.assertEqual(lookup.operation,'SELECT');self.assertEqual(calc.operation,'DIVIDE');self.assertEqual(calc.operand_slots[1].metric,'Số cổ phiếu phổ thông bình quân gia quyền')
 def test_eps_positive_and_wrong_denominator(self):
  a=fact('earnings_attributable_to_common',120000);b=fact('weighted_average_common_shares',100,kind='quantity',unit='shares',currency='')
  result=compute_named('eps',{'earnings_attributable_to_common':a,'weighted_average_common_shares':b});self.assertEqual(result.decimal_value,1200);self.assertEqual(result.output.unit.symbol,'VND/share')
 def test_pe_positive_and_source_date(self):
  a=fact('price_per_share',24000,unit='VND/share',period='2032-12-31');b=fact('earnings_per_share',1200,unit='VND/share',period='2032')
  result=compute_named('pe',{'price_per_share':a,'eps':b},price_date='2032-12-31');self.assertEqual(result.decimal_value,20)
  with self.assertRaises(ValueError):compute_named('pe',{'price_per_share':a,'eps':b},price_date='2032-12-30')
 def test_pe_missing_date_and_margin_missing_metric(self):
  self.assertTrue(financial_calculation_plan('Tính P/E năm 2032')[0].abstention_reason)
  self.assertTrue(financial_calculation_plan('Tính biên lợi nhuận năm 2032')[0].abstention_reason)
 def test_margin_recomputed(self):
  result=compute_named('margin',{'profit':fact('profit_after_tax',25),'revenue':fact('net_revenue',100)});self.assertEqual(result.decimal_value,25)
if __name__=='__main__':main()
