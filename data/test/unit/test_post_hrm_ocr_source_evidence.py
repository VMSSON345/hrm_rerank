"""OCR recovery needs a physical row, governing header, and unambiguous units."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from types import SimpleNamespace

from hrm_finance_qa.document_sources import OCRConfig, build_chunks
from hrm_finance_qa.post_hrm_source import (SourceSnapshot, build_source_facts,
    _narrative_endpoint_witness)
from hrm_finance_qa.post_hrm_engine import ContractDocumentEngine
from hrm_finance_qa.program_data import merge_candidates


def facts_for(text):
    with TemporaryDirectory() as directory:
        source = Path(directory) / "source.md"
        source.write_text(text, encoding="utf8")
        snapshot = SourceSnapshot([source])
        candidates = merge_candidates(
            "", build_chunks([source], ocr_config=OCRConfig(enabled=False)),
            max_candidates=128)
        return build_source_facts(candidates, snapshot)[0]


class OCRSourceEvidenceTests(TestCase):
    def test_same_metric_year_value_narrative_proves_closing(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.md"
            source.write_text("""## Page 1
Chỉ tiêu | Năm 2034
--- | ---
Tổng tài sản | Năm 2034: 1.028,43
## Page 2
Tổng tài sản của công ty cuối năm 2034 đạt 1.028,43 tỷ đồng.
""", encoding="utf8")
            snapshot = SourceSnapshot([source])
            candidate = SimpleNamespace(source_name=source.name, line_number=4)
            witness = _narrative_endpoint_witness(
                snapshot, candidate, "total_assets", "1.028,43", "2034")
            self.assertEqual(witness["line_number"], 6)
            self.assertEqual(witness["basis"], "equivalent_narrative_endpoint")
            self.assertIsNone(_narrative_endpoint_witness(
                snapshot, candidate, "total_assets", "1.028,43", "2033"))
            self.assertIsNone(_narrative_endpoint_witness(
                snapshot, candidate, "current_assets", "1.028,43", "2034"))

    def test_no_fact_pool_exposes_typed_fact_gate(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.md"
            source.write_text("""## Page 1
Đơn vị: đồng
Chi tiêu | Năm 2034
--- | ---
Tổng tài sản | 100.000.000
""", encoding="utf8")
            bundle = SimpleNamespace(
                model=SimpleNamespace(config=SimpleNamespace(max_candidates=64)),
                rank_raw=lambda question, candidates: {
                    "pointers": [[0.0] * len(candidates)] * 4,
                    "select": [0.0] * len(candidates),
                    "pairs": {}, "segments": 0})
            answer = ContractDocumentEngine(bundle, numeric_program_mode="baseline").answer(
                "Lợi nhuận sau thuế năm 2034 là bao nhiêu?", source)
            self.assertEqual(answer["status"], "ABSTAIN")
            self.assertNotIn("fact_pool", answer.get("program_trace") or {})
            self.assertEqual(answer["program_gate_trace"]["stop_reason"],
                             "typed_fact_missing")
            self.assertEqual(answer["program_gate_trace"]["planner_slot_count"], 1)

    def test_ocr_label_and_local_unit_are_bound_to_table(self):
        facts = facts_for("""## Page 1
Tình hình tài chính
Đon v tinh: Đồng
Chi tiêu | Năm 2033 | Năm 2034
--- | --- | ---
Tồng tài sn | Năm 2033: 100.000.000 | Năm 2034: 110.000.000
A. N phåi trå | Năm 2033: 40.000.000 | Năm 2034: 41.000.000
I. N ngån han | Năm 2033: 30.000.000 | Năm 2034: 31.000.000
""")
        chosen = {(fact.metric, fact.period): fact for fact in facts
                  if fact.period == "2034"}
        self.assertIn(("total_assets", "2034"), chosen)
        self.assertIn(("total_liabilities", "2034"), chosen)
        self.assertEqual(chosen[("total_liabilities", "2034")].unit, "VND")
        self.assertTrue(any("Đon v tinh" in w["text"]
                            for w in chosen[("total_liabilities", "2034")].unit_evidence))
        self.assertIn(("current_liabilities", "2034"), chosen)
        self.assertNotEqual(chosen[("total_liabilities", "2034")].metric,
                            chosen[("current_liabilities", "2034")].metric)


    def test_accounting_code_rejects_long_term_asset_as_current(self):
        facts = facts_for("""## Page 1
BẢNG CÂN ĐỐI KẾ TOÁN tại ngày 31/12/2034
Đơn vị: Đồng
Mã số | TÀI SẢN | Số cuối năm | Số đầu năm
--- | --- | --- | ---
Mã số: 100 | A. TÀI SẢN NGẢN HẠN | Số cuối năm: 418.000.000 | Số đầu năm: 400.000.000
Mã số: 200 | B. TÀI SẢN DÀI HẠN | Số cuối năm: 160.000.000 | Số đầu năm: 150.000.000
""")
        self.assertTrue(any(f.metric == "literal:tai san ngan han"
                            and f.raw_value == "418.000.000" for f in facts))
        self.assertFalse(any(f.metric == "literal:tai san ngan han"
                             and f.raw_value == "160.000.000" for f in facts))

    def test_unique_accounting_code_can_repair_undated_ocr_label_only(self):
        facts = facts_for("""## Page 1
NGUỒN VỐN
 | | NGUỒN VỐN | Mã số | | Số cuối kỳ | Số đầu kỳ
--- | --- | --- | --- | --- | --- | ---
NGUỒN VỐN | | C. NQ PHÅI TRÅ | 300 | | 41.000.000 | 40.000.000
NGUỒN VỐN | | I. Nơ ngn hąn | 310 | | 31.000.000 | 30.000.000
""")
        current = [f for f in facts if f.value == 41000000]
        short = [f for f in facts if f.value == 31000000]
        self.assertEqual(len(current), 1)
        self.assertEqual(len(short), 1)
        self.assertEqual(current[0].metric, "total_liabilities")
        self.assertEqual(short[0].metric, "current_liabilities")
        self.assertFalse(current[0].period or current[0].unit_resolved)
        self.assertFalse(short[0].period or short[0].unit_resolved)

    def test_conflicting_or_multiple_codes_do_not_repair_ocr_label(self):
        facts = facts_for("""## Page 1
NGUỒN VỐN
 | | NGUỒN VỐN | Mã số | | Số cuối kỳ | Số đầu kỳ
--- | --- | --- | --- | --- | --- | ---
NGUỒN VỐN | | C. NQ PHÅI TRÅ | 310 | | 41.000.000 | 40.000.000
NGUỒN VỐN | | C. NQ PHÅI TRÅ | 300 310 | | 42.000.000 | 39.000.000
""")
        self.assertFalse(any(f.metric == "total_liabilities"
                             and f.value in {41000000, 42000000}
                             for f in facts))
    def test_unit_from_other_table_is_not_inherited(self):
        facts = facts_for("""## Page 1
Đon v tinh: Đồng
Chi tiêu | Năm 2034
--- | ---
Nợ phải trả | 100.000.000
Bảng khác
Chi tiêu | Năm 2034
--- | ---
Nợ phải trả | 200.000.000
""")
        second = [fact for fact in facts if fact.raw_value == "200.000.000"]
        self.assertTrue(second)
        self.assertTrue(all(not fact.unit_resolved for fact in second))

    def test_continuation_uses_witnessed_year_then_stops_at_new_title(self):
        facts = facts_for("""## Page 1
BẢNG CÂN ĐỐI KẾ TOÁN Tại ngày 31/12/2034
Đơn vị: VND
TÀI SẢN | Mã số | Thuyết minh | Số cuối năm | Số đầu năm
--- | --- | --- | --- | ---
A. Tài sản ngắn hạn | 100 | | 60.000.000 | 50.000.000
## Page 2
TỔNG CỘNG TÀI SẢN | 270 | | 100.000.000 | 90.000.000
BẢNG CÂN ĐỐI KẾ TOÁN Tại ngày 31/12/2032
NGUỒN VỐN | Mã số | Thuyết minh | Số cuối năm | Số đầu năm
--- | --- | --- | --- | ---
C. Nợ phải trả | 300 | | 30.000.000 | 20.000.000
""")
        assets = [f for f in facts if f.metric == "total_assets"
                  and f.raw_value == "100.000.000"]
        self.assertEqual(len(assets), 1)
        self.assertEqual(assets[0].period, "2034")
        self.assertEqual(assets[0].period_role, "closing")
        self.assertFalse(any(f.metric == "total_liabilities" and f.period == "2034"
                             for f in facts))

    def test_two_by_two_glued_ratio_keeps_metric_and_year(self):
        facts = facts_for("""## Page 1
Các chỉ tiêu tài chính chủ yếu
Chi tiêu | Năm 2034 | Năm 2033 | Ghi chú
--- | --- | --- | ---
Chi tiu v kh năng thanh toán: - H s thanh toán ngn hn - H s thanh toán nhanh | Năm 2034: 3,51 3,52 | Năm 2033: 3,81 3,82 |
""")
        recovered = [(f.metric, f.period, str(f.value)) for f in facts
                     if f.metric in {"current_ratio", "quick_ratio"}]
        self.assertCountEqual(recovered, [
            ("current_ratio", "2034", "3.51"),
            ("quick_ratio", "2034", "3.52"),
            ("current_ratio", "2033", "3.81"),
            ("quick_ratio", "2033", "3.82"),
        ])

    def test_glued_ratio_rejects_extra_unmapped_number(self):
        facts = facts_for("""## Page 1
Chi tiêu | Năm 2034 | Năm 2033
--- | --- | ---
Chi tiu v kh năng thanh toán: - H s thanh toán ngn hn - H s thanh toán nhanh | Năm 2034: 3,51 3,52 9,00 | Năm 2033: 3,81 3,82
""")
        self.assertFalse(any(f.metric in {"current_ratio", "quick_ratio"}
                             for f in facts))
