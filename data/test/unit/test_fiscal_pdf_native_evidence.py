"""Independent witnesses for FY labels and ambiguous OCR units."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase

from hrm_finance_qa.fiscal_period_evidence import (
    fiscal_binding, fiscal_question_surface)
from hrm_finance_qa.native_pdf_unit import native_column_unit
from hrm_finance_qa.post_hrm_source import SourceSnapshot
from hrm_finance_qa.question_planner import (
    compile_question_plan, resolve_public_question)


def word(text, x1, y1, x2, y2):
    return {"text": text, "bbox": (x1, y1, x2, y2)}


class NativeColumnUnitTests(TestCase):
    def snapshot(self, words):
        return SimpleNamespace(
            pdf_sources={"source.md": (Path("witness.pdf"), "a" * 64)},
            _pdf_word_cache={("source.md", 1): tuple(words)})

    def test_year_unit_and_value_share_one_pdf_column(self):
        words = [word("2035", 100, 20, 130, 30),
                 word("Tỷ", 100, 35, 116, 45),
                 word("đồng", 118, 35, 146, 45),
                 word("1.028,43", 100, 100, 146, 112)]
        witness = native_column_unit(
            self.snapshot(words), "source.md", 1, "2035T dng", "1.028,43")
        self.assertEqual(witness["unit"], "tỷ đồng")
        self.assertEqual(witness["raw_value"], "1.028,43")
        self.assertIsNone(native_column_unit(
            self.snapshot(words[:-1] + [word("1.028,43", 400, 100, 446, 112)]),
            "source.md", 1, "2035T dng", "1.028,43"))

    def test_duplicate_value_or_conflicting_units_stays_unresolved(self):
        words = [word("2035", 100, 20, 130, 30),
                 word("Tỷ", 100, 35, 116, 45),
                 word("đồng", 118, 35, 146, 45),
                 word("1.028,43", 100, 100, 146, 112)]
        self.assertIsNone(native_column_unit(
            self.snapshot(words + [word("1.028,43", 300, 100, 346, 112)]),
            "source.md", 1, "2035T đồng", "1.028,43"))
        conflict = words[:3] + [
            word("2035", 100, 52, 130, 62),
            word("Triệu", 100, 67, 118, 77),
            word("đồng", 120, 67, 146, 77),
            words[-1]]
        self.assertIsNone(native_column_unit(
            self.snapshot(conflict), "source.md", 1,
            "2035T đồng", "1.028,43"))


class FiscalPeriodEvidenceTests(TestCase):
    def test_planner_keeps_fy_label_separate_from_calendar_date(self):
        question = ("Tổng tài sản hợp nhất tại ngày 31/03/2035, "
                    "cuối năm tài chính 2034 là bao nhiêu?")
        plan = compile_question_plan(resolve_public_question(question))
        self.assertEqual(plan.requested_years, ("2034",))
        self.assertEqual(plan.requested_accounting_scope, "consolidated")
        self.assertEqual([(slot.metric, slot.year, slot.period_role)
                          for slot in plan.operand_slots],
                         [("tong tai san", "2034", "closing")])
        self.assertIsNone(fiscal_question_surface(
            "Tổng tài sản tại ngày 31/03/2035 là bao nhiêu?"))

    def test_source_binding_requires_both_title_and_local_statement(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "report.md"
            source.write_text("""## Page 1
Báo cáo thường niên năm tài chính 2034
## Page 2
BẢNG CÂN ĐỐI KẾ TOÁN HỢP NHẤT Tại ngày 31/03/2035
Chỉ tiêu | Mã số | Số cuối năm
--- | --- | ---
Tổng tài sản | 270 | 1.000.000
""", encoding="utf8")
            snapshot = SourceSnapshot([source])
            cell = SimpleNamespace(table_semantics={"accounting_scope": ""},
                                   column="31/03/2035")
            candidate = SimpleNamespace(source_name=source.name, line_number=7)
            binding = fiscal_binding(snapshot, candidate, cell, "total_assets",
                                     "2035", "2035-03-31", "actual")
            self.assertEqual(binding["fiscal_year"], "2034")
            self.assertEqual(binding["scope"], "consolidated")
            self.assertEqual(len(binding["evidence"]), 2)
            source.write_text(source.read_text(encoding="utf8").replace(
                "năm tài chính 2034", "năm tài chính 2032"), encoding="utf8")
            changed = SourceSnapshot([source])
            self.assertIsNone(fiscal_binding(
                changed, candidate, cell, "total_assets",
                "2035", "2035-03-31", "actual"))

    def test_continuation_conflict_retains_both_printed_years(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "report.md"
            source.write_text("""## Page 1
Bảng cân đối kế toán tại ngày 31/12/2034
## Page 2
Bảng cân đối kế toán (tiếp theo) tại ngày 31/12/2032
""", encoding="utf8")
            conflicts = SourceSnapshot([source]).period_conflicts[source.name]
            self.assertEqual(len(conflicts), 1)
            self.assertEqual(conflicts[0]["previous"]["year"], "2034")
            self.assertEqual(conflicts[0]["continuation"]["year"], "2032")
