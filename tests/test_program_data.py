from __future__ import annotations

import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hrm_finance_qa.program_data import (
    NumericCandidate,
    ReasoningProgram,
    _candidate_quota_indices,
    _table_subtable_provenance,
    build_program_dataset,
    extract_numeric_candidates,
    induce_program_example,
    merge_candidates,
    execute_program,
    execute_program_dag,
    materialize_program_dag,
    program_row_rejection_reason,
    _synthetic_question,
)
from hrm_finance_qa.ontology import normalize_text


class MultiStepProgramDAGTests(unittest.TestCase):
    @staticmethod
    def _candidate(
        index: int,
        value: str,
        row: str,
        *,
        derived_operation: str = "",
        derived_operands: tuple[int, ...] = (),
        derivation_id: str = "",
    ) -> NumericCandidate:
        return NumericCandidate(
            index=index,
            value=value,
            raw=value,
            context=f"ROW: {row}\nPERIOD: 2025\nCOLUMN: 2025",
            line_number=index + 1,
            char_start=0,
            char_end=len(value),
            unit="VND",
            is_percent=False,
            is_year_like=False,
            relevance=1.0,
            source_name="report.pdf",
            extraction_method="native" if not derived_operation else "symbolic-derived",
            derived_operation=derived_operation,
            derived_operands=derived_operands,
            derivation_id=derivation_id,
        )

    def test_materializes_and_executes_share_of_total_dag(self) -> None:
        candidates = (
            self._candidate(0, "80", "Vay ngan han"),
            self._candidate(1, "20", "Vay dai han"),
            self._candidate(2, "1000", "Tong nguon von"),
            self._candidate(
                3,
                "100",
                "Tong no vay chiu lai",
                derived_operation="ADD",
                derived_operands=(0, 1),
                derivation_id="interest_bearing_debt",
            ),
        )
        flat = ReasoningProgram("RATIO_PERCENT", (3, 2))

        dag = materialize_program_dag(flat, candidates)

        self.assertIsNotNone(dag)
        assert dag is not None
        self.assertEqual(
            tuple(node.operation for node in dag.nodes),
            ("ADD", "SHARE_OF_TOTAL"),
        )
        self.assertEqual(dag.leaf_operands, (0, 1, 2))
        self.assertEqual(execute_program(flat, candidates), Decimal("10"))
        self.assertEqual(execute_program_dag(dag, candidates), Decimal("10"))
        self.assertEqual(dag.to_dict()["output_node"], "answer")

    def test_dag_execution_detects_tampered_materialized_value(self) -> None:
        candidates = (
            self._candidate(0, "80", "Vay ngan han"),
            self._candidate(1, "20", "Vay dai han"),
            self._candidate(2, "1000", "Tong nguon von"),
            self._candidate(
                3,
                "999",
                "Tong no vay chiu lai",
                derived_operation="ADD",
                derived_operands=(0, 1),
                derivation_id="interest_bearing_debt",
            ),
        )
        flat = ReasoningProgram("RATIO_PERCENT", (3, 2))
        dag = materialize_program_dag(flat, candidates)

        assert dag is not None
        self.assertNotEqual(
            execute_program(flat, candidates), execute_program_dag(dag, candidates)
        )


class SubtableProvenanceTests(unittest.TestCase):
    def test_distinguishes_side_by_side_entities(self) -> None:
        lines = [
            "CÔNG TY TNHH CHĂN NUÔI BẢO NGỌC Đơn vị: Đồng "
            "CÔNG TY TNHH BẮC AN KHÁNH Đơn vị: Đồng",
            "| STT | Chỉ tiêu | TH 2024 | TH 2025 | STT | Chỉ tiêu | TH 2024 | TH 2025 |",
            "| 1 | Tổng tài sản | 70 | 68 | 1 | Tổng tài sản | 509 | 418 |",
        ]

        left = _table_subtable_provenance(lines, 2, 2)
        right = _table_subtable_provenance(lines, 2, 6)

        self.assertEqual(left, ("table-2-3/subtable-1", "entity:cong ty tnhh chan nuoi bao ngoc"))
        self.assertEqual(right, ("table-2-3/subtable-2", "entity:cong ty tnhh bac an khanh"))

    def test_marks_single_logical_table_as_whole_table(self) -> None:
        lines = [
            "BÁO CÁO TÌNH HÌNH TÀI CHÍNH",
            "| Chỉ tiêu | 2024 | 2025 |",
            "| Tổng tài sản | 100 | 120 |",
        ]

        self.assertEqual(
            _table_subtable_provenance(lines, 2, 2),
            ("table-2-3/whole-table", ""),
        )


class SyntheticQuestionTests(unittest.TestCase):
    LABELS = ("Doanh thu", "Tổng tài sản", "Nợ phải trả", "Lợi nhuận")

    def test_percent_change_includes_external_style_paraphrase(self) -> None:
        questions = {
            _synthetic_question(
                "PERCENT_CHANGE", self.LABELS, 2025, variant=variant
            )[0]
            for variant in range(4)
        }

        self.assertEqual(len(questions), 4)
        self.assertTrue(any("thay đổi bao nhiêu phần trăm" in item for item in questions))

    def test_ratio_includes_chiem_bao_nhieu_paraphrase(self) -> None:
        questions = {
            _synthetic_question(
                "RATIO_PERCENT", self.LABELS, 2025, variant=variant
            )[0]
            for variant in range(4)
        }

        self.assertTrue(any("chiếm bao nhiêu phần trăm" in item for item in questions))

    def test_subtract_includes_period_comparison_paraphrases(self) -> None:
        questions = {
            _synthetic_question(
                "SUBTRACT", self.LABELS, 2025, variant=variant
            )[0]
            for variant in range(6)
        }

        self.assertTrue(any("cuối kỳ" in item and "đầu kỳ" in item for item in questions))
        self.assertTrue(any("năm nay" in item and "năm trước" in item for item in questions))


class ProgramScreeningTests(unittest.TestCase):
    def _row(self, **overrides: object) -> dict[str, object]:
        row: dict[str, object] = {
            "id": "valid-1",
            "task": "QA",
            "difficulty": "Medium_QA",
            "question": "Doanh thu thuần là bao nhiêu?",
            "context": (
                "| Chỉ tiêu | Giá trị |\n"
                "|---|---:|\n"
                "| Doanh thu thuần | 100 VND |"
            ),
            "answer": "Doanh thu thuần là 100 VND.",
        }
        row.update(overrides)
        return row

    def test_rejects_non_qa_tasks(self) -> None:
        row = self._row(task="SA")
        self.assertEqual(program_row_rejection_reason(row), "unsupported_task")
        self.assertIsNone(induce_program_example(row))

    def test_rejects_unanswerable_qa(self) -> None:
        row = self._row(answer="Không đủ dữ liệu để tính tỷ lệ này.")
        self.assertEqual(program_row_rejection_reason(row), "unanswerable")

    def test_rejects_qualitative_qa(self) -> None:
        row = self._row(question="Giải thích nguồn gốc của thu nhập khác.")
        self.assertEqual(
            program_row_rejection_reason(row), "qualitative_question"
        )

    def test_rejects_non_program_temporal_or_why_question(self) -> None:
        temporal = self._row(question="Khi nào Công ty nhận giấy chứng nhận?")
        why = self._row(question="Tại sao Công ty không tính chỉ tiêu này?")
        self.assertEqual(
            program_row_rejection_reason(temporal), "non_program_question"
        )
        self.assertEqual(program_row_rejection_reason(why), "non_program_question")

    def test_accepts_single_value_qa(self) -> None:
        example = induce_program_example(self._row())
        self.assertIsNotNone(example)
        assert example is not None
        self.assertEqual(example.program.operation, "SELECT")
        self.assertEqual(example.answer_value, "100")

    def test_accepts_direct_disclosed_percentage(self) -> None:
        row = self._row(
            question="Thuế suất thuế thu nhập doanh nghiệp là bao nhiêu phần trăm?",
            context=(
                "| Chỉ tiêu | Giá trị |\n"
                "|---|---:|\n"
                "| Thuế suất thuế thu nhập doanh nghiệp | 20% |"
            ),
            answer="Thuế suất thuế thu nhập doanh nghiệp là 20%.",
        )
        example = induce_program_example(row)
        self.assertIsNotNone(example)
        assert example is not None
        self.assertEqual(example.program.operation, "SELECT")

    def test_company_name_does_not_imply_sum(self) -> None:
        row = self._row(
            question="Khoản chi trả của Tổng Công ty A là bao nhiêu?",
            context=(
                "| Chỉ tiêu | Giá trị |\n"
                "|---|---:|\n"
                "| Khoản chi trả của Tổng Công ty A | 100 VND |"
            ),
            answer="Khoản chi trả của Tổng Công ty A là 100 VND.",
        )
        example = induce_program_example(row)
        self.assertIsNotNone(example)
        assert example is not None
        self.assertEqual(example.program.operation, "SELECT")

    def test_zero_balance_delta_is_induced_as_subtract(self) -> None:
        row = self._row(
            question="Tính tổng mức giảm của khoản phải thu.",
            context=(
                "| Chỉ tiêu | Đầu năm | Cuối năm |\n"
                "|---|---:|---:|\n"
                "| Khoản phải thu | 100 VND | 0 VND |"
            ),
            answer="Tổng mức giảm là 100 VND, từ 100 VND xuống 0 VND.",
        )
        example = induce_program_example(row)
        self.assertIsNotNone(example)
        assert example is not None
        self.assertEqual(example.program.operation, "SUBTRACT")

    def test_build_report_tracks_rejections(self) -> None:
        rows = [
            self._row(),
            self._row(id="sa-1", task="SA"),
            self._row(
                id="unanswerable-1",
                answer="Không có đủ thông tin để tính kết quả.",
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            output = Path(directory) / "programs.jsonl"
            source.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n" for row in rows
                ),
                encoding="utf-8",
            )
            report = build_program_dataset(source, output)

        self.assertEqual(report.source_rows, 3)
        self.assertEqual(report.eligible_rows, 1)
        self.assertEqual(report.induced_examples, 1)
        self.assertEqual(report.source_task_counts, {"QA": 2, "SA": 1})
        self.assertEqual(report.rejection_counts["unsupported_task"], 1)
        self.assertEqual(report.rejection_counts["unanswerable"], 1)


class CandidateTableMetadataTests(unittest.TestCase):
    def test_extracts_table_section_and_statement_scope(self) -> None:
        context = """BÁO CÁO HỢP NHẤT
Kết quả kinh doanh
| Chỉ tiêu | Năm 2024 |
|---|---:|
| Doanh thu thuần | 100 VND |

BÁO CÁO TÀI CHÍNH RIÊNG CÔNG TY MẸ
Kết quả kinh doanh
| Chỉ tiêu | Năm 2024 |
|---|---:|
| Doanh thu thuần | 80 VND |
"""

        candidates = extract_numeric_candidates(
            "Doanh thu thuần năm 2024 là bao nhiêu?", context
        )
        current = [
            candidate
            for candidate in candidates
            if candidate.row_label == "Doanh thu thuần"
        ]

        self.assertEqual(len(current), 2)
        self.assertNotEqual(current[0].table_id, current[1].table_id)
        self.assertEqual(current[0].section, "Kết quả kinh doanh")
        self.assertEqual(current[0].scope, "consolidated")
        self.assertEqual(current[1].scope, "parent_company")

    def test_normalize_text_decodes_pdf_unicode_glyph_names(self) -> None:
        self.assertEqual(
            normalize_text("T/uni1ED4NG TÀI S/uni1EA2N"),
            "tong tai san",
        )

    def test_splits_native_note_code_and_adjacent_financial_columns(self) -> None:
        context = (
            "321 9. Dự phòng phải trả ngắn hạn "
            "26 11.678.880.397 23.109.698.971"
        )

        candidates = extract_numeric_candidates("Dự phòng phải trả ngắn hạn", context)

        self.assertEqual(
            [item.decimal for item in candidates],
            [Decimal(11678880397), Decimal(23109698971)],
        )

    def test_splits_financial_value_with_space_before_group_separator(self) -> None:
        context = (
            "210 I. Các khoản phải thu dài hạn "
            "107 .672.573.473 36.440.426.781"
        )

        candidates = extract_numeric_candidates("Các khoản phải thu dài hạn", context)

        self.assertEqual(
            [item.decimal for item in candidates],
            [Decimal(107672573473), Decimal(36440426781)],
        )

    def test_excludes_plaintext_row_and_note_codes_before_metric_label(self) -> None:
        context = (
            "60 18. Lợi nhuận sau thuế TNDN 3.150.404.939.011 2.396.000.255.249\n"
            "61 19 . Lợi nhuận sau thuế của cổ đông công ty mẹ "
            "2.900.000.000.000 2.100.000.000.000"
        )

        candidates = extract_numeric_candidates("Lợi nhuận sau thuế của công ty mẹ", context)
        values = [item.decimal for item in candidates]

        self.assertNotIn(Decimal(60), values)
        self.assertNotIn(Decimal(18), values)
        self.assertNotIn(Decimal(61), values)
        self.assertNotIn(Decimal(19), values)
        self.assertIn(Decimal(2900000000000), values)

    def test_recovers_native_pdf_row_and_period_columns_without_pipes(self) -> None:
        context = (
            "CHỈ TIÊU 31/12/2025 01/01/2025\n"
            "270 TỔNG CỘNG TÀI SẢN 35.486.201.000.000 32.100.500.000.000"
        )

        candidates = extract_numeric_candidates(
            "Tổng cộng tài sản cuối năm 2025", context
        )
        current = next(item for item in candidates if item.decimal == 35486201000000)
        prior = next(item for item in candidates if item.decimal == 32100500000000)

        self.assertIn("ROW: TỔNG CỘNG TÀI SẢN", current.context)
        self.assertIn("PERIOD: 2025", current.context)
        self.assertIn("COLUMN: 31/12/2025", current.context)
        self.assertIn("COLUMN: 01/01/2025", prior.context)
        self.assertEqual(current.row_label, "TỔNG CỘNG TÀI SẢN")
        self.assertEqual(current.column_label, "31/12/2025")
        self.assertEqual(current.period, "2025")
        self.assertEqual(current.source_span, (current.char_start, current.char_end))

    def test_candidate_structured_fields_load_from_legacy_context(self) -> None:
        candidate = NumericCandidate(
            index=0,
            value="100",
            raw="100",
            context=(
                "ROW: Doanh thu thuần\n"
                "ROW_TYPE: detail\n"
                "COLUMN: Năm 2024\n"
                "PERIOD: 2024\n"
                "CELL: Doanh thu thuần | 100"
            ),
            line_number=10,
            char_start=20,
            char_end=23,
            unit="VND",
            is_percent=False,
            is_year_like=False,
            relevance=1.0,
            source_span=[20, 23],
        )

        self.assertEqual(candidate.row_label, "Doanh thu thuần")
        self.assertEqual(candidate.column_label, "Năm 2024")
        self.assertEqual(candidate.period, "2024")
        self.assertEqual(candidate.row_type, "detail")
        self.assertEqual(candidate.source_span, (20, 23))

    def test_legacy_merge_applies_metric_aware_global_quota(self) -> None:
        candidates = tuple(
            NumericCandidate(
                index=index,
                value=str(index + 1),
                raw=str(index + 1),
                context=f"ROW: metric {index}\nCELL: {index + 1}",
                line_number=index + 1,
                char_start=0,
                char_end=1,
                unit="",
                is_percent=False,
                is_year_like=False,
                relevance=float(index),
                extraction_method="ocr",
            )
            for index in range(10)
        )
        chunks = [
            SimpleNamespace(text="first", source_name="report.pdf"),
            SimpleNamespace(text="second", source_name="report.pdf"),
        ]
        with (
            patch(
                "hrm_finance_qa.program_data.extract_numeric_candidates",
                side_effect=(candidates[:5], candidates[5:]),
            ),
            patch(
                "hrm_finance_qa.program_data._candidate_quota_indices",
                return_value=[1, 3, 7, 9],
            ) as quota,
        ):
            merged = merge_candidates(
                "metric requested", chunks, max_candidates=4, legacy_mode=True
            )

        quota.assert_called_once()
        self.assertEqual({item.value for item in merged}, {"2", "4", "8", "10"})

    def test_legacy_merge_deduplicates_identical_overlapping_cells(self) -> None:
        candidate = NumericCandidate(
            index=0,
            value="284331609643",
            raw="284.331.609.643",
            context="ROW: TỔNG CỘNG TÀI SẢN\nPERIOD: cuoi nam",
            line_number=10,
            char_start=0,
            char_end=15,
            unit="VND",
            is_percent=False,
            is_year_like=False,
            relevance=1.0,
            source_name="ABR.pdf",
            page=33,
            extraction_method="ocr",
        )
        chunks = [
            SimpleNamespace(text="first", source_name="ABR.pdf"),
            SimpleNamespace(text="overlap", source_name="ABR.pdf"),
        ]
        with patch(
            "hrm_finance_qa.program_data.extract_numeric_candidates",
            side_effect=((candidate,), (candidate,)),
        ):
            merged = merge_candidates(
                "Tổng cộng tài sản", chunks, max_candidates=64, legacy_mode=True
            )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].value, "284331609643")

    def test_ocr_metric_row_survives_full_document_candidate_quota(self) -> None:
        candidates = [
            NumericCandidate(
                index=index,
                value=str(1000 + index),
                raw=str(1000 + index),
                context=f"ROW: Chi phí quản lý {index}\nCELL: {1000 + index}",
                line_number=index + 1,
                char_start=0,
                char_end=4,
                unit="VND",
                is_percent=False,
                is_year_like=False,
                relevance=10.0 - index / 100,
                extraction_method="ocr",
            )
            for index in range(10)
        ]
        candidates.append(
            NumericCandidate(
                index=10,
                value="284331609643",
                raw="284.331.609.643",
                context="ROW: TỔNG CỘNG TÀI SẢN\nPERIOD: 2025\nCELL: 284.331.609.643",
                line_number=100,
                char_start=0,
                char_end=15,
                unit="VND",
                is_percent=False,
                is_year_like=False,
                relevance=0.1,
                extraction_method="ocr",
            )
        )

        selected = _candidate_quota_indices(
            "Theo bảng cân đối kế toán, tổng cộng tài sản cuối năm 2025 là bao nhiêu?",
            candidates,
            4,
        )

        self.assertIn(10, selected)

    def test_ratio_quota_reserves_both_named_metric_roles(self) -> None:
        candidates = [
            NumericCandidate(
                index=index,
                value=str(1000 + index),
                raw=str(1000 + index),
                context=f"ROW: Chi phí quản lý {index}\nCELL: {1000 + index}",
                line_number=index + 1,
                char_start=0,
                char_end=4,
                unit="VND",
                is_percent=False,
                is_year_like=False,
                relevance=10.0 - index / 100,
                extraction_method="mineru",
            )
            for index in range(10)
        ]
        candidates.extend(
            (
                NumericCandidate(
                    index=10,
                    value="44393950887086",
                    raw="44.393.950.887.086",
                    context="ROW: NỢ PHẢI TRẢ\nPERIOD: 2025",
                    line_number=100,
                    char_start=0,
                    char_end=18,
                    unit="VND",
                    is_percent=False,
                    is_year_like=False,
                    relevance=0.1,
                    extraction_method="mineru",
                ),
                NumericCandidate(
                    index=11,
                    value="88141991634625",
                    raw="88.141.991.634.625",
                    context="ROW: TỔNG CỘNG TÀI SẢN\nPERIOD: 2025",
                    line_number=101,
                    char_start=0,
                    char_end=18,
                    unit="VND",
                    is_percent=False,
                    is_year_like=False,
                    relevance=0.2,
                    extraction_method="mineru",
                ),
            )
        )

        selected = _candidate_quota_indices(
            "Tính tỷ lệ nợ phải trả trên tổng cộng tài sản cuối năm 2025.",
            candidates,
            4,
        )

        self.assertIn(10, selected)
        self.assertIn(11, selected)

    def test_add_quota_reserves_roles_across_conjunction_hypotheses(self) -> None:
        candidates = [
            NumericCandidate(
                index=index,
                value=str(1000 + index),
                raw=str(1000 + index),
                context=f"ROW: No phai tra {index}\nCELL: {1000 + index}",
                line_number=index + 1,
                char_start=0,
                char_end=4,
                unit="VND",
                is_percent=False,
                is_year_like=False,
                relevance=10.0 - index / 100,
                extraction_method="native",
            )
            for index in range(10)
        ]
        candidates.extend(
            (
                NumericCandidate(
                    index=10,
                    value="80",
                    raw="80",
                    context="ROW: I. Tai san ngan han\nPERIOD: dau nam",
                    line_number=100,
                    char_start=0,
                    char_end=2,
                    unit="VND",
                    is_percent=False,
                    is_year_like=False,
                    relevance=0.1,
                    extraction_method="native",
                ),
                NumericCandidate(
                    index=11,
                    value="20",
                    raw="20",
                    context=(
                        "ROW: Tien va cac khoan tuong duong tien\n"
                        "PERIOD: dau nam"
                    ),
                    line_number=101,
                    char_start=0,
                    char_end=2,
                    unit="VND",
                    is_percent=False,
                    is_year_like=False,
                    relevance=0.1,
                    extraction_method="native",
                ),
            )
        )

        selected = _candidate_quota_indices(
            "Tong I. Tai san ngan han va 1. Tien va cac khoan tuong duong "
            "tien tai dau nam la bao nhieu?",
            candidates,
            4,
        )

        self.assertTrue({10, 11}.issubset(selected))

    def test_quota_preserves_disclosed_coefficient_despite_small_value(self) -> None:
        candidates = [
            NumericCandidate(
                index=index,
                value=str(10_000_000_000 + index),
                raw=str(10_000_000_000 + index),
                context=f"ROW: Chi phi quan ly {index}\nCELL: {index}",
                line_number=index + 1,
                char_start=0,
                char_end=4,
                unit="VND",
                is_percent=False,
                is_year_like=False,
                relevance=10.0 - index / 100,
                extraction_method="native",
            )
            for index in range(10)
        ]
        candidates.append(
            NumericCandidate(
                index=10,
                value="0.03",
                raw="0,03",
                context="ROW: He so no tren tong tai san\nPERIOD: 2024",
                line_number=100,
                char_start=0,
                char_end=4,
                unit="x",
                is_percent=False,
                is_year_like=False,
                relevance=0.01,
                extraction_method="native",
            )
        )

        selected = _candidate_quota_indices(
            "He so no tren tong tai san nam 2024 la bao nhieu?",
            candidates,
            4,
            prefer_magnitude=True,
        )

        self.assertIn(10, selected)

    def test_quota_preserves_inputs_of_par_value_identity(self) -> None:
        candidates = [
            NumericCandidate(
                index=index,
                value=str(10_000_000_000 + index),
                raw=str(10_000_000_000 + index),
                context=f"ROW: Chi phi quan ly {index}\nCELL: {index}",
                line_number=index + 1,
                char_start=0,
                char_end=4,
                unit="VND",
                is_percent=False,
                is_year_like=False,
                relevance=10.0 - index / 100,
                extraction_method="native",
            )
            for index in range(10)
        ]
        for index, (value, row, unit) in enumerate(
            (
                ("123464110000", "Von dieu le", "VND"),
                ("12346411", "So luong co phan da phat hanh", "shares"),
                ("10451182", "So luong co phan dang luu hanh", "shares"),
            ),
            start=10,
        ):
            candidates.append(
                NumericCandidate(
                    index=index,
                    value=value,
                    raw=value,
                    context=f"ROW: {row}\nCELL: {value}",
                    line_number=index + 1,
                    char_start=0,
                    char_end=len(value),
                    unit=unit,
                    is_percent=False,
                    is_year_like=False,
                    relevance=0.01,
                    extraction_method="native",
                )
            )

        selected = _candidate_quota_indices(
            "Gia tri theo menh gia cua so co phieu dang luu hanh la bao nhieu?",
            candidates,
            5,
            prefer_magnitude=True,
        )

        self.assertTrue({10, 11, 12}.issubset(selected))

    def test_excludes_leading_ocr_row_code_without_header(self) -> None:
        context = (
            "270 | TỔNG CỘNG TÀI SẢN\n"
            "TỔNG CỘNG TÀI SẢN | 284.331.609.643 | 354.781.425.043"
        )
        candidates = extract_numeric_candidates(
            "Tổng cộng tài sản cuối năm", context,
            extraction_method="ocr",
        )
        values = [item.decimal for item in candidates]
        self.assertNotIn(Decimal(270), values)

    def test_excludes_four_digit_report_code_before_financial_row(self) -> None:
        context = (
            "6119 | Doanh thu thuần | 284.331.609.643 | 354.781.425.043"
        )

        candidates = extract_numeric_candidates(
            "Doanh thu thuần năm 2025 là bao nhiêu?",
            context,
            extraction_method="ocr",
        )

        values = {item.decimal for item in candidates}
        self.assertNotIn(Decimal(6119), values)
        self.assertIn(Decimal(284331609643), values)
        self.assertIn(Decimal(284331609643), values)

    def test_excludes_nonleading_code_after_squashed_metadata_label(self) -> None:
        context = (
            "- Nguyên giá | 228 | | 105.407.208.045 | 105.307.528.045\n"
            "- Giá trị hao mòn | 229a | | (37.327.544.350) | (35.358.506.103)\n"
            "Tài sản dài hạn khác | 250 | | 21.466.844.674 | 31.551.000.741\n"
            "TÀI SẢN | Mãsố | Thuyết | 31/12/2025 | 01/01/2025\n"
            "--- | --- | --- | --- | ---\n"
            "TỔNG CỘNG TÀI SẢN | 270 | | "
            "4.714.032.275.888 | 3.390.042.913.247"
        )

        candidates = extract_numeric_candidates(
            "Tổng cộng tài sản cuối năm 2025 là bao nhiêu?",
            context,
            extraction_method="mineru",
        )
        values = {item.decimal for item in candidates}

        self.assertNotIn(Decimal(270), values)
        self.assertIn(Decimal(4714032275888), values)
        current = next(item for item in candidates if item.decimal == 4714032275888)
        self.assertIn("COLUMN: 31/12/2025", current.context)

    def test_excludes_label_first_row_code_without_any_header(self) -> None:
        context = (
            "TỔNG CỘNG NGUỒN VỐN | 440 | | "
            "4.714.032.275.888 | 3.390.042.913.247"
        )

        candidates = extract_numeric_candidates(
            "Tổng cộng nguồn vốn cuối năm là bao nhiêu?",
            context,
            extraction_method="ocr",
        )
        values = {item.decimal for item in candidates}

        self.assertNotIn(Decimal(440), values)
        self.assertIn(Decimal(4714032275888), values)

    def test_right_aligns_ocr_period_columns_when_header_has_extra_cell(self) -> None:
        context = (
            "Mã | Chỉ tiêu | Thuyết | minh | Năm nay | Năm trước\n"
            "01 | Doanh thu thuần | VI.1 | 25.651.868.351 | 24.896.355.359"
        )
        candidates = extract_numeric_candidates(
            "Doanh thu thuần thay đổi từ 2024 đến 2025", context,
            extraction_method="ocr",
        )
        current = next(item for item in candidates if item.decimal == 25651868351)
        prior = next(item for item in candidates if item.decimal == 24896355359)
        self.assertIn("PERIOD: nam nay", current.context)
        self.assertIn("PERIOD: nam truoc", prior.context)

    def test_period_columns_ignore_split_note_header_and_trailing_ocr_text(self) -> None:
        context = (
            "Mã số | NGUỒN VỐN | Thuyết | minh | Số cuối năm | Số đầu năm\n"
            "440 | TỔNG CỘNG NGUỒN VỐN | 284.331.609.643 | "
            "354.781.425.043 | KIEMT"
        )
        candidates = extract_numeric_candidates(
            "Tổng cộng nguồn vốn cuối năm 2025", context,
            extraction_method="ocr",
        )
        current = next(item for item in candidates if item.decimal == 284331609643)
        prior = next(item for item in candidates if item.decimal == 354781425043)
        self.assertIn("PERIOD: cuoi nam", current.context)
        self.assertIn("PERIOD: dau nam", prior.context)

    def test_question_focus_keeps_squashed_ocr_metric_candidates(self) -> None:
        context = (
            "Chỉ tiêu | 31/12/2025 | 01/01/2025\n"
            "NPHẢITRẢ | 3.117.771.571.701 | 1.954.212.130.333\n"
            "Nguyên tắc ghi nhận | 4.2 | 36\n"
            "Địa chỉ | 16 | 36\n"
            "Chi phí | 100.000 | 90.000"
        )
        candidates = extract_numeric_candidates(
            "Theo báo cáo tình hình tài chính, nợ phải trả năm 2025 là bao nhiêu?",
            context,
            max_candidates=4,
            extraction_method="ocr",
        )
        values = {item.decimal for item in candidates}
        self.assertIn(Decimal(3117771571701), values)

    def test_skips_implausibly_long_ocr_number(self) -> None:
        candidates = extract_numeric_candidates(
            "Tổng tài sản",
            "Tổng tài sản | 12345678901234567890123456789012345",
            extraction_method="ocr",
        )
        self.assertEqual(candidates, ())

    def test_plaintext_split_headers_ignore_year_over_year_ratio(self) -> None:
        context = (
            "Chi tieu Nam 2022\n"
            "(dong)\n"
            "Nam 2023\n"
            "(dong)\n"
            "Ti le (%)\n"
            "2023/2022\n"
            "Tong gia tri tai san 31.660.423.604 34.890.804.353 110,20%\n"
            "Doanh thu thuan 49.276.104.876 40.494.083.152 82,18%"
        )
        candidates = extract_numeric_candidates(
            "Doanh thu thuan nam 2023 la bao nhieu?",
            context,
            extraction_method="native",
        )
        year_2022 = next(
            item for item in candidates if item.decimal == Decimal(49276104876)
        )
        year_2023 = next(
            item for item in candidates if item.decimal == Decimal(40494083152)
        )

        self.assertIn("COLUMN: 2022", year_2022.context)
        self.assertIn("PERIOD: 2022", year_2022.context)
        self.assertIn("COLUMN: 2023", year_2023.context)
        self.assertIn("PERIOD: 2023", year_2023.context)

    def test_plaintext_single_line_year_and_share_headers_map_all_columns(self) -> None:
        context = (
            "Chi tieu Nam 2022 Ty trong 2022 Nam 2023 Ty trong 2023 %2023/2022\n"
            "Tai san ngan han 2.285.865 85,62% 1.802.905 77,51% 78,87%\n"
            "Tai san dai han 383.837 14,38% 523.014 22,49% 136,26%"
        )
        candidates = extract_numeric_candidates(
            "Tong tai san ngan han va tai san dai han tai 2022 la bao nhieu?",
            context,
            extraction_method="native",
        )

        values = {candidate.decimal: candidate for candidate in candidates}
        self.assertEqual(values[Decimal(2285865)].period, "2022")
        self.assertEqual(values[Decimal(1802905)].period, "2023")
        self.assertEqual(values[Decimal("85.62")].unit, "%")
        self.assertEqual(values[Decimal("85.62")].period, "2022")

    def test_plaintext_related_company_table_has_embedded_entity_scope(self) -> None:
        context = (
            "3. Tinh hinh dau tu, tinh hinh thuc hien cac du an\n"
            "Cong ty lien ket: Cong ty co phan Be tong ly tam Dien luc Khanh Hoa\n"
            "Chi tieu Nam 2022\n"
            "Nam 2023\n"
            "Ti le (%)\n"
            "2023/2022\n"
            "Tong gia tri tai san 31.660.423.604 34.890.804.353 110,20%\n"
            "4. Tinh hinh tai chinh\n"
            "4.1. Tinh hinh tai chinh\n"
            "Chi tieu Nam 2022\n"
            "Nam 2023\n"
            "Ti le (%)\n"
            "2023/2022\n"
            "Tong gia tri tai san 175.058.817.104 167.468.103.992 95,66%"
        )
        candidates = extract_numeric_candidates(
            "Tong tai san nam 2023 la bao nhieu?",
            context,
            extraction_method="native",
        )
        embedded = next(
            item for item in candidates if item.decimal == Decimal(34890804353)
        )
        reporting = next(
            item for item in candidates if item.decimal == Decimal(167468103992)
        )

        self.assertEqual(
            embedded.scope,
            "entity:cong ty co phan be tong ly tam dien luc khanh hoa",
        )
        self.assertNotEqual(embedded.unit, "shares")
        self.assertIn("Cong ty lien ket", embedded.section)
        self.assertEqual(reporting.scope, "")
        self.assertIn("Tinh hinh tai chinh", reporting.section)

    def test_explicit_currency_beats_adjacent_landscape_coefficient_heading(self) -> None:
        context = (
            "Đơn vị tính: tỷ đồng                         Hệ số thanh toán\n"
            "| Chỉ tiêu | Năm 2024 | Năm 2023 |\n"
            "|---|---:|---:|\n"
            "| Tổng tài sản | 4.877 | 4.699 |"
        )
        candidates = extract_numeric_candidates(
            "Tổng tài sản năm 2024 là bao nhiêu?",
            context,
            extraction_method="native",
        )
        value = next(item for item in candidates if item.decimal == Decimal(4877))

        self.assertEqual(value.unit, "VND")

    def test_plaintext_single_value_right_aligns_to_last_period(self) -> None:
        context = (
            "NO PHAI TRA 01/01/2023 31/12/2023\n"
            "II. No dai han 1.707.170.000"
        )
        candidates = extract_numeric_candidates(
            "No dai han nam 2023 la bao nhieu?",
            context,
            extraction_method="native",
        )
        value = next(
            item for item in candidates if item.decimal == Decimal(1707170000)
        )

        self.assertEqual(value.column_label, "31/12/2023")
        self.assertEqual(value.period, "2023")

    def test_plaintext_blank_prior_zero_requires_table_and_narrative_support(
        self,
    ) -> None:
        context = (
            "Dvt: dong\n"
            "NO PHAI TRA 01/01/2023 31/12/2023\n"
            "II. No dai han 1.707.170.000\n"
            "No dai han tang 1,7 ty dong do phat sinh no vay trung han."
        )
        candidates = extract_numeric_candidates(
            "No dai han thay doi bao nhieu tu nam 2022 den nam 2023?",
            context,
            extraction_method="native",
        )
        implied = next(
            item
            for item in candidates
            if item.decimal == 0
            and "IMPLIED_BLANK_ZERO" in item.context
        )

        self.assertEqual(implied.column_label, "01/01/2023")
        self.assertEqual(implied.value, "0")
        self.assertEqual(implied.unit, "VND")

    def test_recovers_wrapped_ocr_header_period(self) -> None:
        context = (
            "Chỉ tiêu | Năm 2025 | Năm 2024 | % tăng\n"
            "(giảm)\n"
            "Tổng giá trị tài sản | 1.452.678.263.060 | 1.455.150.771.623 | (0,17)\n"
            "Doanh thu thuần | 844.346.681.090 | 1.172.121.547.172 | (27,96)"
        )
        candidates = extract_numeric_candidates(
            "Doanh thu thuần của AAH năm 2024 là bao nhiêu?", context
        )
        prior_year = next(
            candidate
            for candidate in candidates
            if candidate.decimal == 1172121547172
        )

        self.assertIn("COLUMN: Năm 2024", prior_year.context)
        self.assertIn("PERIOD: 2024", prior_year.context)
        self.assertNotIn("COLUMN: 1.455.150.771.623", prior_year.context)
        self.assertNotEqual(prior_year.unit, "%")
        percent_change = next(
            candidate
            for candidate in candidates
            if candidate.decimal == -Decimal("27.96")
        )
        self.assertEqual(percent_change.unit, "%")

    def test_uses_actual_markdown_header_and_period(self) -> None:
        context = (
            "| Chỉ tiêu | Cuối năm 2024 | Đầu năm 2024 |\n"
            "|---|---:|---:|\n"
            "| Tổng tài sản | 200 VND | 100 VND |\n"
            "| Chi phí lao động | 50 VND | 40 VND |"
        )
        candidates = extract_numeric_candidates(
            "Tổng tài sản cuối năm 2024", context
        )
        current_total = next(item for item in candidates if item.decimal == 200)
        labor = next(item for item in candidates if item.decimal == 50)

        self.assertIn("ROW: Tổng tài sản", current_total.context)
        self.assertIn("ROW_TYPE: total", current_total.context)
        self.assertIn("COLUMN: Cuối năm 2024", current_total.context)
        self.assertIn("PERIOD: cuoi nam 2024", current_total.context)
        self.assertIn("COLUMN: Cuối năm 2024", labor.context)
        self.assertNotIn("COLUMN: 200 VND", labor.context)

    def test_does_not_promote_previous_value_row_to_header(self) -> None:
        context = (
            "| Doanh thu | 2024 | 2023 |\n"
            "| Tổng tài sản | 200 VND | 100 VND |\n"
            "| Chi phí lao động | 50 VND | 40 VND |"
        )
        candidates = extract_numeric_candidates("Chi phí lao động", context)
        labor = next(item for item in candidates if item.decimal == 50)
        self.assertNotIn("COLUMN: 200 VND", labor.context)

    def test_excludes_dates_codes_notes_years_and_ordinals(self) -> None:
        context = (
            "Báo cáo tại ngày 31 tháng 12 năm 2024\n"
            "| Mã số | Chỉ tiêu | Thuyết minh | Cuối năm 2024 | Đầu năm 2024 |\n"
            "|---|---|---|---:|---:|\n"
            "| 300 | 1. Doanh thu thuần | 13 | 100 VND | 90 VND |"
        )
        candidates = extract_numeric_candidates("Doanh thu thuần", context)
        self.assertEqual(
            [candidate.decimal for candidate in candidates],
            [100, 90],
        )

    def test_prefers_semantic_row_label_over_alphanumeric_code(self) -> None:
        context = (
            "| Mã số | Chỉ tiêu | Cuối năm 2024 |\n"
            "|---|---|---:|\n"
            "| 411a | Cổ phiếu phổ thông có quyền biểu quyết | 82 VND |"
        )
        candidates = extract_numeric_candidates("Cổ phiếu phổ thông", context)
        value = next(candidate for candidate in candidates if candidate.decimal == 82)
        self.assertIn(
            "ROW: Cổ phiếu phổ thông có quyền biểu quyết", value.context
        )

    def test_rejects_chunk_number_as_arithmetic_target(self) -> None:
        row = {
            "id": "chunk-metadata",
            "task": "QA",
            "difficulty": "Medium_QA",
            "question": "So sánh khoản đầu tư với tổng đầu tư tài chính dài hạn.",
            "context": (
                "Báo cáo tại ngày 31 tháng 12 năm 2024\n"
                "| Chỉ tiêu | Giá trị |\n"
                "|---|---:|\n"
                "| Khoản đầu tư | 33.000.000.000 VND |\n"
                "| Tổng đầu tư tài chính dài hạn | 33.000.000.000 VND |"
            ),
            "answer": (
                "Khoản đầu tư là 33.000.000.000 VND. Theo chunk 19, tổng đầu "
                "tư tài chính dài hạn cũng là 33.000.000.000 VND."
            ),
        }
        self.assertIsNone(induce_program_example(row))

    def test_header_duration_is_not_used_as_ratio_operand(self) -> None:
        row = {
            "id": "duration-header",
            "task": "QA",
            "difficulty": "Hard_QA",
            "question": (
                "Tính tỷ lệ tài sản tài chính ngắn hạn trên tổng tài sản tài chính."
            ),
            "context": (
                "| Chỉ tiêu | ≤ 1 năm | Trên 1 năm | Tổng |\n"
                "| Tài sản tài chính | 184.000 VND | 0 VND | 184.000 VND |"
            ),
            "answer": (
                "Tài sản tài chính ngắn hạn là 184.000 VND và tổng tài sản tài "
                "chính là 184.000 VND. Tỷ lệ = 100%."
            ),
        }
        example = induce_program_example(row)
        self.assertIsNotNone(example)
        assert example is not None
        self.assertEqual(example.program.operation, "RATIO_PERCENT")
        values = [example.candidates[index].decimal for index in example.program.operands]
        self.assertEqual(values, [184000, 184000])


if __name__ == "__main__":
    unittest.main()
