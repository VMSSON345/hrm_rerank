from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from hrm_finance_qa.program_data import NumericCandidate


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "generate_document_derived_programs",
    ROOT / "scripts" / "generate_document_derived_programs.py",
)
assert SPEC is not None and SPEC.loader is not None
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


class DocumentDerivedProgramGenerationTests(unittest.TestCase):
    def test_rejects_broken_table_coordinates_before_generation(self) -> None:
        context = """| Số cuối năm | 22.302.401.624 | 59.624.651.364 |
| Số đầu năm | 454.717.622 | 2.852.489.649 |
| triển phẩm | 100 | 90 |
| Doanh thu thuần | 120 | 100 |
"""
        examples, _ = GENERATOR._page_examples(
            context,
            source_name="broken-report.pdf",
            page=4,
            max_examples=24,
            id_start=1,
            extraction_method="native",
        )

        self.assertEqual(examples, [])

    def test_accepts_year_embedded_in_temporal_header(self) -> None:
        self.assertEqual(GENERATOR._period_order("Thực hiện năm 2024"), (2, 2024))
        self.assertIsNone(
            GENERATOR._period_order(
                "Thực hiện 2023 Kế hoạch 2024 So sánh thực hiện"
            )
        )

    def test_native_layout_extraction_falls_back_for_unknown_font_encoding(self) -> None:
        class UnknownEncodingPage:
            def extract_text(self, *args, **kwargs):
                if kwargs.get("extraction_mode") == "layout":
                    raise LookupError("unknown encoding: /SymbolSetEncoding")
                return "Chỉ tiêu  100  90"

        self.assertEqual(
            GENERATOR._extract_native_page_text(UnknownEncodingPage()),
            "Chỉ tiêu  100  90",
        )

    def test_embedded_company_identity_is_hash_locked_and_complete(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "company-a.pdf"
            source.write_bytes(b"pdf-a")
            manifest = {
                "documents": [
                    {
                        "document": str(source),
                        "sha256": GENERATOR._sha256(source),
                        "company_key": "opaque-company-01",
                    }
                ]
            }
            self.assertEqual(
                GENERATOR._load_embedded_company_keys(manifest, [source]),
                {source.name: "opaque-company-01"},
            )
            manifest["documents"][0]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                GENERATOR._load_embedded_company_keys(manifest, [source])

    def test_ocr_scored_manifest_requires_explicit_mineru_markdown(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "MinerU Markdown"):
            GENERATOR._validate_manifest_extraction(
                {"ocr_performed": True}, mineru_markdown_dir=None
            )
        GENERATOR._validate_manifest_extraction(
            {"ocr_performed": True}, mineru_markdown_dir=Path("ocr-pages")
        )
        GENERATOR._validate_manifest_extraction(
            {"ocr_performed": False}, mineru_markdown_dir=Path("ocr-pages")
        )

    def test_company_manifest_is_hash_locked_and_complete(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "company-a.pdf"
            source.write_bytes(b"pdf-a")
            manifest = root / "companies.json"
            manifest.write_text(
                json.dumps(
                    {
                        "protocol": "pdf-development-company-manifest-v1",
                        "documents": {
                            source.name: {
                                "company_key": "opaque-company-01",
                                "source_sha256": GENERATOR._sha256(source),
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            companies = GENERATOR._load_company_manifest(manifest, [source])

            self.assertEqual(companies, {source.name: "opaque-company-01"})
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["documents"][source.name]["source_sha256"] = "0" * 64
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                GENERATOR._load_company_manifest(manifest, [source])

    def test_records_specific_subtotal_negative_alongside_wrong_row(self) -> None:
        def candidate(
            index: int, row: str, row_type: str, period: str, value: str
        ) -> NumericCandidate:
            return NumericCandidate(
                index=index,
                value=value,
                raw=value,
                context=(
                    f"ROW: {row}\nROW_TYPE: {row_type}\nCOLUMN: {period}\n"
                    f"PERIOD: {period}\nHEADER: test\nCELL: <VALUE>{value}</VALUE>"
                ),
                line_number=index + 1,
                char_start=0,
                char_end=len(value),
                unit="VND",
                is_percent=False,
                is_year_like=False,
                relevance=1.0,
                source_name="development-report.pdf",
                page=5,
                extraction_method="native",
                row_label=row,
                column_label=period,
                period=period,
                row_type=row_type,
                source_span=(0, len(value)),
            )

        candidates = (
            candidate(0, "Tổng tài sản ngắn hạn", "total", "2024", "10000"),
            candidate(1, "Tài sản ngắn hạn", "detail", "2024", "9000"),
            candidate(2, "Tổng tài sản ngắn hạn", "total", "2023", "8000"),
        )

        supervision = GENERATOR._structured_supervision("SELECT", candidates, (0,))
        negative_types = {item["type"] for item in supervision["hard_negatives"]}

        self.assertIn("same_period_wrong_row", negative_types)
        self.assertIn("subtotal_detail_near_match", negative_types)
        self.assertIn("same_row_wrong_period", negative_types)

    def test_records_same_metric_period_wrong_provenance_negative(self) -> None:
        def candidate(
            index: int, value: str, table_id: str, scope: str
        ) -> NumericCandidate:
            return NumericCandidate(
                index=index,
                value=value,
                raw=value,
                context=(
                    "ROW: Doanh thu thuần\nROW_TYPE: detail\n"
                    "COLUMN: 2024\nPERIOD: 2024\n"
                    f"TABLE: {table_id}\nSCOPE: {scope}\n"
                    f"HEADER: test\nCELL: <VALUE>{value}</VALUE>"
                ),
                line_number=index + 1,
                char_start=0,
                char_end=len(value),
                unit="VND",
                is_percent=False,
                is_year_like=False,
                relevance=1.0,
                source_name="development-report.pdf",
                page=5,
                extraction_method="native",
            )

        candidates = (
            candidate(0, "100", "table-1", "consolidated"),
            candidate(1, "80", "table-2", "parent_company"),
            candidate(2, "70", "table-1", "consolidated"),
        )

        supervision = GENERATOR._structured_supervision("SELECT", candidates, (0,))
        negatives = [
            item
            for item in supervision["hard_negatives"]
            if item["type"] == "same_metric_period_wrong_provenance"
        ]

        self.assertEqual(negatives[0]["candidate_indices"], [1])

    def test_generates_priority_operations_from_real_table_structure(self) -> None:
        context = """Đơn vị: VND
| Chỉ tiêu | Kỳ này | Kỳ trước |
|---|---:|---:|
| Tổng tài sản | 10.000 | 9.000 |
| Tài sản ngắn hạn | 6.000 | 5.500 |
| Tài sản dài hạn | 4.000 | 3.500 |
| Nợ ngắn hạn | 2.000 | 1.800 |
| Nợ dài hạn | 1.000 | 900 |
| Nợ phải trả | 3.000 | 2.700 |
| Vốn chủ sở hữu | 7.000 | 6.300 |
| Tiền và các khoản tương đương tiền | 1.000 | 900 |
| Các khoản phải thu ngắn hạn | 2.000 | 1.800 |
| Hàng tồn kho | 2.500 | 2.300 |
| Tài sản ngắn hạn khác | 500 | 500 |
| Doanh thu thuần | 12.000 | 10.000 |
| Lợi nhuận sau thuế | 1.200 | 1.000 |
"""
        examples, _ = GENERATOR._page_examples(
            context,
            source_name="development-report.pdf",
            page=7,
            max_examples=50,
            id_start=1,
            extraction_method="mineru",
        )

        operations = {example.program.operation for example in examples}
        self.assertTrue(
            {
                "SUBTRACT",
                "RATIO_PERCENT",
                "ADD",
                "DIVIDE",
                "AVERAGE",
                "SUM3",
                "SUM4",
            }
            <= operations
        )

    def test_cross_row_sums_exclude_parent_child_but_allow_defined_ratios(
        self,
    ) -> None:
        context = """Đơn vị: VND
| Chỉ tiêu | Năm 2024 | Năm 2023 |
|---|---:|---:|
| Tổng tài sản | 10.000 | 9.000 |
| Tài sản ngắn hạn | 6.000 | 5.500 |
| Hàng tồn kho | 2.500 | 2.300 |
| Phải trả người bán ngắn hạn | 1.500 | 1.200 |
"""
        examples, _ = GENERATOR._page_examples(
            context,
            source_name="development-report.pdf",
            page=8,
            max_examples=50,
            id_start=1,
            extraction_method="mineru",
        )

        sums = [
            example
            for example in examples
            if example.program.operation
            in {"ADD", "SUM3", "SUM4"}
        ]
        ratios = [
            example
            for example in examples
            if example.program.operation in {"DIVIDE", "RATIO_PERCENT"}
        ]
        self.assertEqual(sums, [])
        self.assertTrue(
            any(
                "Tài sản ngắn hạn" in example.question
                and "Tổng tài sản" in example.question
                for example in ratios
            )
        )

    def test_generates_multiple_distinct_sum4_sibling_combinations(self) -> None:
        context = """Đơn vị: VND
| Chỉ tiêu | Năm 2024 | Năm 2023 |
|---|---:|---:|
| Tiền | 1.000 | 900 |
| Các khoản phải thu ngắn hạn | 2.000 | 1.800 |
| Hàng tồn kho | 3.000 | 2.700 |
| Đầu tư tài chính ngắn hạn | 4.000 | 3.600 |
| Tài sản ngắn hạn khác | 5.000 | 4.500 |
"""
        examples, _ = GENERATOR._page_examples(
            context,
            source_name="development-report.pdf",
            page=81,
            max_examples=100,
            id_start=1,
            extraction_method="mineru",
        )

        sum4_operands = {
            example.program.operands
            for example in examples
            if example.program.operation == "SUM4"
        }
        self.assertGreaterEqual(len(sum4_operands), 2)

    def test_generates_share_count_times_par_value(self) -> None:
        context = """Đơn vị: VND
| Chỉ tiêu | Năm 2024 | Năm 2023 |
|---|---:|---:|
| Số cổ phiếu đang lưu hành | 1.000.000 | 900.000 |
| Mệnh giá cổ phiếu | 10.000 | 10.000 |
| Vốn chủ sở hữu | 20.000.000 | 18.000.000 |
| Lợi nhuận sau thuế | 8.000.000 | 7.000.000 |
"""
        examples, _ = GENERATOR._page_examples(
            context,
            source_name="development-report.pdf",
            page=9,
            max_examples=40,
            id_start=1,
            extraction_method="mineru",
        )

        multiply = next(
            example
            for example in examples
            if example.program.operation == "MULTIPLY"
        )
        self.assertEqual(len(multiply.program.operands), 2)
        self.assertIn("mệnh giá", multiply.question.lower())

    def test_generates_multiply_from_native_share_disclosure(self) -> None:
        context = """THÔNG TIN CỔ PHẦN
Tổng số cổ phiếu đang lưu hành: 10.263.942 cổ phiếu
Mệnh giá cổ phần: 10.000 đồng/cổ phiếu
"""
        examples, _ = GENERATOR._page_examples(
            context,
            source_name="NSS_25CN_BCTN.pdf",
            page=22,
            max_examples=20,
            id_start=1,
            extraction_method="native",
        )

        multiply = next(
            example
            for example in examples
            if example.program.operation == "MULTIPLY"
        )
        selected = [
            multiply.candidates[index] for index in multiply.program.operands
        ]
        self.assertEqual(multiply.answer_value, "102639420000")
        self.assertEqual({item.table_id for item in selected}, {"share-capital-page-22"})
        self.assertTrue(all(item.entity_id == "entity:source:nss" for item in selected))
        self.assertTrue(
            all(
                item.table_id
                and item.entity_id
                and item.subtable_id
                and "CELL:" in item.context
                for item in multiply.candidates
            )
        )

    def test_records_operand_roles_provenance_and_structural_negatives(self) -> None:
        context = """Đơn vị: VND
| Chỉ tiêu | Kỳ này | Kỳ trước |
|---|---:|---:|
| Doanh thu thuần | 12.000 | 10.000 |
| Lợi nhuận sau thuế | 3.000 | 2.000 |
| Tổng tài sản | 20.000 | 18.000 |
| Vốn chủ sở hữu | 8.000 | 7.000 |
"""
        examples, _ = GENERATOR._page_examples(
            context,
            source_name="development-report.pdf",
            page=11,
            max_examples=32,
            id_start=1,
            extraction_method="mineru",
        )
        ratio = next(
            example for example in examples if example.program.operation == "RATIO_PERCENT"
        )
        supervision = ratio.supervision
        assert supervision is not None
        gold = supervision["gold_operands"]
        self.assertEqual([item["role"] for item in gold], ["numerator", "denominator"])
        self.assertTrue(all(item["page"] == 11 for item in gold))
        self.assertTrue(all(item["row"] and item["column"] and item["period"] for item in gold))
        negative_types = {item["type"] for item in supervision["hard_negatives"]}
        self.assertIn("same_row_wrong_period", negative_types)
        self.assertIn("same_period_wrong_row", negative_types)
        self.assertIn("reversed_operand_order", negative_types)


if __name__ == "__main__":
    unittest.main()
