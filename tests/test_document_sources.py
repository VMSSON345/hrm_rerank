from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hrm_finance_qa.document_sources import (
    DocumentSourceError,
    EvidenceChunk,
    OCRConfig,
    SourceLine,
    _read_pdf,
    assess_native_text,
    _load_or_generate_mineru_pages,
    _load_mineru_pages,
    _mineru_table_to_lines,
    _partition_mineru_table_lines,
    _repair_wrapped_numeric_lines,
    _clean_line,
    chunk_lines,
    expand_neighbor_chunks,
)
from hrm_finance_qa.program_data import extract_numeric_candidates


class MinerURoutingTests(unittest.TestCase):
    def test_clean_line_decodes_pdf_unicode_glyph_names(self) -> None:
        self.assertEqual(
            _clean_line("T/uni1ED4NG TÀI S/uni1EA2N"),
            "TỔNG TÀI SẢN",
        )

    def test_mineru_artifact_is_not_required_when_native_text_is_usable(self) -> None:
        OCRConfig(enabled=True).validate()

    def test_clean_native_financial_text_skips_ocr(self) -> None:
        assessment = assess_native_text(
            "BÁO CÁO TÌNH HÌNH TÀI CHÍNH\n"
            "Tổng cộng tài sản | 4.714.032.275.888 | 3.390.042.913.247"
        )

        self.assertTrue(assessment.usable)
        self.assertEqual(assessment.reason, "usable")

    def test_long_broken_cid_text_requires_ocr(self) -> None:
        assessment = assess_native_text(
            " ".join(f"(cid:{index})" for index in range(1, 30))
        )

        self.assertFalse(assessment.usable)
        self.assertEqual(assessment.reason, "broken_encoding")

    def test_replacement_glyph_text_requires_ocr(self) -> None:
        assessment = assess_native_text(
            "Doanh thu thuần ������������ năm 2024 là 100.000.000 đồng"
        )

        self.assertFalse(assessment.usable)
        self.assertEqual(assessment.reason, "broken_encoding")

    def test_short_native_overlay_does_not_hide_scanned_page(self) -> None:
        assessment = assess_native_text("Trang 25")

        self.assertFalse(assessment.usable)
        self.assertEqual(assessment.reason, "too_short")

    def test_ocr_digit_substitutions_inside_words_require_ocr(self) -> None:
        assessment = assess_native_text(
            (
                "Th6ng tin t6ng quan C6ng ty g6m ho4t dQng kinh doanh "
                "v5 k6 ho4ch ph6t tri6n trong n5m t5i ch1nh. "
            )
            * 3
        )

        self.assertFalse(assessment.usable)
        self.assertEqual(assessment.reason, "ocr_digit_substitutions")

    def test_fragmented_vietnamese_diacritics_require_ocr(self) -> None:
        assessment = assess_native_text(
            (
                "Ti ếp t ục phát tri ển vùng nuôi, t ăng n ăng l ực qu ản lý "
                "và xu ất kh ẩu s ản ph ẩm đế n ng ười tiêu th ụ. "
            )
            * 3
        )

        self.assertFalse(assessment.usable)
        self.assertEqual(assessment.reason, "fragmented_diacritics")

    def test_normal_vietnamese_word_spacing_remains_native(self) -> None:
        assessment = assess_native_text(
            "Báo cáo đã ổn định và có ảnh hưởng tích cực đến hoạt động kinh doanh. "
            "Tổng tài sản, doanh thu thuần và lợi nhuận sau thuế đều được trình bày "
            "theo đúng kỳ kế toán hiện hành."
        )

        self.assertTrue(assessment.usable)
        self.assertEqual(assessment.reason, "usable")

    def test_repairs_wrapped_last_digits_in_financial_table(self) -> None:
        lines = [
            "Chỉ tiêu | Năm 2025 | Năm 2024 | % tăng",
            "(giảm)",
            "Tổng giá trị tài sản | 1.452.678.263.060 | 1.455.150.771.62 | (0,17)",
            "3",
            "Doanh thu thuần | 844.346.681.090 | 1.172.121.547.17 | (27,96)",
            "2",
        ]

        repaired = _repair_wrapped_numeric_lines(lines)

        self.assertEqual(len(repaired), 4)
        self.assertIn("1.455.150.771.623", repaired[2])
        self.assertIn("1.172.121.547.172", repaired[3])

    def test_auto_generates_fingerprinted_mineru_page_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "scan.pdf"
            source.write_bytes(b"pdf-placeholder")
            command = root / "mineru"
            command.write_text("placeholder", encoding="utf-8")
            config = OCRConfig(
                enabled=True,
                mineru_cache_dir=root / "cache",
                mineru_command=command,
            )
            completed = SimpleNamespace(returncode=0, stdout="ok", stderr="")
            with patch(
                "hrm_finance_qa.document_sources._load_mineru_pages",
                side_effect=[
                    DocumentSourceError("artifact missing"),
                    {2: ["Doanh thu thuần | 100"]},
                ],
            ) as load_pages:
                with patch(
                    "hrm_finance_qa.document_sources.subprocess.run",
                    return_value=completed,
                ) as run_mineru:
                    pages = _load_or_generate_mineru_pages(
                        source, [False, True, True, False], config
                    )

        self.assertEqual(pages, {2: ["Doanh thu thuần | 100"]})
        command_args = run_mineru.call_args.args[0]
        self.assertEqual(command_args[command_args.index("-s") + 1], "1")
        self.assertEqual(command_args[command_args.index("-e") + 1], "2")
        generated_config = load_pages.call_args_list[1].args[1]
        self.assertEqual(generated_config.mineru_page_offset, 1)

    def test_sparse_cli_selection_generates_only_contiguous_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "scan.pdf"
            source.write_bytes(b"pdf-placeholder")
            command = root / "mineru"
            command.write_text("placeholder", encoding="utf-8")
            config = OCRConfig(
                enabled=True,
                mineru_cache_dir=root / "cache",
                mineru_command=command,
            )
            completed = SimpleNamespace(returncode=0, stdout="ok", stderr="")
            with patch(
                "hrm_finance_qa.document_sources._load_mineru_pages",
                side_effect=(
                    DocumentSourceError("full artifact missing"),
                    DocumentSourceError("first run missing"),
                    {2: ["first"]},
                    DocumentSourceError("second run missing"),
                    {5: ["second"]},
                ),
            ), patch(
                "hrm_finance_qa.document_sources.subprocess.run",
                return_value=completed,
            ) as run_mineru:
                pages = _load_or_generate_mineru_pages(
                    source,
                    [False, True, False, False, True],
                    config,
                )

        self.assertEqual(pages, {2: ["first"], 5: ["second"]})
        self.assertEqual(run_mineru.call_count, 2)
        ranges = []
        for call in run_mineru.call_args_list:
            command_args = call.args[0]
            ranges.append(
                (
                    command_args[command_args.index("-s") + 1],
                    command_args[command_args.index("-e") + 1],
                )
            )
        self.assertEqual(ranges, [("1", "1"), ("4", "4")])

    def test_disabled_auto_generation_requires_existing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "scan.pdf"
            source.write_bytes(b"pdf-placeholder")

            with self.assertRaisesRegex(
                DocumentSourceError, "auto-generation is disabled"
            ):
                _load_or_generate_mineru_pages(
                    source,
                    [True],
                    OCRConfig(enabled=True, mineru_auto_generate=False),
                )

    def test_disabled_auto_generation_still_reads_compatible_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "scan.pdf"
            source.write_bytes(b"pdf-placeholder")
            cached = {1: ["Doanh thu thuần | 1.200.000"]}
            with patch(
                "hrm_finance_qa.document_sources._load_mineru_pages",
                return_value=cached,
            ) as load_pages, patch(
                "hrm_finance_qa.document_sources.subprocess.run"
            ) as run_mineru:
                pages = _load_or_generate_mineru_pages(
                    source,
                    [True],
                    OCRConfig(enabled=True, mineru_auto_generate=False),
                )

        self.assertEqual(pages, cached)
        load_pages.assert_called_once()
        run_mineru.assert_not_called()

    def test_generates_cached_content_list_through_persistent_mineru_api(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "scan.pdf"
            source.write_bytes(b"pdf-placeholder")
            content = [
                {
                    "type": "text",
                    "text": "Doanh thu thuần | 100",
                    "page_idx": 0,
                }
            ]
            response = MagicMock()
            response.__enter__.return_value.read.return_value = json.dumps(
                {
                    "results": {
                        "scan": {"content_list": json.dumps(content)}
                    }
                }
            ).encode("utf-8")
            with patch(
                "hrm_finance_qa.document_sources.urllib.request.urlopen",
                return_value=response,
            ) as urlopen:
                pages = _load_or_generate_mineru_pages(
                    source,
                    [False, True],
                    OCRConfig(
                        enabled=True,
                        mineru_cache_dir=root / "cache",
                        mineru_api_url="http://127.0.0.1:8000",
                    ),
                )

        self.assertEqual(pages, {2: ["Doanh thu thuần | 100"]})
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8000/file_parse")
        self.assertIn(b'name="return_content_list"', request.data)
        self.assertIn(b'name="start_page_id"', request.data)

    def test_persistent_mineru_api_splits_large_page_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "scan.pdf"
            source.write_bytes(b"pdf-placeholder")
            content = [{"type": "text", "text": "row | 100", "page_idx": 0}]
            response = MagicMock()
            response.__enter__.return_value.read.return_value = json.dumps(
                {"results": {"scan": {"content_list": json.dumps(content)}}}
            ).encode("utf-8")
            with patch(
                "hrm_finance_qa.document_sources.urllib.request.urlopen",
                return_value=response,
            ) as urlopen:
                pages = _load_or_generate_mineru_pages(
                    source,
                    [True, True, True, True, True],
                    OCRConfig(
                        enabled=True,
                        mineru_cache_dir=root / "cache",
                        mineru_api_url="http://127.0.0.1:8000",
                        mineru_api_page_batch_size=4,
                    ),
                )

        self.assertEqual(set(pages), {1, 5})
        self.assertEqual(urlopen.call_count, 2)
        first_request = urlopen.call_args_list[0].args[0]
        second_request = urlopen.call_args_list[1].args[0]
        self.assertIn(b'\r\n0\r\n', first_request.data)
        self.assertIn(b'\r\n3\r\n', first_request.data)
        self.assertIn(b'\r\n4\r\n', second_request.data)

class NativeTextRoutingTests(unittest.TestCase):
    class _Page:
        def __init__(self, text: str) -> None:
            self.text = text

        def extract_text(self) -> str:
            return self.text

    @staticmethod
    def _pypdf_module(page_texts: list[str]) -> SimpleNamespace:
        pages = [NativeTextRoutingTests._Page(text) for text in page_texts]
        return SimpleNamespace(PdfReader=lambda _path: SimpleNamespace(pages=pages))

    def test_clean_native_page_never_calls_ocr(self) -> None:
        native = (
            "BÁO CÁO TÌNH HÌNH TÀI CHÍNH\n"
            "Tổng cộng tài sản | 4.714.032.275.888 | 3.390.042.913.247"
        )
        with patch.dict("sys.modules", {"pypdf": self._pypdf_module([native])}):
            with patch(
                "hrm_finance_qa.document_sources._load_or_generate_mineru_pages"
            ) as run_mineru:
                lines = _read_pdf(
                    Path("native.pdf"),
                    OCRConfig(enabled=True),
                )

        run_mineru.assert_not_called()
        self.assertTrue(lines)
        self.assertEqual({line.extraction_method for line in lines}, {"native"})

    def test_mixed_pdf_ocrs_only_unusable_page(self) -> None:
        native = (
            "BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH\n"
            "Doanh thu thuần | 537.023.929.923 | 379.455.976.931"
        )
        page_texts = [native, "Trang 2"]
        with patch.dict("sys.modules", {"pypdf": self._pypdf_module(page_texts)}):
            with patch(
                "hrm_finance_qa.document_sources._load_or_generate_mineru_pages",
                return_value={2: ["Nợ phải trả | 3.117.771.571.701"]},
            ) as run_mineru:
                lines = _read_pdf(
                    Path("mixed.pdf"),
                    OCRConfig(enabled=True),
                )

        self.assertEqual(run_mineru.call_args.args[1], [False, True])
        methods_by_page = {(line.page, line.extraction_method) for line in lines}
        self.assertIn((1, "native"), methods_by_page)
        self.assertIn((2, "mineru"), methods_by_page)


class MinerUContentTests(unittest.TestCase):
    TABLE = """
    <table>
      <tr><td>TÀI SẢN</td><td>Mã số</td><td>Thuyết minh</td><td>31/12/2025</td><td>01/01/2025</td></tr>
      <tr><td></td><td></td><td></td><td>VND</td><td>VND</td></tr>
      <tr><td>TỔNG CỘNG TÀI SẢN</td><td>270</td><td></td><td>4.714.032.275.888</td><td>3.390.042.913.247</td></tr>
    </table>
    """

    def test_table_partition_does_not_shift_middle_rows_to_partial_final_page(self) -> None:
        lines = [f"line-{index}" for index in range(70)]
        pages = _partition_mineru_table_lines(
            lines,
            23,
            (100, 760, 900, 960),
            (
                (24, (100, 60, 900, 940)),
                (25, (100, 70, 900, 470)),
            ),
        )

        self.assertIn("line-52", pages[24])
        self.assertNotIn("line-52", pages[25])

    def test_converts_mineru_html_table_to_candidate_grounding(self) -> None:
        lines = _mineru_table_to_lines(self.TABLE)
        candidates = extract_numeric_candidates(
            "Tổng cộng tài sản cuối năm 2025 là bao nhiêu?",
            "\n".join(lines),
            extraction_method="ocr",
        )

        current = next(item for item in candidates if item.value == "4714032275888")
        prior = next(item for item in candidates if item.value == "3390042913247")
        self.assertIn("ROW: TỔNG CỘNG TÀI SẢN", current.context)
        self.assertIn("COLUMN: 31/12/2025", current.context)
        self.assertIn("PERIOD: 2025", current.context)
        self.assertIn("COLUMN: 01/01/2025", prior.context)

    def test_inline_period_survives_when_chunk_starts_at_data_row(self) -> None:
        lines = _mineru_table_to_lines(self.TABLE)
        data_line = next(line for line in lines if "4.714.032.275.888" in line)
        candidates = extract_numeric_candidates(
            "Tổng cộng tài sản cuối năm 2025 là bao nhiêu?",
            data_line,
            extraction_method="ocr",
        )

        current = next(item for item in candidates if item.value == "4714032275888")
        prior = next(item for item in candidates if item.value == "3390042913247")
        self.assertIn("COLUMN: 31/12/2025", current.context)
        self.assertIn("COLUMN: 01/01/2025", prior.context)

    def test_mineru_table_rejects_statement_code_without_header_chunk(self) -> None:
        table = """
        <table>
          <tr><td>Chỉ tiêu</td><td>Mã số</td><td>Thuyết minh</td><td>Năm 2025 VND</td><td>Năm 2024 VND</td></tr>
          <tr><td>LỢI NHUẬN KẾ TOÁN SAU THUẾ TNDN</td><td>200</td><td></td><td>173.171.512.356</td><td>103.203.054.235</td></tr>
        </table>
        """
        lines = _mineru_table_to_lines(table)
        data_line = next(line for line in lines if "173.171.512.356" in line)
        candidates = extract_numeric_candidates(
            "Lợi nhuận kế toán sau thuế TNDN năm 2025 là bao nhiêu?",
            data_line,
            extraction_method="mineru",
        )

        self.assertNotIn("200", {item.value for item in candidates})
        self.assertIn("173171512356", {item.value for item in candidates})

    def test_expands_spans_and_recovers_only_explicit_formula_cell(self) -> None:
        table = """
        <table>
          <tr><th rowspan="2">Mã số</th><th rowspan="2">Chỉ tiêu</th><th rowspan="2">Thuyết minh</th><th colspan="2">Giá trị</th></tr>
          <tr><th>Năm 2025</th><th>Năm 2024</th></tr>
          <tr><td>01</td><td>Doanh thu bán hàng</td><td>25</td><td>1.250.000</td><td>1.100.000</td></tr>
          <tr><td>02</td><td>Các khoản giảm trừ doanh thu</td><td>25</td><td>50.000</td><td>40.000</td></tr>
          <tr><td>10</td><td>Doanh thu thuần (10 = 01 - 02)</td><td>26</td><td>1.200.000</td></tr>
        </table>
        """

        lines = _mineru_table_to_lines(table)
        candidates = extract_numeric_candidates(
            "Doanh thu thuần thay đổi bao nhiêu phần trăm từ năm 2024 đến năm 2025?",
            "\n".join(lines),
            extraction_method="mineru",
        )

        current = next(item for item in candidates if item.value == "1200000")
        prior = next(item for item in candidates if item.value == "1060000")
        self.assertIn("PERIOD: 2025", current.context)
        self.assertIn("PERIOD: 2024", prior.context)
        self.assertNotIn(
            "VALUE_ORIGIN: DERIVED_FROM_TABLE_FORMULA", current.context
        )
        self.assertIn(
            "VALUE_ORIGIN: DERIVED_FROM_TABLE_FORMULA", prior.context
        )

    def test_recovers_dropped_description_header_before_note_and_periods(self) -> None:
        table = """
        <table>
          <tr><th rowspan="2">Mã số</th><th rowspan="2">Thuyết minh</th><th colspan="2">Năm tài chính kết thúc</th></tr>
          <tr><th>2025</th><th>2024</th></tr>
          <tr><td>01</td><td>Doanh thu bán hàng</td><td>8.400.000</td><td>7.900.000</td></tr>
          <tr><td>02</td><td>Các khoản giảm trừ doanh thu</td><td>400.000</td><td>300.000</td></tr>
          <tr><td>10</td><td>Doanh thu thuần (10 = 01 - 02)</td><td>26</td><td>8.000.000</td></tr>
        </table>
        """

        lines = _mineru_table_to_lines(table)
        candidates = extract_numeric_candidates(
            "Doanh thu thuần thay đổi bao nhiêu phần trăm giữa hai năm?",
            "\n".join(lines),
            extraction_method="mineru",
        )

        current = next(item for item in candidates if item.value == "8000000")
        prior = next(item for item in candidates if item.value == "7600000")
        self.assertIn("ROW: Doanh thu thuần", current.context)
        self.assertIn("COLUMN: 2025", current.context)
        self.assertNotIn(
            "VALUE_ORIGIN: DERIVED_FROM_TABLE_FORMULA", current.context
        )
        self.assertIn("COLUMN: 2024", prior.context)
        self.assertIn(
            "VALUE_ORIGIN: DERIVED_FROM_TABLE_FORMULA", prior.context
        )

    def test_does_not_widen_ordinary_four_column_period_table(self) -> None:
        table = """
        <table>
          <tr><th rowspan="2">Mã số</th><th rowspan="2">Thuyết minh</th><th colspan="2">Giá trị</th></tr>
          <tr><th>2025</th><th>2024</th></tr>
          <tr><td>01</td><td>Doanh thu bán hàng</td><td>700.000</td><td>600.000</td></tr>
          <tr><td>02</td><td>Các khoản giảm trừ</td><td>10.000</td><td>20.000</td></tr>
        </table>
        """

        lines = _mineru_table_to_lines(table)
        data_line = next(line for line in lines if "700.000" in line)

        self.assertEqual(len(data_line.split("|")), 4)
        self.assertNotIn("DERIVED_FROM_TABLE_FORMULA", data_line)

    def test_does_not_fill_truncated_row_without_explicit_formula(self) -> None:
        table = """
        <table>
          <tr><td>Mã số</td><td>Chỉ tiêu</td><td>Năm 2025</td><td>Năm 2024</td></tr>
          <tr><td>10</td><td>Doanh thu thuần</td><td>70.112.825.100.710</td></tr>
        </table>
        """

        lines = _mineru_table_to_lines(table)

        self.assertFalse(any("DERIVED_FROM_TABLE_FORMULA" in line for line in lines))

    def test_loads_nested_content_list_with_source_page_offset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "ABS_25CN_BCTN.pdf"
            source.write_bytes(b"pdf-placeholder")
            output = root / "mineru" / source.stem / "hybrid_ocr"
            output.mkdir(parents=True)
            content_list = output / f"{source.stem}_content_list.json"
            content_list.write_text(
                json.dumps(
                    [
                        {
                            "type": "table",
                            "table_caption": ["Báo cáo tình hình tài chính"],
                            "table_body": self.TABLE,
                            "page_idx": 0,
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            pages = _load_mineru_pages(
                source,
                OCRConfig(
                    enabled=True,
                    force=True,
                    mineru_output_dir=root / "mineru",
                    mineru_page_offset=30,
                ),
            )

            self.assertEqual(list(pages), [31])
            self.assertTrue(any("4.714.032.275.888" in line for line in pages[31]))

    def test_distributes_merged_table_rows_across_empty_continuation_page(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "ABI.pdf"
            source.write_bytes(b"pdf-placeholder")
            output = root / "mineru" / source.stem / "hybrid_ocr"
            output.mkdir(parents=True)
            rows = "".join(
                f"<tr><td>{index}</td><td>Chi tieu {index}</td>"
                f"<td>{index}.000.000</td><td>{index}.100.000</td></tr>"
                for index in range(1, 41)
            )
            table = (
                "<table><tr><td>Ma so</td><td>TAI SAN</td>"
                "<td>31/12/2025</td><td>01/01/2025</td></tr>"
                + rows
                + "</table>"
            )
            content_list = output / f"{source.stem}_content_list.json"
            content_list.write_text(
                json.dumps(
                    [
                        {
                            "type": "table",
                            "table_body": table,
                            "bbox": [100, 760, 900, 960],
                            "page_idx": 23,
                        },
                        {
                            "type": "table",
                            "bbox": [100, 60, 900, 940],
                            "page_idx": 24,
                        },
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            pages = _load_mineru_pages(
                source,
                OCRConfig(enabled=True, force=True, mineru_output_dir=root / "mineru"),
            )

            self.assertTrue(any("1.000.000" in line for line in pages[24]))
            self.assertFalse(any("40.000.000" in line for line in pages[24]))
            self.assertTrue(any("40.000.000" in line for line in pages[25]))

    def test_mineru_citation_identifies_backend(self) -> None:
        chunk = EvidenceChunk(
            chunk_id="ABS:1-3",
            source_path="ABS.pdf",
            source_name="ABS.pdf",
            page_start=31,
            page_end=31,
            line_start=1,
            line_end=3,
            text="TỔNG CỘNG TÀI SẢN | 4.714.032.275.888",
            extraction_methods=("mineru",),
        )

        self.assertEqual(chunk.citation(), "ABS.pdf, page 31 [MinerU OCR]")

    def test_chunks_never_cross_pdf_page_boundaries(self) -> None:
        lines = [
            SourceLine("report.pdf", "report.pdf", 1, 1, "page one", "mineru"),
            SourceLine("report.pdf", "report.pdf", 1, 2, "page one row", "mineru"),
            SourceLine("report.pdf", "report.pdf", 2, 3, "page two", "mineru"),
            SourceLine("report.pdf", "report.pdf", 2, 4, "page two row", "mineru"),
        ]

        chunks = chunk_lines(lines, chunk_lines_count=4, overlap_lines=1)

        self.assertEqual(len(chunks), 2)
        self.assertTrue(all(chunk.page_start == chunk.page_end for chunk in chunks))

    def test_neighbor_expansion_does_not_ignore_later_ranked_hits(self) -> None:
        ranked: list[EvidenceChunk] = []
        all_chunks: list[EvidenceChunk] = []
        for index in range(5):
            hit = EvidenceChunk(
                chunk_id=f"source-{index}:1-2",
                source_path=f"source-{index}.pdf",
                source_name=f"source-{index}.pdf",
                page_start=index + 1,
                page_end=index + 1,
                line_start=1,
                line_end=2,
                text=f"metric section {index}",
                extraction_methods=("native",),
                score=float(10 - index),
            )
            ranked.append(hit)
            all_chunks.append(hit)
        later_neighbor = EvidenceChunk(
            chunk_id="source-4:3-4",
            source_path="source-4.pdf",
            source_name="source-4.pdf",
            page_start=5,
            page_end=5,
            line_start=3,
            line_end=4,
            text="adjacent financial values",
            extraction_methods=("native",),
        )
        all_chunks.append(later_neighbor)

        expanded = expand_neighbor_chunks(
            ranked,
            all_chunks,
            neighbor_window=1,
            max_chunks=6,
        )

        self.assertIn(later_neighbor.chunk_id, {item.chunk_id for item in expanded})

if __name__ == "__main__":
    unittest.main()
