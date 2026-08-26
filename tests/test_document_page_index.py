from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hrm_finance_qa.document_page_index import build_page_indexed_chunks
from hrm_finance_qa.document_sources import DocumentSourceError, OCRConfig


class DocumentPageIndexTests(unittest.TestCase):
    class _Page:
        def __init__(self, text: str) -> None:
            self.text = text

        def extract_text(self) -> str:
            return self.text

    @classmethod
    def _pypdf_module(cls, texts: list[str], calls: list[str]) -> SimpleNamespace:
        def reader(path: str) -> SimpleNamespace:
            calls.append(path)
            return SimpleNamespace(pages=[cls._Page(text) for text in texts])

        return SimpleNamespace(PdfReader=reader)

    def test_native_pdf_is_indexed_once_without_ocr(self) -> None:
        pages = [
            "BÁO CÁO TÀI CHÍNH NĂM 2025\nTổng tài sản cuối năm là 1.000.000.000 đồng",
            "BÁO CÁO KẾT QUẢ KINH DOANH\nDoanh thu thuần năm 2025 là 900.000.000 đồng",
        ]
        reader_calls: list[str] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "native.pdf"
            source.write_bytes(b"pdf-placeholder")
            index_path = root / "page-index.sqlite3"
            with patch.dict(
                "sys.modules", {"pypdf": self._pypdf_module(pages, reader_calls)}
            ):
                with patch(
                    "hrm_finance_qa.document_page_index._load_or_generate_mineru_pages"
                ) as run_mineru:
                    cold = build_page_indexed_chunks(
                        [source],
                        "Tổng tài sản cuối năm 2025",
                        index_path=index_path,
                        ocr_config=OCRConfig(enabled=True),
                        top_k_pages=1,
                        neighbor_pages=0,
                    )
                    warm = build_page_indexed_chunks(
                        [source],
                        "Doanh thu thuần năm 2025",
                        index_path=index_path,
                        ocr_config=OCRConfig(enabled=True),
                        top_k_pages=1,
                        neighbor_pages=0,
                    )

            run_mineru.assert_not_called()
            self.assertEqual(len(reader_calls), 1)
            self.assertEqual(cold.stats.native_pages, 2)
            self.assertEqual(cold.stats.documents_rebuilt, 1)
            self.assertEqual(warm.stats.documents_rebuilt, 0)
            self.assertGreaterEqual(warm.stats.cache_hits, 2)
            self.assertEqual(warm.stats.pages_selected, 1)

    def test_scan_materializes_only_ranked_fallback_page_then_reuses_it(self) -> None:
        pages = ["Trang 1", "Trang 2", "Trang 3", "Trang 4"]
        routing_text = {
            1: ["Mục lục báo cáo tài chính"],
            2: ["Doanh thu thuần 900.000.000"],
            3: ["Tổng nợ phải trả 700.000.000"],
            4: ["Vốn chủ sở hữu 300.000.000"],
        }
        calls: list[list[bool]] = []

        def run_mineru(_path, selected, _config):
            mask = list(selected)
            calls.append(mask)
            return {
                page: routing_text[page]
                for page, chosen in enumerate(mask, start=1)
                if chosen
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "scan.pdf"
            source.write_bytes(b"scan-placeholder")
            with patch.dict(
                "sys.modules", {"pypdf": self._pypdf_module(pages, [])}
            ):
                with patch(
                    "hrm_finance_qa.document_page_index._load_or_generate_mineru_pages",
                    side_effect=run_mineru,
                ):
                    cold = build_page_indexed_chunks(
                        [source],
                        "Tổng nợ phải trả",
                        index_path=root / "page-index.sqlite3",
                        ocr_config=OCRConfig(enabled=True),
                        top_k_pages=1,
                        neighbor_pages=0,
                    )
                    warm = build_page_indexed_chunks(
                        [source],
                        "Tổng nợ phải trả",
                        index_path=root / "page-index.sqlite3",
                        ocr_config=OCRConfig(enabled=True),
                        top_k_pages=1,
                        neighbor_pages=0,
                    )

        self.assertEqual(calls, [[True, False, False, False]])
        self.assertEqual(cold.stats.mineru_pages, 0)
        self.assertEqual(cold.stats.selected_mineru_pages, 1)
        self.assertEqual(cold.stats.pages_selected, 1)
        self.assertEqual(warm.stats.selected_mineru_pages, 0)
        self.assertGreaterEqual(warm.stats.cache_hits, 5)
        self.assertTrue(all(chunk.page_start == 1 for chunk in warm.chunks))
        self.assertEqual(warm.ranked_pages, (1,))

    def test_source_fingerprint_change_invalidates_cached_pages(self) -> None:
        pages = [
            "BÁO CÁO TÀI CHÍNH NĂM 2025\nTổng tài sản cuối năm là 1.000.000.000 đồng"
        ]
        reader_calls: list[str] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "changed.pdf"
            source.write_bytes(b"version-one")
            with patch.dict(
                "sys.modules", {"pypdf": self._pypdf_module(pages, reader_calls)}
            ):
                first = build_page_indexed_chunks(
                    [source],
                    "Tổng tài sản",
                    index_path=root / "page-index.sqlite3",
                    ocr_config=OCRConfig(enabled=False),
                    top_k_pages=1,
                    neighbor_pages=0,
                )
                source.write_bytes(b"version-two-is-different")
                second = build_page_indexed_chunks(
                    [source],
                    "Tổng tài sản",
                    index_path=root / "page-index.sqlite3",
                    ocr_config=OCRConfig(enabled=False),
                    top_k_pages=1,
                    neighbor_pages=0,
                )

        self.assertEqual(first.stats.documents_rebuilt, 1)
        self.assertEqual(second.stats.documents_rebuilt, 1)
        self.assertEqual(len(reader_calls), 2)

    def test_force_ocr_preserves_all_page_semantics(self) -> None:
        pages = [
            "BÁO CÁO TÀI CHÍNH NĂM 2025\nTổng tài sản cuối năm là 1.000.000.000 đồng",
            "BÁO CÁO KẾT QUẢ KINH DOANH\nDoanh thu thuần năm 2025 là 900.000.000 đồng",
        ]
        masks: list[list[bool]] = []

        def run_mineru(_path, selected, _config):
            mask = list(selected)
            masks.append(mask)
            return {
                page: [f"OCR page {page} | {page}.000.000"]
                for page, chosen in enumerate(mask, start=1)
                if chosen
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "force.pdf"
            source.write_bytes(b"force-placeholder")
            with patch.dict(
                "sys.modules", {"pypdf": self._pypdf_module(pages, [])}
            ):
                with patch(
                    "hrm_finance_qa.document_page_index._load_or_generate_mineru_pages",
                    side_effect=run_mineru,
                ):
                    result = build_page_indexed_chunks(
                        [source],
                        "Tổng tài sản",
                        index_path=root / "page-index.sqlite3",
                        ocr_config=OCRConfig(enabled=True, force=True),
                        top_k_pages=1,
                        neighbor_pages=0,
                    )

        self.assertEqual(masks, [[True, True]])
        self.assertEqual(result.stats.selected_mineru_pages, 2)
        self.assertEqual(result.stats.pages_selected, 2)

    def test_no_ocr_does_not_use_cached_ocr_text_for_ranking_or_evidence(self) -> None:
        pages = [
            "Trang 1",
            "BÁO CÁO TÀI CHÍNH NĂM 2025\nTổng tài sản cuối năm là 1.000.000.000 đồng",
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "mixed.pdf"
            source.write_bytes(b"mixed-placeholder")
            module = self._pypdf_module(pages, [])
            with patch.dict("sys.modules", {"pypdf": module}):
                with patch(
                    "hrm_finance_qa.document_page_index._load_or_generate_mineru_pages",
                    return_value={1: ["Bí mật OCR chỉ có ở cache 9.999"]},
                ):
                    build_page_indexed_chunks(
                        [source],
                        "Bí mật OCR",
                        index_path=root / "page-index.sqlite3",
                        ocr_config=OCRConfig(enabled=True),
                        top_k_pages=1,
                        neighbor_pages=0,
                    )
                    native_only = build_page_indexed_chunks(
                        [source],
                        "Tổng tài sản",
                        index_path=root / "page-index.sqlite3",
                        ocr_config=OCRConfig(enabled=False),
                        top_k_pages=1,
                        neighbor_pages=0,
                    )

        text = "\n".join(chunk.text for chunk in native_only.chunks)
        self.assertIn("Tổng tài sản", text)
        self.assertNotIn("Bí mật OCR", text)
        self.assertTrue(
            all("ocr" not in chunk.extraction_methods for chunk in native_only.chunks)
        )

    def test_selective_mineru_override_reconstructs_only_ranked_native_page(self) -> None:
        pages = [
            "Company overview and governance information with enough clean native text.",
            "Revenue table native extraction has vertical columns and value fragments.",
        ]
        masks: list[list[bool]] = []

        def run_mineru(_path, selected, config):
            mask = list(selected)
            masks.append(mask)
            self.assertTrue(config.mineru_auto_generate)
            return {
                2: [
                    "ROW: Net revenue | COLUMN: Year 2024 | "
                    "CELL: Net revenue | Year 2024: 284331609643"
                ]
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "vertical-table.pdf"
            source.write_bytes(b"vertical-table-placeholder")
            index_path = root / "page-index.sqlite3"
            with patch.dict(
                "sys.modules", {"pypdf": self._pypdf_module(pages, [])}
            ), patch(
                "hrm_finance_qa.document_page_index._load_or_generate_mineru_pages",
                side_effect=run_mineru,
            ):
                cold = build_page_indexed_chunks(
                    [source],
                    "Revenue table value fragments",
                    index_path=index_path,
                    ocr_config=OCRConfig(
                        enabled=True, mineru_auto_generate=True
                    ),
                    top_k_pages=1,
                    neighbor_pages=0,
                    prefer_mineru_for_selected_pages=True,
                )
                warm = build_page_indexed_chunks(
                    [source],
                    "Net revenue Year 2024",
                    index_path=index_path,
                    ocr_config=OCRConfig(
                        enabled=True, mineru_auto_generate=False
                    ),
                    top_k_pages=1,
                    neighbor_pages=0,
                    prefer_mineru_for_selected_pages=True,
                )

        self.assertEqual(masks, [[False, True]])
        self.assertEqual(cold.stats.selected_mineru_pages, 1)
        self.assertEqual(warm.stats.selected_mineru_pages, 0)
        self.assertGreaterEqual(warm.stats.cache_hits, 3)
        self.assertIn(
            "284331609643", "\n".join(chunk.text for chunk in warm.chunks)
        )
        self.assertTrue(
            all("mineru" in chunk.extraction_methods for chunk in warm.chunks)
        )

    def test_page_index_build_defers_ocr_until_after_question_page_ranking(self) -> None:
        pages = [
            "scan",
            "Revenue table and current year value 284331609643 with enough native text.",
        ]
        masks: list[list[bool]] = []

        def run_mineru(_path, selected, _config):
            masks.append(list(selected))
            return {1: ["OCR-only unrelated page"]}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "cold-index.pdf"
            source.write_bytes(b"cold-index-placeholder")
            with patch.dict(
                "sys.modules", {"pypdf": self._pypdf_module(pages, [])}
            ), patch(
                "hrm_finance_qa.document_page_index._load_or_generate_mineru_pages",
                side_effect=run_mineru,
            ):
                result = build_page_indexed_chunks(
                    [source],
                    "Revenue table current year value",
                    index_path=root / "page-index.sqlite3",
                    ocr_config=OCRConfig(enabled=True),
                    top_k_pages=1,
                    neighbor_pages=0,
                )

        self.assertEqual(masks, [])
        self.assertEqual(result.ranked_pages, (2,))
        self.assertIn(
            "284331609643", "\n".join(chunk.text for chunk in result.chunks)
        )

    def test_selected_mineru_materialization_does_not_promote_disabled_generation(self) -> None:
        pages = [
            "Revenue table native extraction remains available with value 100000000."
        ]
        observed_auto_generate: list[bool] = []

        def unavailable(_path, _selected, config):
            observed_auto_generate.append(config.mineru_auto_generate)
            raise DocumentSourceError("cache unavailable")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "disabled-generation.pdf"
            source.write_bytes(b"disabled-generation-placeholder")
            with patch.dict(
                "sys.modules", {"pypdf": self._pypdf_module(pages, [])}
            ), patch(
                "hrm_finance_qa.document_page_index._load_or_generate_mineru_pages",
                side_effect=unavailable,
            ):
                result = build_page_indexed_chunks(
                    [source],
                    "Revenue table",
                    index_path=root / "page-index.sqlite3",
                    ocr_config=OCRConfig(
                        enabled=True, mineru_auto_generate=False
                    ),
                    top_k_pages=1,
                    neighbor_pages=0,
                    prefer_mineru_for_selected_pages=True,
                )

        self.assertEqual(observed_auto_generate, [False])
        self.assertIn(
            "100000000", "\n".join(chunk.text for chunk in result.chunks)
        )
        self.assertTrue(
            any(
                "Selective MinerU override was unavailable" in warning
                for warning in result.warnings
            )
        )

    def test_selective_mineru_override_retains_native_when_api_unavailable(self) -> None:
        pages = [
            "Revenue table native extraction remains available with value 100000000."
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "native-fallback.pdf"
            source.write_bytes(b"native-fallback-placeholder")
            with patch.dict(
                "sys.modules", {"pypdf": self._pypdf_module(pages, [])}
            ), patch(
                "hrm_finance_qa.document_page_index._load_or_generate_mineru_pages",
                side_effect=DocumentSourceError("API unavailable"),
            ):
                result = build_page_indexed_chunks(
                    [source],
                    "Revenue table",
                    index_path=root / "page-index.sqlite3",
                    ocr_config=OCRConfig(enabled=True),
                    top_k_pages=1,
                    neighbor_pages=0,
                    prefer_mineru_for_selected_pages=True,
                )

        self.assertEqual(result.stats.selected_mineru_pages, 0)
        self.assertIn(
            "100000000", "\n".join(chunk.text for chunk in result.chunks)
        )
        self.assertTrue(
            any(
                "Selective MinerU override was unavailable" in warning
                for warning in result.warnings
            )
        )


if __name__ == "__main__":
    unittest.main()
