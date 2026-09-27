from pathlib import Path
from tempfile import TemporaryDirectory
import json
from unittest import TestCase

from hrm_finance_qa.document_sources import OCRConfig, _load_mineru_pages


class MinerUUnitHeaderOrderTests(TestCase):
    def pages(self, payload):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "REPORT_content_list.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf8"
            )
            return _load_mineru_pages(
                root / "REPORT.pdf",
                OCRConfig(mineru_output_dir=root),
            )

    def test_physical_unit_header_precedes_table_even_if_emitted_last(self):
        table = {
            "type": "table",
            "page_idx": 0,
            "bbox": [100, 100, 900, 250],
            "table_body": "<table><tr><td>Chỉ tiêu</td><td>2023</td></tr><tr><td>Doanh thu</td><td>12</td></tr></table>",
        }
        pages = self.pages([
            table,
            {"type": "header", "page_idx": 0, "bbox": [750, 70, 850, 85], "text": "DVT:VND"},
            {"type": "header", "page_idx": 0, "bbox": [50, 10, 500, 30], "text": "Annual Report 2023"},
        ])
        self.assertEqual(pages[1][0], "DVT:VND")
        self.assertTrue(any("Doanh thu" in line for line in pages[1][1:]))

    def test_unrelated_or_below_table_header_is_not_moved(self):
        table = {
            "type": "table",
            "page_idx": 0,
            "bbox": [100, 100, 900, 250],
            "table_body": "<table><tr><td>Chỉ tiêu</td><td>2023</td></tr><tr><td>Doanh thu</td><td>12</td></tr></table>",
        }
        pages = self.pages([
            table,
            {"type": "header", "page_idx": 0, "bbox": [750, 300, 850, 315], "text": "DVT:VND"},
        ])
        self.assertNotEqual(pages[1][0], "DVT:VND")
        self.assertEqual(pages[1][-1], "DVT:VND")
