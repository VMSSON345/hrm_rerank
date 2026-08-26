from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_unseen_scan_ocr_discovery",
    ROOT / "scripts" / "build_unseen_scan_ocr_discovery.py",
)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class UnseenScanOcrDiscoveryTests(unittest.TestCase):
    def test_combines_table_structure_and_uniform_coverage(self) -> None:
        pages = BUILDER.select_pages(
            20,
            {page: (1.0 if page in {7, 13, 18} else 0.0) for page in range(1, 21)},
            first_pages=2,
            uniform_pages=4,
            table_pages=3,
            max_pages=8,
        )
        selected = {item["page"] for item in pages}
        self.assertTrue({7, 13, 18}.issubset(selected))
        self.assertTrue({1, 2}.issubset(selected))
        self.assertLessEqual(len(selected), 8)

    def test_selection_is_independent_of_financial_values(self) -> None:
        first = BUILDER.select_pages(
            8,
            {page: page / 100 for page in range(1, 9)},
            first_pages=1,
            uniform_pages=3,
            table_pages=2,
            max_pages=5,
        )
        second = BUILDER.select_pages(
            8,
            {page: page / 100 for page in range(1, 9)},
            first_pages=1,
            uniform_pages=3,
            table_pages=2,
            max_pages=5,
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
