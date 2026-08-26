from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_native_subtract_page_manifest",
    ROOT / "scripts" / "build_native_subtract_page_manifest.py",
)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class NativeSubtractPageManifestTests(unittest.TestCase):
    def test_layout_conversion_preserves_financial_header_and_row(self) -> None:
        text = """Chỉ tiêu                 Năm 2024       Năm 2023
Tổng tài sản             1.200.000      1.000.000
Thuyết minh thông thường.
"""
        converted = BUILDER.layout_to_table_context(text)

        self.assertIn("| Chỉ tiêu | Năm 2024 | Năm 2023 |", converted)
        self.assertIn("| Tổng tài sản | 1.200.000 | 1.000.000 |", converted)
        self.assertIn("Thuyết minh thông thường.", converted)

    def test_selects_only_explicit_native_layout_families(self) -> None:
        selected = BUILDER._selected_source_names(
            {"A.pdf", "B.pdf", "C.pdf"},
            {"C.pdf"},
            {
                "A.pdf": {"layout_family": "native_text_portrait"},
                "B.pdf": {"layout_family": "scanned_image_dominant"},
                "C.pdf": {"layout_family": "native_text_mixed_orientation"},
            },
            {"native_text_portrait", "native_text_mixed_orientation"},
        )

        self.assertEqual(selected, ["A.pdf"])

if __name__ == "__main__":
    unittest.main()
