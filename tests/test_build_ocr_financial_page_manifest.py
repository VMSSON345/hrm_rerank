from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_ocr_financial_page_manifest",
    ROOT / "scripts" / "build_ocr_financial_page_manifest.py",
)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class OcrFinancialPageManifestTests(unittest.TestCase):
    def test_scores_only_hash_locked_discovery_pages(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scan.md").write_text(
                "## Page 2\n\nBao cao ket qua hoat dong kinh doanh\n"
                "Don vi tinh: VND\nChi tieu | Nam nay | Nam truoc\n"
                "--- | --- | ---\n"
                "Doanh thu thuan | 10.000.000 | 9.000.000\n"
                "Loi nhuan sau thue | 2.000.000 | 1.000.000\n",
                encoding="utf-8",
            )
            discovery = {
                "documents": [
                    {
                        "document": "/workspace/pdf/scan.pdf",
                        "sha256": "a" * 64,
                        "company_key": "company-a",
                        "total_pages": 10,
                        "pages": [{"page": 2, "selection": "discovery"}],
                    }
                ]
            }
            result = BUILDER.build_manifest(discovery, root)
            self.assertTrue(result["ocr_performed"])
            self.assertEqual(result["protocol"], "ocr-financial-page-manifest-v2")
            self.assertIsNone(result["selection_threshold"])
            self.assertEqual(result["metrics"]["core_financial_pages"], 1)
            self.assertEqual(result["metrics"]["documents_with_financial_summary"], 1)
            page = result["documents"][0]["pages"][0]
            self.assertEqual(page["page"], 2)
            self.assertEqual(page["selection"], "core")


if __name__ == "__main__":
    unittest.main()
