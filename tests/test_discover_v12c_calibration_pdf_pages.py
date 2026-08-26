from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import fitz

from scripts.discover_v12c_calibration_pdf_pages import discover


class DiscoverV12CCalibrationPagesTests(unittest.TestCase):
    def test_excludes_repair_ticker_before_independent_discovery(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_root = root / "pdf"
            pdf_root.mkdir()
            excluded_pdf = self._pdf(pdf_root / "OLD_24CN_BCTN.pdf")
            independent_pdf = self._pdf(pdf_root / "NEW_24CN_BCTN.pdf")
            cohort = root / "development-repair.jsonl"
            cohort.write_text(json.dumps({
                "classification": "development_only_visual_qa_approved",
                "report_id": "OLD_23CN_BCTN.pdf",
            }) + "\n", encoding="utf-8")
            catalog = root / "development-catalog.json"
            catalog.write_text(json.dumps({
                "classification": "development_only",
                "documents": {
                    excluded_pdf.name: {
                        "company_key": "old-company",
                        "source_sha256": self._hash(excluded_pdf),
                    },
                    independent_pdf.name: {
                        "company_key": "new-company",
                        "source_sha256": self._hash(independent_pdf),
                    },
                },
            }), encoding="utf-8")

            report = discover(catalog, cohort, pdf_root)

            self.assertEqual(report["excluded_tickers"], ["OLD"])
            self.assertEqual(report["counts"]["excluded_same_company_documents"], 1)
            self.assertEqual(report["counts"]["candidate_pages"], 1)
            self.assertEqual(report["records"][0]["source_name"], independent_pdf.name)
            self.assertFalse(report["gate"]["ready_for_calibration"])

    @staticmethod
    def _pdf(path: Path) -> Path:
        document = fitz.open()
        page = document.new_page()
        page.insert_text(
            (72, 72),
            "BANG CAN DOI KE TOAN\nVay ngan han 100\n"
            "Vay dai han 20\nTong nguon von 1000\n"
            "Don vi VND So cuoi nam So dau nam",
        )
        document.save(path)
        document.close()
        return path

    @staticmethod
    def _hash(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
