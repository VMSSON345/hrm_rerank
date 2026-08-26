import json
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

from scripts.audit_pdf_corpus import (
    documents_from_company_manifest,
    documents_from_evaluation_manifest,
    summarize_native_pages,
)


class NativePdfCorpusAuditTests(unittest.TestCase):
    def test_evaluation_manifest_loader_preserves_company_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected = root / "AAA_25CN_BCTN.pdf"
            selected.write_bytes(b"selected-final")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "protocol": (
                            "document-hrm-final-evaluation-pdf-manifest-v1"
                        ),
                        "documents": [
                            {
                                "document": str(selected),
                                "source_sha256": sha256(
                                    b"selected-final"
                                ).hexdigest(),
                                "company_key": "final-eval-company-aaa",
                                "review_status": "pending_independent_review",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            documents = documents_from_evaluation_manifest(manifest)

            self.assertEqual(
                documents,
                {
                    selected.resolve(): {
                        "company_key": "final-eval-company-aaa",
                        "review_status": "pending_independent_review",
                    }
                },
            )

    def test_manifest_loader_returns_only_hash_locked_documents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected = root / "AAA_23CN_BCTN.pdf"
            unselected = root / "BBB_23CN_BCTN.pdf"
            selected.write_bytes(b"selected")
            unselected.write_bytes(b"unselected")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "protocol": "pdf-development-company-manifest-v1",
                        "documents": {
                            selected.name: {
                                "company_key": "development-company-aaa",
                                "source_sha256": sha256(b"selected").hexdigest(),
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            documents = documents_from_company_manifest(manifest, root)

            self.assertEqual(documents, [selected.resolve()])

    def test_manifest_loader_rejects_changed_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected = root / "AAA_23CN_BCTN.pdf"
            selected.write_bytes(b"changed")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "protocol": "pdf-development-company-manifest-v1",
                        "documents": {
                            selected.name: {
                                "company_key": "development-company-aaa",
                                "source_sha256": "0" * 64,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "checksum"):
                documents_from_company_manifest(manifest, root)

    def test_reports_mixed_native_and_ocr_candidate_pages(self) -> None:
        report = summarize_native_pages(
            [
                "BÁO CÁO TÀI CHÍNH\nDoanh thu thuần | 100.000.000 | 90.000.000",
                "",
                "Trang 3",
            ],
            min_native_chars=40,
        )

        self.assertEqual(report["classification"], "mixed")
        self.assertEqual(report["native_usable_pages"], 1)
        self.assertEqual(report["native_unusable_pages"], 2)
        self.assertEqual(report["ocr_candidate_pages"], [2, 3])
        self.assertEqual(
            report["reason_counts"], {"empty": 1, "too_short": 1, "usable": 1}
        )

    def test_all_empty_pages_are_scan_or_unusable_native(self) -> None:
        report = summarize_native_pages(
            ["", ""],
            min_native_chars=40,
        )

        self.assertEqual(report["classification"], "scan_or_unusable_native")
        self.assertEqual(report["native_usable_rate"], 0.0)
        self.assertEqual(report["ocr_candidate_pages"], [1, 2])


if __name__ == "__main__":
    unittest.main()
