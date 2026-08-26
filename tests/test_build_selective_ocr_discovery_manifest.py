from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_selective_ocr_discovery_manifest",
    ROOT / "scripts" / "build_selective_ocr_discovery_manifest.py",
)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class SelectiveOcrDiscoveryManifestTests(unittest.TestCase):
    def test_only_queues_missing_new_scan_groups_with_bounded_pages(self) -> None:
        companies = {
            "documents": {
                "scan.pdf": {
                    "company_key": "new-scan",
                    "source_sha256": "a" * 64,
                    "identity_source": "deterministic_filename_group",
                },
                "native.pdf": {
                    "company_key": "new-native",
                    "source_sha256": "b" * 64,
                    "identity_source": "deterministic_filename_group",
                },
                "known.pdf": {
                    "company_key": "known",
                    "source_sha256": "c" * 64,
                    "identity_source": "existing_development_manifest",
                },
            }
        }
        native = {
            "documents": [
                {
                    "document": "/workspace/pdf/scan.pdf",
                    "sha256": "a" * 64,
                    "total_pages": 30,
                    "ocr_candidate_pages": list(range(1, 31)),
                },
                {
                    "document": "/workspace/pdf/native.pdf",
                    "sha256": "b" * 64,
                    "total_pages": 20,
                    "ocr_candidate_pages": [],
                },
                {
                    "document": "/workspace/pdf/known.pdf",
                    "sha256": "c" * 64,
                    "total_pages": 20,
                    "ocr_candidate_pages": list(range(1, 21)),
                },
            ]
        }

        result = BUILDER.build_manifest(
            companies,
            native,
            eligible_companies=set(),
            new_groups_only=True,
            first_pages=4,
            uniform_pages=8,
            max_pages=12,
        )

        self.assertEqual(len(result["documents"]), 1)
        queued = result["documents"][0]
        self.assertEqual(Path(queued["document"]).name, "scan.pdf")
        self.assertLessEqual(len(queued["pages"]), 12)
        self.assertTrue(all(page["selection"] == "discovery" for page in queued["pages"]))
        self.assertEqual(result["metrics"]["native_followup_documents"], 1)
        self.assertEqual(result["native_followup"][0]["company_key"], "new-native")

    def test_skips_company_already_present_in_eligible_supervision(self) -> None:
        companies = {
            "documents": {
                "scan.pdf": {
                    "company_key": "new-scan",
                    "source_sha256": "a" * 64,
                    "identity_source": "deterministic_filename_group",
                }
            }
        }
        native = {
            "documents": [
                {
                    "document": "/workspace/pdf/scan.pdf",
                    "sha256": "a" * 64,
                    "total_pages": 10,
                    "ocr_candidate_pages": list(range(1, 11)),
                }
            ]
        }
        result = BUILDER.build_manifest(
            companies,
            native,
            eligible_companies={"new-scan"},
            new_groups_only=True,
            first_pages=4,
            uniform_pages=8,
            max_pages=12,
        )
        self.assertEqual(result["documents"], [])
        self.assertEqual(result["metrics"]["companies_already_eligible"], 1)

    def test_mixed_document_keeps_native_followup_and_bounded_ocr_queue(self) -> None:
        companies = {
            "documents": {
                "mixed.pdf": {
                    "company_key": "new-mixed",
                    "source_sha256": "d" * 64,
                    "identity_source": "deterministic_filename_group",
                }
            }
        }
        native = {
            "documents": [
                {
                    "document": "/workspace/pdf/mixed.pdf",
                    "sha256": "d" * 64,
                    "total_pages": 20,
                    "native_usable_pages": 8,
                    "ocr_candidate_pages": list(range(9, 21)),
                }
            ]
        }

        result = BUILDER.build_manifest(
            companies,
            native,
            eligible_companies=set(),
            new_groups_only=True,
            first_pages=4,
            uniform_pages=8,
            max_pages=12,
        )

        self.assertEqual(len(result["documents"]), 1)
        self.assertEqual(len(result["native_followup"]), 1)
        self.assertEqual(
            result["native_followup"][0]["reason"],
            "native_usable_pages_available",
        )

    def test_rejects_hash_mismatch(self) -> None:
        companies = {
            "documents": {
                "scan.pdf": {
                    "company_key": "new-scan",
                    "source_sha256": "a" * 64,
                    "identity_source": "deterministic_filename_group",
                }
            }
        }
        native = {
            "documents": [
                {
                    "document": "/workspace/pdf/scan.pdf",
                    "sha256": "b" * 64,
                    "total_pages": 10,
                    "ocr_candidate_pages": [1],
                }
            ]
        }
        with self.assertRaisesRegex(ValueError, "checksum"):
            BUILDER.build_manifest(
                companies,
                native,
                eligible_companies=set(),
                new_groups_only=True,
                first_pages=4,
                uniform_pages=8,
                max_pages=12,
            )


if __name__ == "__main__":
    unittest.main()
