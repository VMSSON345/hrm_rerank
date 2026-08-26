from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_native_supervision_discovery_manifest",
    ROOT / "scripts" / "build_native_supervision_discovery_manifest.py",
)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class NativeSupervisionDiscoveryManifestTests(unittest.TestCase):
    def test_selects_only_structurally_strong_native_followup_pages(self) -> None:
        ocr_manifest = {
            "native_followup": [
                {
                    "document": "/workspace/pdf/native.pdf",
                    "company_key": "new-native",
                }
            ]
        }
        financial = {
            "documents": [
                {
                    "document": "/workspace/pdf/native.pdf",
                    "sha256": "a" * 64,
                    "total_pages": 10,
                    "pages": [
                        {"page": 2, "metric_hits": ["a", "b", "c"], "numeric_values": 5},
                        {"page": 3, "metric_hits": ["a", "b"], "numeric_values": 20},
                    ],
                }
            ]
        }
        result = BUILDER.build_manifest(
            ocr_manifest,
            financial,
            minimum_metric_hits=3,
            minimum_numeric_values=5,
            maximum_pages_per_document=4,
        )
        self.assertEqual(len(result["documents"]), 1)
        self.assertEqual(result["documents"][0]["pages"][0]["page"], 2)
        self.assertEqual(result["documents"][0]["pages"][0]["selection"], "core")


if __name__ == "__main__":
    unittest.main()
