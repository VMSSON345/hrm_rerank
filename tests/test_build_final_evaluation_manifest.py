from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_final_evaluation_manifest",
    ROOT / "scripts" / "build_final_evaluation_manifest.py",
)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class FinalEvaluationManifestTests(unittest.TestCase):
    def test_selection_is_company_disjoint_and_prefers_latest_report(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for ticker in ("AAA", "BBB", "CCC"):
                company = root / ticker
                company.mkdir()
                (company / f"{ticker}_24CN_BCTN.pdf").write_bytes(
                    f"{ticker}-2024".encode()
                )
                (company / f"{ticker}_25CN_BCTN.pdf").write_bytes(
                    f"{ticker}-2025".encode()
                )
            reference = root / "reference.jsonl"
            reference.write_text(
                json.dumps(
                    {
                        "context": "Mã cổ phiếu: CCC",
                        "candidates": [
                            {"source_name": "BBB_23CN_BCTN.pdf"}
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            manifest = BUILDER.build_manifest(
                root,
                reference_datasets=[reference],
                company_count=1,
                seed=17,
            )

        documents = manifest["documents"]
        self.assertEqual({item["ticker"] for item in documents}, {"AAA"})
        self.assertTrue(
            all(item["source_name"].endswith("_25CN_BCTN.pdf") for item in documents)
        )
        self.assertEqual(
            manifest["classification"],
            "external_blind_candidate_pending_independent_review",
        )
        self.assertFalse(manifest["selection_policy"]["model_outputs_used"])
        self.assertFalse(manifest["selection_policy"]["native_text_quality_used"])


if __name__ == "__main__":
    unittest.main()
