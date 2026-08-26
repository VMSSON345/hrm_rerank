from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "combine_pdf_company_manifests",
    ROOT / "scripts" / "combine_pdf_company_manifests.py",
)
assert SPEC is not None and SPEC.loader is not None
COMBINER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMBINER)


class CombinePdfCompanyManifestsTests(unittest.TestCase):
    def test_combines_same_role_manifests_and_reuses_cross_year_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for year in ("23", "24"):
                path = root / f"{year}.json"
                path.write_text(
                    json.dumps(
                        {
                            "protocol": "pdf-development-company-manifest-v1",
                            "classification": "development_only",
                            "documents": {
                                f"ABC_{year}CN_BCTN.pdf": {
                                    "company_key": "company-abc",
                                    "source_sha256": year[0] * 64,
                                }
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                paths.append(path)

            result = COMBINER.combine_manifests(paths)

            self.assertEqual(result["metrics"], {"documents": 2, "company_groups": 1})

    def test_rejects_conflicting_cross_year_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for year, key in (("23", "one"), ("24", "two")):
                path = root / f"{year}.json"
                path.write_text(
                    json.dumps(
                        {
                            "protocol": "pdf-development-company-manifest-v1",
                            "classification": "development_only",
                            "documents": {
                                f"ABC_{year}CN_BCTN.pdf": {
                                    "company_key": key,
                                    "source_sha256": year[0] * 64,
                                }
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                paths.append(path)

            with self.assertRaisesRegex(ValueError, "Conflicting company key"):
                COMBINER.combine_manifests(paths)


if __name__ == "__main__":
    unittest.main()
