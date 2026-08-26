from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_pdf_company_manifest",
    ROOT / "scripts" / "build_pdf_company_manifest.py",
)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class PdfCompanyManifestTests(unittest.TestCase):
    def test_reuses_frozen_exclusions_without_protected_source_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = root / "frozen.json"
            frozen.write_text(
                json.dumps(
                    {
                        "protocol": "pdf-development-company-manifest-v1",
                        "excluded_company_identities": ["ABC", "xyz"],
                        "documents": {},
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                BUILDER.frozen_excluded_identities([frozen]),
                {"abc", "xyz"},
            )

    def test_loads_protected_test_and_blind_manifest_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protected = root / "protected.json"
            protected.write_text(
                json.dumps(
                    {
                        "protocol": "document-hrm-final-evaluation-pdf-manifest-v1",
                        "documents": [
                            {
                                "ticker": "ABC",
                                "source_name": "ABC_25CN_BCTN.pdf",
                            },
                            {"source_name": "XYZ_24CN_BCTN.pdf"},
                        ],
                        "excluded_tickers": ["HLD"],
                    }
                ),
                encoding="utf-8",
            )

            identities = BUILDER.protected_company_identities([protected])

            self.assertEqual(identities, {"abc", "xyz", "hld"})

    def test_reads_only_source_names_from_development_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "development.jsonl"
            metadata.write_text(
                json.dumps(
                    {
                        "question": "must not affect selection",
                        "answer_value": 123,
                        "candidates": [
                            {"source_name": "AAA_23CN_BCTN.pdf", "value": 999},
                            {"source_name": "AAA_23CN_BCTN.pdf", "value": 111},
                            {"source_name": "BBB_24CN_BCTN.pdf", "value": 222},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            identities = BUILDER.development_metadata_identities([metadata])

            self.assertEqual(identities, {"aaa", "bbb"})

    def test_excludes_protected_company_across_report_years(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protected_2024 = root / "ABT_24CN_BCTN.pdf"
            safe_2023 = root / "XYZ_23CN_BCTN.pdf"
            protected_2024.write_bytes(b"protected")
            safe_2023.write_bytes(b"safe")
            split = root / "split.json"
            split.write_text(
                json.dumps(
                    {
                        "schema": "financial-pdf-corpus-split-v1",
                        "excluded": [{"file": "ACE_23CN_BCTN.pdf"}],
                        "calibration": [{"file": "ABT_23CN_BCTN.pdf"}],
                        "holdout_v2": [{"file": "ADG_23CN_BCTN.pdf"}],
                        "development": [{"file": "XYZ_23CN_BCTN.pdf"}],
                    }
                ),
                encoding="utf-8",
            )

            protected = BUILDER.protected_company_identities([split])
            result = BUILDER.build_manifest(
                [protected_2024, safe_2023],
                existing_manifests=[],
                excluded_names=set(),
                excluded_company_identities=protected,
            )

            self.assertEqual(set(result["documents"]), {safe_2023.name})
            self.assertIn("abt", result["excluded_company_identities"])
            self.assertNotIn("xyz", result["excluded_company_identities"])

    def test_selects_hash_ordered_new_company_cohort_before_file_hashing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            for ticker in ("AAA", "BBB", "CCC", "DDD"):
                path = root / f"{ticker}_23CN_BCTN.pdf"
                path.write_bytes(ticker.encode("ascii"))
                sources.append(path)
            prior = root / "prior.json"
            prior.write_text(
                json.dumps(
                    {
                        "protocol": "pdf-development-company-manifest-v1",
                        "documents": {
                            "AAA_22CN_BCTN.pdf": {
                                "company_key": "development-company-aaa",
                                "source_sha256": "0" * 64,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = BUILDER.build_manifest(
                list(reversed(sources)),
                existing_manifests=[prior],
                excluded_names=set(),
                new_companies_only=True,
                max_company_groups=2,
            )

            expected = sorted(
                ("bbb", "ccc", "ddd"),
                key=lambda identity: (
                    sha256(identity.encode("utf-8")).hexdigest(),
                    identity,
                ),
            )[:2]
            actual = sorted(
                BUILDER._company_identity(name) for name in result["documents"]
            )
            self.assertEqual(actual, sorted(expected))
            self.assertEqual(result["metrics"]["eligible_company_groups"], 3)
            self.assertEqual(result["metrics"]["company_groups"], 2)
            self.assertEqual(
                result["selection_policy"]["ordering"],
                "sha256(company_identity), then company_identity",
            )

    def test_selects_next_hash_ordered_cohort_with_offset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            for ticker in ("AAA", "BBB", "CCC", "DDD"):
                path = root / f"{ticker}_23CN_BCTN.pdf"
                path.write_bytes(ticker.encode("ascii"))
                sources.append(path)

            result = BUILDER.build_manifest(
                sources,
                existing_manifests=[],
                excluded_names=set(),
                company_group_offset=2,
                max_company_groups=2,
            )

            ordered = sorted(
                ("aaa", "bbb", "ccc", "ddd"),
                key=lambda identity: (
                    sha256(identity.encode("utf-8")).hexdigest(),
                    identity,
                ),
            )
            self.assertEqual(
                sorted(BUILDER._company_identity(name) for name in result["documents"]),
                sorted(ordered[2:4]),
            )
            self.assertEqual(result["selection_policy"]["company_group_offset"], 2)

    def test_reuses_company_key_across_report_years(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prior_pdf = root / "ABC_23CN_BCTN.pdf"
            current_pdf = root / "ABC_24CN_BCTN.pdf"
            new_pdf = root / "XYZ_24CN_BCTN.pdf"
            prior_pdf.write_bytes(b"prior")
            current_pdf.write_bytes(b"current")
            new_pdf.write_bytes(b"new")
            prior_manifest = root / "prior.json"
            prior_manifest.write_text(
                json.dumps(
                    {
                        "protocol": "pdf-development-company-manifest-v1",
                        "documents": {
                            prior_pdf.name: {
                                "company_key": "development-company-abc",
                                "source_sha256": sha256(b"prior").hexdigest(),
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = BUILDER.build_manifest(
                [current_pdf, new_pdf],
                existing_manifests=[prior_manifest],
                excluded_names=set(),
            )

            self.assertEqual(
                result["documents"][current_pdf.name]["company_key"],
                "development-company-abc",
            )
            self.assertNotEqual(
                result["documents"][new_pdf.name]["company_key"],
                "development-company-abc",
            )
            self.assertEqual(
                result["documents"][current_pdf.name]["source_sha256"],
                sha256(b"current").hexdigest(),
            )

    def test_excludes_named_regression_document_without_hardcoding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            keep = root / "NEW_24CN_BCTN.pdf"
            excluded = root / "REG_24CN_BCTN.pdf"
            keep.write_bytes(b"keep")
            excluded.write_bytes(b"excluded")

            result = BUILDER.build_manifest(
                [keep, excluded],
                existing_manifests=[],
                excluded_names={excluded.name},
            )

            self.assertEqual(set(result["documents"]), {keep.name})
            self.assertEqual(result["excluded_documents"], [excluded.name])

    def test_rejects_conflicting_existing_company_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = []
            for year, key in (("23", "company-one"), ("22", "company-two")):
                name = f"ABC_{year}CN_BCTN.pdf"
                path = root / f"manifest-{year}.json"
                path.write_text(
                    json.dumps(
                        {
                            "protocol": "pdf-development-company-manifest-v1",
                            "documents": {
                                name: {
                                    "company_key": key,
                                    "source_sha256": "0" * 64,
                                }
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                manifests.append(path)

            with self.assertRaisesRegex(ValueError, "Conflicting company keys"):
                BUILDER.build_manifest(
                    [root / "ABC_24CN_BCTN.pdf"],
                    existing_manifests=manifests,
                    excluded_names=set(),
                )


if __name__ == "__main__":
    unittest.main()
