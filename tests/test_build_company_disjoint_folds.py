from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_company_disjoint_folds",
    ROOT / "scripts" / "build_company_disjoint_folds.py",
)
assert SPEC is not None and SPEC.loader is not None
FOLDS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FOLDS)

CACHE_SPEC = importlib.util.spec_from_file_location(
    "build_v12b2_fold_caches",
    ROOT / "scripts" / "build_v12b2_fold_caches.py",
)
assert CACHE_SPEC is not None and CACHE_SPEC.loader is not None
FOLD_CACHE = importlib.util.module_from_spec(CACHE_SPEC)
CACHE_SPEC.loader.exec_module(FOLD_CACHE)

from hrm_finance_qa.program_data import (  # noqa: E402
    NumericCandidate,
    ProgramExample,
    ReasoningProgram,
)


OPERATIONS = (
    "SELECT",
    "SUBTRACT",
    "RATIO_PERCENT",
    "ADD",
    "DIVIDE",
    "PERCENT_CHANGE",
)


def _rows(company_count: int) -> list[dict[str, object]]:
    return [
        {
            "example_id": f"company-{company:02d}-{operation}",
            "program": {"operation": operation, "operands": [0]},
            "supervision": {"company_key": f"company-{company:02d}"},
        }
        for company in range(company_count)
        for operation in OPERATIONS
    ]


class CompanyFoldTests(unittest.TestCase):
    def test_development_only_fold_does_not_declare_sealed_validation(self) -> None:
        paths = FOLD_CACHE._declared_source_paths(
            {
                "input": "development.jsonl",
                "sealed_validation": "validation.jsonl",
                "sealed_validation_loaded": False,
            }
        )

        self.assertEqual(paths, (Path("development.jsonl"),))

    def test_legacy_fold_keeps_both_declared_sources(self) -> None:
        paths = FOLD_CACHE._declared_source_paths(
            {
                "input": "train.jsonl",
                "sealed_validation": "validation.jsonl",
            }
        )

        self.assertEqual(
            paths, (Path("train.jsonl"), Path("validation.jsonl"))
        )

    def test_builds_default_five_company_disjoint_folds(self) -> None:
        folds = FOLDS._build_folds(_rows(20))

        self.assertEqual(len(folds), 5)
        seen: set[str] = set()
        for fold in folds:
            validation = set(fold["validation_companies"])
            self.assertEqual(len(validation), 4)
            self.assertFalse(seen & validation)
            self.assertEqual(fold["company_overlap"], 0)
            self.assertEqual(fold["report_overlap"], 0)
            self.assertEqual(fold["group_overlap"], 0)
            seen.update(validation)
        self.assertEqual(len(seen), 20)

    def test_builds_four_disjoint_balanced_folds(self) -> None:
        folds = FOLDS._build_folds(_rows(16), fold_count=4, minimum_companies=16)

        self.assertEqual(len(folds), 4)
        validation_companies: list[set[str]] = []
        for fold in folds:
            train = {row["supervision"]["company_key"] for row in fold["train"]}
            validation = {
                row["supervision"]["company_key"] for row in fold["validation"]
            }
            self.assertEqual(len(train), 12)
            self.assertEqual(len(validation), 4)
            self.assertFalse(train & validation)
            self.assertEqual(
                {row["program"]["operation"] for row in fold["validation"]},
                set(OPERATIONS),
            )
            validation_companies.append(validation)
        self.assertEqual(set().union(*validation_companies), {f"company-{i:02d}" for i in range(16)})
        self.assertEqual(sum(len(group) for group in validation_companies), 16)

    def test_balances_heterogeneous_company_sizes_deterministically(self) -> None:
        sizes = (60, 48, 36, 24, 18, 18, 12, 12, 12, 12, 6, 6, 6, 6, 6, 6)
        rows = []
        for company, size in enumerate(sizes):
            repeats = size // len(OPERATIONS)
            for repeat in range(repeats):
                for operation in OPERATIONS:
                    rows.append(
                        {
                            "example_id": f"{company}-{repeat}-{operation}",
                            "program": {"operation": operation, "operands": [0]},
                            "supervision": {"company_key": f"company-{company:02d}"},
                        }
                    )

        first = FOLDS._build_folds(rows, fold_count=4, minimum_companies=16)
        second = FOLDS._build_folds(
            list(reversed(rows)), fold_count=4, minimum_companies=16
        )

        first_groups = [
            sorted(
                {
                    row["supervision"]["company_key"]
                    for row in fold["validation"]
                }
            )
            for fold in first
        ]
        second_groups = [
            sorted(
                {
                    row["supervision"]["company_key"]
                    for row in fold["validation"]
                }
            )
            for fold in second
        ]
        validation_sizes = [len(fold["validation"]) for fold in first]
        company_counts = [len(group) for group in first_groups]
        self.assertEqual(first_groups, second_groups)
        self.assertLessEqual(max(validation_sizes) - min(validation_sizes), max(sizes))
        self.assertLessEqual(max(company_counts) - min(company_counts), 1)

    def test_rejects_too_few_companies(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 16"):
            FOLDS._build_folds(_rows(15), fold_count=4, minimum_companies=16)

    def test_distributes_remainder_company_groups_evenly(self) -> None:
        folds = FOLDS._build_folds(_rows(21), fold_count=4, minimum_companies=16)
        validation_counts = [
            len(
                {
                    row["supervision"]["company_key"]
                    for row in fold["validation"]
                }
            )
            for fold in folds
        ]
        self.assertEqual(sorted(validation_counts), [5, 5, 5, 6])

    def test_group_metadata_has_all_generalization_dimensions(self) -> None:
        row = {
            "example_id": "group-row",
            "question": "Doanh thu năm 2025 là bao nhiêu?",
            "program": {"operation": "SELECT", "operands": [0]},
            "supervision": {"company_key": "company-a"},
            "candidates": [
                {
                    "source_name": "A_25CN_BCTN.pdf",
                    "extraction_method": "native",
                    "row_label": "Doanh thu",
                    "column_label": "Năm 2025",
                    "period": "2025",
                    "scope": "consolidated",
                }
            ],
        }

        metadata = FOLDS._group_metadata(row)

        self.assertEqual(
            set(metadata),
            {
                "company",
                "report",
                "source",
                "layout_family",
                "period",
                "group_key",
            },
        )
        self.assertEqual(metadata["company"], "company-a")
        self.assertEqual(metadata["source"], "native")
        self.assertEqual(metadata["period"], "2025")

    def test_shared_report_companies_stay_in_one_fold(self) -> None:
        rows = _rows(20)
        for row in rows:
            company = row["supervision"]["company_key"]
            report = (
                "shared.pdf"
                if company in {"company-00", "company-01"}
                else f"{company}.pdf"
            )
            row["candidates"] = [{"source_name": report}]

        folds = FOLDS._build_folds(rows)

        fold_by_company = {
            company: fold["fold"]
            for fold in folds
            for company in fold["validation_companies"]
        }
        self.assertEqual(
            fold_by_company["company-00"], fold_by_company["company-01"]
        )

    def test_semantic_dedup_keeps_exact_gold_not_same_value_metric(self) -> None:
        def candidate(index: int, metric: str) -> NumericCandidate:
            return NumericCandidate(
                index=index,
                value="10",
                raw="10",
                context=f"ROW: {metric}\nCOLUMN: 2025\nPERIOD: 2025",
                line_number=index + 1,
                char_start=index * 3,
                char_end=index * 3 + 2,
                unit="VND",
                is_percent=False,
                is_year_like=False,
                relevance=1.0,
            )

        example = ProgramExample(
            example_id="semantic-dedup",
            question="Doanh thu nam 2025 la bao nhieu?",
            context="",
            answer_text="10",
            candidates=(
                candidate(0, "Doanh thu"),
                candidate(1, "Doanh thu"),
                candidate(2, "Loi nhuan"),
            ),
            program=ReasoningProgram("SELECT", (1,)),
            answer_value="10",
        )
        record = {
            "example_id": example.example_id,
            "operation": "SELECT",
            "candidates": [
                {
                    "program_key": "SELECT:0",
                    "exact_gold": False,
                    "positive_class": "grounded_equivalent",
                    "runtime_eligible": True,
                    "training_state": "VALID",
                },
                {
                    "program_key": "SELECT:1",
                    "exact_gold": True,
                    "positive_class": "canonical_exact",
                    "runtime_eligible": True,
                    "training_state": "VALID",
                },
                {
                    "program_key": "SELECT:2",
                    "exact_gold": False,
                    "positive_class": "execution_only",
                    "runtime_eligible": True,
                    "training_state": "VALID",
                },
            ],
        }

        deduplicated, dropped = FOLD_CACHE._deduplicate_record(record, example)

        self.assertEqual(dropped, 1)
        self.assertEqual(
            [item["program_key"] for item in deduplicated["candidates"]],
            ["SELECT:1", "SELECT:2"],
        )
        self.assertEqual(
            {
                member["program_key"]
                for member in deduplicated["candidates"][0][
                    "semantic_equivalent_members"
                ]
            },
            {"SELECT:0", "SELECT:1"},
        )


if __name__ == "__main__":
    unittest.main()
