from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "split_document_derived_development",
    ROOT / "scripts" / "split_document_derived_development.py",
)
assert SPEC is not None and SPEC.loader is not None
SPLITTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SPLITTER)


OPERATIONS = (
    "SELECT",
    "SUBTRACT",
    "RATIO_PERCENT",
    "ADD",
    "DIVIDE",
    "SUM3",
    "SUM4",
    "PERCENT_CHANGE",
)


def _row(company: str, source: str, operation: str, index: int) -> dict[str, object]:
    return {
        "example_id": f"{company}-{source}-{operation}-{index}",
        "candidates": [{"source_name": source}],
        "program": {"operation": operation, "operands": [0]},
        "supervision": {"company_key": company},
    }


class CompanyDisjointSplitTests(unittest.TestCase):
    def test_same_company_sources_never_cross_splits(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows: list[dict[str, object]] = []
            for company in ("company-a", "company-b", "company-c", "company-d"):
                sources = (f"{company}-separate.txt", f"{company}-consolidated.txt")
                for index, operation in enumerate(OPERATIONS):
                    rows.append(_row(company, sources[index % 2], operation, index))
            input_path = root / "input.jsonl"
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            output_dir = root / "split"
            report_path = root / "report.json"
            argv = [
                "split_document_derived_development.py",
                "--input",
                str(input_path),
                "--output-dir",
                str(output_dir),
                "--report",
                str(report_path),
            ]

            with patch.object(sys, "argv", argv):
                self.assertEqual(SPLITTER.main(), 0)

            report = json.loads(report_path.read_text(encoding="utf-8"))
            groups = [set(item["company_groups"]) for item in report["splits"].values()]
            self.assertFalse(groups[0] & groups[1])
            self.assertFalse(groups[0] & groups[2])
            self.assertFalse(groups[1] & groups[2])
            for split in report["splits"].values():
                sources = split["documents"]
                for company in split["company_groups"]:
                    self.assertEqual(
                        sum(source.startswith(company) for source in sources), 2
                    )

    def test_fraction_selects_multiple_company_groups(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                _row(company, f"{company}.txt", operation, index)
                for company in (f"company-{index:02d}" for index in range(10))
                for index, operation in enumerate(OPERATIONS)
            ]
            input_path = root / "input.jsonl"
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            output_dir = root / "split"
            report_path = root / "report.json"
            argv = [
                "split_document_derived_development.py",
                "--input",
                str(input_path),
                "--output-dir",
                str(output_dir),
                "--report",
                str(report_path),
                "--calibration-fraction",
                "0.2",
                "--eval-fraction",
                "0.2",
            ]

            with patch.object(sys, "argv", argv):
                self.assertEqual(SPLITTER.main(), 0)

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(len(report["splits"]["calibration"]["company_groups"]), 2)
            self.assertEqual(len(report["splits"]["eval"]["company_groups"]), 2)
            self.assertEqual(len(report["splits"]["train"]["company_groups"]), 6)
            groups = [
                set(report["splits"][name]["company_groups"])
                for name in ("train", "calibration", "eval")
            ]
            self.assertFalse(groups[0] & groups[1])
            self.assertFalse(groups[0] & groups[2])
            self.assertFalse(groups[1] & groups[2])

    def test_large_equal_company_pool_retains_disjoint_choices(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                _row(company, f"{company}.txt", operation, index)
                for company in (f"company-{index:02d}" for index in range(24))
                for index, operation in enumerate(OPERATIONS)
            ]
            input_path = root / "input.jsonl"
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            output_dir = root / "split"
            report_path = root / "report.json"
            argv = [
                "split_document_derived_development.py",
                "--input",
                str(input_path),
                "--output-dir",
                str(output_dir),
                "--report",
                str(report_path),
            ]

            with patch.object(sys, "argv", argv):
                self.assertEqual(SPLITTER.main(), 0)

            report = json.loads(report_path.read_text(encoding="utf-8"))
            calibration = set(report["splits"]["calibration"]["company_groups"])
            evaluation = set(report["splits"]["eval"]["company_groups"])
            self.assertEqual(len(calibration), 4)
            self.assertEqual(len(evaluation), 4)
            self.assertFalse(calibration & evaluation)


if __name__ == "__main__":
    unittest.main()
