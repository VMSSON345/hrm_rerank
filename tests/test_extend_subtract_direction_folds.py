from __future__ import annotations

import importlib.util
import hashlib
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "extend_subtract_direction_folds",
    ROOT / "scripts" / "extend_subtract_direction_folds.py",
)
assert SPEC is not None and SPEC.loader is not None
EXTENDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXTENDER)


class ExtendSubtractDirectionFoldsTests(unittest.TestCase):
    def test_extended_manifest_is_bound_to_development_only_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            development = Path(directory) / "development.jsonl"
            development.write_text('{"example_id":"one"}\n', encoding="utf-8")
            manifest = {
                "input": "old.jsonl",
                "sealed_validation": "validation.jsonl",
                "sealed_validation_sha256": "old",
                "sealed_validation_loaded": True,
            }

            EXTENDER.seal_development_manifest(manifest, development)

            self.assertEqual(manifest["input"], str(development.resolve()))
            self.assertEqual(
                manifest["input_sha256"],
                hashlib.sha256(development.read_bytes()).hexdigest(),
            )
            self.assertFalse(manifest["sealed_validation_loaded"])
            self.assertNotIn("sealed_validation", manifest)
            self.assertNotIn("sealed_validation_sha256", manifest)

    def test_greedy_assignment_closes_sparse_calibration_folds(self) -> None:
        assignments, final = EXTENDER.assign_new_groups(
            [f"group-{index:02d}" for index in range(46)],
            [12, 8, 7, 7, 2],
        )

        increments = [sum(fold == index for fold in assignments.values()) for index in range(5)]
        final_train = [
            base + 46 - increments[index]
            for index, base in enumerate([24, 28, 29, 29, 34])
        ]
        self.assertEqual(final, [17, 17, 16, 16, 16])
        self.assertEqual(final_train, [65, 65, 66, 66, 66])
        self.assertTrue(all(value >= 30 for value in final_train))
        self.assertTrue(all(value >= 15 for value in final))


if __name__ == "__main__":
    unittest.main()
