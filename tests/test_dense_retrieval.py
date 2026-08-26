from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from hrm_finance_qa.dense_retrieval import _find_cached_fastembed_snapshot


class FastEmbedCacheTests(unittest.TestCase):
    def test_finds_complete_matching_snapshot(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            repository = (
                root
                / "models--qdrant--paraphrase-multilingual-MiniLM-L12-v2-onnx-Q"
            )
            snapshot = repository / "snapshots" / "revision-1"
            snapshot.mkdir(parents=True)
            (repository / "refs").mkdir()
            (repository / "refs" / "main").write_text(
                "revision-1", encoding="utf-8"
            )
            for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
                (snapshot / name).write_text("{}", encoding="utf-8")
            (snapshot / "model_optimized.onnx").write_bytes(b"onnx")

            found = _find_cached_fastembed_snapshot(
                root,
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            )

            self.assertEqual(found, snapshot)

    def test_rejects_incomplete_snapshot(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = (
                root
                / "models--qdrant--paraphrase-multilingual-MiniLM-L12-v2-onnx-Q"
                / "snapshots"
                / "revision-1"
            )
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")

            found = _find_cached_fastembed_snapshot(
                root,
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            )

            self.assertIsNone(found)


if __name__ == "__main__":
    unittest.main()
