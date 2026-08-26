from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PackageImportTests(unittest.TestCase):
    def test_page_index_import_does_not_load_torch(self) -> None:
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT / "src")
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import hrm_finance_qa; "
                    "from hrm_finance_qa import DocumentPageIndex; "
                    "assert DocumentPageIndex.__name__ == 'DocumentPageIndex'; "
                    "assert 'torch' not in sys.modules"
                ),
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout={completed.stdout}\nstderr={completed.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
