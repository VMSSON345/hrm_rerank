"""Physical page counts using the PDFium dependency in the MinerU environment.

This module never extracts PDF text, tables, or QA evidence.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path


def pdf_page_count(path: str | Path) -> int:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"PDF not found: {source}")
    try:
        import pypdfium2 as pdfium
    except ImportError:
        root = Path(__file__).resolve().parents[2]
        python = root / ".venv-mineru" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not python.is_file():
            raise RuntimeError("MinerU metadata environment unavailable; run scripts/setup_mineru_venv.sh and prepare-document in that environment. No native-text fallback.")
        code = "import sys; import pypdfium2 as p; d=p.PdfDocument(sys.argv[1]); print(len(d)); d.close()"
        try:
            result = subprocess.run([str(python), "-I", "-c", code, str(source)], check=True, capture_output=True, text=True, timeout=60)
            count = int(result.stdout.strip())
        except (subprocess.SubprocessError, ValueError) as exc:
            raise RuntimeError(f"MinerU PDF metadata unavailable for {source.name}") from exc
    else:
        with pdfium.PdfDocument(str(source)) as document:
            count = len(document)
    if count < 1:
        raise ValueError(f"PDF contains no physical pages: {source.name}")
    return count
