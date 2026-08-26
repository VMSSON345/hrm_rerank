from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import fitz


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRECOMMIT = ROOT / "config/finhrm_qp3_core_add_diverse_development_cohort_precommit_v1.json"
DEFAULT_MANIFEST = ROOT / "data/finhrm_qp3_core_add_diversity_generation_manifest_v1.json"
DEFAULT_OUTPUT = ROOT / "data/finhrm_qp3_core_add_diversity_generation_manifest_v2.json"
PAGE_SCORER = ROOT / "scripts/build_financial_page_manifest.py"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_page_scorer():
    spec = importlib.util.spec_from_file_location("financial_page_scorer", PAGE_SCORER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load financial page scorer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_manifest(
    precommit_path: Path = DEFAULT_PRECOMMIT,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    precommit = json.loads(precommit_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scorer = _load_page_scorer()
    expanded_documents = []
    for source in manifest["documents"]:
        item = dict(source)
        if item["layout_stratum"] == "OCR_OR_HYBRID_TABLE":
            source_path = Path(item["document"])
            if _digest(source_path) != item["sha256"]:
                raise RuntimeError(f"source hash changed: {source_path.name}")
            pages = []
            document = fitz.open(source_path)
            try:
                for index, page in enumerate(document):
                    text = page.get_text("text", sort=True) or ""
                    result = scorer.score_financial_page(text)
                    if result["page_class"] == "financial_summary":
                        pages.append(
                            {
                                "page": index + 1,
                                "selection": "core",
                                "selection_reason": "generic_financial_page_scorer",
                                "score": result["score"],
                            }
                        )
            finally:
                document.close()
            item["pages"] = pages
        expanded_documents.append(item)
    result = {
        **manifest,
        "protocol": "finhrm-qp3-core-add-diversity-generation-manifest-v2",
        "inputs": {
            **manifest["inputs"],
            "v1_manifest": {
                "path": manifest_path.relative_to(ROOT).as_posix(),
                "sha256": _digest(manifest_path),
            },
            "page_scorer": {
                "path": PAGE_SCORER.relative_to(ROOT).as_posix(),
                "sha256": _digest(PAGE_SCORER),
            },
            "precommit_v2": {
                "path": precommit_path.relative_to(ROOT).as_posix(),
                "sha256": _digest(precommit_path),
            },
        },
        "metrics": {
            "documents": len(expanded_documents),
            "company_groups": len({row["company_key"] for row in expanded_documents}),
            "pages": sum(len(row["pages"]) for row in expanded_documents),
            "layout_strata": manifest["metrics"]["layout_strata"],
            "ocr_or_hybrid_core_pages": sum(
                len(row["pages"])
                for row in expanded_documents
                if row["layout_stratum"] == "OCR_OR_HYBRID_TABLE"
            ),
        },
        "documents": expanded_documents,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precommit", type=Path, default=DEFAULT_PRECOMMIT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = build_manifest(args.precommit, args.manifest)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
