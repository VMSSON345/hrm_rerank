from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRECOMMIT = ROOT / "config/finhrm_qp3_core_add_diverse_development_cohort_precommit_v1.json"
DEFAULT_AUDIT = ROOT / "reports/finhrm_qp3_core_add_diverse_development_source_audit_v1.json"
DEFAULT_OUTPUT = ROOT / "data/finhrm_qp3_core_add_diversity_generation_manifest_v1.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stratum(family: str, strata: dict[str, list[str]]) -> str:
    matches = [name for name, families in strata.items() if family in families]
    if len(matches) != 1:
        raise ValueError(f"layout family does not map uniquely: {family}")
    return matches[0]


def build_manifest(
    precommit_path: Path = DEFAULT_PRECOMMIT,
    audit_path: Path = DEFAULT_AUDIT,
) -> dict[str, Any]:
    precommit = json.loads(precommit_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    selection = precommit["deterministic_selection"]
    maximum_pages = int(selection["maximum_keyword_pages_per_source"])
    documents: list[dict[str, Any]] = []
    for row in audit["details"]:
        pages = sorted(
            {
                *[int(page) for page in row["balance_term_pages"]],
                *[int(page) for page in row["statement_keyword_pages"]],
            }
        )[:maximum_pages]
        if not pages:
            continue
        documents.append(
            {
                "document": str((ROOT / row["source_path"]).resolve()),
                "sha256": row["source_sha256"],
                "source_key": row["source_key"],
                "company_key": row["company_key"],
                "layout_family": row["layout_family"],
                "layout_stratum": _stratum(
                    row["layout_family"], selection["layout_strata"]
                ),
                "pages": [
                    {
                        "page": page,
                        "selection": "core",
                        "selection_reason": (
                            "balance_terms"
                            if page in row["balance_term_pages"]
                            else "financial_statement_heading"
                        ),
                    }
                    for page in pages
                ],
            }
        )
    documents.sort(key=lambda row: (row["layout_stratum"], row["sha256"]))
    return {
        "protocol": "finhrm-qp3-core-add-diversity-generation-manifest-v1",
        "classification": "development_only",
        "ocr_performed": False,
        "model_invoked": False,
        "protected_evaluation_loaded": False,
        "selection_used_filename_company_value_or_question_literal": False,
        "inputs": {
            "precommit": {
                "path": precommit_path.relative_to(ROOT).as_posix(),
                "sha256": _digest(precommit_path),
            },
            "source_audit": {
                "path": audit_path.relative_to(ROOT).as_posix(),
                "sha256": _digest(audit_path),
            },
        },
        "metrics": {
            "documents": len(documents),
            "company_groups": len({row["company_key"] for row in documents}),
            "pages": sum(len(row["pages"]) for row in documents),
            "layout_strata": {
                name: sum(row["layout_stratum"] == name for row in documents)
                for name in selection["layout_strata"]
            },
        },
        "documents": documents,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precommit", type=Path, default=DEFAULT_PRECOMMIT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    manifest = build_manifest(args.precommit, args.audit)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["metrics"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
