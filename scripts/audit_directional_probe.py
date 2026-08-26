#!/usr/bin/env python3
"""Audit one proven directional example in the frozen candidate search trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrm_finance_qa.document_hrm import DocumentHRMBundle  # noqa: E402
from hrm_finance_qa.program_data import load_program_examples  # noqa: E402


def _reject_protected(path: Path) -> None:
    normalized = str(path).lower().replace("\\", "/")
    if any(
        marker in normalized
        for marker in ("validation.jsonl", "test.jsonl", "blind", "final_eval")
    ):
        raise ValueError(f"Protected evaluation input is not allowed: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--example-id", required=True)
    parser.add_argument(
        "--base-model",
        type=Path,
        default=ROOT / "models" / "document_hrm_V8_candidate.pt",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    _reject_protected(args.input)
    examples = load_program_examples(args.input)
    example = next(
        (item for item in examples if item.example_id == args.example_id), None
    )
    if example is None:
        raise ValueError(f"Unknown example ID: {args.example_id}")
    if len(example.program.operands) != 2:
        raise ValueError("Directional probe audit requires a binary gold program")

    bundle = DocumentHRMBundle.load(args.base_model, device=args.device)
    bundle.metadata = dict(bundle.metadata)
    bundle.metadata.update(
        {
            "typed_program_synthesis_enabled": True,
            "typed_program_synthesis_version": "typed-slot-synthesis-v1",
            "financial_identity_catalog_enabled": True,
            "candidate_recall_v12_4_enabled": True,
        }
    )
    prediction = bundle.predict_many([example])[0]
    direct = tuple(example.program.operands)
    reverse = tuple(reversed(direct))
    rows = [
        {
            "program": f"{trace.program.operation}:"
            + ",".join(str(value) for value in trace.program.operands),
            "training_state": trace.training_state,
            "runtime_eligible": trace.eligible,
            "rejection_reasons": list(trace.rejection_reasons),
            "generation_sources": list(trace.generation_sources),
        }
        for trace in prediction.program_search_trace
        if trace.program.operation == example.program.operation
        and tuple(trace.program.operands) in {direct, reverse}
    ]
    reversed_rows = [
        row
        for row in rows
        if row["program"].endswith(",".join(str(value) for value in reverse))
    ]
    gate = bool(
        reversed_rows
        and all(row["training_state"] == "INVALID" for row in reversed_rows)
        and all(not row["runtime_eligible"] for row in reversed_rows)
        and all(
            "directional_reversal_probe" in row["generation_sources"]
            for row in reversed_rows
        )
    )
    payload = {
        "example_id": example.example_id,
        "blind_test_used": False,
        "probe_gate": gate,
        "rows": rows,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not gate:
        raise RuntimeError("Directional reversal probe gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
