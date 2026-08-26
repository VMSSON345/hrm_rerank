from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrm_finance_qa.document_hrm import DocumentHRMBundle


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download/cache the frozen semantic encoder used by the local demo"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=ROOT / "models" / "document_hrm_V8_candidate.pt",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    bundle = DocumentHRMBundle.load(args.model_path, device=args.device)
    if bundle.semantic_encoder is None:
        print("Checkpoint does not use the semantic encoder; no warmup is required.")
        return 0
    vectors = bundle.semantic_encoder.encode(
        ["Doanh thu thuần năm nay là bao nhiêu?"]
    )
    print(
        "Semantic encoder is ready: "
        f"{bundle.model.config.semantic_model_name} ({vectors.shape[-1]} dimensions)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
