from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRECOMMIT = ROOT / "config/finhrm_qp3_core_add_diverse_development_cohort_precommit_v1.json"
COHORT = ROOT / "data/finhrm_qp3_core_add_diverse_development_reviewed_cohort_v1.jsonl"
COHORT_REPORT = ROOT / "reports/finhrm_qp3_core_add_diverse_development_reviewed_cohort_v1.json"
VISUAL_REVIEW = ROOT / "reports/finhrm_qp3_core_add_diverse_development_visual_review_v1.json"
PRE_FIX = ROOT / "reports/finhrm_qp3_core_add_relative_period_counterfactual_audit_v1.json"
POST_FIX = ROOT / "reports/finhrm_qp3_core_add_relative_period_counterfactual_post_fix_v1.json"
SAFE_REPLAY = ROOT / "reports/finhrm_qp3_core_s4_safe_replay_post_add_diverse_v1.json"
MARKDOWN = ROOT / "reports/finhrm_qp3_core_add_diverse_development_reviewed_cohort_v1.md"
LOCK = ROOT / "reports/finhrm_qp3_core_add_diverse_development_reviewed_cohort_lock_v1.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> tuple[str, dict[str, object]]:
    cohort = json.loads(COHORT_REPORT.read_text(encoding="utf-8"))
    review = json.loads(VISUAL_REVIEW.read_text(encoding="utf-8"))
    pre = json.loads(PRE_FIX.read_text(encoding="utf-8"))
    post = json.loads(POST_FIX.read_text(encoding="utf-8"))
    replay = json.loads(SAFE_REPLAY.read_text(encoding="utf-8"))
    stage_status = {
        "document_discovery": True,
        "native_or_ocr_extraction": True,
        "chunk_retrieval": False,
        "numeric_candidate_materialization": True,
        "neural_operation_and_operand_proposal": False,
        "symbolic_binding_and_grounding": True,
        "deterministic_verification_and_execution": True,
    }
    checks = {
        "reviewed_cohort_gate_passed": cohort["reviewed_cohort_gate_passed"],
        "four_companies_per_layout": cohort["metrics"]["selected_by_layout"]
        == {"NATIVE_TEXT_TABLE": 4, "OCR_OR_HYBRID_TABLE": 4},
        "eight_company_disjoint_examples": cohort["metrics"]["selected_examples"] == 8
        and cohort["metrics"]["selected_companies"] == 8,
        "independent_review_complete": review["metrics"]["approved"] == 8,
        "visual_rejection_excluded": cohort["metrics"]["visual_rejections"] == 1
        and cohort["gate_checks"]["visual_rejection_not_selected"],
        "relative_period_false_accepts_zero": pre["metrics"]["false_accepts"] > 0
        and post["metrics"]["false_accepts"] == 0,
        "frozen_safe_replay_passed": replay["safe_replay_gate_passed"],
        "gold_execution_363_of_363": replay["results"]["gold_execution_correct"] == 363,
        "permutation_363_of_363": replay["results"]["permutation_exact"] == 363,
        "counterfactual_false_accepts_zero": replay["results"]["counterfactual_false_accepts"] == 0,
    }
    cohort_ready = all(checks.values())
    end_to_end_ready = all(stage_status.values())
    markdown = "\n".join(
        [
            "# FinHRM QP3-CORE diverse development ADD cohort",
            "",
            f"- Reviewed cohort gate: **{'PASS' if cohort_ready else 'FAIL'}**.",
            "- Coverage: `8` examples / `8` companies; `4 NATIVE_TEXT_TABLE` + `4 OCR_OR_HYBRID_TABLE`; one positive per company and company-disjoint across strata.",
            "- Visual review: `9` candidates inspected, `8` approved, `1` rejected for a cross-region row/value provenance collision and excluded.",
            f"- Explicit wrong-period counterfactual false accepts: `{pre['metrics']['false_accepts']} -> {post['metrics']['false_accepts']}` across the development audit.",
            "- Frozen replay: gold execution `363/363`, permutation `363/363`, counterfactual false accepts `0/141`.",
            "",
            "## Qualification standard",
            "",
            "A positive requires explicit two-role ADD syntax, exactly two distinct logical and physical cells, same explicit period/source/entity/unit, complete page/table/row/column/span provenance, exact execution, a generic hard negative, independent visual approval, permutation invariance, and a semantic counterfactual that is not ANSWER.",
            "",
            "## Remaining locked gate",
            "",
            "The cohort is ready for frozen end-to-end replay, but it does not by itself close QP3-CORE: chunk retrieval and neural operation/operand proposal have not yet been replayed from PDF+question on these eight reviewed records. QP3 answer presentation, QP4 and Phase D remain locked.",
            "",
        ]
    )
    artifacts = []
    for path in (
        PRECOMMIT,
        COHORT,
        COHORT_REPORT,
        VISUAL_REVIEW,
        PRE_FIX,
        POST_FIX,
        SAFE_REPLAY,
        Path(__file__),
    ):
        artifacts.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _digest(path),
            }
        )
    lock = {
        "schema": "finhrm-qp3-core-add-diverse-development-reviewed-cohort-lock-v1",
        "created_at": "2026-08-20",
        "classification": "development_only_reviewed_cohort_ready_for_end_to_end_replay",
        "reviewed_cohort_gate_passed": cohort_ready,
        "end_to_end_confirmation_passed": end_to_end_ready,
        "status": "cohort_ready_end_to_end_pending" if cohort_ready and not end_to_end_ready else "failed",
        "checks": checks,
        "pipeline_stage_status": stage_status,
        "artifacts": artifacts,
        "invariants": {
            "protected_evaluation_loaded": False,
            "abw_loaded": False,
            "synthetic_layout_counted": False,
            "threshold_checkpoint_or_model_changed": False,
            "qp3_answer_presentation_unlocked": False,
            "qp4_or_phase_d_unlocked": False,
        },
        "authorized_next_step": "Run the frozen PDF+question end-to-end engine on the eight reviewed cohort records, including chunk retrieval and neural operation/operand proposal; do not open answer presentation yet.",
    }
    return markdown, lock


if __name__ == "__main__":
    if MARKDOWN.exists() or LOCK.exists():
        raise FileExistsError("refusing to overwrite reviewed cohort summary artifacts")
    markdown, lock = build()
    MARKDOWN.write_text(markdown, encoding="utf-8")
    LOCK.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cohort_gate": lock["reviewed_cohort_gate_passed"], "end_to_end": lock["end_to_end_confirmation_passed"], "status": lock["status"]}, sort_keys=True))
