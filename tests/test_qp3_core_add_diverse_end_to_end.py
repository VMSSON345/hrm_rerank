from pathlib import Path
import unittest

from scripts.evaluate_qp3_core_add_diverse_end_to_end import (
    InferenceCase,
    _coverage_gate_summary,
    _physical_role_match,
    _run_frozen_inference,
)


class _RecordingEngine:
    def __init__(self) -> None:
        self.calls = []

    def answer(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return "answer"


class Qp3CoreAddDiverseEndToEndTests(unittest.TestCase):
    def test_frozen_inference_boundary_receives_only_question_and_pdf(self) -> None:
        engine = _RecordingEngine()
        case = InferenceCase(
            example_id="diagnostic-id",
            question="Tổng hai vai trò là bao nhiêu?",
            source_path=Path("report.pdf"),
            source_sha256="0" * 64,
            company_key="development-company",
            layout_stratum="NATIVE_TEXT_TABLE",
            layout_family="native_text_portrait",
        )

        result = _run_frozen_inference(engine, case)

        self.assertEqual(result, "answer")
        self.assertEqual(
            engine.calls,
            [((case.question, case.source_path), {})],
        )

    def test_physical_role_scoring_allows_extractor_header_path_variants(self) -> None:
        gold = {
            "value": "480086",
            "source_name": "report.pdf",
            "page": 24,
            "period": "2022",
            "row_label": "1. Tài sản dài hạn",
            "column_label": "31/12/2022 Giá trị",
        }
        observed = type(
            "Candidate",
            (),
            {
                "value": 480086,
                "source_name": "report.pdf",
                "page": 24,
                "period": "2022",
                "row_label": "Tài sản dài hạn",
                "column_label": "31/12/2022",
            },
        )()

        self.assertTrue(_physical_role_match(gold, observed))

    def test_physical_role_scoring_keeps_period_and_page_strict(self) -> None:
        gold = {
            "value": "480086",
            "source_name": "report.pdf",
            "page": 24,
            "period": "2022",
            "row_label": "Tài sản dài hạn",
            "column_label": "31/12/2022",
        }
        observed = type(
            "Candidate",
            (),
            {
                "value": 480086,
                "source_name": "report.pdf",
                "page": 25,
                "period": "2023",
                "row_label": "Tài sản dài hạn",
                "column_label": "31/12/2022",
            },
        )()

        self.assertFalse(_physical_role_match(gold, observed))

    def test_coverage_gate_requires_three_companies_per_layout_and_safe_residuals(self) -> None:
        results = [
            {
                "end_to_end_positive_passed": True,
                "layout_family": layout.lower(),
                "layout_stratum": layout,
                "company_key": f"{layout}-company-{index}",
                "source_sha256": f"{layout}-source-{index}",
                "observed": {"abstained": False},
            }
            for layout in ("NATIVE", "OCR")
            for index in range(3)
        ]
        results.append(
            {
                "end_to_end_positive_passed": False,
                "layout_family": "mixed",
                "layout_stratum": "OCR",
                "company_key": "residual-company",
                "source_sha256": "residual-source",
                "observed": {"abstained": True},
            }
        )

        safe = _coverage_gate_summary(
            results,
            full_cohort=True,
            required_layout_strata=("NATIVE", "OCR"),
            minimum_companies_per_layout=3,
        )
        results[-1]["observed"]["abstained"] = False
        unsafe = _coverage_gate_summary(
            results,
            full_cohort=True,
            required_layout_strata=("NATIVE", "OCR"),
            minimum_companies_per_layout=3,
        )

        self.assertTrue(safe["gate_passed"])
        self.assertEqual(safe["unsafe_false_accepts"], 0)
        self.assertFalse(unsafe["gate_passed"])
        self.assertEqual(unsafe["unsafe_false_accepts"], 1)


if __name__ == "__main__":
    unittest.main()
