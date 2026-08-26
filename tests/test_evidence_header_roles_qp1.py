from __future__ import annotations

from decimal import Decimal
import unittest

from hrm_finance_qa.cache_schema import (
    CANDIDATE_CACHE_SCHEMA,
    candidate_cache_schema_identity,
)
from hrm_finance_qa.evidence_schema import (
    EVIDENCE_SCHEMA_FINGERPRINT,
    EVIDENCE_SCHEMA_VERSION,
    evidence_cells,
)
from hrm_finance_qa.program_data import NumericCandidate, extract_numeric_candidates


def _candidate(
    index: int,
    *,
    value: str = "100",
    column: str = "",
    header: str,
    row: str,
    extraction_method: str = "native",
    source_span: tuple[int, int] | None = None,
) -> NumericCandidate:
    parts = ["ROW: Doanh thu hoạt động", "ROW_TYPE: detail"]
    if column:
        parts.append(f"COLUMN: {column}")
    parts.extend(
        (
            "TABLE: table-1",
            "SECTION: Tình hình hoạt động sản xuất kinh doanh",
            f"HEADER: {header}",
            f"CELL: {row}",
        )
    )
    return NumericCandidate(
        index=index,
        value=value,
        raw=value,
        context="\n".join(parts),
        line_number=10,
        char_start=index * 10,
        char_end=index * 10 + len(value),
        unit="VND",
        is_percent=False,
        is_year_like=False,
        relevance=1.0,
        source_name="development-report.pdf",
        page=10,
        extraction_method=extraction_method,
        source_span=source_span,
    )


class EvidenceHeaderRoleQP1Tests(unittest.TestCase):
    def test_flattened_native_plan_actual_header_recovers_all_five_roles(self) -> None:
        context = "\n".join(
            (
                "STT Chỉ tiêu",
                "Thực hiện Kế hoạch Thực hiện % tăng giảm",
                "TH2023/",
                "KH 2023",
                "% tăng giảm",
                "TH2023/",
                "TH 2022 Năm 2022 Năm 2023 Năm 2023",
                (
                    "2 Doanh thu hoạt động 363,638,445,420 "
                    "472,723,394,169 304,912,639,828 -35.50% -16.15%"
                ),
            )
        )
        candidates = extract_numeric_candidates(
            "Doanh thu hoạt động thực hiện năm 2023 là bao nhiêu?",
            context,
            extraction_method="native",
        )
        cells = evidence_cells(candidates)
        targets = list(cells)

        self.assertEqual(
            tuple(cell.column_role for cell in targets),
            (
                "actual",
                "plan",
                "actual",
                "actual_to_plan_ratio",
                "actual_to_prior_ratio",
            ),
        )
        self.assertEqual(targets[2].column, "Thực hiện 2023")
        self.assertEqual(targets[2].explicit_years, ("2023",))
        self.assertTrue(targets[2].raw_header)

    def test_extractor_reconstructs_merged_header_path_end_to_end(self) -> None:
        context = "\n".join(
            (
                "| Chi tieu | 2022 | 2023 | |",
                "| | Thuc hien | Ke hoach | Thuc hien |",
                "|---|---:|---:|---:|",
                "| Doanh thu | 90.000 | 110.000 | 100.000 |",
            )
        )
        candidates = extract_numeric_candidates(
            "Doanh thu thuc hien nam 2023 la bao nhieu?",
            context,
            extraction_method="native",
        )
        cells = evidence_cells(candidates)
        target = next(cell for cell in cells if cell.value == Decimal("100000"))

        self.assertEqual(target.column_role, "actual")
        self.assertEqual(target.header_path, ("2023", "Thuc hien"))
        self.assertEqual(
            target.header_path_provenance,
            ("inherited_merged_header", "explicit_header"),
        )

    def test_repeated_header_uses_the_nearest_table_header(self) -> None:
        context = "\n".join(
            (
                "| Chi tieu | Ke hoach 2022 | Thuc hien 2022 |",
                "|---|---:|---:|",
                "| Doanh thu | 80.000 | 90.000 |",
                "",
                "| Chi tieu | Ke hoach 2023 | Thuc hien 2023 |",
                "|---|---:|---:|",
                "| Doanh thu | 110.000 | 100.000 |",
            )
        )
        candidates = extract_numeric_candidates(
            "Doanh thu thuc hien nam 2023 la bao nhieu?",
            context,
            extraction_method="native",
        )
        cells = evidence_cells(candidates)
        target = next(cell for cell in cells if cell.value == Decimal("100000"))

        self.assertEqual(target.column_role, "actual")
        self.assertEqual(target.header_path, ("Thuc hien 2023",))
        self.assertEqual(target.explicit_years, ("2023",))

    def test_hierarchical_merged_header_distinguishes_roles(self) -> None:
        upper = "| Chỉ tiêu | 2022 | 2023 | | % so sánh | |"
        lower = "| | Thực hiện | Kế hoạch | Thực hiện | TH/KH | TH/2022 |"
        header = f"{upper} || {lower}"
        values = ("90", "110", "100", "90.91", "11.11")
        columns = (
            "2022 / Thực hiện",
            "2023 / Kế hoạch",
            "Thực hiện",
            "% so sánh / TH/KH",
            "% TH/2022",
        )
        expected_roles = (
            "actual",
            "plan",
            "actual",
            "actual_to_plan_ratio",
            "actual_to_prior_ratio",
        )
        candidates = []
        for offset, (value, column) in enumerate(zip(values, columns), start=1):
            cells = ["Doanh thu hoạt động", *values]
            cells[offset] = f"<VALUE> {value} </VALUE>"
            candidates.append(
                _candidate(
                    offset,
                    value=value,
                    column=column,
                    header=header,
                    row="| " + " | ".join(cells) + " |",
                )
            )

        observed = evidence_cells(candidates)

        self.assertEqual(
            tuple(cell.column_role for cell in observed), expected_roles
        )
        self.assertEqual(observed[2].header_path, ("2023", "Thực hiện"))
        self.assertEqual(
            observed[2].header_path_provenance,
            ("inherited_merged_header", "explicit_header"),
        )
        self.assertTrue(
            all(cell.column_role_confidence == "EXPLICIT" for cell in observed)
        )

    def test_parallel_panels_preserve_scope_path_without_changing_role(self) -> None:
        header = (
            "| Chỉ tiêu | Công ty mẹ | | Hợp nhất | | || "
            "| | Kế hoạch | Thực hiện | Kế hoạch | Thực hiện |"
        )
        row = "| Doanh thu | 90 | <VALUE> 100 </VALUE> | 190 | 200 |"
        cell = evidence_cells(
            (
                _candidate(
                    1,
                    column="Công ty mẹ / Thực hiện",
                    header=header,
                    row=row,
                ),
            )
        )[0]

        self.assertEqual(cell.column_role, "actual")
        self.assertEqual(cell.header_path, ("Công ty mẹ", "Thực hiện"))
        self.assertFalse(cell.column_role_conflict)

    def test_revised_plan_forecast_and_no_diacritic_actual_are_canonical(self) -> None:
        cases = (
            ("Ke hoach dieu chinh 2024", "revised_plan"),
            ("Du bao 2024", "forecast"),
            ("Thuc hien 2024", "actual"),
        )
        for index, (label, expected) in enumerate(cases):
            with self.subTest(label=label):
                cell = evidence_cells(
                    (
                        _candidate(
                            index,
                            column=label,
                            header=f"| Chi tieu | {label} |",
                            row="| Doanh thu | <VALUE> 100 </VALUE> |",
                        ),
                    )
                )[0]
                self.assertEqual(cell.column_role, expected)
                self.assertEqual(cell.column_role_confidence, "EXPLICIT")

    def test_bare_years_remain_unknown_instead_of_using_column_order(self) -> None:
        cell = evidence_cells(
            (
                _candidate(
                    1,
                    column="2023",
                    header="| Chỉ tiêu | 2022 | 2023 |",
                    row="| Doanh thu | 90 | <VALUE> 100 </VALUE> |",
                ),
            )
        )[0]

        self.assertEqual(cell.column_role, "")
        self.assertEqual(cell.column_role_confidence, "UNKNOWN")
        self.assertFalse(cell.column_role_conflict)

    def test_cross_extractor_role_disagreement_is_conflict(self) -> None:
        common = {
            "value": "100",
            "row": "| Doanh thu | <VALUE> 100 </VALUE> |",
        }
        candidates = (
            _candidate(
                1,
                column="Kế hoạch 2023",
                header="| Chỉ tiêu | Kế hoạch 2023 |",
                extraction_method="native",
                source_span=(120, 123),
                **common,
            ),
            _candidate(
                2,
                column="Thực hiện 2023",
                header="| Chỉ tiêu | Thực hiện 2023 |",
                extraction_method="mineru",
                source_span=(480, 483),
                **common,
            ),
        )

        observed = evidence_cells(candidates)

        for cell in observed:
            self.assertEqual(cell.column_role, "")
            self.assertEqual(cell.column_role_source, "cross_extractor_conflict")
            self.assertEqual(cell.column_role_confidence, "CONFLICT")
            self.assertTrue(cell.column_role_conflict)
            self.assertEqual(
                set(cell.column_role_evidence),
                {"Kế hoạch 2023", "Thực hiện 2023"},
            )

    def test_candidate_permutation_does_not_change_role_assignment(self) -> None:
        plan = _candidate(
            7,
            value="110",
            column="Kế hoạch 2023",
            header="| Chỉ tiêu | Kế hoạch 2023 |",
            row="| Doanh thu | <VALUE> 110 </VALUE> |",
        )
        actual = _candidate(
            3,
            value="100",
            column="Thực hiện 2023",
            header="| Chỉ tiêu | Thực hiện 2023 |",
            row="| Doanh thu | <VALUE> 100 </VALUE> |",
        )

        forward = {cell.index: cell for cell in evidence_cells((plan, actual))}
        reverse = {cell.index: cell for cell in evidence_cells((actual, plan))}

        self.assertEqual(
            {index: cell.column_role for index, cell in forward.items()},
            {index: cell.column_role for index, cell in reverse.items()},
        )
        self.assertEqual(forward[7].value, Decimal("110"))

    def test_cache_identity_contract_includes_qp1_schema_version(self) -> None:
        self.assertEqual(EVIDENCE_SCHEMA_VERSION, "evidence-cell-header-role-v2")
        self.assertEqual(len(EVIDENCE_SCHEMA_FINGERPRINT), 64)
        self.assertEqual(
            CANDIDATE_CACHE_SCHEMA,
            "v12b2-applicability-proof-candidate-cache-v5",
        )
        self.assertEqual(
            candidate_cache_schema_identity(),
            {
                "cache_schema": CANDIDATE_CACHE_SCHEMA,
                "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
                "evidence_schema_fingerprint": EVIDENCE_SCHEMA_FINGERPRINT,
            },
        )


if __name__ == "__main__":
    unittest.main()
