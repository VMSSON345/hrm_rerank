from scripts.annotate_document_provenance import (
    annotate_row,
    _infer_explicit_scope,
    _infer_financial_section,
)


def _candidate(row: str, context: str = "") -> dict[str, str]:
    return {
        "row_label": row,
        "row_type": "detail",
        "context": context,
    }


def test_infers_financial_statement_sections_from_unambiguous_rows() -> None:
    assert (
        _infer_financial_section(_candidate("Tổng tài sản"))
        == "statement_of_financial_position"
    )
    assert _infer_financial_section(_candidate("Doanh thu thuần")) == "income_statement"
    assert (
        _infer_financial_section(
            _candidate("Hệ số Lợi nhuận sau thuế/Doanh thu thuần")
        )
        == "financial_ratios"
    )


def test_keeps_ambiguous_section_empty() -> None:
    assert _infer_financial_section(_candidate("Bình Thái Nguyên")) == ""


def test_scope_requires_explicit_heading_or_header() -> None:
    assert (
        _infer_explicit_scope(
            _candidate(
                "Tổng tài sản",
                "HEADER: BÁO CÁO TÌNH HÌNH TÀI CHÍNH HỢP NHẤT\nCELL: Tổng tài sản",
            )
        )
        == "consolidated"
    )
    assert (
        _infer_explicit_scope(
            _candidate(
                "Lợi nhuận sau thuế",
                "HEADER: Chỉ tiêu | Năm 2025\n"
                "CELL: Lợi nhuận sau thuế của cổ đông Công ty mẹ",
            )
        )
        == ""
    )


def test_annotates_side_by_side_entity_and_subtable() -> None:
    context = (
        "CÔNG TY TNHH A Đơn vị: Đồng CÔNG TY TNHH B Đơn vị: Đồng\n"
        "| STT | Chỉ tiêu | 2024 | STT | Chỉ tiêu | 2024 |\n"
        "| 1 | Tổng tài sản | 100 | 1 | Tổng tài sản | 200 |"
    )
    row = {
        "context": context,
        "candidates": [
            {
                "index": 0,
                "raw": "100",
                "context": "ROW: Tổng tài sản\nPERIOD: 2024",
                "line_number": 3,
                "char_start": context.splitlines()[2].index("100"),
                "char_end": context.splitlines()[2].index("100") + 3,
                "source_name": "sample.pdf",
            },
            {
                "index": 1,
                "raw": "200",
                "context": "ROW: Tổng tài sản\nPERIOD: 2024",
                "line_number": 3,
                "char_start": context.splitlines()[2].index("200"),
                "char_end": context.splitlines()[2].index("200") + 3,
                "source_name": "sample.pdf",
            },
        ],
        "program": {"operation": "SELECT", "operands": [0]},
        "supervision": {"hard_negatives": []},
    }

    annotate_row(row)

    assert row["candidates"][0]["entity_id"] == "entity:cong ty tnhh a"
    assert row["candidates"][1]["entity_id"] == "entity:cong ty tnhh b"
    assert row["candidates"][0]["subtable_id"].endswith("subtable-1")
    assert row["candidates"][1]["subtable_id"].endswith("subtable-2")
    assert row["supervision"]["hard_negatives"][0]["type"] == (
        "same_metric_period_wrong_provenance"
    )
