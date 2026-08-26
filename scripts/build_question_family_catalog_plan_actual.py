from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
import unicodedata


ROOT = Path(__file__).resolve().parents[1]


def _slot(metric: str, year: str = "", role: str = "", scope: str = "reporting") -> dict[str, str]:
    return {"metric": metric, "year": year, "column_role": role, "scope": scope}


def _expected(
    operation: str | None,
    *,
    years: tuple[str, ...] = (),
    roles: tuple[str, ...] = (),
    slots: tuple[dict[str, str], ...] = (),
    answer_mode: str = "SINGLE_SCALAR",
    direction: str | None = None,
    unit: str = "",
    dag: tuple[str, ...] = (),
    behavior: str = "ANSWER",
    abstention_reason: str = "",
    entity_mode: str = "reporting",
    cross_scope_allowed: bool = False,
) -> dict[str, object]:
    return {
        "behavior": behavior,
        "root_operation": operation,
        "requested_years": list(years),
        "column_roles": list(roles),
        "operand_slots": list(slots),
        "answer_mode": answer_mode,
        "direction_intent": direction,
        "requested_unit_family": unit,
        "program_dag_operations": list(dag or ((operation,) if operation else ())),
        "entity_mode": entity_mode,
        "cross_scope_allowed": cross_scope_allowed,
        "abstention_reason": abstention_reason,
    }


def _spec(
    family_id: str,
    category: str,
    level: str,
    question: str,
    expected: dict[str, object],
) -> dict[str, object]:
    return {
        "family_id": family_id,
        "category": category,
        "level": level,
        "question": question,
        "expected": expected,
    }


def _base_specs() -> list[dict[str, object]]:
    specs: list[dict[str, object]] = [
        _spec("L01", "lookup", "Q0", "Doanh thu thuần năm 2023 là bao nhiêu?", _expected("SELECT", years=("2023",), slots=(_slot("doanh thu thuần", "2023"),))),
        _spec("L02", "lookup", "Q1", "Doanh thu thực hiện năm 2023 là bao nhiêu?", _expected("SELECT", years=("2023",), roles=("ACTUAL",), slots=(_slot("doanh thu", "2023", "ACTUAL"),))),
        _spec("L03", "lookup", "Q1", "Doanh thu kế hoạch năm 2023 là bao nhiêu?", _expected("SELECT", years=("2023",), roles=("PLAN",), slots=(_slot("doanh thu", "2023", "PLAN"),))),
        _spec("L04", "lookup", "Q1", "Kế hoạch điều chỉnh doanh thu năm 2024 là bao nhiêu?", _expected("SELECT", years=("2024",), roles=("REVISED_PLAN",), slots=(_slot("doanh thu", "2024", "REVISED_PLAN"),))),
        _spec("L05", "lookup", "Q1", "Tiền và tương đương tiền cuối kỳ là bao nhiêu?", _expected("SELECT", slots=(_slot("tiền và tương đương tiền"),))),
        _spec("L06", "lookup", "Q1", "Tổng tài sản tại ngày 31/12/2023 là bao nhiêu?", _expected("SELECT", years=("2023",), slots=(_slot("tổng tài sản", "2023"),))),
        _spec("L07", "lookup", "Q1", "Doanh thu thực hiện năm 2023 theo đơn vị triệu đồng là bao nhiêu?", _expected("SELECT", years=("2023",), roles=("ACTUAL",), slots=(_slot("doanh thu", "2023", "ACTUAL"),), unit="currency")),
        _spec("L08", "lookup", "Q1", "Doanh thu năm 2023 của Công ty cổ phần ABC là bao nhiêu?", _expected("SELECT", years=("2023",), slots=(_slot("doanh thu", "2023", scope="named"),), entity_mode="named")),
        _spec("L09", "lookup", "Q1", "Lợi nhuận sau thuế hợp nhất năm 2023 là bao nhiêu?", _expected("SELECT", years=("2023",), slots=(_slot("lợi nhuận sau thuế", "2023", scope="consolidated"),))),
        _spec("L10", "lookup", "Q5", "Doanh thu hoạt động năm 2023 là bao nhiêu khi bảng có hai panel cùng tên chỉ tiêu?", _expected(None, years=("2023",), behavior="ABSTAIN", abstention_reason="logical_cell_not_unique")),
        _spec("A01", "single_step", "Q2", "Tổng nợ ngắn hạn và nợ dài hạn năm 2023 là bao nhiêu?", _expected("ADD", years=("2023",), slots=(_slot("nợ ngắn hạn", "2023"), _slot("nợ dài hạn", "2023")))),
        _spec("A02", "single_step", "Q3", "Tổng tiền, đầu tư ngắn hạn và hàng tồn kho năm 2023 là bao nhiêu?", _expected("SUM3", years=("2023",), slots=(_slot("tiền", "2023"), _slot("đầu tư ngắn hạn", "2023"), _slot("hàng tồn kho", "2023")))),
        _spec("A03", "single_step", "Q2", "Doanh thu năm 2023 thay đổi bao nhiêu so với năm 2022?", _expected("SUBTRACT", years=("2023", "2022"), direction="NEW_MINUS_OLD", slots=(_slot("doanh thu", "2023"), _slot("doanh thu", "2022")))),
        _spec("A04", "single_step", "Q2", "Doanh thu năm 2023 thấp hơn năm 2022 bao nhiêu?", _expected("SUBTRACT", years=("2023", "2022"), direction="DECREASE_AMOUNT", slots=(_slot("doanh thu", "2022"), _slot("doanh thu", "2023")))),
        _spec("A05", "single_step", "Q2", "Doanh thu năm 2023 tăng bao nhiêu phần trăm so với năm 2022?", _expected("PERCENT_CHANGE", years=("2023", "2022"), direction="NEW_MINUS_OLD", unit="percent", slots=(_slot("doanh thu", "2023"), _slot("doanh thu", "2022")))),
        _spec("A06", "single_step", "Q2", "Lợi nhuận sau thuế chiếm bao nhiêu phần trăm doanh thu thuần năm 2023?", _expected("RATIO_PERCENT", years=("2023",), unit="percent", slots=(_slot("lợi nhuận sau thuế", "2023"), _slot("doanh thu thuần", "2023")))),
        _spec("A07", "single_step", "Q2", "Hệ số nợ trên tổng tài sản năm 2023 là bao nhiêu?", _expected(None, years=("2023",), answer_mode="SINGLE_SCALAR", unit="ratio", dag=("SELECT_OR_DIVIDE",), slots=(_slot("nợ", "2023"), _slot("tổng tài sản", "2023")))),
        _spec("A08", "single_step", "Q2", "Tài sản bình quân giữa đầu kỳ và cuối kỳ năm 2023 là bao nhiêu?", _expected("AVERAGE", years=("2023",), slots=(_slot("tài sản", "2023"), _slot("tài sản", "2023")))),
        _spec("A09", "single_step", "Q2", "Số cổ phiếu đang lưu hành nhân với mệnh giá là bao nhiêu?", _expected("MULTIPLY", slots=(_slot("số cổ phiếu đang lưu hành"), _slot("mệnh giá")))),
        _spec("A10", "single_step", "Q2", "Tổng tài sản có bằng tổng nguồn vốn năm 2023 không, nếu lệch thì chênh lệch là bao nhiêu?", _expected("SUBTRACT", years=("2023",), direction="LEFT_MINUS_RIGHT", answer_mode="SCALAR_WITH_DERIVED_LABEL", slots=(_slot("tổng tài sản", "2023"), _slot("tổng nguồn vốn", "2023")))),
    ]

    pa = [
        ("PA01", "Q1", "Doanh thu hoạt động thực hiện năm 2023 là bao nhiêu?", _expected("SELECT", years=("2023",), roles=("ACTUAL",), slots=(_slot("doanh thu hoạt động", "2023", "ACTUAL"),))),
        ("PA02", "Q1", "Doanh thu hoạt động kế hoạch năm 2023 là bao nhiêu?", _expected("SELECT", years=("2023",), roles=("PLAN",), slots=(_slot("doanh thu hoạt động", "2023", "PLAN"),))),
        ("PA03", "Q2", "Doanh thu hoạt động thực hiện năm 2023 đạt bao nhiêu phần trăm kế hoạch năm 2023?", _expected("RATIO_PERCENT", years=("2023",), roles=("ACTUAL", "PLAN"), unit="percent", slots=(_slot("doanh thu hoạt động", "2023", "ACTUAL"), _slot("doanh thu hoạt động", "2023", "PLAN")))),
        ("PA04", "Q2", "Doanh thu hoạt động thực hiện năm 2023 trừ kế hoạch năm 2023 là bao nhiêu?", _expected("SUBTRACT", years=("2023",), roles=("ACTUAL", "PLAN"), direction="LEFT_MINUS_RIGHT", slots=(_slot("doanh thu hoạt động", "2023", "ACTUAL"), _slot("doanh thu hoạt động", "2023", "PLAN")))),
        ("PA05", "Q2", "Doanh thu hoạt động thực hiện năm 2023 thấp hơn kế hoạch năm 2023 bao nhiêu đồng?", _expected("SUBTRACT", years=("2023",), roles=("PLAN", "ACTUAL"), direction="DECREASE_AMOUNT", unit="currency", slots=(_slot("doanh thu hoạt động", "2023", "PLAN"), _slot("doanh thu hoạt động", "2023", "ACTUAL")))),
        ("PA06", "Q2", "Doanh thu hoạt động thực hiện năm 2023 vượt kế hoạch năm 2023 bao nhiêu đồng?", _expected("SUBTRACT", years=("2023",), roles=("ACTUAL", "PLAN"), direction="LEFT_MINUS_RIGHT", unit="currency", slots=(_slot("doanh thu hoạt động", "2023", "ACTUAL"), _slot("doanh thu hoạt động", "2023", "PLAN")))),
        ("PA07", "Q2", "Doanh thu hoạt động thực hiện năm 2023 thay đổi bao nhiêu phần trăm so với kế hoạch năm 2023?", _expected("PERCENT_CHANGE", years=("2023",), roles=("ACTUAL", "PLAN"), direction="LEFT_MINUS_RIGHT", unit="percent", slots=(_slot("doanh thu hoạt động", "2023", "ACTUAL"), _slot("doanh thu hoạt động", "2023", "PLAN")))),
        ("PA08", "Q3", "Doanh thu hoạt động thực hiện năm 2023 còn thiếu bao nhiêu phần trăm để đạt kế hoạch năm 2023?", _expected("RATIO_PERCENT", years=("2023",), roles=("PLAN", "ACTUAL", "PLAN"), unit="percent", dag=("SUBTRACT", "RATIO_PERCENT"), slots=(_slot("doanh thu hoạt động", "2023", "PLAN"), _slot("doanh thu hoạt động", "2023", "ACTUAL"), _slot("doanh thu hoạt động", "2023", "PLAN")))),
        ("PA09", "Q3", "Doanh thu hoạt động thực hiện năm 2023 vượt kế hoạch năm 2023 bao nhiêu phần trăm?", _expected("RATIO_PERCENT", years=("2023",), roles=("ACTUAL", "PLAN", "PLAN"), unit="percent", dag=("SUBTRACT", "RATIO_PERCENT"), slots=(_slot("doanh thu hoạt động", "2023", "ACTUAL"), _slot("doanh thu hoạt động", "2023", "PLAN"), _slot("doanh thu hoạt động", "2023", "PLAN")))),
        ("PA10", "Q3", "Kế hoạch doanh thu năm 2024 tăng bao nhiêu phần trăm so với thực hiện năm 2023?", _expected("PERCENT_CHANGE", years=("2024", "2023"), roles=("PLAN", "ACTUAL"), direction="LEFT_MINUS_RIGHT", unit="percent", slots=(_slot("doanh thu", "2024", "PLAN"), _slot("doanh thu", "2023", "ACTUAL")))),
        ("PA11", "Q3", "Tổng doanh thu và lợi nhuận thực hiện năm 2023 đạt bao nhiêu phần trăm tổng doanh thu và lợi nhuận kế hoạch năm 2023?", _expected("RATIO_PERCENT", years=("2023",), roles=("ACTUAL", "ACTUAL", "PLAN", "PLAN"), unit="percent", dag=("ADD", "ADD", "RATIO_PERCENT"), slots=(_slot("doanh thu", "2023", "ACTUAL"), _slot("lợi nhuận", "2023", "ACTUAL"), _slot("doanh thu", "2023", "PLAN"), _slot("lợi nhuận", "2023", "PLAN")))),
        ("PA12", "Q3", "Doanh thu công ty mẹ thực hiện năm 2023 đạt bao nhiêu phần trăm kế hoạch hợp nhất năm 2023?", _expected("RATIO_PERCENT", years=("2023",), roles=("ACTUAL", "PLAN"), unit="percent", slots=(_slot("doanh thu", "2023", "ACTUAL", "parent"), _slot("doanh thu", "2023", "PLAN", "consolidated")), cross_scope_allowed=True)),
    ]
    specs.extend(_spec(fid, "plan_actual", level, question, expected) for fid, level, question, expected in pa)

    c_specs = [
        ("C01", "Q3", "Theo báo cáo hợp nhất của Công ty cổ phần ABC, doanh thu hoạt động thực hiện năm 2023 theo đơn vị triệu đồng đạt bao nhiêu phần trăm kế hoạch cùng năm?", _expected("RATIO_PERCENT", years=("2023",), roles=("ACTUAL", "PLAN"), unit="percent", entity_mode="named", slots=(_slot("doanh thu hoạt động", "2023", "ACTUAL", "consolidated"), _slot("doanh thu hoạt động", "2023", "PLAN", "consolidated")))),
        ("C02", "Q3", "Theo bảng kết quả hoạt động sản xuất kinh doanh và căn cứ nghị quyết đã được phê duyệt, doanh thu thực hiện năm 2023 đạt bao nhiêu phần trăm kế hoạch cùng năm?", _expected("RATIO_PERCENT", years=("2023",), roles=("ACTUAL", "PLAN"), unit="percent")),
        ("C03", "Q2", "Tính hệ số thanh toán hiện hành (Current Ratio) năm 2023 theo đơn vị lần.", _expected(None, years=("2023",), unit="ratio", dag=("SELECT_OR_DIVIDE",))),
        ("C04", "Q3", "Kế hoạch doanh thu năm 2024 tăng bao nhiêu phần trăm so với thực hiện năm 2023?", _expected("PERCENT_CHANGE", years=("2024", "2023"), roles=("PLAN", "ACTUAL"), unit="percent", direction="LEFT_MINUS_RIGHT")),
        ("C05", "Q3", "Lợi nhuận thực hiện năm 2023 chiếm bao nhiêu phần trăm doanh thu kế hoạch năm 2023?", _expected("RATIO_PERCENT", years=("2023",), roles=("ACTUAL", "PLAN"), unit="percent")),
        ("C06", "Q3", "Tổng doanh thu và lợi nhuận thực hiện đạt bao nhiêu phần trăm tổng doanh thu và lợi nhuận kế hoạch năm 2023?", _expected("RATIO_PERCENT", years=("2023",), roles=("ACTUAL", "ACTUAL", "PLAN", "PLAN"), unit="percent", dag=("ADD", "ADD", "RATIO_PERCENT"))),
        ("C07", "Q4", "Doanh thu năm 2023 thay đổi bao nhiêu phần trăm so với năm 2022 và vì sao?", _expected("PERCENT_CHANGE", years=("2023", "2022"), unit="percent", answer_mode="SCALAR_WITH_EVIDENCE_LIMITED_EXPLANATION", direction="NEW_MINUS_OLD")),
        ("C08", "Q4", "Doanh thu thực hiện năm 2023 là bao nhiêu, kế hoạch là bao nhiêu và tỷ lệ hoàn thành là bao nhiêu phần trăm?", _expected(None, years=("2023",), roles=("ACTUAL", "PLAN"), answer_mode="MULTI_ANSWER", dag=("SELECT", "SELECT", "RATIO_PERCENT"))),
        ("C09", "Q4", "Doanh thu năm 2023 đã hoàn thành kế hoạch chưa và vượt hoặc thiếu bao nhiêu?", _expected("SUBTRACT", years=("2023",), roles=("ACTUAL", "PLAN"), answer_mode="SCALAR_WITH_DERIVED_LABEL")),
        ("C10", "Q4", "Doanh thu thực hiện năm 2023 sau khi loại doanh thu tài chính là bao nhiêu?", _expected("SUBTRACT", years=("2023",), roles=("ACTUAL",), direction="LEFT_MINUS_RIGHT")),
        ("C11", "Q5", "Chỉ tiêu này trong năm đó là bao nhiêu?", _expected(None, behavior="ABSTAIN", abstention_reason="missing_conversation_state")),
        ("C12", "Q5", "So sánh doanh thu năm 2023 giữa báo cáo Công ty ABC và báo cáo Công ty XYZ.", _expected(None, years=("2023",), behavior="ABSTAIN", abstention_reason="multi_document_protocol_required")),
        ("C13", "Q5", "Chênh lệch giữa doanh thu thực hiện và kế hoạch năm 2023 là bao nhiêu?", _expected(None, years=("2023",), roles=("ACTUAL", "PLAN"), behavior="ABSTAIN", abstention_reason="ambiguous_signed_direction")),
        ("C14", "Q5", "Doanh thu năm 2023 lớn hơn bao nhiêu phần trăm?", _expected(None, years=("2023",), unit="percent", behavior="ABSTAIN", abstention_reason="missing_reference_base")),
        ("C15", "Q5", "Doanh thu hoạt động năm 2023 là bao nhiêu khi bảng có cả kế hoạch và thực hiện?", _expected(None, years=("2023",), behavior="ABSTAIN", abstention_reason="missing_column_role")),
    ]
    specs.extend(_spec(fid, "complex", level, question, expected) for fid, level, question, expected in c_specs)

    ty = [
        ("TY01", "Q2", "Doanh thu thực hiện năm 2023 tăng bao nhiêu so với thực hiện năm 2022?", _expected("SUBTRACT", years=("2023", "2022"), roles=("ACTUAL", "ACTUAL"), direction="NEW_MINUS_OLD", slots=(_slot("doanh thu", "2023", "ACTUAL"), _slot("doanh thu", "2022", "ACTUAL")))),
        ("TY02", "Q2", "Doanh thu thực hiện năm 2023 tăng bao nhiêu phần trăm so với thực hiện năm 2022?", _expected("PERCENT_CHANGE", years=("2023", "2022"), roles=("ACTUAL", "ACTUAL"), direction="NEW_MINUS_OLD", unit="percent", slots=(_slot("doanh thu", "2023", "ACTUAL"), _slot("doanh thu", "2022", "ACTUAL")))),
        ("TY03", "Q2", "Kế hoạch doanh thu năm 2024 tăng bao nhiêu so với kế hoạch năm 2023?", _expected("SUBTRACT", years=("2024", "2023"), roles=("PLAN", "PLAN"), direction="NEW_MINUS_OLD", slots=(_slot("doanh thu", "2024", "PLAN"), _slot("doanh thu", "2023", "PLAN")))),
        ("TY04", "Q3", "Kế hoạch doanh thu năm 2024 cao hơn thực hiện năm 2023 bao nhiêu phần trăm?", _expected("PERCENT_CHANGE", years=("2024", "2023"), roles=("PLAN", "ACTUAL"), direction="LEFT_MINUS_RIGHT", unit="percent", slots=(_slot("doanh thu", "2024", "PLAN"), _slot("doanh thu", "2023", "ACTUAL")))),
        ("TY05", "Q4", "Tỷ lệ hoàn thành kế hoạch doanh thu năm 2022 và năm 2023 là bao nhiêu?", _expected(None, years=("2022", "2023"), roles=("ACTUAL", "PLAN", "ACTUAL", "PLAN"), answer_mode="MULTI_ANSWER", unit="percent", dag=("RATIO_PERCENT", "RATIO_PERCENT"))),
        ("TY06", "Q3", "Tỷ lệ hoàn thành kế hoạch doanh thu năm 2023 cao hơn năm 2022 bao nhiêu điểm phần trăm?", _expected("SUBTRACT", years=("2023", "2022"), roles=("ACTUAL", "PLAN", "ACTUAL", "PLAN"), unit="percentage_point", dag=("RATIO_PERCENT", "RATIO_PERCENT", "SUBTRACT"))),
        ("TY07", "Q3", "Tỷ lệ hoàn thành kế hoạch doanh thu năm 2023 tăng bao nhiêu phần trăm so với năm 2022?", _expected("PERCENT_CHANGE", years=("2023", "2022"), roles=("ACTUAL", "PLAN", "ACTUAL", "PLAN"), unit="percent", dag=("RATIO_PERCENT", "RATIO_PERCENT", "PERCENT_CHANGE"))),
        ("TY08", "Q4", "Năm 2022 hay năm 2023 có tỷ lệ hoàn thành kế hoạch doanh thu cao hơn và cao hơn bao nhiêu?", _expected("SUBTRACT", years=("2022", "2023"), roles=("ACTUAL", "PLAN", "ACTUAL", "PLAN"), answer_mode="SCALAR_WITH_DERIVED_LABEL", dag=("RATIO_PERCENT", "RATIO_PERCENT", "COMPARE", "SUBTRACT"))),
        ("TY09", "Q4", "Cho biết thực hiện, kế hoạch và tỷ lệ hoàn thành doanh thu của cả năm 2022 và năm 2023.", _expected(None, years=("2022", "2023"), roles=("ACTUAL", "PLAN", "ACTUAL", "PLAN"), answer_mode="MULTI_ANSWER", dag=("SELECT", "SELECT", "SELECT", "SELECT", "RATIO_PERCENT", "RATIO_PERCENT"))),
        ("TY10", "Q3", "Tổng doanh thu và lợi nhuận thực hiện năm 2023 tăng bao nhiêu phần trăm so với tổng doanh thu và lợi nhuận thực hiện năm 2022?", _expected("PERCENT_CHANGE", years=("2023", "2022"), roles=("ACTUAL", "ACTUAL", "ACTUAL", "ACTUAL"), unit="percent", dag=("ADD", "ADD", "PERCENT_CHANGE"))),
        ("TY11", "Q3", "Biên lợi nhuận thực hiện năm 2023 so với năm 2022 thay đổi bao nhiêu điểm phần trăm?", _expected("SUBTRACT", years=("2023", "2022"), roles=("ACTUAL", "ACTUAL"), unit="percentage_point", dag=("RATIO_PERCENT", "RATIO_PERCENT", "SUBTRACT"))),
        ("TY12", "Q3", "Doanh thu công ty mẹ thực hiện năm 2023 tăng bao nhiêu phần trăm so với doanh thu hợp nhất thực hiện năm 2022?", _expected("PERCENT_CHANGE", years=("2023", "2022"), roles=("ACTUAL", "ACTUAL"), unit="percent", cross_scope_allowed=True, slots=(_slot("doanh thu", "2023", "ACTUAL", "parent"), _slot("doanh thu", "2022", "ACTUAL", "consolidated")))),
    ]
    specs.extend(_spec(fid, "two_year", level, question, expected) for fid, level, question, expected in ty)
    return specs


def _without_diacritics(text: str) -> str:
    text = text.replace("Đ", "D").replace("đ", "d")
    return "".join(
        char for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )


def _swap_roles_text(text: str) -> str:
    replacements = (
        (r"\bthực hiện\b", "__ROLE_ACTUAL__"),
        (r"\bthực tế\b", "__ROLE_ACTUAL__"),
        (r"\bkế hoạch\b", "__ROLE_PLAN__"),
        (r"\bactual\b", "__ROLE_ACTUAL__"),
        (r"\bplan\b", "__ROLE_PLAN__"),
    )
    result = text
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result.replace("__ROLE_ACTUAL__", "kế hoạch").replace("__ROLE_PLAN__", "thực hiện")


def _swap_roles_expected(expected: dict[str, object]) -> dict[str, object]:
    value = deepcopy(expected)
    swap = {"ACTUAL": "PLAN", "PLAN": "ACTUAL"}
    value["column_roles"] = [swap.get(str(role), role) for role in value["column_roles"]]
    for slot in value["operand_slots"]:
        slot["column_role"] = swap.get(str(slot.get("column_role", "")), slot.get("column_role", ""))
    return value


def _shift_years_text(text: str) -> str:
    return re.sub(r"\b(20\d{2})\b", lambda match: str(int(match.group(1)) + 1), text)


def _shift_years_expected(expected: dict[str, object]) -> dict[str, object]:
    value = deepcopy(expected)
    value["requested_years"] = [str(int(year) + 1) for year in value["requested_years"]]
    for slot in value["operand_slots"]:
        year = str(slot.get("year", ""))
        if year:
            slot["year"] = str(int(year) + 1)
    return value


def _strip_roles(text: str) -> str:
    value = re.sub(r"\b(?:thực hiện|thực tế|kế hoạch|actual|plan)\b", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", value).strip()


def _record(spec: dict[str, object], variant: str, question: str, expected: dict[str, object]) -> dict[str, object]:
    return {
        "schema": "question-family-contract-v1",
        "classification": "development_only_synthetic_compiler_contract",
        "example_id": f"{spec['family_id']}-{variant}",
        "family_id": spec["family_id"],
        "category": spec["category"],
        "difficulty_level": spec["level"],
        "variant": variant,
        "question": question,
        "expected_ir": expected,
        "document_content_used": False,
        "numeric_answer_used": False,
    }


def _expanded_records(specs: list[dict[str, object]]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    expanded = {f"PA{index:02d}" for index in range(1, 13)} | {
        f"TY{index:02d}" for index in range(1, 13)
    }
    for spec in specs:
        question = str(spec["question"])
        expected = deepcopy(spec["expected"])
        variants: list[tuple[str, str, dict[str, object]]] = [("canonical", question, expected)]
        if spec["family_id"] in expanded:
            lower = question[0].lower() + question[1:]
            variants.extend(
                [
                    ("paraphrase_1", f"Hãy cho biết {lower}", deepcopy(expected)),
                    ("paraphrase_2", f"Theo báo cáo, {lower}", deepcopy(expected)),
                    ("paraphrase_3", f"Dựa trên số liệu được công bố, {lower}", deepcopy(expected)),
                    (
                        "long_clause",
                        "Căn cứ đúng bảng số liệu tài chính trong báo cáo và giữ nguyên năm, "
                        "đơn vị, phạm vi cùng vai trò của từng cột, hãy xác định: " + lower,
                        deepcopy(expected),
                    ),
                    ("no_diacritics", _without_diacritics(question), deepcopy(expected)),
                    (
                        "clause_reordered",
                        "Đối với cùng chỉ tiêu và phạm vi báo cáo, " + lower,
                        deepcopy(expected),
                    ),
                    (
                        "minimal_pair_role_swap",
                        _swap_roles_text(question),
                        _swap_roles_expected(expected),
                    ),
                    (
                        "minimal_pair_year_shift",
                        _shift_years_text(question),
                        _shift_years_expected(expected),
                    ),
                    (
                        "hard_negative_missing_role",
                        _strip_roles(question),
                        _expected(
                            None,
                            years=tuple(expected["requested_years"]),
                            behavior="ABSTAIN",
                            abstention_reason="missing_required_column_role",
                        ),
                    ),
                    (
                        "ambiguity_control",
                        "Chỉ tiêu tài chính trong năm được hỏi là bao nhiêu khi bảng có nhiều cột "
                        "cùng năm nhưng câu hỏi không xác định vai trò cột?",
                        _expected(None, behavior="ABSTAIN", abstention_reason="ambiguous_metric_or_column_role"),
                    ),
                ]
            )
        records.extend(_record(spec, variant, text, target) for variant, text, target in variants)
    return records


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the frozen QP0 question-family contract catalog")
    parser.add_argument("--output", type=Path, default=ROOT / "config" / "question_family_catalog_plan_actual_v1.jsonl")
    parser.add_argument("--lock", type=Path, default=ROOT / "reports" / "question_family_catalog_plan_actual_v1_lock.json")
    args = parser.parse_args()

    specs = _base_specs()
    records = _expanded_records(specs)
    example_ids = [str(row["example_id"]) for row in records]
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("Duplicate catalog example_id")
    family_ids = {str(row["family_id"]) for row in records}
    expected_families = {
        *(f"L{index:02d}" for index in range(1, 11)),
        *(f"A{index:02d}" for index in range(1, 11)),
        *(f"PA{index:02d}" for index in range(1, 13)),
        *(f"C{index:02d}" for index in range(1, 16)),
        *(f"TY{index:02d}" for index in range(1, 13)),
    }
    if family_ids != expected_families:
        raise ValueError(f"Family coverage mismatch: {sorted(expected_families - family_ids)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    counts: dict[str, int] = {}
    for row in records:
        counts[str(row["category"])] = counts.get(str(row["category"]), 0) + 1
    lock = {
        "schema": "question-family-catalog-plan-actual-lock-v1",
        "classification": "development_only_synthetic_compiler_contract",
        "catalog": str(args.output.relative_to(ROOT)),
        "catalog_sha256": _digest(args.output),
        "builder_sha256": _digest(Path(__file__)),
        "family_count": len(family_ids),
        "example_count": len(records),
        "category_counts": dict(sorted(counts.items())),
        "pa_variants_per_family": 11,
        "ty_variants_per_family": 11,
        "document_content_used": False,
        "numeric_answer_used": False,
        "global_validation_loaded": False,
        "sealed_validation_loaded": False,
        "blind_test_loaded": False,
    }
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    args.lock.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(lock, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
