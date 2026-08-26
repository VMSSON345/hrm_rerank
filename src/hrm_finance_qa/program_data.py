from __future__ import annotations

import itertools
import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation, localcontext
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Sequence

from .ontology import normalize_text
from .financial_identities import get_financial_identity
from .flattened_table_reconstruction import qp3d_manifest_from_flattened_record
from .question_planner import compile_question_plan


OPERATIONS: tuple[str, ...] = (
    "SELECT",
    "ADD",
    "SUBTRACT",
    "MULTIPLY",
    "DIVIDE",
    "RATIO_PERCENT",
    "PERCENT_CHANGE",
    "AVERAGE",
    "SUM3",
    "SUM4",
)
OPERATION_TO_ID = {name: index for index, name in enumerate(OPERATIONS)}
OPERATION_ARITY = {
    "SELECT": 1,
    "ADD": 2,
    "SUBTRACT": 2,
    "MULTIPLY": 2,
    "DIVIDE": 2,
    "RATIO_PERCENT": 2,
    "PERCENT_CHANGE": 2,
    "AVERAGE": 2,
    "SUM3": 3,
    "SUM4": 4,
}
COMMUTATIVE = {"ADD", "MULTIPLY", "AVERAGE", "SUM3", "SUM4"}
MAX_OPERANDS = 4


def currency_unit_scale(unit: str) -> Decimal | None:
    """Return the explicit currency scale without converting disclosed values."""

    normalized = f" {normalize_text(unit)} "
    if not normalized.strip():
        return None
    if " ty " in normalized and " dong " in normalized:
        return Decimal("1000000000")
    if " trieu " in normalized and " dong " in normalized:
        return Decimal("1000000")
    if (
        " nghin " in normalized
        or " ngan " in normalized
        or " 000 " in normalized
    ) and " dong " in normalized:
        return Decimal("1000")
    if any(marker in normalized for marker in (" vnd ", " dong ")):
        return Decimal(1)
    return None


class ProgramDataError(ValueError):
    pass


@dataclass(frozen=True)
class NumericCandidate:
    index: int
    value: str
    raw: str
    context: str
    line_number: int
    char_start: int
    char_end: int
    unit: str
    is_percent: bool
    is_year_like: bool
    relevance: float
    source_name: str = "context"
    page: int | None = None
    extraction_method: str = ""
    row_label: str = ""
    column_label: str = ""
    period: str = ""
    row_type: str = ""
    table_id: str = ""
    section: str = ""
    scope: str = ""
    entity_id: str = ""
    subtable_id: str = ""
    source_span: tuple[int, int] | None = None
    derived_operation: str = ""
    derived_operands: tuple[int, ...] = ()
    derivation_id: str = ""

    def __post_init__(self) -> None:
        fields = {
            "row_label": "ROW",
            "column_label": "COLUMN",
            "period": "PERIOD",
            "row_type": "ROW_TYPE",
            "table_id": "TABLE",
            "section": "SECTION",
            "scope": "SCOPE",
            "entity_id": "ENTITY",
            "subtable_id": "SUBTABLE",
        }
        for attribute, label in fields.items():
            if getattr(self, attribute):
                continue
            match = re.search(
                rf"(?im)(?:^|\n){label}:\s*(.*?)(?=\n[A-Z_]+:|$)",
                self.context,
            )
            if match:
                object.__setattr__(self, attribute, match.group(1).strip())
        if self.source_span is None:
            object.__setattr__(
                self, "source_span", (self.char_start, self.char_end)
            )
        elif not isinstance(self.source_span, tuple):
            object.__setattr__(self, "source_span", tuple(self.source_span))
        if not isinstance(self.derived_operands, tuple):
            object.__setattr__(
                self, "derived_operands", tuple(self.derived_operands)
            )

    @property
    def decimal(self) -> Decimal:
        return Decimal(self.value)


@dataclass(frozen=True)
class ReasoningProgram:
    operation: str
    operands: tuple[int, ...]

    def validate(self, candidate_count: int) -> None:
        if self.operation not in OPERATION_ARITY:
            raise ProgramDataError(f"Unknown operation: {self.operation}")
        expected = OPERATION_ARITY[self.operation]
        if len(self.operands) != expected:
            raise ProgramDataError(
                f"{self.operation} needs {expected} operands, got {len(self.operands)}"
            )
        if any(index < 0 or index >= candidate_count for index in self.operands):
            raise ProgramDataError("Operand pointer is outside the candidate list")
        if len(set(self.operands)) != len(self.operands):
            raise ProgramDataError("A program cannot reuse the same candidate pointer")


def canonicalize_program(program: ReasoningProgram) -> ReasoningProgram:
    """Return the unique pointer ordering used for exact program metrics.

    Pointer order is semantic for directional operations, but only a
    serialization choice for commutative operations.  Keeping the definition
    here lets runtime evaluation, supervision and reranker caches agree.
    """

    if program.operation not in COMMUTATIVE:
        return program
    operands = tuple(sorted(program.operands))
    if operands == program.operands:
        return program
    return ReasoningProgram(program.operation, operands)


def canonical_program_equal(
    predicted: ReasoningProgram, gold: ReasoningProgram
) -> bool:
    return canonicalize_program(predicted) == canonicalize_program(gold)


def canonical_pointer_equal(
    operation: str,
    predicted: Sequence[int],
    gold: Sequence[int],
) -> bool:
    if operation in COMMUTATIVE:
        return tuple(sorted(predicted)) == tuple(sorted(gold))
    return tuple(predicted) == tuple(gold)


DAG_OPERATION_ARITY = {**OPERATION_ARITY, "SHARE_OF_TOTAL": 2}


@dataclass(frozen=True)
class ProgramDAGInput:
    kind: str
    reference: int | str

    @classmethod
    def candidate(cls, index: int) -> "ProgramDAGInput":
        return cls("candidate", index)

    @classmethod
    def node(cls, node_id: str) -> "ProgramDAGInput":
        return cls("node", node_id)


@dataclass(frozen=True)
class ProgramDAGNode:
    node_id: str
    operation: str
    inputs: tuple[ProgramDAGInput, ...]
    result_unit_family: str = ""


@dataclass(frozen=True)
class ReasoningDAGProgram:
    """Topologically ordered executable program with raw-evidence leaves."""

    nodes: tuple[ProgramDAGNode, ...]
    output_node: str

    def validate(self, candidate_count: int) -> None:
        if not self.nodes:
            raise ProgramDataError("A DAG program must contain at least one node")
        produced: set[str] = set()
        for node in self.nodes:
            if not node.node_id or node.node_id in produced:
                raise ProgramDataError("DAG node IDs must be non-empty and unique")
            operation = node.operation.upper()
            if operation not in DAG_OPERATION_ARITY:
                raise ProgramDataError(
                    f"Unsupported DAG operation: {node.operation}"
                )
            if len(node.inputs) != DAG_OPERATION_ARITY[operation]:
                raise ProgramDataError(
                    f"{operation} needs {DAG_OPERATION_ARITY[operation]} inputs, "
                    f"got {len(node.inputs)}"
                )
            for item in node.inputs:
                if item.kind == "candidate":
                    if not isinstance(item.reference, int) or not (
                        0 <= item.reference < candidate_count
                    ):
                        raise ProgramDataError(
                            "DAG candidate input is outside the candidate list"
                        )
                elif item.kind == "node":
                    if not isinstance(item.reference, str) or item.reference not in produced:
                        raise ProgramDataError(
                            "DAG node inputs must reference an earlier node"
                        )
                else:
                    raise ProgramDataError(f"Unknown DAG input kind: {item.kind}")
            produced.add(node.node_id)
        if self.output_node not in produced:
            raise ProgramDataError("DAG output node is missing")

    @property
    def leaf_operands(self) -> tuple[int, ...]:
        output: list[int] = []
        for node in self.nodes:
            for item in node.inputs:
                if (
                    item.kind == "candidate"
                    and isinstance(item.reference, int)
                    and item.reference not in output
                ):
                    output.append(item.reference)
        return tuple(output)

    def to_dict(self) -> dict[str, object]:
        return {
            "nodes": [
                {
                    "id": node.node_id,
                    "operation": node.operation,
                    "inputs": [
                        {"kind": item.kind, "reference": item.reference}
                        for item in node.inputs
                    ],
                    "result_unit_family": node.result_unit_family,
                }
                for node in self.nodes
            ],
            "output_node": self.output_node,
        }

    def render(self) -> str:
        rendered: list[str] = []
        for node in self.nodes:
            inputs = ", ".join(
                f"candidate[{item.reference}]"
                if item.kind == "candidate"
                else str(item.reference)
                for item in node.inputs
            )
            rendered.append(f"{node.node_id}={node.operation}({inputs})")
        return " -> ".join(rendered)


@dataclass(frozen=True)
class ProgramExample:
    example_id: str
    question: str
    context: str
    answer_text: str
    candidates: tuple[NumericCandidate, ...]
    program: ReasoningProgram
    answer_value: str
    task: str = "QA"
    difficulty: str = ""
    supervision: dict[str, object] | None = None


@dataclass(frozen=True)
class ProgramBuildReport:
    source_rows: int
    eligible_rows: int
    numeric_rows: int
    induced_examples: int
    operation_counts: dict[str, int]
    skipped_no_target: int
    skipped_no_program: int
    source_task_counts: dict[str, int]
    rejection_counts: dict[str, int]
    rejection_examples: dict[str, tuple[str, ...]]


# Match one numeric cell at a time.  The previous permissive ``[\d .,'’]*``
# body merged note codes and adjacent PDF table columns, e.g.
# ``26 11.678.880.397 23.109.698.971`` became one 24-digit operand.
_NUMBER_BODY = r"(?:" + "|".join(
    (
        # Vietnamese thousands groups; tolerate PDF whitespace before dots.
        r"\d{1,3}(?:\s*\.\s?\d{3})+(?:,\d{1,2})?",
        # English thousands groups.
        r"\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?",
        # Thousands separated by ordinary spaces.
        r"\d{1,3}(?:[ \u00a0]\d{3})+",
        # Ungrouped integer or short decimal.
        r"\d+(?:[.,]\d{1,2})?",
    )
) + r")"
_NUMBER_RE = re.compile(
    rf"(?<![\w/])(?P<paren>\()?[-+]?{_NUMBER_BODY}(?P<close>\))?"
    r"\s*(?P<pct>%|phần\s*trăm)?",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")

_UNANSWERABLE_MARKERS = (
    "khong the tinh",
    "khong tinh duoc",
    "khong du du lieu",
    "khong co du du lieu",
    "khong du thong tin",
    "khong co du thong tin",
    "khong co du lieu",
    "khong co thong tin",
    "chua du du lieu",
    "chua du thong tin",
    "khong co trong chunk",
    "khong duoc cung cap",
    "khong xac dinh duoc",
    "thong tin khong co trong du lieu",
)
_NON_PROGRAM_QUESTION_MARKERS = (
    "khi nao",
    "tai sao",
    "vi sao",
    "muc dich",
    "ly do gi",
    "co can",
    "co duoc the chap",
)
_QUALITATIVE_QUESTION_MARKERS = (
    "phan tich",
    "danh gia",
    "giai thich",
    "nhan xet",
    "thao luan",
    "mo ta",
    "rui ro tiem an",
)
_ARITHMETIC_QUESTION_MARKERS = (
    "tinh ty le",
    "tinh toan ty le",
    "tinh ty suat",
    "tinh toan ty suat",
    "tren tong",
    "chiem bao nhieu phan tram",
    "tang truong",
    "muc tang",
    "muc giam",
    "chenh lech",
    "tang bao nhieu",
    "giam bao nhieu",
    "tong muc",
    "cong lai",
    "trung binh",
    "binh quan",
    "bao nhieu lan",
    "so sanh",
)


def parse_decimal(raw: str) -> Decimal | None:
    text = raw.strip().replace("\u00a0", " ")
    negative = text.startswith("(") and ")" in text
    text = re.sub(r"(?i)phần\s*trăm|%", "", text)
    text = text.strip(" ()+")
    text = text.replace("'", "").replace("’", "").replace(" ", "")
    if not text or not re.search(r"\d", text):
        return None
    # Vietnamese convention: dots group thousands and comma is decimal. English
    # reports often use commas for thousands and dot for decimal.
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind(".") and len(text) - text.rfind(",") <= 4:
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        pieces = text.split(",")
        if len(pieces) == 2 and 1 <= len(pieces[-1]) <= 2:
            text = pieces[0] + "." + pieces[1]
        else:
            text = "".join(pieces)
    elif "." in text:
        pieces = text.split(".")
        if len(pieces) > 2 or (len(pieces) == 2 and len(pieces[-1]) == 3):
            text = "".join(pieces)
    try:
        value = Decimal(text)
    except InvalidOperation:
        return None
    return -value if negative else value


def _is_natural_date_component(text: str, match: re.Match[str]) -> bool:
    prefix = normalize_text(text[max(0, match.start() - 24) : match.start()])
    suffix = normalize_text(text[match.end() : match.end() + 24])
    prefix_tokens = prefix.split()
    if "ngay" in prefix_tokens[-3:] and suffix.startswith("thang"):
        return True
    if "thang" in prefix_tokens[-3:] and suffix.startswith("nam"):
        return True
    return False


def _is_ordinal_number(text: str, match: re.Match[str], value: Decimal) -> bool:
    if value != value.to_integral_value() or not Decimal(0) <= value <= Decimal(100):
        return False
    before_cell = text[: match.start()].rsplit("|", 1)[-1].strip()
    suffix = text[match.end() :]
    return not before_cell and bool(re.match(r"\s*[.)](?:\s|$)", suffix))


def _is_context_metadata_number(
    line: str,
    match: re.Match[str],
    candidate_context: str,
    value: Decimal,
) -> bool:
    if _is_leading_table_code(line, match, value):
        return True
    if _is_leading_plaintext_code(line, match, value):
        return True
    if _is_plaintext_note_code(line, match, value):
        return True
    if _is_table_metadata_cell_code(line, match, value):
        return True
    prefix = normalize_text(line[max(0, match.start() - 32) : match.start()])
    # ``normalize_text`` removes punctuation, including the ``:`` inserted by
    # the MinerU table renderer, so match the normalized label itself.
    if re.search(r"(?:ma so|maso|thuyet minh|thuyetminh|stt)\s*$", prefix):
        return True
    column_line = next(
        (
            item
            for item in candidate_context.splitlines()
            if item.startswith("COLUMN:")
        ),
        "",
    )
    normalized_column = normalize_text(column_line)
    if any(
        field in normalized_column
        for field in (
            "column ma so",
            "column maso",
            "column thuyet minh",
            "column thuyetminh",
            "column stt",
        )
    ):
        return True
    return _is_natural_date_component(line, match) or _is_ordinal_number(
        line, match, value
    )


def _is_table_metadata_cell_code(
    line: str, match: re.Match[str], value: Decimal
) -> bool:
    """Reject a small value placed in/after an OCR-damaged metadata cell."""

    if "|" not in line or value != value.to_integral_value():
        return False
    if not Decimal(0) <= value < Decimal(10000):
        return False
    cells = _markdown_cells(line)
    if not cells:
        return False
    cell_index = max(
        0,
        line[: match.start()].count("|")
        - (1 if line.lstrip().startswith("|") else 0),
    )
    if cell_index >= len(cells):
        return False
    metadata = {"ma so", "maso", "thuyet minh", "thuyetminh", "stt"}
    current = normalize_text(cells[cell_index])
    previous = normalize_text(cells[cell_index - 1]) if cell_index else ""
    if any(
        current == label
        or current.startswith(label + " ")
        or previous == label
        for label in metadata
    ):
        return True

    # A common statement layout is ``metric | row-code | note | current |
    # prior``. Even when the header is outside the chunk, a small integer after
    # a textual metric and before two financial-scale cells is structural
    # metadata, not an operand. This is the table analogue of the existing
    # plaintext note-code rule and is independent of company/report vocabulary.
    has_textual_label = any(
        re.search(r"[^\W\d_]", cell, re.UNICODE)
        for cell in cells[:cell_index]
    )
    following_values = 0
    for cell in cells[cell_index + 1 :]:
        for token in _NUMBER_RE.finditer(cell):
            parsed = parse_decimal(token.group(0))
            if parsed is not None and abs(parsed) >= Decimal(1000):
                following_values += 1
    return has_textual_label and following_values >= 2


def _is_leading_table_code(
    line: str, match: re.Match[str], value: Decimal
) -> bool:
    """Reject OCR row codes even when the table header was lost."""

    if "|" not in line or value != value.to_integral_value():
        return False
    if not Decimal(0) <= value < Decimal(10000):
        return False
    cells = _markdown_cells(line)
    if len(cells) < 2:
        return False
    raw_cell = line[match.start() : match.end()].strip()
    first_nonempty = next((cell for cell in cells if cell), "")
    if normalize_text(first_nonempty) != normalize_text(raw_cell):
        return False
    return any(re.search(r"[^\W\d_]", cell, re.UNICODE) for cell in cells[1:])


def _is_leading_plaintext_code(
    line: str, match: re.Match[str], value: Decimal
) -> bool:
    """Reject row/note codes before a metric label when PDF lost table pipes."""

    if "|" in line or value != value.to_integral_value():
        return False
    if not Decimal(0) <= value < Decimal(10000):
        return False
    first_letter = re.search(r"[^\W\d_]", line, re.UNICODE)
    if first_letter is None or match.end() > first_letter.start():
        return False
    # Require a financial-looking value after the textual label.  This avoids
    # treating arbitrary prose that starts with a small quantity as a row.
    suffix = line[first_letter.start() :]
    return any(
        parsed is not None and abs(parsed) >= Decimal(1000)
        for token in _NUMBER_RE.finditer(suffix)
        if (parsed := parse_decimal(token.group(0))) is not None
    )


def _is_plaintext_note_code(
    line: str, match: re.Match[str], value: Decimal
) -> bool:
    """Reject a small note reference before two native-PDF value columns."""

    if "|" in line or abs(value) >= Decimal(1000):
        return False
    if not re.search(r"[^\W\d_]", line[: match.start()], re.UNICODE):
        return False
    following_values = 0
    for token in _NUMBER_RE.finditer(line, match.end()):
        parsed = parse_decimal(token.group(0))
        if parsed is not None and abs(parsed) >= Decimal(1000):
            following_values += 1
    return following_values >= 2


def _is_answer_metadata_number(
    text: str,
    match: re.Match[str],
    value: Decimal,
) -> bool:
    if _is_probable_year(value) or _is_natural_date_component(text, match):
        return True
    prefix = normalize_text(text[max(0, match.start() - 28) : match.start()])
    if re.search(r"(?:chunk|ma so|muc)\s*$", prefix):
        return True
    return _is_ordinal_number(text, match, value)


def extract_numeric_candidates(
    question: str,
    context: str,
    *,
    max_candidates: int = 64,
    source_name: str = "context",
    page: int | None = None,
    source_line_start: int = 1,
    retrieval_score: float = 0.0,
    extraction_method: str = "",
    financial_identity_catalog_enabled: bool = False,
) -> tuple[NumericCandidate, ...]:
    if max_candidates < 4:
        raise ProgramDataError("max_candidates must be at least 4")
    lines = context.splitlines() or [context]
    question_tokens = _content_tokens(_question_focus_text(question))
    year_tokens = {token for token in normalize_text(question).split() if _YEAR_RE.match(token)}
    raw_candidates: list[NumericCandidate] = []
    page_unit = _explicit_page_unit(lines)
    for line_index, line in enumerate(lines):
        if not line.strip():
            continue
        if _is_table_header_or_separator(lines, line_index):
            continue
        normalized_line = normalize_text(line)
        line_tokens = set(normalized_line.split())
        overlap = _fuzzy_content_overlap(question_tokens, line_tokens)
        for match in _NUMBER_RE.finditer(line):
            if (match.start() > 0 and line[match.start() - 1] == "/") or (
                match.end() < len(line) and line[match.end()] == "/"
            ):
                continue
            raw = match.group(0).strip()
            value = parse_decimal(raw)
            if value is None:
                continue
            # Values with 25+ integer digits are almost certainly several OCR
            # table cells accidentally glued together, not financial operands.
            if value and value.adjusted() >= 24:
                continue
            digits = re.sub(r"\D", "", raw)
            is_year_like = bool(_YEAR_RE.match(digits))
            if is_year_like:
                # Years are period metadata, never arithmetic operands in the
                # single-result financial program schema.
                continue
            candidate_context = _candidate_context(
                lines,
                line_index,
                match.start(),
                match.end(),
                source_name=source_name,
                page=page,
                source_line_start=source_line_start,
                extraction_method=extraction_method,
            )
            is_percent = bool(match.group("pct")) or _is_percent_candidate_column(
                candidate_context
            )
            if _is_context_metadata_number(line, match, candidate_context, value):
                continue
            unit = _infer_unit(candidate_context, is_percent) or (
                "" if is_percent else page_unit
            )
            relevance = overlap + 0.20 * max(0.0, min(1.0, retrieval_score))
            column_line = next(
                (item for item in candidate_context.splitlines() if item.startswith("COLUMN:")),
                "",
            )
            normalized_column = normalize_text(column_line)
            if year_tokens and any(year in normalized_column for year in year_tokens):
                relevance += 0.35
            elif year_tokens and any(year in normalize_text(candidate_context) for year in year_tokens):
                relevance += 0.05
            normalized_question = normalize_text(question)
            if any(term in normalized_question for term in ("cuoi nam", "cuoi ky")) and any(
                term in normalized_column for term in ("cuoi nam", "cuoi ky", "31 12")
            ):
                relevance += 0.12
            if "dau nam" in normalized_question and any(
                term in normalized_column for term in ("dau nam", "01 01")
            ):
                relevance += 0.12
            if is_percent and any(
                term in normalize_text(question)
                for term in ("ty le", "ty suat", "phan tram", "tang truong", "bien dong")
            ):
                relevance += 0.15
            if "tong" in normalized_question and "row type total" in normalize_text(
                candidate_context
            ):
                relevance += 0.15
            if is_year_like and digits not in year_tokens:
                relevance -= 0.30
            # Values on table rows get a small boost over metadata dates/codes.
            if "|" in line:
                relevance += 0.12
            if re.search(r"(?i)ma\s*so|thuyet\s*minh", line) and abs(value) < 1000:
                relevance -= 0.15
            raw_candidates.append(
                NumericCandidate(
                    index=-1,
                    value=_decimal_string(value),
                    raw=raw,
                    context=candidate_context,
                    line_number=source_line_start + line_index,
                    char_start=match.start(),
                    char_end=match.end(),
                    unit=unit,
                    is_percent=is_percent,
                    is_year_like=is_year_like,
                    relevance=relevance,
                    source_name=source_name,
                    page=page,
                    extraction_method=extraction_method,
                    entity_id=_source_entity_id(
                        source_name,
                        candidate_context,
                    ),
                )
            )
    raw_candidates.extend(
        _implied_plaintext_blank_zero_candidates(
            question,
            lines,
            source_name=source_name,
            page=page,
            source_line_start=source_line_start,
            retrieval_score=retrieval_score,
            extraction_method=extraction_method,
            page_unit=page_unit,
        )
    )
    # Keep document order among the strongest candidates. This avoids answer
    # leakage: selection depends on the question and source only.
    if len(raw_candidates) > max_candidates:
        strongest = _candidate_quota_indices(
            question,
            raw_candidates,
            max_candidates,
            prefer_magnitude=True,
            financial_identity_catalog_enabled=(
                financial_identity_catalog_enabled
            ),
        )
        strongest_set = set(strongest)
        raw_candidates = [
            candidate
            for index, candidate in enumerate(raw_candidates)
            if index in strongest_set
        ]
    return tuple(
        NumericCandidate(**{**candidate.__dict__, "index": index})
        for index, candidate in enumerate(raw_candidates)
    )


def _implied_plaintext_blank_zero_candidates(
    question: str,
    lines: Sequence[str],
    *,
    source_name: str,
    page: int | None,
    source_line_start: int,
    retrieval_score: float,
    extraction_method: str,
    page_unit: str,
) -> list[NumericCandidate]:
    """Recover a prior zero only when table structure and prose both support it."""

    plan = compile_question_plan(question)
    if (
        plan.operation not in {"SUBTRACT", "PERCENT_CHANGE"}
        or not plan.temporal_change
    ):
        return []

    question_terms = _metric_alignment_tokens(plan.metric_text)
    output: list[NumericCandidate] = []
    for line_index, line in enumerate(lines):
        if "|" in line or not line.strip():
            continue
        matches: list[tuple[re.Match[str], Decimal]] = []
        for token in _NUMBER_RE.finditer(line):
            parsed = parse_decimal(token.group(0))
            if parsed is not None and abs(parsed) >= Decimal(1000):
                matches.append((token, parsed))
        if len(matches) != 1:
            continue
        first_letter = re.search(r"[^\W\d_]", line, re.UNICODE)
        if first_letter is None:
            continue
        label = line[first_letter.start() : matches[0][0].start()]
        label = re.sub(
            r"(?:\s+\d+(?:[.,]\d{1,2})?)+\s*$",
            "",
            label,
        ).strip(" .:-")
        if not label:
            continue
        row_terms = _metric_alignment_tokens(label)
        if len(question_terms & row_terms) < min(2, len(row_terms)):
            continue

        period_labels = _nearby_plaintext_period_labels(lines, line_index)
        if len(period_labels) != 2:
            continue
        # The real value is right-aligned by _plaintext_financial_row_context,
        # leaving exactly the first/start-period cell blank.
        blank_column = period_labels[0]
        value_column = period_labels[1]
        if not _column_period(blank_column) or not _column_period(value_column):
            continue
        if not _blank_zero_supported_by_nearby_narrative(
            lines, line_index, label
        ):
            continue

        section, scope = _nearby_plaintext_table_provenance(lines, line_index)
        parts = [
            f"ROW: {label}",
            f"ROW_TYPE: {_row_type(label)}",
            f"COLUMN: {blank_column}",
            f"PERIOD: {_column_period(blank_column)}",
        ]
        if section:
            parts.append(f"SECTION: {section}")
        if scope:
            parts.append(f"SCOPE: {scope}")
        parts.extend(
            (
                "VALUE_ORIGIN: IMPLIED_BLANK_ZERO_FROM_TABLE_AND_NARRATIVE",
                f"CELL: {label} <VALUE> 0 </VALUE> "
                f"[blank {blank_column}; disclosed value {value_column}]",
            )
        )
        context = "\n".join(parts)
        relevance = (
            _fuzzy_content_overlap(
                question_terms,
                set(normalize_text(line).split()),
            )
            + 0.20 * max(0.0, min(1.0, retrieval_score))
            + 0.20
        )
        char_start = matches[0][0].start()
        output.append(
            NumericCandidate(
                index=-1,
                value="0",
                raw="0",
                context=context,
                line_number=source_line_start + line_index,
                char_start=char_start,
                char_end=char_start,
                unit=_infer_unit(context, False) or page_unit,
                is_percent=False,
                is_year_like=False,
                relevance=relevance,
                source_name=source_name,
                page=page,
                extraction_method=extraction_method,
                entity_id=_source_entity_id(source_name, context),
            )
        )
    return output


def _blank_zero_supported_by_nearby_narrative(
    lines: Sequence[str],
    line_index: int,
    row_label: str,
) -> bool:
    """Require prose saying the balance increased because it newly arose."""

    narrative = normalize_text(
        " ".join(lines[line_index + 1 : min(len(lines), line_index + 10)])
    )
    if "tang" not in narrative or "phat sinh" not in narrative:
        return False
    row_terms = {
        term
        for term in _metric_alignment_tokens(row_label)
        if len(term) >= 3
    }
    return bool(row_terms & set(narrative.split()))


def merge_candidates(
    question: str,
    chunks: Sequence[object],
    *,
    max_candidates: int = 64,
    legacy_mode: bool = False,
    financial_identity_catalog_enabled: bool = False,
) -> tuple[NumericCandidate, ...]:
    """Extract, score and de-duplicate candidates from relevant source chunks.

    Overlapping chunks frequently contain the same table cell. Earlier versions
    treated those duplicates as separate operands and also discarded the BM25
    chunk score. This function preserves real source-line provenance, adds a
    bounded retrieval prior and collapses duplicate cells before HRM inference.
    """
    output: list[NumericCandidate] = []
    if not chunks:
        return ()
    if legacy_mode:
        per_chunk = max(8, math.ceil(max_candidates / max(1, len(chunks))))
        for chunk in chunks:
            output.extend(
                extract_numeric_candidates(
                    question,
                    str(chunk.text),
                    max_candidates=per_chunk,
                    source_name=str(chunk.source_name),
                    page=getattr(chunk, "page_start", None),
                    extraction_method=_chunk_extraction_method(chunk),
                    financial_identity_catalog_enabled=(
                        financial_identity_catalog_enabled
                    ),
                )
            )
        # Legacy checkpoints receive overlapping chunks but historically had no
        # source-line-aware de-duplication. Identical cells then split grounded
        # probability mass and can turn a correct SELECT into an unnecessary
        # abstention. Preserve the first/highest-relevance copy of an identical
        # source-page context without changing genuine repeated values.
        unique_legacy: dict[tuple[object, ...], NumericCandidate] = {}
        for candidate in output:
            key = (
                candidate.source_name,
                candidate.page,
                candidate.value,
                candidate.context,
            )
            current = unique_legacy.get(key)
            if current is None or candidate.relevance > current.relevance:
                unique_legacy[key] = candidate
        output = list(unique_legacy.values())
        if len(output) > max_candidates:
            selected = set(
                _candidate_quota_indices(
                    question,
                    output,
                    max_candidates,
                    financial_identity_catalog_enabled=(
                        financial_identity_catalog_enabled
                    ),
                )
            )
            output = [
                candidate
                for index, candidate in enumerate(output)
                if index in selected
            ]
        output.sort(key=lambda item: (item.relevance, -item.line_number), reverse=True)
        return tuple(
            NumericCandidate(**{**candidate.__dict__, "index": index})
            for index, candidate in enumerate(output)
        )
    maximum_score = max(float(getattr(chunk, "score", 0.0)) for chunk in chunks)
    per_chunk = max(10, math.ceil(max_candidates * 1.5 / max(1, len(chunks))))
    # Flattened native/OCR tables are often split across overlapping retrieval
    # chunks. Reassemble only conflict-free, contiguous global-line runs from
    # the same source/page before local candidate pruning. The reconstruction
    # still has to satisfy the frozen QP3E proof before it can emit a manifest.
    for merged in _merged_flattened_chunk_runs(chunks):
        output.extend(
            extract_numeric_candidates(
                question,
                merged["text"],
                max_candidates=max(max_candidates * 2, 64),
                source_name=merged["source_name"],
                page=merged["page"],
                source_line_start=merged["line_start"],
                retrieval_score=(
                    merged["score"] / maximum_score if maximum_score > 0 else 0.0
                ),
                extraction_method=merged["extraction_method"],
                financial_identity_catalog_enabled=(
                    financial_identity_catalog_enabled
                ),
            )
        )
    for chunk in chunks:
        raw_score = float(getattr(chunk, "score", 0.0))
        normalized_score = raw_score / maximum_score if maximum_score > 0 else 0.0
        output.extend(
            extract_numeric_candidates(
                question,
                str(chunk.text),
                max_candidates=per_chunk,
                source_name=str(chunk.source_name),
                page=getattr(chunk, "page_start", None),
                source_line_start=int(getattr(chunk, "line_start", 1)),
                retrieval_score=normalized_score,
                extraction_method=_chunk_extraction_method(chunk),
                financial_identity_catalog_enabled=(
                    financial_identity_catalog_enabled
                ),
            )
        )

    unique: dict[tuple[object, ...], NumericCandidate] = {}
    for candidate in output:
        key = (
            candidate.source_name,
            candidate.page,
            candidate.line_number,
            candidate.char_start,
            candidate.char_end,
            candidate.value,
        )
        current = unique.get(key)
        if (
            current is None
            or candidate.relevance > current.relevance
            or (
                math.isclose(candidate.relevance, current.relevance)
                and _candidate_provenance_richness(candidate)
                > _candidate_provenance_richness(current)
            )
        ):
            unique[key] = candidate

    output = list(unique.values())
    if len(output) > max_candidates:
        selected = set(
            _candidate_quota_indices(
                question,
                output,
                max_candidates,
                financial_identity_catalog_enabled=(
                    financial_identity_catalog_enabled
                ),
            )
        )
        output = [candidate for index, candidate in enumerate(output) if index in selected]
    output.sort(key=_candidate_rank_key, reverse=True)
    return tuple(
        NumericCandidate(**{**candidate.__dict__, "index": index})
        for index, candidate in enumerate(output)
    )


def _merged_flattened_chunk_runs(
    chunks: Sequence[object],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, int, str], list[object]] = {}
    for chunk in chunks:
        page_start = getattr(chunk, "page_start", None)
        page_end = getattr(chunk, "page_end", page_start)
        if not isinstance(page_start, int) or page_end != page_start:
            continue
        text = str(getattr(chunk, "text", ""))
        if "|" in text:
            continue
        key = (
            str(getattr(chunk, "source_name", "")),
            page_start,
            _chunk_extraction_method(chunk),
        )
        groups.setdefault(key, []).append(chunk)

    output: list[dict[str, object]] = []
    for (source_name, page, method), group in groups.items():
        if len(group) < 2:
            continue
        line_map: dict[int, str] = {}
        conflict = False
        for chunk in group:
            line_start = int(getattr(chunk, "line_start", 1))
            for offset, line in enumerate(str(getattr(chunk, "text", "")).splitlines()):
                global_line = line_start + offset
                current = line_map.get(global_line)
                if current is not None and current != line:
                    conflict = True
                    break
                line_map[global_line] = line
            if conflict:
                break
        if conflict or not line_map:
            continue
        ordered = sorted(line_map)
        run: list[int] = []
        runs: list[list[int]] = []
        for line_number in ordered:
            if run and line_number != run[-1] + 1:
                runs.append(run)
                run = []
            run.append(line_number)
        if run:
            runs.append(run)
        group_score = max(float(getattr(chunk, "score", 0.0)) for chunk in group)
        for current_run in runs:
            if len(current_run) < 4:
                continue
            text = "\n".join(line_map[line_number] for line_number in current_run)
            normalized = normalize_text(text)
            if not any(
                token in normalized
                for token in ("thuc hien", "ke hoach", "actual", "plan")
            ):
                continue
            output.append(
                {
                    "source_name": source_name,
                    "page": page,
                    "extraction_method": method or "text",
                    "line_start": current_run[0],
                    "score": group_score,
                    "text": text,
                }
            )
    return output


def _candidate_rank_key(candidate: NumericCandidate) -> tuple[float, bool, int]:
    return candidate.relevance, not candidate.is_year_like, -candidate.line_number


def _candidate_provenance_richness(candidate: NumericCandidate) -> int:
    """Prefer an overlapping extraction that retained table provenance."""

    return sum(
        bool(str(value).strip())
        for value in (
            candidate.row_label,
            candidate.column_label,
            candidate.period,
            candidate.table_id,
            candidate.section,
            candidate.scope,
            candidate.entity_id,
            candidate.subtable_id,
        )
    )


def _ocr_row_metric_score(question: str, candidate: NumericCandidate) -> float:
    # Historical public name retained for diagnostic-script compatibility.
    # Runtime native rows now carry the same structured metadata as OCR rows.
    if candidate.extraction_method.lower() not in {
        "native",
        "text",
        "ocr",
        "mineru",
    }:
        return 0.0
    row = next(
        (
            line.removeprefix("ROW:").strip()
            for line in candidate.context.splitlines()
            if line.startswith("ROW:")
        ),
        "",
    )
    if not row:
        return 0.0
    expected = _metric_alignment_tokens(_question_focus_text(question))
    observed = _metric_alignment_tokens(row)
    score = 0.0
    for term in expected:
        best = max(
            (
                1.0
                if term == item
                else 0.92
                if len(term) >= 3 and term in item
                else 0.85
                if len(item) >= 4 and item in term
                else SequenceMatcher(None, term, item).ratio()
                for item in observed
            ),
            default=0.0,
        )
        if best >= 0.72:
            score += best
    return score


def _metric_alignment_tokens(text: str) -> set[str]:
    stopwords = {
        "bao",
        "bang",
        "cho",
        "cua",
        "cuoi",
        "dau",
        "duoc",
        "den",
        "la",
        "nam",
        "ngay",
        "nhieu",
        "phan",
        "tai",  # restored below when the phrase denotes tài sản
        "theo",
        "tinh",
        "tri",
        "tu",
        "vnd",
        "voi",
    }
    normalized = normalize_text(text)
    tokens = re.findall(r"[a-z0-9]+", normalized)
    # ``tại`` and ``tài`` both fold to ``tai``. In financial metric phrases,
    # retain it only when adjacent to ``san`` (tài sản).
    output = {
        token
        for token in tokens
        if len(token) > 1 and not token.isdigit() and token not in stopwords
    }
    if any(tokens[index : index + 2] == ["tai", "san"] for index in range(len(tokens) - 1)):
        output.add("tai")
    return output


def _candidate_quota_indices(
    question: str,
    candidates: Sequence[NumericCandidate],
    maximum: int,
    *,
    prefer_magnitude: bool = False,
    financial_identity_catalog_enabled: bool = False,
) -> list[int]:
    """Preserve metric and operand-role evidence before applying a hard quota."""

    def rank_key(index: int) -> tuple[float, bool, Decimal | int]:
        candidate = candidates[index]
        final: Decimal | int = (
            abs(candidate.decimal) if prefer_magnitude else -candidate.line_number
        )
        return candidate.relevance, not candidate.is_year_like, final

    ranked = sorted(
        range(len(candidates)),
        key=rank_key,
        reverse=True,
    )
    metric_scores = {
        index: _ocr_row_metric_score(question, candidate)
        for index, candidate in enumerate(candidates)
    }
    best_metric = max(metric_scores.values(), default=0.0)
    reserved = (
        [
            index
            for index in sorted(
                metric_scores,
                key=lambda item: (
                    metric_scores[item],
                    *rank_key(item),
                ),
                reverse=True,
            )
            if metric_scores[index] >= best_metric - 0.30
        ][: min(8, maximum)]
        if best_metric >= 1.5
        else []
    )
    # Questions may name multiple typed evidence roles. Reserving only rows
    # similar to the complete question systematically favors verbose or large
    # values (for example total assets) and can evict concise evidence (for
    # example a disclosed coefficient, issued-share count, or par value) before
    # the HRM pointer ever sees it. The roles come from the report-independent
    # question plan; identity inputs are added only for a financial identity
    # explicitly requested by the question.
    role_parts = _quota_required_role_parts(
        question,
        financial_identity_catalog_enabled=financial_identity_catalog_enabled,
    )
    role_reserved: list[int] = []
    if role_parts:
        per_role = max(1, min(4, maximum // max(1, len(role_parts))))
        for role_text in role_parts:
            role_scores = {
                index: _quota_metric_score(role_text, candidate)
                for index, candidate in enumerate(candidates)
            }
            best_role = max(role_scores.values(), default=0.0)
            if best_role < 1.2:
                continue
            role_ranked = [
                index
                for index in sorted(
                    role_scores,
                    key=lambda item: (role_scores[item], *rank_key(item)),
                    reverse=True,
                )
                if role_scores[index] >= best_role - 0.30
            ]
            role_reserved.extend(role_ranked[:per_role])
            role_reserved = list(dict.fromkeys(role_reserved))

    selected = list(dict.fromkeys([*role_reserved, *reserved]))[:maximum]
    selected_set = set(selected)
    for index in ranked:
        if len(selected) >= maximum:
            break
        if index not in selected_set:
            selected.append(index)
            selected_set.add(index)
    return selected


def _quota_candidate_row(candidate: NumericCandidate) -> str:
    return next(
        (
            line.removeprefix("ROW:").strip()
            for line in candidate.context.splitlines()
            if line.startswith("ROW:")
        ),
        "",
    )


def _quota_metric_score(expected_text: str, candidate: NumericCandidate) -> float:
    row = _quota_candidate_row(candidate)
    if not row:
        return 0.0
    expected = _metric_alignment_tokens(expected_text)
    observed = _metric_alignment_tokens(row)
    score = 0.0
    for term in expected:
        best = max(
            (
                1.0
                if term == item
                else 0.92
                if len(term) >= 3 and term in item
                else 0.85
                if len(item) >= 4 and item in term
                else SequenceMatcher(None, term, item).ratio()
                for item in observed
            ),
            default=0.0,
        )
        if best >= 0.72:
            score += best
    return score


def _quota_operand_role_parts(question: str) -> tuple[str, str] | None:
    normalized = f" {normalize_text(_question_focus_text(question))} "
    marker = next(
        (
            item
            for item in (
                " chiem bao nhieu phan tram ",
                " chia cho ",
                " tren ",
                " trong tong ",
            )
            if item in normalized
        ),
        None,
    )
    if marker is None:
        return None
    left, right = normalized.split(marker, 1)
    for prefix in (" ty le ", " ty trong ", " he so "):
        if prefix in left:
            left = left.rsplit(prefix, 1)[-1]
    left = left.strip()
    right = right.strip()
    if not _metric_alignment_tokens(left) or not _metric_alignment_tokens(right):
        return None
    return left, right


def _quota_required_role_parts(
    question: str,
    *,
    financial_identity_catalog_enabled: bool = False,
) -> tuple[str, ...]:
    """Return generic evidence roles that must survive candidate truncation."""

    plan = compile_question_plan(question)
    roles: list[str] = []
    if plan.direct_metric_lookup and plan.metric_text:
        # Preserve an explicitly disclosed coefficient before considering a
        # recomputation from numerator and denominator.
        roles.append(plan.metric_text)
    roles.extend(plan.operand_roles)
    if plan.operation == "ADD":
        # Internal conjunctions can yield several structural two-role parses.
        # Keep proposal candidates for every parse through the hard quota;
        # symbolic binding still has to prove one complete two-row assignment.
        roles.extend(
            role
            for hypothesis in plan.addition_role_hypotheses
            for role in hypothesis
        )
    if financial_identity_catalog_enabled:
        for identity_id in plan.financial_identity_ids:
            identity = get_financial_identity(identity_id)
            roles.extend(identity.operand_texts)

    normalized = plan.normalized
    if (
        "gia tri theo menh gia" in normalized
        and any(term in normalized for term in ("co phieu", "co phan"))
    ):
        # These are the two disclosed inputs of the standard identity
        # par value = charter capital / issued shares. They are evidence roles,
        # not issuer-, page-, year-, or value-specific shortcuts.
        roles.extend(
            (
                "von dieu le",
                "so luong co phieu da phat hanh",
                "so luong co phan da phat hanh",
            )
        )

    legacy_roles = _quota_operand_role_parts(question)
    if legacy_roles is not None:
        roles.extend(legacy_roles)
    return tuple(dict.fromkeys(role for role in roles if role.strip()))


def _chunk_extraction_method(chunk: object) -> str:
    methods = tuple(str(item) for item in getattr(chunk, "extraction_methods", ()))
    if any(method in {"ocr", "mineru"} for method in methods):
        return "ocr"
    return methods[0] if methods else ""


def _execute_operation_values(operation: str, values: Sequence[Decimal]) -> Decimal:
    with localcontext() as context:
        context.prec = 34
        if operation == "SELECT":
            return values[0]
        if operation == "ADD":
            return values[0] + values[1]
        if operation == "SUBTRACT":
            return values[0] - values[1]
        if operation == "MULTIPLY":
            return values[0] * values[1]
        if operation == "DIVIDE":
            if values[1] == 0:
                raise ZeroDivisionError("DIVIDE denominator is zero")
            return values[0] / values[1]
        if operation == "RATIO_PERCENT":
            if values[1] == 0:
                raise ZeroDivisionError("RATIO_PERCENT denominator is zero")
            return values[0] / values[1] * Decimal(100)
        if operation == "PERCENT_CHANGE":
            if values[1] == 0:
                raise ZeroDivisionError("PERCENT_CHANGE baseline is zero")
            return (values[0] - values[1]) / values[1] * Decimal(100)
        if operation == "AVERAGE":
            return (values[0] + values[1]) / Decimal(2)
        if operation in {"SUM3", "SUM4"}:
            return sum(values, Decimal(0))
        if operation == "SHARE_OF_TOTAL":
            if values[1] == 0:
                raise ZeroDivisionError("SHARE_OF_TOTAL denominator is zero")
            return values[0] / values[1] * Decimal(100)
    raise ProgramDataError(f"Unsupported operation: {operation}")


def execute_program(
    program: ReasoningProgram,
    candidates: Sequence[NumericCandidate],
) -> Decimal:
    program.validate(len(candidates))
    values = [candidates[index].decimal for index in program.operands]
    return _execute_operation_values(program.operation, values)


def execute_program_dag(
    program: ReasoningDAGProgram,
    candidates: Sequence[NumericCandidate],
) -> Decimal:
    program.validate(len(candidates))
    results: dict[str, Decimal] = {}
    for node in program.nodes:
        values = [
            candidates[item.reference].decimal
            if item.kind == "candidate" and isinstance(item.reference, int)
            else results[str(item.reference)]
            for item in node.inputs
        ]
        results[node.node_id] = _execute_operation_values(
            node.operation, values
        )
    return results[program.output_node]


def materialize_program_dag(
    program: ReasoningProgram,
    candidates: Sequence[NumericCandidate],
) -> ReasoningDAGProgram | None:
    """Expand lineage-carrying derived candidates into an executable DAG.

    The neural checkpoint continues to emit the frozen flat DSL.  This lowering
    step replaces every selected derived candidate with its explicit upstream
    node and uses ``SHARE_OF_TOTAL`` for a percent-of-total output.
    """

    program.validate(len(candidates))
    nodes: list[ProgramDAGNode] = []
    emitted: dict[int, str] = {}
    visiting: set[int] = set()

    def candidate_input(index: int) -> ProgramDAGInput:
        if not 0 <= index < len(candidates):
            raise ProgramDataError(
                "Derived candidate operand is outside the candidate list"
            )
        candidate = candidates[index]
        operation = candidate.derived_operation.strip().upper()
        if not operation:
            return ProgramDAGInput.candidate(index)
        if index in emitted:
            return ProgramDAGInput.node(emitted[index])
        if index in visiting:
            raise ProgramDataError("Derived candidate lineage contains a cycle")
        if operation not in DAG_OPERATION_ARITY:
            raise ProgramDataError(
                f"Unsupported derived operation: {candidate.derived_operation}"
            )
        if len(candidate.derived_operands) != DAG_OPERATION_ARITY[operation]:
            raise ProgramDataError(
                f"Derived {operation} candidate has invalid operand arity"
            )
        visiting.add(index)
        inputs = tuple(
            candidate_input(operand) for operand in candidate.derived_operands
        )
        visiting.remove(index)
        node_id = candidate.derivation_id.strip() or f"derived_candidate_{index}"
        if any(node.node_id == node_id for node in nodes):
            raise ProgramDataError("Derived candidate lineage reuses a node ID")
        nodes.append(
            ProgramDAGNode(
                node_id=node_id,
                operation=operation,
                inputs=inputs,
                result_unit_family="currency" if candidate.unit else "",
            )
        )
        emitted[index] = node_id
        return ProgramDAGInput.node(node_id)

    final_inputs = tuple(candidate_input(index) for index in program.operands)
    if not nodes:
        return None
    final_operation = (
        "SHARE_OF_TOTAL"
        if program.operation == "RATIO_PERCENT"
        else program.operation
    )
    output_id = "answer"
    if any(node.node_id == output_id for node in nodes):
        output_id = "program_output"
    nodes.append(
        ProgramDAGNode(
            node_id=output_id,
            operation=final_operation,
            inputs=final_inputs,
            result_unit_family=(
                "percent"
                if final_operation in {"SHARE_OF_TOTAL", "RATIO_PERCENT"}
                else ""
            ),
        )
    )
    dag = ReasoningDAGProgram(tuple(nodes), output_id)
    dag.validate(len(candidates))
    return dag


def program_row_rejection_reason(
    row: dict[str, object],
    *,
    allowed_tasks: Sequence[str] = ("QA",),
) -> str | None:
    """Return a stable audit reason when a source row is not program-trainable.

    Document HRM emits one executable numerical program. Qualitative SA/NER
    prompts, unanswerable prompts and incomplete records must not be forced into
    that schema merely because their prose happens to contain numbers.
    """

    task = str(row.get("task") or "").strip().upper()
    normalized_allowed = {item.strip().upper() for item in allowed_tasks}
    if task not in normalized_allowed:
        return "unsupported_task"

    difficulty = str(row.get("difficulty") or "").strip().upper()
    difficulty_suffix = difficulty.rsplit("_", 1)[-1] if "_" in difficulty else ""
    if difficulty_suffix in {"QA", "SA", "NER"} and difficulty_suffix != task:
        return "difficulty_task_mismatch"

    question = str(row.get("question") or "").strip()
    context = str(row.get("context") or "").strip()
    answer_text = str(row.get("answer") or "").strip()
    if not question or not context or not answer_text:
        return "missing_required_text"

    normalized_answer = normalize_text(answer_text)
    if any(marker in normalized_answer for marker in _UNANSWERABLE_MARKERS):
        return "unanswerable"

    normalized_question = normalize_text(question)
    if any(marker in normalized_question for marker in _NON_PROGRAM_QUESTION_MARKERS):
        return "non_program_question"
    if any(marker in normalized_question for marker in _QUALITATIVE_QUESTION_MARKERS):
        if not _question_implies_calculation(question):
            return "qualitative_question"
    return None


def induced_program_rejection_reason(example: ProgramExample) -> str | None:
    """Validate that an induced program has a unique, answer-grounded target."""

    answer_numbers = _answer_numbers(example.answer_text)
    grounded = set(
        _answer_grounded_candidate_indices(answer_numbers, example.candidates)
    )
    if not all(pointer in grounded for pointer in example.program.operands):
        return "ungrounded_operand"

    if example.program.operation == "SELECT":
        if _question_requires_arithmetic(example.question):
            return "select_for_arithmetic_question"
        unique_values: list[Decimal] = []
        for value, _start, _end, _raw in answer_numbers:
            if _is_probable_year(value):
                continue
            if not any(_close(value, current) for current in unique_values):
                unique_values.append(value)
        if len(unique_values) > 1:
            return "ambiguous_multi_value_select"
    return None


def _is_degenerate_arithmetic(
    operation: str,
    operand_values: Sequence[Decimal],
) -> bool:
    """Reject trivial arithmetic without discarding valid zero deltas.

    A previous blanket check rejected any result equal to an operand. That also
    removed legitimate cases such as ``100 - 0`` when a balance fell to zero.
    """

    if operation == "ADD":
        return any(value == 0 for value in operand_values)
    if operation == "MULTIPLY":
        return any(value in {Decimal(0), Decimal(1)} for value in operand_values)
    if operation == "DIVIDE":
        return len(operand_values) == 2 and operand_values[1] == 1
    if operation == "AVERAGE":
        return len(operand_values) == 2 and operand_values[0] == operand_values[1]
    if operation in {"SUM3", "SUM4"}:
        return any(value == 0 for value in operand_values)
    return False


def induce_program_example(
    row: dict[str, object],
    *,
    max_candidates: int = 64,
    strict: bool = True,
) -> ProgramExample | None:
    if strict and program_row_rejection_reason(row) is not None:
        return None
    question = str(row.get("question") or "")
    context = str(row.get("context") or "")
    answer_text = str(row.get("answer") or "")
    if not question or not context or not answer_text:
        return None
    candidates = extract_numeric_candidates(
        question, context, max_candidates=max_candidates
    )
    if not candidates:
        return None
    answer_numbers = _answer_numbers(answer_text)
    if not answer_numbers:
        return None
    explicit_targets = _explicit_result_values(answer_text)
    if len(explicit_targets) > 1:
        # The current output schema represents one numerical program. Multi-part
        # questions are retained in the source corpus but not mislabeled as a
        # single-output training example.
        return None
    target_values = explicit_targets or _target_values(question, answer_text, answer_numbers)
    if not target_values:
        return None

    operation_order = _operation_preference(question, answer_text)
    input_indices = _answer_grounded_candidate_indices(answer_numbers, candidates)
    pool = input_indices or tuple(range(min(len(candidates), 28)))
    # Keep at most 18 answer-grounded pointers for controlled combinatorics.
    pool = tuple(dict.fromkeys(pool))[:18]

    best: tuple[float, ReasoningProgram, Decimal] | None = None
    for target_rank, target in enumerate(target_values[:6]):
        for operation_rank, operation in enumerate(operation_order):
            arity = OPERATION_ARITY[operation]
            for operands in _operand_tuples(operation, pool, candidates, target):
                program = ReasoningProgram(operation, operands)
                try:
                    value = execute_program(program, candidates)
                except (ZeroDivisionError, ProgramDataError, InvalidOperation):
                    continue
                if not _close(value, target):
                    continue
                operand_values = [candidates[index].decimal for index in operands]
                if _is_degenerate_arithmetic(operation, operand_values):
                    # Reject identities such as amount + 0 or amount * 1 while
                    # retaining meaningful SUBTRACT cases whose second value is 0.
                    continue
                if operation in {"SUM3", "SUM4"} and max(map(abs, operand_values)) < Decimal(1000):
                    continue
                if operation == "DIVIDE" and target == 0 and operand_values[0] != 0:
                    continue
                score = 5.0 - target_rank * 0.35 - operation_rank * 0.12
                score += sum(candidates[index].relevance for index in operands) * 0.15
                if operation == "SELECT" and _question_implies_calculation(question):
                    score -= 1.5
                if operation != "SELECT" and not _question_implies_calculation(question):
                    score -= 0.4
                if input_indices and all(index in input_indices for index in operands):
                    score += 1.0
                if best is None or score > best[0]:
                    best = (score, program, target)
        if best is not None and best[0] >= 5.0:
            break
    if best is None:
        return None
    _, program, target = best
    if program.operation in COMMUTATIVE:
        program = ReasoningProgram(program.operation, tuple(sorted(program.operands)))
    example = ProgramExample(
        example_id=str(row.get("id") or "unknown"),
        question=question,
        context=context,
        answer_text=answer_text,
        candidates=candidates,
        program=program,
        answer_value=_decimal_string(target),
        task=str(row.get("task") or "QA"),
        difficulty=str(row.get("difficulty") or ""),
    )
    if strict and induced_program_rejection_reason(example) is not None:
        return None
    return example


def build_program_dataset(
    source_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    max_candidates: int = 64,
    allowed_tasks: Sequence[str] = ("QA",),
) -> ProgramBuildReport:
    source = Path(source_jsonl).expanduser().resolve()
    destination = Path(output_jsonl).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_rows = eligible_rows = numeric_rows = induced = no_target = no_program = 0
    counts = {name: 0 for name in OPERATIONS}
    source_task_counts: dict[str, int] = {}
    rejection_counts: dict[str, int] = {}
    rejection_examples: dict[str, list[str]] = {}

    def reject(reason: str, row: dict[str, object]) -> None:
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        examples = rejection_examples.setdefault(reason, [])
        if len(examples) < 25:
            examples.append(str(row.get("id") or "unknown"))

    with source.open("r", encoding="utf-8") as handle, destination.open(
        "w", encoding="utf-8"
    ) as output:
        for line in handle:
            if not line.strip():
                continue
            source_rows += 1
            row = json.loads(line)
            task = str(row.get("task") or "missing").strip().upper() or "missing"
            source_task_counts[task] = source_task_counts.get(task, 0) + 1
            screening_reason = program_row_rejection_reason(
                row, allowed_tasks=allowed_tasks
            )
            if screening_reason is not None:
                reject(screening_reason, row)
                continue
            eligible_rows += 1
            answer_numbers = _answer_numbers(str(row.get("answer") or ""))
            if not answer_numbers:
                no_target += 1
                reject("no_numeric_target", row)
                continue
            numeric_rows += 1
            example = induce_program_example(
                row, max_candidates=max_candidates, strict=False
            )
            if example is None:
                no_program += 1
                reject("no_matching_program", row)
                continue
            validation_reason = induced_program_rejection_reason(example)
            if validation_reason is not None:
                no_program += 1
                reject(validation_reason, row)
                continue
            induced += 1
            counts[example.program.operation] += 1
            output.write(json.dumps(program_example_to_dict(example), ensure_ascii=False) + "\n")
    return ProgramBuildReport(
        source_rows=source_rows,
        eligible_rows=eligible_rows,
        numeric_rows=numeric_rows,
        induced_examples=induced,
        operation_counts={key: value for key, value in counts.items() if value},
        skipped_no_target=no_target,
        skipped_no_program=no_program,
        source_task_counts=dict(sorted(source_task_counts.items())),
        rejection_counts=dict(
            sorted(rejection_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        rejection_examples={
            reason: tuple(example_ids)
            for reason, example_ids in sorted(rejection_examples.items())
        },
    )


def load_program_examples(path: str | Path) -> list[ProgramExample]:
    source = Path(path).expanduser().resolve()
    output: list[ProgramExample] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            candidates = tuple(NumericCandidate(**item) for item in payload["candidates"])
            program = ReasoningProgram(
                operation=payload["program"]["operation"],
                operands=tuple(payload["program"]["operands"]),
            )
            output.append(
                ProgramExample(
                    example_id=str(payload["example_id"]),
                    question=payload["question"],
                    context=payload["context"],
                    answer_text=payload["answer_text"],
                    candidates=candidates,
                    program=program,
                    answer_value=payload["answer_value"],
                    task=payload.get("task", "QA"),
                    difficulty=payload.get("difficulty", ""),
                    supervision=payload.get("supervision"),
                )
            )
    return output


def program_example_to_dict(example: ProgramExample) -> dict[str, object]:
    payload: dict[str, object] = {
        "example_id": example.example_id,
        "question": example.question,
        "context": example.context,
        "answer_text": example.answer_text,
        "candidates": [asdict(candidate) for candidate in example.candidates],
        "program": {
            "operation": example.program.operation,
            "operands": list(example.program.operands),
        },
        "answer_value": example.answer_value,
        "task": example.task,
        "difficulty": example.difficulty,
    }
    if example.supervision is not None:
        payload["supervision"] = example.supervision
    return payload


def _answer_numbers(text: str) -> list[tuple[Decimal, int, int, str]]:
    output: list[tuple[Decimal, int, int, str]] = []
    for match in _NUMBER_RE.finditer(text):
        if (match.start() > 0 and text[match.start() - 1] == "/") or (
            match.end() < len(text) and text[match.end()] == "/"
        ):
            continue
        raw = match.group(0).strip()
        value = parse_decimal(raw)
        if value is None:
            continue
        if _is_answer_metadata_number(text, match, value):
            continue
        output.append((value, match.start(), match.end(), raw))
    return output



def _explicit_result_values(text: str) -> tuple[Decimal, ...]:
    values: list[Decimal] = []
    marker = re.compile(r"(?:≈|~=|=|xấp\s*xỉ|khoảng)\s*", re.IGNORECASE)
    for match in marker.finditer(text):
        number_match = _NUMBER_RE.search(text, match.end())
        if number_match is None or number_match.start() - match.end() > 6:
            continue
        value = parse_decimal(number_match.group(0))
        if value is None or _is_probable_year(value):
            continue
        if not any(_close(value, existing) for existing in values):
            values.append(value)
    return tuple(values)


def _labeled_result_values(text: str) -> tuple[Decimal, ...]:
    """Extract values introduced by an explicit natural-language conclusion.

    Financial answers often state the result first and then repeat the starting
    and ending balances, for example ``mức giảm là 100, từ 100 xuống 0``. Using
    the final number as target would incorrectly select 0 in that structure.
    """

    markers = re.compile(
        r"(?:"
        r"(?:tổng\s+)?mức\s+(?:tăng|giảm|thay\s+đổi|chênh\s+lệch)"
        r"|chênh\s+lệch"
        r"|kết\s+quả"
        r"|giá\s+trị\s+tính\s+được"
        r")\s*(?:là|:)",
        re.IGNORECASE,
    )
    values: list[Decimal] = []
    for marker in markers.finditer(text):
        number_match = _NUMBER_RE.search(text, marker.end())
        if number_match is None or number_match.start() - marker.end() > 24:
            continue
        value = parse_decimal(number_match.group(0))
        if value is None or _is_probable_year(value):
            continue
        if not any(_close(value, existing) for existing in values):
            values.append(value)
    return tuple(values)


def _target_values(
    question: str,
    answer_text: str,
    numbers: Sequence[tuple[Decimal, int, int, str]],
) -> tuple[Decimal, ...]:
    ranked: list[Decimal] = list(_labeled_result_values(answer_text))
    last_equals = answer_text.rfind("=")
    if last_equals >= 0:
        ranked.extend(value for value, start, _end, _raw in numbers if start > last_equals)
    normalized_question = normalize_text(question)
    if any(term in normalized_question for term in ("ty le", "ty suat", "tang truong", "phan tram")):
        ranked.extend(
            value
            for value, _start, _end, raw in numbers
            if "%" in raw or "phần trăm" in raw.lower()
        )
    # Final numerical claim is usually the computed answer in this dataset.
    ranked.extend(value for value, *_rest in reversed(numbers))
    unique: list[Decimal] = []
    for value in ranked:
        if _is_probable_year(value):
            continue
        if not any(_close(value, existing) for existing in unique):
            unique.append(value)
    return tuple(unique)


def _answer_grounded_candidate_indices(
    answer_numbers: Sequence[tuple[Decimal, int, int, str]],
    candidates: Sequence[NumericCandidate],
) -> tuple[int, ...]:
    output: list[int] = []
    # Every number copied into the explanation is evidence for operand selection.
    for value, _start, _end, _raw in answer_numbers:
        if _is_probable_year(value):
            continue
        matches = [
            candidate
            for candidate in candidates
            if _close(candidate.decimal, value)
        ]
        matches.sort(key=lambda item: item.relevance, reverse=True)
        for candidate in matches[:2]:
            output.append(candidate.index)
    return tuple(dict.fromkeys(output))


def _operation_preference(question: str, answer: str) -> tuple[str, ...]:
    text = normalize_text(f"{question} {answer[:500]}")
    preferred: list[str] = []
    if any(term in text for term in ("tang truong", "tang giam", "bien dong", "thay doi")):
        preferred.append("PERCENT_CHANGE")
    if any(term in text for term in ("ty le", "ty suat", "tren tong", "chiem bao nhieu phan tram")):
        preferred.append("RATIO_PERCENT")
    if any(term in text for term in ("tong", "cong lai", "tong cong")):
        preferred.extend(("SUM3", "SUM4", "ADD"))
    if any(
        term in text
        for term in (
            "chenh lech",
            "giam bao nhieu",
            "tang bao nhieu",
            "muc giam",
            "muc tang",
        )
    ):
        preferred.append("SUBTRACT")
    if any(term in text for term in ("binh quan", "trung binh")):
        preferred.append("AVERAGE")
    if any(term in text for term in ("nhan", "tich")):
        preferred.append("MULTIPLY")
    if any(term in text for term in ("chia", "bao nhieu lan")):
        preferred.append("DIVIDE")
    if not _question_implies_calculation(question):
        preferred.append("SELECT")
    preferred.extend(OPERATIONS)
    return tuple(dict.fromkeys(preferred))


def _operand_tuples(
    operation: str,
    pool: Sequence[int],
    candidates: Sequence[NumericCandidate],
    target: Decimal,
) -> Iterable[tuple[int, ...]]:
    arity = OPERATION_ARITY[operation]
    if operation == "SELECT":
        exact = [index for index in pool if _close(candidates[index].decimal, target)]
        yield from ((index,) for index in exact)
        return
    if len(pool) < arity:
        return
    if operation in COMMUTATIVE:
        # SUM operations are only attempted on answer-grounded/small pools.
        combination_pool = pool[:14] if arity >= 3 else pool
        yield from itertools.combinations(combination_pool, arity)
    else:
        for pair in itertools.permutations(pool, arity):
            yield pair


def _close(left: Decimal, right: Decimal) -> bool:
    difference = abs(left - right)
    scale = max(abs(right), Decimal(1))
    if scale < Decimal(1000):
        tolerance = Decimal("0.035")
    else:
        tolerance = max(Decimal("0.5"), scale * Decimal("0.00000000001"))
    return difference <= tolerance


def _decimal_string(value: Decimal) -> str:
    if value == value.to_integral_value():
        return format(value, "f")
    return format(value.normalize(), "f")


def _is_probable_year(value: Decimal) -> bool:
    return value == value.to_integral_value() and Decimal(1900) <= value <= Decimal(2100)


def _question_implies_calculation(question: str) -> bool:
    text = normalize_text(question)
    return any(
        term in text
        for term in (
            "tinh",
            "ty le",
            "ty suat",
            "tang truong",
            "chenh lech",
            "tong",
            "bao nhieu phan tram",
            "so sanh",
        )
    )


def _question_requires_arithmetic(question: str) -> bool:
    text = normalize_text(question)
    return any(term in text for term in _ARITHMETIC_QUESTION_MARKERS)


def _content_tokens(text: str) -> set[str]:
    stop = {
        "la",
        "cua",
        "va",
        "cho",
        "nam",
        "trong",
        "tai",
        "vao",
        "bao",
        "nhieu",
        "tinh",
        "hay",
        "duoc",
        "theo",
    }
    return {
        token
        for token in normalize_text(text).split()
        if len(token) > 1 and token not in stop
    }


def _question_focus_text(question: str) -> str:
    """Drop a leading report/statement qualifier from a user question."""

    if "," in question and normalize_text(question).startswith("theo "):
        return question.split(",", 1)[-1].strip()
    return question


def _fuzzy_content_overlap(expected: set[str], observed: set[str]) -> float:
    if not expected:
        return 0.0
    score = 0.0
    for term in expected:
        best = 0.0
        for candidate in observed:
            if term == candidate:
                best = 1.0
            elif len(term) >= 3 and term in candidate:
                # OCR frequently removes spaces: NPHAITRA, TONGCONGTAISAN.
                best = max(best, 0.92)
            elif len(candidate) >= 4 and candidate in term:
                best = max(best, 0.85)
            else:
                ratio = SequenceMatcher(None, term, candidate).ratio()
                if ratio >= 0.78:
                    best = max(best, ratio)
        score += best
    return score / len(expected)


def _markdown_cells(line: str) -> list[str]:
    stripped = line.strip()
    if "|" not in stripped:
        return []
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def _is_markdown_separator(line: str) -> bool:
    cells = _markdown_cells(line)
    return bool(cells) and all(
        not cell or bool(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")))
        for cell in cells
    )


def _is_table_header_or_separator(
    lines: Sequence[str], line_index: int
) -> bool:
    line = lines[line_index].strip()
    if _is_markdown_separator(line):
        return True
    if "|" not in line:
        return False
    normalized_cells = {normalize_text(cell) for cell in _markdown_cells(line)}
    if normalized_cells & {"chi tieu", "column name"}:
        return True
    # Standard and multi-row Markdown headers are immediately followed by a
    # separator. Header numbers such as "≤ 1 năm" are period labels, not values.
    for following in range(line_index + 1, min(len(lines), line_index + 4)):
        candidate = lines[following].strip()
        if not candidate or "|" not in candidate:
            break
        if _is_markdown_separator(candidate):
            return True
    return False


def _table_header_lines(lines: Sequence[str], line_index: int) -> list[str]:
    """Return only real header rows from the current Markdown table block."""

    block_start = line_index
    while block_start > 0:
        previous = lines[block_start - 1].strip()
        if not previous or "|" not in previous:
            break
        block_start -= 1

    separator_index = next(
        (
            index
            for index in range(line_index - 1, block_start - 1, -1)
            if _is_markdown_separator(lines[index])
        ),
        None,
    )
    if separator_index is not None:
        headers: list[str] = []
        # Page partitioning/chunk overlap can leave data rows immediately before
        # a repeated header. Walk backward from the nearest separator and stop
        # at the first financial row instead of promoting the entire pipe block
        # to a multi-row header.
        for index in range(
            separator_index - 1,
            max(block_start - 1, separator_index - 4),
            -1,
        ):
            candidate = lines[index].strip()
            cells = _markdown_cells(candidate)
            if not cells or _financial_value_cell_indices(cells):
                break
            headers.append(candidate)
        return list(reversed(headers))

    # PDF-to-text output occasionally omits Markdown's separator. Only trust
    # the first row when it contains explicit header vocabulary; never scan
    # arbitrary preceding value rows for a year and call them a header.
    if block_start < line_index:
        candidate = lines[block_start].strip()
        normalized = normalize_text(candidate)
        if any(
            token in normalized
            for token in (
                "chi tieu",
                "ma so",
                "thuyet minh",
                "cuoi ky",
                "dau nam",
                "nam nay",
                "nam truoc",
            )
        ):
            return [candidate]

    # OCR tables often wrap part of a header (for example ``(giảm)``) onto a
    # non-table line. Search a small window above the current value block, but
    # accept only explicit header vocabulary plus at least two table cells.
    for index in range(block_start - 1, max(-1, block_start - 8), -1):
        candidate = lines[index].strip()
        cells = _markdown_cells(candidate)
        if len(cells) < 2:
            continue
        normalized = normalize_text(candidate)
        years = re.findall(r"\b(?:19|20)\d{2}\b", normalized)
        has_header_label = any(
            token in normalized
            for token in (
                "chi tieu",
                "ch tieu",
                "nam nay",
                "nam truoc",
                "cuoi nam",
                "cuoi ky",
                "dau nam",
                "dau ky",
            )
        )
        if has_header_label and (years or len(cells) >= 3):
            return [candidate]
    return []


def _row_type(row_label: str) -> str:
    normalized = normalize_text(row_label)
    if re.search(r"(?:^| )(?:tong|tong cong|cong)(?: |$)", normalized):
        return "total"
    if re.match(r"^(?:I|II|III|IV|V|VI|VII|VIII|IX|X)\.[ \t]+", row_label.strip()):
        return "subtotal"
    if re.match(r"^[A-ZĐ]\.[ \t]+", row_label.strip()):
        return "total"
    if any(term in normalized for term in ("tieu tong", "cong nhom")):
        return "subtotal"
    return "detail"


def _column_period(column_label: str) -> str:
    normalized = normalize_text(column_label)
    qualifiers = [
        term
        for term in (
            "cuoi nam",
            "cuoi ky",
            "dau nam",
            "dau ky",
            "nam nay",
            "nam truoc",
        )
        if term in normalized
    ]
    years = re.findall(r"\b(?:19|20)\d{2}\b", normalized)
    return " ".join(dict.fromkeys([*qualifiers, *years]))


def _best_row_label(row_cells: Sequence[str], cell_index: int) -> str:
    search_order = [*row_cells[:cell_index], *row_cells[cell_index:]]
    textual = [
        cell
        for cell in search_order
        if re.search(r"[^\W\d_]", cell, re.UNICODE)
        and not _inline_period_column((cell,), 0)
    ]
    if not textual:
        return row_cells[0] if row_cells else ""

    def score(cell: str) -> tuple[int, int]:
        letters = sum(character.isalpha() for character in cell)
        looks_like_code = bool(re.fullmatch(r"\d+[A-Za-z]?", cell.strip()))
        return (letters - (100 if looks_like_code else 0), len(cell))

    return max(textual, key=score)


def _candidate_context(
    lines: Sequence[str],
    line_index: int,
    start: int,
    end: int,
    *,
    source_name: str = "context",
    page: int | None = None,
    source_line_start: int = 1,
    extraction_method: str = "",
) -> str:
    raw_line = lines[line_index]
    line = raw_line.strip()
    derived_formula_value = bool(
        re.match(
            r"\s*\[DERIVED_FROM_TABLE_FORMULA\]",
            raw_line[end:],
        )
    )
    marked = (
        raw_line[:start]
        + " <VALUE> "
        + raw_line[start:end].strip()
        + " </VALUE> "
        + raw_line[end:]
    )
    marked = re.sub(r"[ \t]+", " ", marked).strip()
    # The rendered row can contain another derived period cell. Remove the
    # row-level marker and attach provenance only to the numeric value currently
    # enclosed by <VALUE>, otherwise a disclosed current-period value would be
    # incorrectly treated as derived because its prior-period neighbour was.
    marked = marked.replace(" [DERIVED_FROM_TABLE_FORMULA]", "")
    if "|" in line:
        cell_index = max(
            0,
            raw_line[:start].count("|")
            - (1 if raw_line.lstrip().startswith("|") else 0),
        )
        row_cells = _markdown_cells(line)
        row_label = _best_row_label(row_cells, cell_index)
        header_lines = _table_header_lines(lines, line_index)
        column_parts: list[str] = []
        period_columns: list[str] = []
        for header_line in header_lines:
            header_cells = _markdown_cells(header_line)
            for header_cell in header_cells:
                if _column_period(header_cell) and header_cell not in period_columns:
                    period_columns.append(header_cell)
            # OCR can drop an empty ``Thuyet minh`` cell from a data row while
            # retaining it in the header. Period/value columns are on the right,
            # so align unequal rows from the right.
            header_index = cell_index + max(0, len(header_cells) - len(row_cells))
            if header_index < len(header_cells):
                part = header_cells[header_index]
                if part and part not in column_parts:
                    column_parts.append(part)
        column_label = " / ".join(column_parts)
        financial_cells = _financial_value_cell_indices(row_cells)
        if (
            cell_index in financial_cells
            and len(financial_cells) == len(period_columns)
        ):
            column_label = period_columns[financial_cells.index(cell_index)]
        inline_column = _inline_period_column(row_cells, cell_index)
        if inline_column:
            column_label = inline_column
        period = _column_period(column_label)
        table_id = _table_identifier(lines, line_index)
        section = _nearest_table_section(lines, line_index)
        scope = _table_scope(lines, line_index)
        subtable_id, entity_id = _table_subtable_provenance(
            lines, line_index, cell_index
        )
        parts = [f"ROW: {row_label}", f"ROW_TYPE: {_row_type(row_label)}"]
        if column_label:
            parts.append(f"COLUMN: {column_label}")
        if period:
            parts.append(f"PERIOD: {period}")
        if table_id:
            parts.append(f"TABLE: {table_id}")
        if section:
            parts.append(f"SECTION: {section}")
        if scope:
            parts.append(f"SCOPE: {scope}")
        if entity_id:
            parts.append(f"ENTITY: {entity_id}")
        if subtable_id:
            parts.append(f"SUBTABLE: {subtable_id}")
        if derived_formula_value:
            parts.append("VALUE_ORIGIN: DERIVED_FROM_TABLE_FORMULA")
        if header_lines:
            parts.append(f"HEADER: {' || '.join(header_lines)}")
        inventory_manifest = _table_role_inventory_manifest(
            lines,
            line_index,
            source_name=source_name,
            page=page,
            source_line_start=source_line_start,
            extraction_method=extraction_method,
        )
        if inventory_manifest:
            parts.append(
                "ROLE_INVENTORY_MANIFEST: "
                + json.dumps(
                    inventory_manifest,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        parts.append(f"CELL: {marked}")
        return "\n".join(parts)

    plaintext_context = _plaintext_financial_row_context(
        lines,
        line_index,
        start,
        end,
        marked,
        source_name=source_name,
        page=page,
        source_line_start=source_line_start,
        extraction_method=extraction_method,
    )
    if plaintext_context:
        return plaintext_context

    previous_text = ""
    for previous in range(line_index - 1, max(-1, line_index - 4), -1):
        candidate = lines[previous].strip()
        if candidate:
            previous_text = candidate
            break
    if previous_text and len(previous_text) < 180:
        return f"CONTEXT: {previous_text}\nVALUE_LINE: {marked}"
    return f"VALUE_LINE: {marked}"


def _table_block_bounds(
    lines: Sequence[str], line_index: int
) -> tuple[int, int]:
    start = line_index
    while start > 0 and "|" in lines[start - 1]:
        start -= 1
    end = line_index
    while end + 1 < len(lines) and "|" in lines[end + 1]:
        end += 1
    return start, end


def _table_identifier(lines: Sequence[str], line_index: int) -> str:
    if "|" not in lines[line_index]:
        return ""
    start, end = _table_block_bounds(lines, line_index)
    return f"table-{start + 1}-{end + 1}"


def _markdown_cells_with_spans(line: str) -> list[tuple[str, tuple[int, int]]]:
    """Return Markdown cells with source spans inside the unstripped line."""

    separators = [index for index, value in enumerate(line) if value == "|"]
    if not separators:
        return []
    boundaries = list(separators)
    if not line[: separators[0]].strip():
        starts_at = separators[0] + 1
    else:
        boundaries.insert(0, -1)
        starts_at = 0
    if not line[separators[-1] + 1 :].strip():
        ends_at = separators[-1]
    else:
        boundaries.append(len(line))
        ends_at = len(line)
    output: list[tuple[str, tuple[int, int]]] = []
    for left, right in zip(boundaries, boundaries[1:]):
        raw_start = left + 1
        raw_end = right
        if raw_start < starts_at or raw_end > ends_at:
            continue
        raw = line[raw_start:raw_end]
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw.rstrip())
        output.append(
            (
                raw.strip(),
                (raw_start + leading, raw_start + trailing),
            )
        )
    return output


def _table_role_inventory_manifest(
    lines: Sequence[str],
    line_index: int,
    *,
    source_name: str,
    page: int | None,
    source_line_start: int,
    extraction_method: str,
) -> dict[str, object] | None:
    """Capture full table provenance before candidate Top-K selection.

    This function deliberately records structure rather than deciding whether
    the role inventory is complete. The evidence-layer producer owns that
    decision under the frozen QP3C contract.
    """

    if "|" not in lines[line_index]:
        return None
    block_start, block_end = _table_block_bounds(lines, line_index)
    header_lines = _table_header_lines(lines, line_index)
    if not header_lines:
        return None
    separator_before = next(
        (
            index
            for index in range(line_index - 1, block_start - 1, -1)
            if _is_markdown_separator(lines[index])
        ),
        None,
    )
    data_start = separator_before + 1 if separator_before is not None else block_start + len(header_lines)
    next_separator = next(
        (
            index
            for index in range(line_index + 1, block_end + 1)
            if _is_markdown_separator(lines[index])
        ),
        None,
    )
    data_end = (
        max(line_index, next_separator - len(header_lines) - 1)
        if next_separator is not None
        else block_end
    )
    row_indices = [
        index
        for index in range(data_start, data_end + 1)
        if "|" in lines[index]
        and not _is_markdown_separator(lines[index])
        and any(_markdown_cells(lines[index]))
    ]
    if not row_indices:
        return None

    row_width = max(len(_markdown_cells(lines[index])) for index in row_indices)
    header_cells = [_markdown_cells(raw) for raw in header_lines]
    header_columns: list[dict[str, object]] = []
    for column_index in range(row_width):
        path: list[str] = []
        provenance: list[str] = []
        for raw_cells in header_cells:
            cells = (
                [""] * (row_width - len(raw_cells)) + raw_cells
                if len(raw_cells) < row_width
                else raw_cells[-row_width:]
            )
            part = cells[column_index].strip() if column_index < len(cells) else ""
            source = "explicit_header"
            if not part:
                for previous in range(column_index - 1, -1, -1):
                    if cells[previous].strip():
                        part = cells[previous].strip()
                        source = "inherited_merged_header"
                        break
            normalized = normalize_text(part)
            if not part or normalized in {
                "chi tieu",
                "noi dung",
                "item",
                "indicator",
                "column name",
                "stt",
                "ma so",
                "thuyet minh",
                "notes",
            }:
                continue
            if normalize_text(part) not in {normalize_text(value) for value in path}:
                path.append(part)
                provenance.append(source)
        header_columns.append(
            {
                "column_id": f"column:{column_index}",
                "header_path": path,
                "header_path_provenance": provenance,
            }
        )
    declared_data_column_ids = [
        str(item["column_id"])
        for item in header_columns
        if item["header_path"]
    ]

    expected_row_ids = [f"row:{source_line_start + index}" for index in row_indices]
    observed_coordinates: list[str] = []
    physical_observations: list[dict[str, object]] = []
    for row_id, index in zip(expected_row_ids, row_indices):
        cells_with_spans = _markdown_cells_with_spans(lines[index])
        cells = [value for value, _ in cells_with_spans]
        for column_index, cell_value in enumerate(cells):
            if not cell_value.strip():
                continue
            if column_index >= len(cells_with_spans):
                continue
            column_id = f"column:{column_index}"
            coordinate = f"{row_id}|{column_id}"
            observed_coordinates.append(coordinate)
            local_start, local_end = cells_with_spans[column_index][1]
            global_line = source_line_start + index
            physical_observations.append(
                {
                    "cell_coordinate": coordinate,
                    "document_id": source_name,
                    "page": page if page is not None else 0,
                    "source_span": [
                        global_line * 100000 + local_start,
                        global_line * 100000 + local_end,
                    ],
                    "extraction_method": extraction_method or "unknown_extractor",
                    "column_id": column_id,
                }
            )

    if len(physical_observations) > 256:
        physical_observations = physical_observations[:256]
        observed_coordinates = observed_coordinates[:256]
        boundary_state = "PARTIAL"
    else:
        boundary_state = (
            "COMPLETE"
            if block_start > 0 and block_end + 1 < len(lines)
            else "PARTIAL"
        )
    header_alignment_state = (
        "ALIGNED"
        if header_cells
        and all(len(cells) == row_width for cells in header_cells)
        and all(len(_markdown_cells(lines[index])) == row_width for index in row_indices)
        else "UNALIGNED"
    )
    return {
        "schema": "finhrm-table-role-inventory-manifest-v1",
        "document_id": source_name,
        "table_id": _table_identifier(lines, line_index),
        "table_boundary_state": boundary_state,
        "header_alignment_state": header_alignment_state,
        "extractor_alignment_state": "CONSISTENT",
        "raw_header_columns": header_columns,
        "declared_data_column_ids": declared_data_column_ids,
        "expected_row_ids": expected_row_ids,
        "observed_cell_coordinates": observed_coordinates,
        "physical_observations": physical_observations,
    }


def _subtable_anchor_indices(lines: Sequence[str], line_index: int) -> list[int]:
    anchors: list[int] = []
    for header_line in _table_header_lines(lines, line_index):
        cells = _markdown_cells(header_line)
        current = [
            index
            for index, cell in enumerate(cells)
            if normalize_text(cell) in {"chi tieu", "noi dung", "item", "indicator"}
        ]
        if len(current) > len(anchors):
            anchors = current
    return anchors


def _section_entities(section: str) -> list[str]:
    if not section:
        return []
    parts = re.split(
        r"(?i)\s+đơn\s+vị\s*:?\s*"
        r"(?:đồng|vnd|nghìn\s+đồng|triệu\s+đồng|tỷ\s+đồng)\s*",
        section,
    )
    entities = [
        f"entity:{normalize_text(part.strip(' ,;:-'))}"
        for part in parts
        if normalize_text(part).startswith(("cong ty ", "tong cong ty ", "tap doan "))
    ]
    return list(dict.fromkeys(entities))


def _table_subtable_provenance(
    lines: Sequence[str], line_index: int, cell_index: int
) -> tuple[str, str]:
    """Identify side-by-side logical tables only when repeated headers prove them."""

    anchors = _subtable_anchor_indices(lines, line_index)
    if len(anchors) < 2:
        table_id = _table_identifier(lines, line_index)
        return (
            (f"{table_id}/whole-table", "") if table_id else ("", "")
        )
    group = max(
        0,
        max(
            (index for index, anchor in enumerate(anchors) if anchor <= cell_index),
            default=0,
        ),
    )
    table_id = _table_identifier(lines, line_index)
    subtable_id = f"{table_id}/subtable-{group + 1}"
    entities = _section_entities(_nearest_table_section(lines, line_index))
    entity_id = entities[group] if group < len(entities) else ""
    return subtable_id, entity_id


def _source_entity_id(source_name: str, context: str = "") -> str:
    """Return a stable document-level entity when no explicit entity is present."""

    explicit = re.search(
        r"(?im)(?:^|\n)ENTITY:\s*(.*?)(?=\n[A-Z_]+:|$)",
        context,
    )
    if explicit and explicit.group(1).strip():
        return explicit.group(1).strip()
    source = Path(source_name).name
    if source_name in {"", "context", "synthetic"} or not source:
        return ""
    ticker = Path(source).stem.split("_", 1)[0]
    identity = normalize_text(ticker or Path(source).stem)
    scope_match = re.search(
        r"(?im)(?:^|\n)SCOPE:\s*(.*?)(?=\n[A-Z_]+:|$)",
        context,
    )
    scope = normalize_text(scope_match.group(1)) if scope_match else ""
    suffix = f":scope:{scope}" if scope else ""
    return f"entity:source:{identity}{suffix}"


def _nearest_table_section(lines: Sequence[str], line_index: int) -> str:
    start, _ = _table_block_bounds(lines, line_index)
    for index in range(start - 1, max(-1, start - 16), -1):
        text = lines[index].strip()
        if not text or "|" in text or len(text) > 120:
            continue
        normalized = normalize_text(text)
        if normalized.startswith(("don vi", "nguon")):
            continue
        if normalized.startswith(("section ", "phan ", "muc ")):
            return text.split(":", 1)[-1].strip()
        letters = [char for char in text if char.isalpha()]
        uppercase_ratio = (
            sum(char.isupper() for char in letters) / len(letters)
            if letters
            else 0.0
        )
        if uppercase_ratio >= 0.6 or any(
            term in normalized
            for term in (
                "ket qua kinh doanh",
                "tinh hinh tai chinh",
                "bang can doi",
                "bao cao ket qua",
            )
        ):
            return text
    return ""


def _table_scope(lines: Sequence[str], line_index: int) -> str:
    start, _ = _table_block_bounds(lines, line_index)
    window = lines[max(0, start - 24) : start]
    preceding = normalize_text(" ".join(window))
    consolidated_at = preceding.rfind("hop nhat")
    parent_at = max(
        preceding.rfind("cong ty me"),
        preceding.rfind("bao cao rieng"),
        preceding.rfind("bao cao tai chinh rieng"),
    )
    if consolidated_at < 0 and parent_at < 0:
        for raw_line in reversed(window):
            text = raw_line.strip(" •*-:")
            normalized = normalize_text(text)
            if normalized.startswith(("cong ty ", "tong cong ty ", "tap doan ")):
                return f"entity:{normalized}"
        return ""
    return "consolidated" if consolidated_at > parent_at else "parent_company"


def _plaintext_financial_row_context(
    lines: Sequence[str],
    line_index: int,
    start: int,
    end: int,
    marked: str,
    *,
    source_name: str,
    page: int | None,
    source_line_start: int,
    extraction_method: str,
) -> str:
    """Recover row/period metadata when native PDF extraction loses columns."""

    line = lines[line_index]
    matches: list[tuple[re.Match[str], Decimal]] = []
    for token in _NUMBER_RE.finditer(line):
        parsed = parse_decimal(token.group(0))
        if parsed is not None and (
            abs(parsed) >= Decimal(1000) or bool(token.group("pct"))
        ):
            matches.append((token, parsed))
    target_index = next(
        (
            index
            for index, (token, _) in enumerate(matches)
            if token.start() == start and token.end() == end
        ),
        -1,
    )
    if target_index < 0:
        return ""
    first_letter = re.search(r"[^\W\d_]", line, re.UNICODE)
    if first_letter is None:
        return ""
    label = line[first_letter.start() : matches[0][0].start()]
    label = re.sub(r"(?:\s+\d+(?:[.,]\d{1,2})?)+\s*$", "", label).strip(" .:-")
    if not label:
        return ""

    column_labels, raw_header_lines = _nearby_plaintext_column_labels(
        lines,
        line_index,
        expected_count=len(matches),
    )
    period_labels = _nearby_plaintext_period_labels(
        lines,
        line_index,
        expected_count=len(matches),
    )
    section, scope = _nearby_plaintext_table_provenance(lines, line_index)
    column = ""
    if len(column_labels) == len(matches):
        column = column_labels[target_index]
    else:
        financial_positions = [
            index
            for index, (_, value) in enumerate(matches)
            if abs(value) >= Decimal(1000)
        ]
        financial_target = (
            financial_positions.index(target_index)
            if target_index in financial_positions
            else -1
        )
        financial_count = len(financial_positions)
        if financial_target >= 0 and financial_count <= len(period_labels):
            # Native extraction removes empty cells. Financial period columns
            # are right-aligned, so a row containing only the current/end value
            # must align to the last header rather than losing period metadata.
            header_index = (
                financial_target + len(period_labels) - financial_count
            )
            if header_index < len(period_labels):
                column = period_labels[header_index]
    parts = [f"ROW: {label}", f"ROW_TYPE: {_row_type(label)}"]
    if column:
        parts.append(f"COLUMN: {column}")
        if period := _column_period(column):
            parts.append(f"PERIOD: {period}")
    if section:
        parts.append(f"SECTION: {section}")
    if scope:
        parts.append(f"SCOPE: {scope}")
    if raw_header_lines:
        parts.append(f"RAW_HEADER: {' || '.join(raw_header_lines)}")
    inventory_manifest = _flattened_table_role_inventory_manifest(
        lines,
        line_index,
        source_name=source_name,
        page=page,
        source_line_start=source_line_start,
        extraction_method=extraction_method,
        column_labels=column_labels,
    )
    if inventory_manifest:
        parts.append(f"TABLE: {inventory_manifest['table_id']}")
        parts.append(
            "ROLE_INVENTORY_MANIFEST: "
            + json.dumps(
                inventory_manifest,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    parts.append(f"CELL: {marked}")
    return "\n".join(parts)


def _flattened_role_key(label: str) -> str:
    normalized = normalize_text(label)
    if "%" in label or "ratio" in normalized:
        if re.search(r"(?:^|\W)(?:kh|ke hoach|plan)(?:\W|$)", normalized):
            return "ACTUAL_TO_PLAN_RATIO"
        if re.search(r"(?:^|\W)(?:th|thuc hien|actual)(?:\W|$)", normalized):
            return "ACTUAL_TO_PRIOR_RATIO"
    if any(
        token in normalized
        for token in (
            "ke hoach dieu chinh",
            "ke hoach sua doi",
            "revised plan",
            "adjusted plan",
        )
    ):
        return "REVISED_PLAN"
    if any(token in normalized for token in ("du bao", "uoc tinh", "forecast")):
        return "FORECAST"
    if any(token in normalized for token in ("ke hoach", " plan")):
        return "PLAN"
    if any(token in normalized for token in ("thuc hien", "thuc te", "actual")):
        return "ACTUAL"
    return ""


def _flattened_numeric_cell_slots(line: str) -> list[tuple[int, int]]:
    """Return numeric-cell spans without retaining or comparing their values."""

    output: list[tuple[int, int]] = []
    for match in _NUMBER_RE.finditer(line):
        raw = match.group(0)
        digits = re.sub(r"\D", "", raw)
        if not digits:
            continue
        if len(digits) == 4 and re.fullmatch(r"(?:19|20)\d{2}", digits):
            continue
        # Cell recognition uses only lexical shape/cardinality. The numeric
        # magnitude and answer value are never retained by the reconstruction.
        if match.group("pct") or len(digits) >= 4:
            output.append((match.start(), match.end()))
    return output


def _flattened_header_slot_sources(
    lines: Sequence[str],
    line_index: int,
    column_labels: Sequence[str],
) -> list[tuple[int, int, int]]:
    def positional_normalize(value: str) -> str:
        value = value.replace("đ", "d").replace("Đ", "D")
        value = "".join(
            character
            for character in unicodedata.normalize("NFKD", value)
            if not unicodedata.combining(character)
        ).lower()
        return "".join(
            character if character.isalnum() or character in {"/", "%"} else " "
            for character in value
        )

    window_start = max(0, line_index - 12)
    role_pattern = re.compile(
        r"ke hoach dieu chinh|ke hoach sua doi|revised plan|adjusted plan|"
        r"thuc hien|thuc te|actual|ke hoach|plan|du bao|uoc tinh|forecast"
    )
    role_lines: list[tuple[int, list[tuple[int, int]]]] = []
    for index in range(window_start, line_index):
        normalized = positional_normalize(lines[index])
        matches = [(match.start(), match.end()) for match in role_pattern.finditer(normalized)]
        if matches:
            role_lines.append((index, matches))
    base_count = sum(not str(label).lstrip().startswith("%") for label in column_labels)
    ratio_count = len(column_labels) - base_count
    if not role_lines:
        return []
    base_line, base_matches = max(
        role_lines,
        key=lambda item: (len(item[1]), item[0]),
    )
    if len(base_matches) != base_count:
        return []

    ratio_matches: list[tuple[int, int, int]] = []
    ratio_pattern = re.compile(
        r"\bth\s*(?:19|20)\d{2}\s*/\s*(?:kh|th)\s*(?:19|20)\d{2}\b"
    )
    for index in range(window_start, line_index):
        normalized = positional_normalize(lines[index])
        ratio_matches.extend(
            (index, match.start(), match.end())
            for match in ratio_pattern.finditer(normalized)
        )
    if len(ratio_matches) != ratio_count:
        return []
    return [
        *((base_line, start, end) for start, end in base_matches),
        *ratio_matches,
    ]


def _flattened_source_observation(
    *,
    source_name: str,
    page: int | None,
    source_line_start: int,
    local_line: int,
    start: int,
    end: int,
    extraction_method: str,
) -> dict[str, object]:
    global_line = source_line_start + local_line
    return {
        "document_id": source_name,
        "page": page if page is not None and page > 0 else 1,
        "source_span": [global_line * 100000 + start, global_line * 100000 + end],
        "source_sequence": global_line * 100000 + start,
        "extraction_method": extraction_method or "text",
        # Native layout text and OCR lines both preserve a local horizontal
        # interval. This is a character/bbox provenance interval, never a
        # numeric-value feature.
        "bbox": [start, global_line, max(start + 1, end), global_line + 1],
    }


def _flattened_table_role_inventory_manifest(
    lines: Sequence[str],
    line_index: int,
    *,
    source_name: str,
    page: int | None,
    source_line_start: int,
    extraction_method: str,
    column_labels: Sequence[str],
) -> dict[str, object] | None:
    """Conservatively reconstruct a flattened table under the QP3E lock."""

    if not column_labels or any(not _flattened_role_key(label) for label in column_labels):
        return None
    header_slots = _flattened_header_slot_sources(lines, line_index, column_labels)
    if len(header_slots) != len(column_labels):
        return None
    header_centers = [(start + end) / 2 for _, start, end in header_slots]
    if any(left >= right for left, right in zip(header_centers, header_centers[1:])):
        return None

    width = len(column_labels)
    header_end = max(index for index, _, _ in header_slots)
    rows: list[tuple[int, str, list[tuple[int, int]]]] = []
    started = False
    block_end = -1
    for index in range(header_end + 1, len(lines)):
        slots = _flattened_numeric_cell_slots(lines[index])
        first_letter = re.search(r"[^\W\d_]", lines[index], re.UNICODE)
        row_like = first_letter is not None and bool(slots)
        if row_like and len(slots) != width:
            if started:
                return None
            continue
        if row_like:
            started = True
            label = lines[index][first_letter.start() : slots[0][0]].strip(" .:-")
            if not label:
                return None
            rows.append((index, label, slots))
            block_end = index
            continue
        if started:
            break
    if not rows or line_index not in {index for index, _, _ in rows}:
        return None

    header_start = min(index for index, _, _ in header_slots)
    boundary_complete = header_start > 0 and block_end + 1 < len(lines)
    column_ids = [f"flat-column:{index}" for index in range(width)]
    headers: list[dict[str, object]] = []
    for ordinal, (label, (source_line, start, end)) in enumerate(
        zip(column_labels, header_slots)
    ):
        headers.append(
            {
                "column_id": column_ids[ordinal],
                "column_ordinal": ordinal,
                "column_band_id": f"flat-band:{ordinal}",
                "header_text": str(label),
                "canonical_role": _flattened_role_key(str(label)),
                "role_confidence": "EXPLICIT",
                "header_provenance": "explicit_flattened_header",
                "inheritance_anchor_id": "",
                "source": _flattened_source_observation(
                    source_name=source_name,
                    page=page,
                    source_line_start=source_line_start,
                    local_line=source_line,
                    start=start,
                    end=end,
                    extraction_method=extraction_method,
                ),
            }
        )

    band_bounds: list[tuple[float, float]] = []
    for ordinal, center in enumerate(header_centers):
        lower = (
            float("-inf")
            if ordinal == 0
            else (header_centers[ordinal - 1] + center) / 2
        )
        upper = (
            float("inf")
            if ordinal + 1 == width
            else (center + header_centers[ordinal + 1]) / 2
        )
        band_bounds.append((lower, upper))

    row_ids: list[str] = []
    row_observations: list[dict[str, object]] = []
    cells: list[dict[str, object]] = []
    for local_line, label, slots in rows:
        global_line = source_line_start + local_line
        row_id = f"flat-row:{global_line}"
        row_ids.append(row_id)
        label_start = lines[local_line].find(label)
        row_observations.append(
            {
                "row_id": row_id,
                "descriptor_id": f"descriptor:{global_line}",
                "source": _flattened_source_observation(
                    source_name=source_name,
                    page=page,
                    source_line_start=source_line_start,
                    local_line=local_line,
                    start=max(0, label_start),
                    end=max(1, slots[0][0]),
                    extraction_method=extraction_method,
                ),
            }
        )
        for ordinal, (start, end) in enumerate(slots):
            center = (start + end) / 2
            lower, upper = band_bounds[ordinal]
            aligned = lower <= center < upper
            column_id = column_ids[ordinal]
            cells.append(
                {
                    "cell_coordinate": f"{row_id}|{column_id}",
                    "row_id": row_id,
                    "column_id": column_id,
                    "column_ordinal": ordinal,
                    "column_band_id": (
                        f"flat-band:{ordinal}" if aligned else "flat-band:unresolved"
                    ),
                    "physical_cell_id": f"{source_name}:{global_line}:{start}:{end}",
                    "source": _flattened_source_observation(
                        source_name=source_name,
                        page=page,
                        source_line_start=source_line_start,
                        local_line=local_line,
                        start=start,
                        end=end,
                        extraction_method=extraction_method,
                    ),
                }
            )

    table_id = (
        f"flat-table:{source_line_start + header_start}:"
        f"{source_line_start + block_end}"
    )
    record: dict[str, object] = {
        "document_id": source_name,
        "table_id": table_id,
        "source_boundary_state": "COMPLETE" if boundary_complete else "PARTIAL",
        "table_continuation_state": "CLOSED" if block_end + 1 < len(lines) else "OPEN",
        "extractor_alignment_state": "CONSISTENT",
        "declared_data_width": width,
        "declared_data_column_ids": column_ids,
        "declared_row_ids": row_ids,
        "header_observations": headers,
        "inheritance_anchors": [],
        "row_observations": row_observations,
        "cell_observations": cells,
        "repeated_header_observations": [],
    }
    return qp3d_manifest_from_flattened_record(record)


def _nearby_plaintext_column_labels(
    lines: Sequence[str],
    line_index: int,
    *,
    expected_count: int,
) -> tuple[list[str], list[str]]:
    """Recover explicit role/year labels from a flattened native PDF header.

    This handles layouts whose visual multi-row header is emitted as separate
    plaintext lines.  A result is returned only when explicit role tokens and
    periods account for every numeric cell; no role is inferred from the
    numeric cell's left/right position alone.
    """

    raw_window = [
        line.strip()
        for line in lines[max(0, line_index - 12) : line_index]
        if line.strip()
    ]
    if not raw_window or expected_count <= 0:
        return [], []

    # Some native landscape tables flatten a complete header onto one line,
    # for example: Nam 2022 | Ty trong 2022 | Nam 2023 | Ty trong 2023 |
    # %2023/2022.  Map only when every numeric cell has one explicit textual
    # column role; no cell value or question answer participates.
    explicit_year_role_pattern = re.compile(
        r"\bnam\s+(?:19|20)\d{2}\b|"
        r"\bty\s+(?:trong|le)\s+(?:19|20)\d{2}\b|"
        r"\b(?:19|20)\d{2}\s+(?:19|20)\d{2}\b"
    )
    for raw_header in reversed(raw_window):
        normalized_header = normalize_text(raw_header)
        explicit_labels = [
            match.group(0)
            for match in explicit_year_role_pattern.finditer(normalized_header)
        ]
        if len(explicit_labels) == expected_count:
            return explicit_labels, raw_window

    def role_tokens(raw: str) -> list[str]:
        normalized = normalize_text(raw)
        pattern = re.compile(
            r"ke hoach dieu chinh|ke hoach sua doi|revised plan|adjusted plan|"
            r"thuc hien|thuc te|actual|ke hoach|plan|du bao|uoc tinh|forecast"
        )
        output: list[str] = []
        for match in pattern.finditer(normalized):
            token = match.group(0)
            if token in {"thuc hien", "thuc te", "actual"}:
                output.append("Thực hiện")
            elif token in {
                "ke hoach dieu chinh",
                "ke hoach sua doi",
                "revised plan",
                "adjusted plan",
            }:
                output.append("Kế hoạch điều chỉnh")
            elif token in {"ke hoach", "plan"}:
                output.append("Kế hoạch")
            else:
                output.append("Dự báo")
        return output

    role_lines = [
        (position, roles)
        for position, raw in enumerate(raw_window)
        for roles in (role_tokens(raw),)
        if roles
    ]
    if not role_lines:
        return [], []
    # The strongest explicit role row is normally the visual header's first
    # level.  Ties use the nearest row to the financial values.
    _, base_roles = max(role_lines, key=lambda item: (len(item[1]), item[0]))

    joined = " ".join(raw_window).replace("/", " QP1SLASH ")
    normalized_joined = normalize_text(joined)
    ratios: list[str] = []
    for match in re.finditer(
        r"\bth\s*((?:19|20)\d{2})\s+qp1slash\s+"
        r"(kh|th)\s*((?:19|20)\d{2})\b",
        normalized_joined,
    ):
        numerator_year, denominator, denominator_year = match.groups()
        if denominator == "kh":
            ratios.append(
                f"% TH {numerator_year} / KH {denominator_year}"
            )
        else:
            ratios.append(
                f"% TH {numerator_year} / TH {denominator_year}"
            )

    years = re.findall(r"\bnam\s+((?:19|20)\d{2})\b", normalized_joined)
    if len(years) < len(base_roles):
        return [], raw_window
    labels = [
        f"{role} {year}" for role, year in zip(base_roles, years)
    ]
    labels.extend(ratios)
    if len(labels) != expected_count:
        return [], raw_window
    return labels, raw_window


def _nearby_plaintext_table_provenance(
    lines: Sequence[str],
    line_index: int,
) -> tuple[str, str]:
    """Distinguish issuer tables from embedded related-company disclosures."""

    for previous in range(line_index - 1, max(-1, line_index - 29), -1):
        text = lines[previous].strip(" \t-*•:")
        if not text:
            continue
        normalized = normalize_text(text)
        related = re.search(
            r"\b(?:cong ty|don vi)\s+"
            r"(?:lien ket|lien doanh|con)\s*:?\s*(.+)$",
            normalized,
        )
        if related:
            identity = related.group(1).strip(" .:-")
            if identity:
                return text, f"entity:{identity}"

        # A numbered report section is a hard boundary.  In particular,
        # ``4.1. Tinh hinh tai chinh`` starts the issuer's own table and must
        # not inherit a related-company marker from an earlier investment
        # disclosure on the same or preceding page.
        if re.match(r"^\d+(?:\.\d+)*\.?\s+\D", text) and any(
            marker in normalized
            for marker in (
                "tinh hinh",
                "bao cao",
                "ket qua",
                "tai chinh",
                "hoat dong",
                "dau tu",
            )
        ):
            return text, ""
    return "", ""


def _nearby_plaintext_period_labels(
    lines: Sequence[str],
    line_index: int,
    *,
    expected_count: int | None = None,
) -> list[str]:
    """Find ordered period columns in nearby native-PDF header lines."""

    collected: list[tuple[int, str]] = []
    for previous in range(line_index - 1, max(-1, line_index - 13), -1):
        header = lines[previous]
        # A comparison/ratio column is commonly rendered on two lines:
        #
        #   Ti le (%)
        #   2023/2022
        #
        # Treating the latter as a pair of value-column headers swaps the
        # actual 2022 and 2023 monetary cells.  Remove year-over-year
        # expressions before looking for period labels.
        header_without_comparisons = re.sub(
            r"\b(?:19|20)\d{2}\s*[/\\]\s*(?:19|20)\d{2}\b",
            " ",
            header,
        )
        dates = re.findall(
            r"\b\d{1,2}[./-]\d{1,2}[./-](?:19|20)\d{2}\b",
            header_without_comparisons,
        )
        if len(dates) >= 2:
            return dates
        normalized = normalize_text(header_without_comparisons)
        normalized = re.sub(
            r"\b(?:19|20)\d{2}\s+(?:so voi|vs)\s+(?:19|20)\d{2}\b",
            " ",
            normalized,
        )
        phrases = re.findall(
            r"(?:so\s+)?(?:cuoi|dau)\s+(?:nam|ky)(?:\s+(?:19|20)\d{2})?",
            normalized,
        )
        if len(phrases) >= 2:
            return phrases
        years = re.findall(r"\b(?:19|20)\d{2}\b", normalized)
        if len(years) >= 2:
            return years

        # Native PDF extraction often emits each header cell on its own line.
        # Keep singular period labels together with their original line index,
        # then restore the visual top-to-bottom (left-to-right) order.
        singular: list[str] = []
        if len(dates) == 1:
            singular = dates
        elif len(phrases) == 1:
            singular = phrases
        elif len(years) == 1 and any(
            token in normalized
            for token in ("nam", "ky", "chi tieu", "column", "period")
        ):
            singular = years
        for label in singular:
            collected.append((previous, label))
        if expected_count and len(collected) >= expected_count:
            break

    ordered = [
        label
        for _, label in sorted(collected, key=lambda item: item[0])
    ]
    return list(dict.fromkeys(ordered))


def _inline_period_column(cells: Sequence[str], cell_index: int) -> str:
    if not 0 <= cell_index < len(cells):
        return ""
    match = re.match(r"^\s*(.+?):\s*[()+\-]?\d", cells[cell_index])
    if match is None:
        return ""
    label = match.group(1).strip()
    return label if _column_period(label) else ""


def _financial_value_cell_indices(cells: Sequence[str]) -> list[int]:
    """Return the likely period-value cells, excluding row and note codes."""

    output: list[int] = []
    for index, cell in enumerate(cells):
        values = [parse_decimal(match.group(0)) for match in _NUMBER_RE.finditer(cell)]
        if any(
            value is not None
            and abs(value) >= Decimal(1000)
            and value.adjusted() < 24
            and not _is_probable_year(value)
            for value in values
        ):
            output.append(index)
    return output


def _infer_unit(line: str, is_percent: bool) -> str:
    normalized = normalize_text(line.replace("Đ", "D").replace("đ", "d"))
    if is_percent:
        return "%"
    if any(
        term in normalized
        for term in (
            "so luong co phieu",
            "so luong co phan",
            "co phieu dang luu hanh",
            "co phan dang luu hanh",
            "co phieu da phat hanh",
            "co phan da phat hanh",
        )
    ):
        return "shares"
    if "menh gia" in normalized and any(
        term in normalized for term in ("vnd", "dong")
    ):
        return "VND/share"
    if any(term in normalized for term in ("vnd", "dong", "trieu", "ty dong")):
        return "VND"
    if any(term in normalized for term in ("usd", "dollar")):
        return "USD"
    # A landscape extraction may concatenate ``Don vi tinh: ty dong`` from
    # one table with a coefficient heading from the adjacent table.  Explicit
    # currency tokens therefore take precedence over a generic ``he so``.
    # Ratio-row contexts still resolve to ``x`` when no currency is declared.
    if "he so" in normalized:
        return "x"
    # ``co phan`` by itself commonly occurs in a company's legal name.  Treating
    # that phrase as a share-count unit contaminates every financial row whose
    # candidate context carries ENTITY/SCOPE metadata (for example, revenue of
    # a ``Cong ty co phan``).  Explicit share-count phrases are handled above;
    # the generic fallback therefore only accepts unambiguous share tokens.
    if any(term in normalized for term in ("co phieu", "shares")):
        return "shares"
    return ""


def _explicit_page_unit(lines: Sequence[str]) -> str:
    """Return one unambiguous unit declared by a page-level unit header."""

    units = {
        unit
        for line in lines
        if any(
            marker in normalize_text(line.replace("Đ", "D").replace("đ", "d"))
            for marker in ("don vi", "dvt")
        )
        for unit in (_infer_unit(line, False),)
        if unit
    }
    return next(iter(units)) if len(units) == 1 else ""


def _is_percent_candidate_column(candidate_context: str) -> bool:
    column = next(
        (
            item.removeprefix("COLUMN:").strip()
            for item in candidate_context.splitlines()
            if item.startswith("COLUMN:")
        ),
        "",
    )
    if not column:
        return False
    normalized = normalize_text(column)
    return "%" in column or any(
        term in normalized
        for term in ("phan tram", "ty le", "ty suat", "percent")
    )


def generate_synthetic_program_examples(
    *,
    count_per_operation: int = 100,
    seed: int = 29,
    max_candidates: int = 64,
) -> list[ProgramExample]:
    """Balanced financial program tasks used as HRM-style augmentation.

    The original HRM experiments heavily augment a small seed set. Here the
    augmentation changes entities, values, row order and distractors while the
    target remains only an input-output program; no chain-of-thought labels are
    used by the model.
    """
    import random

    rng = random.Random(seed)
    output: list[ProgramExample] = []
    labels = (
        "Doanh thu thuần",
        "Lợi nhuận sau thuế",
        "Vốn chủ sở hữu",
        "Tổng tài sản",
        "Nợ phải trả",
        "Chi phí quản lý doanh nghiệp",
        "Tài sản ngắn hạn",
        "Tiền và tương đương tiền",
        "Hàng tồn kho",
        "Lợi nhuận gộp",
    )
    for operation in OPERATIONS:
        for sample_index in range(count_per_operation):
            year = rng.choice((2022, 2023, 2024, 2025))
            chosen = rng.sample(labels, 4)
            amounts = [Decimal(rng.randint(20_000, 900_000)) * Decimal(1_000_000) for _ in range(8)]
            # Distinct small multiplicands for MULTIPLY and ratio diversity.
            if operation == "MULTIPLY":
                operands = (Decimal(rng.randint(2, 80)), Decimal(rng.randint(100_000, 9_000_000)))
            elif operation == "PERCENT_CHANGE":
                baseline = amounts[1]
                current = (baseline * Decimal(rng.randint(60, 180)) / Decimal(100)).quantize(Decimal(1))
                operands = (current, baseline)
            elif operation in {"DIVIDE", "RATIO_PERCENT"}:
                operands = (amounts[0], amounts[1])
            elif operation == "SUBTRACT":
                high = max(amounts[0], amounts[1])
                low = min(amounts[0], amounts[1])
                operands = (high, low)
            elif operation == "AVERAGE":
                operands = (amounts[0], amounts[1])
            elif operation == "ADD":
                operands = (amounts[0], amounts[1])
            elif operation == "SUM3":
                operands = tuple(amounts[:3])
            elif operation == "SUM4":
                operands = tuple(amounts[:4])
            else:
                operands = (amounts[0],)

            question, row_labels = _synthetic_question(
                operation, chosen, year, variant=sample_index
            )
            rows_text: list[str] = []
            for position, value in enumerate(operands):
                label = row_labels[position]
                rows_text.append(f"| {label} | {_format_vietnamese(value)} VND |")
            distractors = rng.sample(list(zip(labels, amounts[4:])), k=3)
            for label, value in distractors:
                rows_text.append(f"| {label} khác | {_format_vietnamese(value)} VND |")
            rng.shuffle(rows_text)
            context = (
                f"Công ty: Doanh nghiệp mô phỏng {sample_index}\n"
                f"Mã cổ phiếu: SYN\nNăm báo cáo: {year}\n"
                "| Chỉ tiêu | Giá trị |\n|---|---:|\n"
                + "\n".join(rows_text)
            )
            candidates = extract_numeric_candidates(
                question, context, max_candidates=max_candidates, source_name="synthetic"
            )
            pointer_indices: list[int] = []
            used: set[int] = set()
            for value, label in zip(operands, row_labels):
                choices = [
                    candidate
                    for candidate in candidates
                    if candidate.index not in used
                    and _close(candidate.decimal, value)
                    and normalize_text(label) in normalize_text(candidate.context)
                ]
                if not choices:
                    choices = [
                        candidate
                        for candidate in candidates
                        if candidate.index not in used and _close(candidate.decimal, value)
                    ]
                if not choices:
                    pointer_indices = []
                    break
                selected = max(choices, key=lambda item: item.relevance)
                used.add(selected.index)
                pointer_indices.append(selected.index)
            if not pointer_indices:
                continue
            program = ReasoningProgram(operation, tuple(pointer_indices))
            if operation in COMMUTATIVE:
                program = ReasoningProgram(operation, tuple(sorted(program.operands)))
            answer = execute_program(program, candidates)
            answer_text = (
                f"Kết quả = {_decimal_string(answer)}"
                + ("%." if operation in {"RATIO_PERCENT", "PERCENT_CHANGE"} else ".")
            )
            output.append(
                ProgramExample(
                    example_id=f"synthetic-{operation.lower()}-{sample_index}",
                    question=question,
                    context=context,
                    answer_text=answer_text,
                    candidates=candidates,
                    program=program,
                    answer_value=_decimal_string(answer),
                    task="QA",
                    difficulty="Synthetic",
                )
            )
    return output


def write_program_examples(
    examples: Sequence[ProgramExample], path: str | Path
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(program_example_to_dict(example), ensure_ascii=False) + "\n")
    return destination


def _synthetic_question(
    operation: str, labels: Sequence[str], year: int, *, variant: int = 0
) -> tuple[str, tuple[str, ...]]:
    a, b, c, d = labels[:4]
    if operation == "SELECT":
        templates = (
            f"{a} năm {year} là bao nhiêu?",
            f"Theo báo cáo tài chính, {a} tại ngày 31/12/{year} là bao nhiêu VND?",
            f"Giá trị {a} cuối năm {year} là bao nhiêu?",
            f"Cho biết số dư {a} năm {year}.",
        )
        return templates[variant % len(templates)], (a,)
    if operation == "ADD":
        templates = (
            f"Tính tổng {a} và {b} năm {year}.",
            f"Cộng {a} với {b} trong năm {year}.",
            f"{a} và {b} có tổng giá trị bao nhiêu trong năm {year}?",
        )
        return templates[variant % len(templates)], (a, b)
    if operation == "SUBTRACT":
        selected = variant % 6
        templates = (
            f"Tính chênh lệch giữa {a} và {b} năm {year}.",
            f"{a} cao hơn {b} bao nhiêu VND trong năm {year}?",
            f"Lấy {a} trừ {b} năm {year} được bao nhiêu?",
            f"Hiệu số giữa {a} và {b} năm {year} là bao nhiêu?",
            f"So sánh {a} cuối kỳ {year} so với đầu kỳ {year}.",
            f"So sánh {a} năm nay và năm trước.",
        )
        row_labels = (
            (f"{a} cuối kỳ", f"{a} đầu kỳ")
            if selected == 4
            else (f"{a} năm nay", f"{a} năm trước")
            if selected == 5
            else (a, b)
        )
        return templates[selected], row_labels
    if operation == "MULTIPLY":
        return (
            f"Tính doanh thu năm {year} bằng số lượng bán nhân với giá bán bình quân.",
            ("Số lượng bán", "Giá bán bình quân"),
        )
    if operation == "DIVIDE":
        templates = (
            f"Tính hệ số {a} trên {b} năm {year}.",
            f"{a} bằng bao nhiêu lần {b} trong năm {year}?",
            f"Lấy {a} chia cho {b} năm {year}.",
        )
        return templates[variant % len(templates)], (a, b)
    if operation == "RATIO_PERCENT":
        templates = (
            f"Tính tỷ lệ {a} trên {b} năm {year} theo phần trăm.",
            f"{a} chiếm bao nhiêu phần trăm {b} năm {year}?",
            f"Trong tổng {b}, {a} chiếm tỷ trọng bao nhiêu phần trăm?",
            f"Tỷ suất {a} so với {b} trong năm {year} là bao nhiêu %?",
        )
        return templates[variant % len(templates)], (a, b)
    if operation == "PERCENT_CHANGE":
        templates = (
            f"Tính tốc độ tăng trưởng của {a} từ đầu năm đến cuối năm {year}.",
            f"{a} thay đổi bao nhiêu phần trăm từ năm {year - 1} đến năm {year}?",
            f"Mức thay đổi phần trăm của {a} trong năm {year} là bao nhiêu?",
            f"So với năm {year - 1}, {a} năm {year} biến động bao nhiêu phần trăm?",
        )
        return templates[variant % len(templates)], (
            f"{a} cuối năm",
            f"{a} đầu năm",
        )
    if operation == "AVERAGE":
        return (
            f"Tính {a} bình quân năm {year} từ số đầu năm và cuối năm.",
            (f"{a} cuối năm", f"{a} đầu năm"),
        )
    if operation == "SUM3":
        return f"Tính tổng {a}, {b} và {c} năm {year}.", (a, b, c)
    if operation == "SUM4":
        return f"Tính tổng {a}, {b}, {c} và {d} năm {year}.", (a, b, c, d)
    raise ProgramDataError(f"Unsupported synthetic operation: {operation}")


def _format_vietnamese(value: Decimal) -> str:
    integer = int(value.to_integral_value())
    return f"{integer:,}".replace(",", ".")
