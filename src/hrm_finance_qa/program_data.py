from __future__ import annotations

from .physical_coordinates import LEGACY_SPAN, LINE_SPAN, TABLE_CELL, SPAN_SCOPES, line_digest

import itertools
import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Sequence

from .ontology import normalize_text
from .exact_executor import UnsupportedOperationError, execute_decimal_operation
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
# OPERATIONS is the frozen neural/checkpoint vocabulary. V14 extends the
# logical program DSL without changing those IDs so V8/V13 operation heads keep
# their exact tensor shapes and historical metrics remain comparable.
SUM_N_OPERATION = "SUM_N"
PROGRAM_OPERATIONS: tuple[str, ...] = (*OPERATIONS, SUM_N_OPERATION)
SUM_OPERATION_ALIASES = frozenset({"ADD", "SUM3", "SUM4", SUM_N_OPERATION})
SUM_N_MIN_OPERANDS = 2
SUM_N_MAX_OPERANDS = 8
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
COMMUTATIVE = {
    "ADD",
    "MULTIPLY",
    "AVERAGE",
    "SUM3",
    "SUM4",
    SUM_N_OPERATION,
}
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


def is_sum_operation(operation: str) -> bool:
    """Return whether operation belongs to the additive program family."""

    return operation in SUM_OPERATION_ALIASES


def operation_arity_bounds(operation: str) -> tuple[int, int]:
    """Return inclusive operand-count bounds for a logical operation.

    Legacy operations remain fixed-arity. SUM_N is the only V14 variable-
    arity operation and is deliberately bounded so decoding/search stays finite.
    """

    if operation == SUM_N_OPERATION:
        return SUM_N_MIN_OPERANDS, SUM_N_MAX_OPERANDS
    try:
        arity = OPERATION_ARITY[operation]
    except KeyError as exc:
        raise ProgramDataError(f"Unknown operation: {operation}") from exc
    return arity, arity


def operation_accepts_arity(operation: str, operand_count: int) -> bool:
    """Return whether an operand count is valid for operation."""

    try:
        minimum, maximum = operation_arity_bounds(operation)
    except ProgramDataError:
        return False
    return minimum <= operand_count <= maximum


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
    row_id: str = ""
    column_id: str = ""
    column_role: str = ""
    source_bbox: tuple[float, float, float, float] | None = None
    derived_operation: str = ""
    derived_operands: tuple[int, ...] = ()
    derivation_id: str = ""
    source_span_scope: str = LEGACY_SPAN
    source_line_sha256: str = ""

    source_section_path: tuple[str, ...] = ()
    table_semantics: dict | None = None
    number_format: dict | None = None
    report_context: dict | None = None
    unit_assumption: str = ""

    @property
    def source_unit(self) -> str:
        """A default is not a unit recovered from the source."""
        return "" if self.unit_assumption else self.unit

    def __post_init__(self) -> None:
        if self.source_span_scope not in SPAN_SCOPES:
            raise ProgramDataError("Unknown source span coordinate scope")
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
            "column_role": "COLUMN_ROLE",
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
        minimum, maximum = operation_arity_bounds(self.operation)
        if not minimum <= len(self.operands) <= maximum:
            expected = (
                str(minimum)
                if minimum == maximum
                else f"between {minimum} and {maximum}"
            )
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


def canonical_sum_family_equal(
    predicted: ReasoningProgram, gold: ReasoningProgram
) -> bool:
    """Compare V14 logical sums without changing historical exact metrics.

    The default evaluator intentionally keeps ``ADD``/``SUM3``/``SUM4``
    distinct from ``SUM_N``. V14 evaluators opt into this helper so a logical
    variable-arity sum is equivalent only to the legacy alias with the same
    operand count and the same unordered physical pointers.
    """

    if not (is_sum_operation(predicted.operation) and is_sum_operation(gold.operation)):
        return canonical_program_equal(predicted, gold)
    if len(predicted.operands) != len(gold.operands):
        return False
    if not operation_accepts_arity(predicted.operation, len(predicted.operands)):
        return False
    if not operation_accepts_arity(gold.operation, len(gold.operands)):
        return False
    return tuple(sorted(predicted.operands)) == tuple(sorted(gold.operands))


def canonical_pointer_equal(
    operation: str,
    predicted: Sequence[int],
    gold: Sequence[int],
) -> bool:
    if operation in COMMUTATIVE:
        return tuple(sorted(predicted)) == tuple(sorted(gold))
    return tuple(predicted) == tuple(gold)


DAG_OPERATION_ARITY = {**OPERATION_ARITY, "SHARE_OF_TOTAL": 2}


def dag_operation_arity_bounds(operation: str) -> tuple[int, int]:
    """Return inclusive arity bounds for an executable DAG operation."""

    if operation == "SHARE_OF_TOTAL":
        return 2, 2
    return operation_arity_bounds(operation)


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
            try:
                minimum, maximum = dag_operation_arity_bounds(operation)
            except ProgramDataError:
                raise ProgramDataError(
                    f"Unsupported DAG operation: {node.operation}"
                ) from None
            if not minimum <= len(node.inputs) <= maximum:
                expected = (
                    str(minimum)
                    if minimum == maximum
                    else f"between {minimum} and {maximum}"
                )
                raise ProgramDataError(
                    f"{operation} needs {expected} inputs, "
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
        # A zero integer part denotes a fraction, including 3+ decimals.
        # Keep the whole token; do not split 0,0004 into 0,000 and 4.
        r"0[.,]\d+(?!\d|[.,]\d)",
        # Vietnamese thousands groups; tolerate PDF whitespace before dots.
        r"\d{1,3}(?:\s*\.\s?\d{3})+(?:,\d+)?(?!\d|[.,]\d)",
        # English thousands groups.
        r"\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\d|[.,]\d)",
        # Thousands separated by ordinary spaces.
        r"\d{1,3}(?:[ \u00a0]\d{3})+",
        # Ungrouped integer or short decimal.
        r"\d+(?:[.,]\d+)*(?!\d|[.,]\d)",
    )
) + r")"
_NUMBER_RE = re.compile(
    rf"(?<![\w/])(?P<paren>\()?[-+]?{_NUMBER_BODY}(?P<close>\))?"
    r"\s*(?P<pct>%|phần\s*trăm)?",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")

# A scalar table cell can retain only some of its thousands spaces. Match
# the whole cell value, never glue tokens across a pipe or in prose.
_CELL_SPACED_INTEGER_RE = re.compile(
    r"(?:(?<=\|)|(?<=:))[ \u00a0]*(?P<paren>\()?[-+]?"
    r"(?P<head>[1-9]\d{3,})(?:[ \u00a0]+\d{3})+"
    r"(?P<close>\))?[ \u00a0]*(?P<pct>%|phần\s*trăm)?"
    r"(?=[ \u00a0]*(?:\||$))", re.IGNORECASE,
)


def _iter_number_matches(line: str):
    """Keep partially space-grouped integers whole inside one table cell.

    A suffix, range, list, note, unbalanced sign or year-like first group
    cannot establish a single value. Original source offsets are preserved.
    """
    whole = [m for m in _CELL_SPACED_INTEGER_RE.finditer(line)
             if "|" in line and not _YEAR_RE.fullmatch(m.group("head"))
             and bool(m.group("paren")) == bool(m.group("close"))]
    if not whole:
        yield from _NUMBER_RE.finditer(line)
        return
    tokens = [m for m in _NUMBER_RE.finditer(line)
              if not any(m.start() < cell.end() and m.end() > cell.start()
                         for cell in whole)]
    yield from sorted([*tokens, *whole], key=lambda m: m.start())

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
    if re.fullmatch(r"[-+]?0[.,]\d+", text):
        text = text.replace(",", ".")
    elif "," in text and "." in text:
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
    return not before_cell and bool(re.match(r"\s*(?:[.)](?:\s|$)|[-–—]\s*[^\W\d_])", suffix, re.UNICODE))


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
    if _is_table_metadata_cell_code(
        line, match, value, column_label=next((part[7:].strip() for part in candidate_context.splitlines() if part.startswith("COLUMN:")), "")
    ):
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
    if "|" in line and normalized_column in {"column noi dung", "column chi tieu", "column indicator", "column item", "column description", "column metric"}:
        from .metric_semantics import row_code_equation_span
        start = line.rfind("|", 0, match.start()) + 1
        end = line.find("|", match.end())
        cell = line[start:end if end >= 0 else len(line)]
        equation = row_code_equation_span(cell)
        if equation and equation[0] <= match.start() - start < equation[1]:
            return True
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
    line: str, match: re.Match[str], value: Decimal, *, column_label: str = ""
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

    # An explicit period-bearing data column or inline period label is
    # stronger evidence than a magnitude/position heuristic for note codes.
    # Retain explicit metadata labels above, regardless of numeric format.
    if (_inline_period_column(cells, cell_index)
            or (_column_period(column_label)
                and not any(label in normalize_text(column_label) for label in metadata))):
        return False

    # A grouped/decimal printed amount is not a bare note code. Explicit
    # metadata labels above still win; magnitude alone cannot reject it.
    if re.search(r"[.,]", match.group(0)):
        return False

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
            raw_token = token.group(0).strip()
            if (
                parsed is not None
                and abs(parsed) >= Decimal(1000)
                and (
                    not _is_probable_year(parsed)
                    or _has_explicit_grouped_integer_format(raw_token)
                )
                and not _is_natural_date_component(cell, token)
            ):
                following_values += 1
    return has_textual_label and following_values >= 2


def _has_explicit_grouped_integer_format(raw: str) -> bool:
    """Return whether punctuation explicitly encodes thousands grouping.

    Plain ``2000`` remains year-like.  ``2.000`` and ``2,000`` are distinct
    source forms: inside a numeric cell the separator is physical evidence
    that the token was printed as an amount rather than as a year label.
    """

    cleaned = raw.strip().replace(" ", "")
    return bool(
        re.fullmatch(r"\(?[+-]?\d{1,3}(?:[.,]\d{3})+\)?", cleaned)
    )


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
    structured_cells: Sequence[object] = (),
) -> tuple[NumericCandidate, ...]:
    if max_candidates < 4:
        raise ProgramDataError("max_candidates must be at least 4")
    lines = context.splitlines() or [context]
    from .number_formats import local_records, for_candidate
    formats = local_records(context, source_name, page, source_line_start)
    question_tokens = _content_tokens(_question_focus_text(question))
    year_tokens = {token for token in normalize_text(question).split() if _YEAR_RE.match(token)}
    raw_candidates: list[NumericCandidate] = []
    for cell in sorted(
        structured_cells,
        key=lambda item: (
            int(getattr(item, "page", 0)),
            str(getattr(item, "table_id", "")),
            str(getattr(item, "row_id", "")),
            str(getattr(item, "column_id", "")),
            tuple(getattr(item, "source_span", ())),
        ),
    ):
        row_label = str(getattr(cell, "row_label", "")).strip()
        column_label = str(getattr(cell, "column_label", "")).strip()
        period = str(getattr(cell, "period", "")).strip()
        table_id = str(getattr(cell, "table_id", "")).strip()
        row_id = str(getattr(cell, "row_id", "")).strip()
        column_id = str(getattr(cell, "column_id", "")).strip()
        column_role = str(getattr(cell, "column_role", "")).strip().lower()
        source_span = tuple(getattr(cell, "source_span", ()))
        source_bbox = tuple(getattr(cell, "bbox", ()))
        if not (
            row_label
            and column_label
            and period
            and table_id
            and row_id
            and column_id
            and column_role in {"actual", "plan"}
            and len(source_span) == 2
            and len(source_bbox) == 4
        ):
            continue
        value = parse_decimal(str(getattr(cell, "value", "")))
        if value is None:
            continue
        context_parts = [
            f"ROW: {row_label}",
            f"ROW_TYPE: {_row_type(row_label)}",
            f"COLUMN: {column_label}",
            f"PERIOD: {period}",
            f"TABLE: {table_id}",
            f"ROW_ID: {row_id}",
            f"COLUMN_ID: {column_id}",
            (
                f"CELL: {row_label} <VALUE> "
                f"{getattr(cell, 'raw_value', getattr(cell, 'value', ''))} "
                f"</VALUE> {column_label}"
            ),
        ]
        context_value = "\n".join(context_parts)
        relevance = (
            _fuzzy_content_overlap(question_tokens, _content_tokens(row_label))
            + 0.20 * max(0.0, min(1.0, retrieval_score))
            + 0.12
        )
        if year_tokens and period in year_tokens:
            relevance += 0.35
        elif year_tokens and period:
            relevance += 0.05
        raw_candidates.append(
            NumericCandidate(
                index=-1,
                value=_decimal_string(value),
                raw=str(getattr(cell, "raw_value", getattr(cell, "value", ""))),
                context=context_value,
                line_number=source_line_start + int(source_span[0]),
                char_start=int(source_span[0]),
                char_end=int(source_span[1]),
                unit=str(getattr(cell, "unit", "")),
                is_percent=False,
                is_year_like=False,
                relevance=relevance,
                source_name=source_name,
                page=int(getattr(cell, "page", page or 0)) or page,
                extraction_method=str(
                    getattr(cell, "extraction_method", "structured_table")
                ),
                row_label=row_label,
                column_label=column_label,
                period=period,
                row_type=_row_type(row_label),
                table_id=table_id,
                entity_id=_source_entity_id(source_name, context_value),
                source_span=(int(source_span[0]), int(source_span[1])),
                source_span_scope=TABLE_CELL,
                row_id=row_id,
                column_id=column_id,
                column_role=column_role,
                source_bbox=tuple(float(item) for item in source_bbox),
            )
        )
    page_unit = _explicit_page_unit(lines)
    for line_index, line in enumerate(lines):
        if not line.strip():
            continue
        if _is_table_header_or_separator(lines, line_index):
            continue
        normalized_line = normalize_text(line)
        line_tokens = set(normalized_line.split())
        overlap = _fuzzy_content_overlap(question_tokens, line_tokens)
        for match in _iter_number_matches(line):
            if (match.start() > 0 and line[match.start() - 1] == "/") or (
                match.end() < len(line) and line[match.end()] == "/"
            ):
                continue
            raw = match.group(0).strip()
            number_format = for_candidate(raw, match.start(), formats.get(line_index, {}).get("number_formats", ()))
            value = Decimal(number_format["value"]) if number_format["state"] == "VALID" else parse_decimal(raw)
            if value is None:
                continue
            # Values with 25+ integer digits are almost certainly several OCR
            # table cells accidentally glued together, not financial operands.
            if value and value.adjusted() >= 24:
                continue
            digits = re.sub(r"\D", "", raw)
            is_year_like = bool(_YEAR_RE.fullmatch(digits)) and not (
                _has_explicit_grouped_integer_format(raw)
            )
            if is_year_like:
                # Years are period metadata, never arithmetic operands in the
                # single-result financial program schema.
                continue
            candidate_context = _candidate_context(
                lines,
                line_index,
                match.start(),
                match.end(),
                question=question,
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
            if "|" in line:
                unit = _table_candidate_unit(
                    lines,
                    line_index,
                    match.start(),
                    is_percent=is_percent,
                )
            else:
                unit = (
                    ("%" if is_percent else _number_local_unit(line, match.end()))
                    or _infer_unit(candidate_context, is_percent)
                    or ("" if is_percent else page_unit)
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
                    number_format=number_format,
                    context=candidate_context,
                    line_number=source_line_start + line_index,
                    source_span_scope=LINE_SPAN,
                    source_line_sha256=line_digest(line),
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
                source_span_scope=LINE_SPAN,
                source_line_sha256=line_digest(lines[line_index]),
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
    # Keep geometry-backed observations outside the effective Candidate pool
    # until query compatibility, inventory uniqueness, conflicts and any
    # plaintext equivalence have all been resolved. A valid table is not, by
    # itself, evidence that the table is authoritative for this question.
    structured_output: list[NumericCandidate] = []
    for chunk in chunks:
        structured_cells = tuple(getattr(chunk, "structured_cells", ()))
        if not structured_cells:
            continue
        raw_score = float(getattr(chunk, "score", 0.0))
        normalized_score = raw_score / maximum_score if maximum_score > 0 else 0.0
        structured_output.extend(
            extract_numeric_candidates(
                question,
                "",
                max_candidates=max(max_candidates * 2, 64),
                source_name=str(chunk.source_name),
                page=getattr(chunk, "page_start", None),
                source_line_start=int(getattr(chunk, "line_start", 1)),
                retrieval_score=normalized_score,
                extraction_method="structured_table",
                financial_identity_catalog_enabled=(
                    financial_identity_catalog_enabled
                ),
                structured_cells=structured_cells,
            )
        )
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

    # Attach source-wide metadata after extraction; retain all source coordinates.
    binding_options: dict[tuple[object, ...], list[dict]] = {}
    for chunk in chunks:
        for item in getattr(chunk, "source_annotations", ()):
            binding_options.setdefault((str(chunk.source_name), getattr(chunk, "page_start", None), item["line_sha256"], item.get("line_number")), []).append(item)
    from dataclasses import replace
    bound_output = []
    for candidate in output:
        options = binding_options.get((candidate.source_name, candidate.page, candidate.source_line_sha256, candidate.line_number), [])
        unique = {json.dumps(item, sort_keys=True): item for item in options}
        if len(unique) != 1:
            bound_output.append(candidate)
            continue
        binding = next(iter(unique.values()))
        unit = candidate.unit
        context = candidate.context
        if not unit and binding["unit"] and not candidate.is_percent:
            unit = binding["unit"]
            context += "\nUNIT_SOURCE: " + json.dumps(binding["unit_evidence"], ensure_ascii=False, sort_keys=True)
        # A header ending in the currency code VND identifies currency, not
        # a contradictory scale of one. Keep a declared table scale when no
        # more specific numeric-cell unit overrides it.
        if (unit == "VND" and binding["unit"] and not binding["unit_conflict"]
                and currency_unit_scale(binding["unit"]) not in {None,Decimal(1)}
                and re.search(r"\bvnd$",normalize_text(candidate.column_label))):
            marked=next((x[5:] for x in context.splitlines() if x.startswith("CELL:")), "")
            cell_text=re.sub(r"</?VALUE>", "", next((part for part in marked.split('|') if '<VALUE>' in part),marked))
            after_number=cell_text.rsplit(candidate.raw,1)[-1] if candidate.raw in cell_text else ""
            if not _explicit_currency_units(after_number):
                unit=binding["unit"]
                context += "\nUNIT_SOURCE: " + json.dumps(binding["unit_evidence"],ensure_ascii=False,sort_keys=True)
        if binding["unit_conflict"]:
            cell_text = re.sub(r"</?VALUE>", "", next((x[5:] for x in context.splitlines() if x.startswith("CELL:")), ""))
            if not _infer_unit(cell_text, candidate.is_percent):
                unit = ""
                context += "\nUNIT_SOURCE_CONFLICT: true"
        from .number_formats import for_candidate
        number_format = for_candidate(candidate.raw, candidate.char_start, binding.get("number_formats", ()))
        value = _decimal_string(Decimal(number_format["value"])) if number_format["state"] == "VALID" else candidate.value
        continuation = binding.get('numbered_continuation')
        if continuation:
            row = re.sub(r'</?VALUE>', '', next((x[5:] for x in context.splitlines() if x.startswith('CELL:')), ''))
            cells = _markdown_cells(row)
            index = row[:candidate.source_span[0]].count('|') - int(row.lstrip().startswith('|'))
            headers = _aligned_table_header_cells(_markdown_cells(continuation['headers'][-1]['text']), len(cells))
            if 0 <= index < len(headers):
                label = headers[index]
                context = re.sub(r'(?m)^COLUMN:.*$', 'COLUMN: '+label, context)
                if not re.search(r'(?m)^COLUMN:',context):context += '\nCOLUMN: '+label
                candidate = replace(candidate, column_label=label)
        semantic = binding.get("table_semantics")
        entity_id = candidate.entity_id
        if semantic and semantic.get("entity_id"):
            if entity_id and not entity_id.startswith("entity:source:") and entity_id != semantic["entity_id"]:
                semantic = {**semantic, "entity_conflict": True}
            else:
                entity_id = semantic["entity_id"]
                context = re.sub(r"(?m)^ENTITY:.*$", "ENTITY: " + entity_id, context)
        report_context = binding.get("report_context")
        if report_context is None and "REPORT_FIELD_SOURCE: " in context:
            # Field identity comes from its own hashed line. A report title is
            # optional metadata until the question explicitly requests it.
            report_context = {"report_year": "", "title_evidence": []}
        owner = binding.get("entity_owner")
        if owner:
            semantic = {**(semantic or {}), "entity_owner": owner}
            # The source segment binds a legal entity, not accounting scope.
            # More specific but conflicting row/table ownership still blocks.
            same_named_context = (entity_id.startswith("entity:context:")
                                  and normalize_text(entity_id[len("entity:context:"):]) == normalize_text(owner["name"]))
            if entity_id and not entity_id.startswith("entity:source:") and entity_id != owner["entity_id"] and not same_named_context:
                semantic["entity_conflict"] = True
            entity_id = owner["entity_id"]
            if re.search(r"(?m)^ENTITY:", context):
                context = re.sub(r"(?m)^ENTITY:.*$", "ENTITY: " + entity_id, context)
            else:
                context += "\nENTITY: " + entity_id
            if report_context:
                report_context = {**report_context, "entity_owner": owner}
        bound_output.append(NumericCandidate(**{**candidate.__dict__, "value": value, "number_format": number_format, "unit": unit, "context": context,
            "source_section_path": tuple(binding["source_section_path"]), "entity_id": entity_id,
            "table_semantics": semantic, "report_context": report_context}))
    from .column_context import resolved_relative_year
    from dataclasses import replace
    for i, candidate in enumerate(bound_output):
        from .column_context import continued_statement_column
        continued_column=continued_statement_column(candidate)
        if continued_column and (not candidate.column_label or re.fullmatch(r"\d{1,3}",candidate.column_label.strip())):
            # A separator after the first data row can make its amount look
            # like a header. The validated continuation supplies the label.
            context=re.sub(r"(?m)^COLUMN:.*$", "COLUMN: "+continued_column, candidate.context)
            if not re.search(r"(?m)^COLUMN:",context):context+="\nCOLUMN: "+continued_column
            candidate=replace(candidate,column_label=continued_column,context=context)
            bound_output[i]=candidate
        from .table_semantics import bound_column_context
        context_year,_ = bound_column_context(candidate)
        if context_year:
            context=re.sub(r"(?m)^PERIOD:.*$", "PERIOD: "+context_year, candidate.context)
            if not re.search(r"(?m)^PERIOD:",context):context+="\nPERIOD: "+context_year
            candidate=replace(candidate,period=context_year,context=context,
                table_semantics={**(candidate.table_semantics or {}),'period_origin':'local_table_context'})
            bound_output[i]=candidate
        year = resolved_relative_year(candidate)
        if year:
            context = re.sub(r"(?m)^PERIOD:.*$", "PERIOD: " + year, candidate.context)
            bound_output[i] = replace(candidate, period=year, context=context)
    from .number_formats import reconcile_exact_repetitions
    from .unit_defaults import apply_default_unit
    output = reconcile_exact_repetitions([apply_default_unit(c) for c in bound_output])

    # Structured evidence participates only through one complete physical row
    # inventory that uniquely accounts for every typed role/period slot in the
    # question plan. Competing physical rows are ambiguity even when their
    # values agree: numeric equality is not proof of authority.
    plan = compile_question_plan(question)
    raw_slots = [
        (str(slot.column_role).strip().lower(), str(slot.year).strip())
        for slot in plan.operand_slots
        if str(slot.column_role).strip().lower() in {"actual", "plan"}
        and str(slot.year).strip()
    ]
    requested_slots = tuple(dict.fromkeys(raw_slots))
    duplicate_requested_slot = len(requested_slots) != len(raw_slots)

    def canonical_metric_tokens(value: str) -> frozenset[str]:
        normalized = normalize_text(value)
        # Source-language variants of the same financial metric head.
        normalized = re.sub(r"\blai\b", "loi nhuan", normalized)
        return frozenset(_metric_alignment_tokens(normalized))

    expected_metric = canonical_metric_tokens(
        plan.metric_text or _question_focus_text(question)
    )
    expected_metric_text = plan.metric_text or _question_focus_text(question)

    def metric_compatible(candidate: NumericCandidate) -> bool:
        if not metric_qualifiers_compatible(
            expected_metric_text, candidate.row_label
        ):
            return False
        observed = canonical_metric_tokens(candidate.row_label)
        return bool(
            expected_metric
            and observed
            and (observed.issubset(expected_metric) or expected_metric.issubset(observed))
        )

    def unit_family(value: str) -> str:
        normalized = normalize_text(value)
        if any(token in normalized for token in ("vnd", "dong", "trieu", "ty")):
            return "currency"
        if "%" in value or "phan tram" in normalized:
            return "percent"
        if any(token in normalized for token in ("share", "co phieu", "co phan")):
            return "shares"
        return normalized

    def plaintext_role(candidate: NumericCandidate) -> str:
        explicit = candidate.column_role.strip().lower()
        if explicit in {"actual", "plan"}:
            return explicit
        normalized = f" {normalize_text(candidate.column_label)} "
        if any(token in normalized for token in (" ke hoach ", " plan ", " kh ")):
            return "plan"
        if any(
            token in normalized
            for token in (" thuc hien ", " thuc te ", " actual ", " th ")
        ):
            return "actual"
        return ""

    # Collapse repeated observations of the same physical structured cell
    # before testing conflicts. A disagreement under one physical identity is
    # itself a conflict and therefore admits no structured evidence.
    physical_cells: dict[tuple[object, ...], NumericCandidate] = {}
    physical_conflict = False
    for candidate in structured_output:
        physical_key = (
            candidate.source_name,
            candidate.page,
            candidate.table_id,
            candidate.row_id,
            candidate.column_id,
        )
        current = physical_cells.get(physical_key)
        if current is None:
            physical_cells[physical_key] = candidate
        elif (
            current.decimal != candidate.decimal
            or current.column_role != candidate.column_role
            or current.period != candidate.period
            or current.source_bbox != candidate.source_bbox
        ):
            physical_conflict = True

    row_inventories: dict[tuple[object, ...], list[NumericCandidate]] = {}
    for candidate in physical_cells.values():
        if not metric_compatible(candidate):
            continue
        row_key = (
            candidate.source_name,
            candidate.page,
            candidate.table_id,
            candidate.row_id,
        )
        row_inventories.setdefault(row_key, []).append(candidate)

    requested_set = set(requested_slots)
    slot_claims: dict[tuple[str, str], set[tuple[object, ...]]] = {
        slot: set() for slot in requested_slots
    }
    row_slots: dict[
        tuple[object, ...], dict[tuple[str, str], list[NumericCandidate]]
    ] = {}
    for row_key, candidates in row_inventories.items():
        inventory: dict[tuple[str, str], list[NumericCandidate]] = {}
        for candidate in candidates:
            slot = (candidate.column_role.strip().lower(), candidate.period.strip())
            if slot not in requested_set:
                continue
            inventory.setdefault(slot, []).append(candidate)
            slot_claims[slot].add(row_key)
        row_slots[row_key] = inventory

    ambiguous_physical_claim = any(
        len(claiming_rows) > 1 for claiming_rows in slot_claims.values()
    )
    complete_rows: list[tuple[object, ...]] = []
    for row_key, inventory in row_slots.items():
        if requested_set and all(
            len(inventory.get(slot, ())) == 1 for slot in requested_slots
        ):
            selected = [inventory[slot][0] for slot in requested_slots]
            families = {
                unit_family(candidate.unit) for candidate in selected if candidate.unit
            }
            if len(families) <= 1:
                complete_rows.append(row_key)

    admitted_structured: list[NumericCandidate] = []
    if (
        structured_output
        and requested_slots
        and not duplicate_requested_slot
        and not physical_conflict
        and not ambiguous_physical_claim
        and len(complete_rows) == 1
    ):
        selected_inventory = row_slots[complete_rows[0]]
        selected_structured = [
            selected_inventory[slot][0] for slot in requested_slots
        ]
        plaintext_by_slot: dict[
            tuple[str, str], list[tuple[int, NumericCandidate]]
        ] = {slot: [] for slot in requested_slots}
        for index, candidate in enumerate(output):
            slot = (plaintext_role(candidate), candidate.period.strip())
            if slot in requested_set and metric_compatible(candidate):
                plaintext_by_slot[slot].append((index, candidate))

        plaintext_conflict = False
        equivalences: dict[int, NumericCandidate] = {}
        unmatched: list[NumericCandidate] = []
        used_plaintext: set[int] = set()
        for structured in selected_structured:
            slot = (
                structured.column_role.strip().lower(),
                structured.period.strip(),
            )
            compatible_plaintext = [
                (index, candidate)
                for index, candidate in plaintext_by_slot.get(slot, ())
                if candidate.source_name == structured.source_name
                and candidate.page == structured.page
                and (
                    not candidate.unit
                    or not structured.unit
                    or unit_family(candidate.unit) == unit_family(structured.unit)
                )
            ]
            if any(
                candidate.decimal != structured.decimal
                for _, candidate in compatible_plaintext
            ):
                plaintext_conflict = True
                break
            exact = [
                (index, candidate)
                for index, candidate in compatible_plaintext
                if candidate.decimal == structured.decimal
            ]
            if len(exact) > 1 or (exact and exact[0][0] in used_plaintext):
                plaintext_conflict = True
                break
            if not exact:
                unmatched.append(structured)
                continue
            index, plaintext = exact[0]
            used_plaintext.add(index)
            equivalences[index] = NumericCandidate(
                **{
                    **plaintext.__dict__,
                    "row_label": structured.row_label,
                    "column_label": structured.column_label,
                    "period": structured.period,
                    "table_id": structured.table_id,
                    "entity_id": structured.entity_id,
                    "source_span": structured.source_span,
                    "row_id": structured.row_id,
                    "column_id": structured.column_id,
                    "column_role": structured.column_role,
                    "source_bbox": structured.source_bbox,
                }
            )
        if not plaintext_conflict:
            for index, enriched in equivalences.items():
                output[index] = enriched
            admitted_structured.extend(unmatched)

    output.extend(apply_default_unit(c) for c in admitted_structured)

    unique: dict[tuple[object, ...], NumericCandidate] = {}
    for candidate in output:
        key = (
            (
                "structured",
                candidate.source_name,
                candidate.page,
                candidate.table_id,
                candidate.row_id,
                candidate.column_id,
                candidate.value,
            )
            if candidate.table_id and candidate.row_id and candidate.column_id
            else (
                candidate.source_name,
                candidate.page,
                candidate.line_number,
                candidate.char_start,
                candidate.char_end,
                candidate.value,
            )
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
        "mineru_markdown",
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
    focus_text = _question_focus_text(question)
    if not metric_qualifiers_compatible(focus_text, row):
        return 0.0
    expected = _metric_alignment_tokens(focus_text)
    observed = _metric_alignment_tokens(row)
    score = 0.0
    for term in sorted(expected):
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


def _quota_column_role(candidate: NumericCandidate) -> str:
    """Read only an explicit role from a candidate's recovered header path."""

    explicit = candidate.column_role.strip().lower()
    if explicit in {"actual", "plan"}:
        return explicit
    normalized = f" {normalize_text(candidate.column_label)} "
    # Comparison/ratio columns may contain both words. They are derived
    # disclosures, not either raw operand lane.
    if (
        " so voi " in normalized
        or "/" in candidate.column_label
        or "%" in candidate.column_label
    ):
        return ""
    if any(
        marker in normalized
        for marker in (
            " ke hoach ",
            " k ho ch ",
            " plan ",
            " kh ",
        )
    ):
        return "plan"
    if any(
        marker in normalized
        for marker in (
            " thuc hien ",
            " th c hi n ",
            " thuc te ",
            " actual ",
            " th ",
        )
    ):
        return "actual"
    return ""


def _quota_slot_period_match(
    candidate: NumericCandidate, requested_year: str
) -> bool:
    if not requested_year:
        return True
    observed = set(
        re.findall(
            r"\b(?:19|20)\d{2}\b",
            f"{candidate.period} {candidate.column_label}",
        )
    )
    return requested_year in observed


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
    if re.search(r"\blnst\b", normalized):
        normalized = f"{normalized} loi nhuan sau thue"
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


def metric_contrastive_qualifiers(text: str) -> dict[str, str]:
    """Return only metric qualifiers whose opposites change financial meaning.

    The result deliberately excludes generic topical words.  Missing qualifiers
    remain unknown, while an explicit contradiction (for example net versus
    gross revenue) can be rejected before fuzzy token overlap is applied.
    """

    normalized = normalize_text(text)
    padded = f" {normalized} "
    qualifiers: dict[str, str] = {}
    if " doanh thu " in padded:
        if " doanh thu thuan " in padded:
            qualifiers["revenue_basis"] = "net"
        elif (
            " doanh thu gop " in padded
            or " tong doanh thu " in padded
            or re.search(r"\bdoanh thu(?:\s+\w+){0,3}\s+ban hang\b", normalized)
        ):
            qualifiers["revenue_basis"] = "gross"

    profit_like = " loi nhuan " in padded or bool(
        re.search(r"\blnst\b", normalized)
    )
    after_tax = (
        " loi nhuan sau thue " in padded
        or " sau thue " in padded
        or bool(re.search(r"\blnst\b", normalized))
    )
    before_tax = " loi nhuan truoc thue " in padded or " truoc thue " in padded
    if after_tax:
        qualifiers["tax_basis"] = "after_tax"
    elif before_tax:
        qualifiers["tax_basis"] = "before_tax"

    parent_share = any(
        marker in padded
        for marker in (
            " co dong cua cong ty me ",
            " co dong cong ty me ",
            " thuoc ve co dong cua cong ty me ",
            " thuoc ve co dong cong ty me ",
        )
    )
    if after_tax or (profit_like and parent_share):
        qualifiers["parent_share"] = "parent" if parent_share else "total"
    return qualifiers


def _is_ratio_metric(text: str) -> bool:
    normalized = normalize_text(text)
    if re.search(r"\b(?:he so|ty le|ty suat|bien loi nhuan)\b", normalized):
        return True
    # Slash in a unit (dong/co phieu) is not a ratio of financial metrics.
    heads = r"\b(?:doanh thu|loi nhuan|tai san|von|no|chi phi|thu nhap)\b"
    parts = [normalize_text(part) for part in text.split("/")]
    return len(parts) == 2 and all(re.search(heads, part) for part in parts)


def metric_qualifiers_compatible(expected_text: str, observed_text: str) -> bool:
    """Return false only for an explicit contrastive metric contradiction."""

    # A disclosed ratio is not its numerator's monetary fact. This guard is
    # shared by candidate quota reservation and structured metric matching.
    if _is_ratio_metric(observed_text) and not _is_ratio_metric(expected_text):
        return False
    from .question_planner import _has_compound_financial_metric
    if _has_compound_financial_metric(normalize_text(observed_text)) and not _has_compound_financial_metric(normalize_text(expected_text)):
        return False
    expected = metric_contrastive_qualifiers(expected_text)
    observed = metric_contrastive_qualifiers(observed_text)
    if (
        observed.get("parent_share") == "parent"
        and expected.get("parent_share") != "parent"
    ):
        return False
    return all(
        observed.get(name) == value
        for name, value in expected.items()
        if name in observed
    )


def _candidate_quota_indices(
    question: str,
    candidates: Sequence[NumericCandidate],
    maximum: int,
    *,
    prefer_magnitude: bool = False,
    financial_identity_catalog_enabled: bool = False,
) -> list[int]:
    """Preserve metric and operand-role evidence before applying a hard quota."""

    def rank_key(index: int) -> tuple[object, ...]:
        candidate = candidates[index]
        final: Decimal | int = (
            abs(candidate.decimal) if prefer_magnitude else -candidate.line_number
        )
        page = candidate.page if candidate.page is not None else 10**9
        return (
            candidate.relevance,
            not candidate.is_year_like,
            final,
            normalize_text(candidate.source_name),
            -page,
            -candidate.char_start,
            normalize_text(candidate.table_id),
            normalize_text(candidate.row_id),
            normalize_text(candidate.column_id),
            candidate.value,
        )

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
    plan = compile_question_plan(question)
    explicit_column_roles = tuple(
        dict.fromkeys(
            role.strip().lower()
            for role in plan.column_roles
            if role.strip().lower() in {"actual", "plan"}
        )
    )
    slot_specs = [
        (slot.metric, slot.column_role.strip().lower(), slot.year)
        for slot in plan.operand_slots
        if slot.column_role.strip().lower() in {"actual", "plan"}
        and slot.year
    ]
    # A scalar lookup can carry one explicit plan/actual lane without a typed
    # operand slot. Preserve that lane under the same quota contract.
    if not slot_specs and len(explicit_column_roles) == 1:
        metric_text = plan.metric_text or _question_focus_text(question)
        slot_specs = [
            (metric_text, explicit_column_roles[0], year)
            for year in (plan.requested_years or ("",))
        ]
    slot_reserved: list[int] = []
    # Source-wide role bindings are attached after per-chunk pruning. Keep
    # metric/period-compatible unresolved observations available for that step;
    # this is candidate recall, never permission to label them actual or plan.
    # Explicit competing roles stay separate and cannot fill the wrong slot.
    if slot_specs:
        per_slot = max(
            1,
            min(4, maximum // max(1, len(slot_specs))),
        )
        for metric_text, role, year in slot_specs:
            slot_scores = {
                index: _quota_metric_score(metric_text, candidate)
                for index, candidate in enumerate(candidates)
                if _quota_column_role(candidate) in {role, ""}
                and _quota_slot_period_match(candidate, year)
            }
            best_slot = max(slot_scores.values(), default=0.0)
            if best_slot < 1.2:
                continue
            matches = [
                index
                for index in sorted(
                    slot_scores,
                    key=lambda item: (
                        slot_scores[item],
                        _quota_column_role(candidates[item]) == role,
                        *rank_key(item),
                    ),
                    reverse=True,
                )
                if slot_scores[index] >= best_slot - 0.30
            ]
            slot_reserved.extend(matches[:per_slot])
        slot_reserved = list(dict.fromkeys(slot_reserved))
    period_reserved: list[int] = []
    requested_years = tuple(dict.fromkeys(plan.requested_years))
    if not slot_specs and not explicit_column_roles and requested_years:
        per_period = max(1, min(4, maximum // max(1, len(requested_years))))
        metric_text = plan.metric_text or _question_focus_text(question)
        for year in requested_years:
            period_scores = {
                index: _quota_metric_score(metric_text, candidate)
                for index, candidate in enumerate(candidates)
                if _quota_slot_period_match(candidate, year)
            }
            best_period = max(period_scores.values(), default=0.0)
            if best_period < 1.2:
                continue
            matches = [
                index
                for index in sorted(
                    period_scores,
                    key=lambda item: (
                        period_scores[item],
                        *rank_key(item),
                    ),
                    reverse=True,
                )
                if period_scores[index] >= best_period - 0.30
            ]
            # Keep the strongest observation from every physical role lane
            # before filling remaining capacity. Roleless questions must not
            # silently evict either plan or actual evidence.
            strongest_by_role: list[int] = []
            seen_roles: set[str] = set()
            # Explicit competing lanes take precedence over an untyped
            # observation when the quota can retain only two cells. Keeping
            # both exposes the ambiguity; it does not infer either role.
            for required_role in ("actual", "plan"):
                strongest = next(
                    (
                        index
                        for index in matches
                        if _quota_column_role(candidates[index])
                        == required_role
                    ),
                    None,
                )
                if strongest is not None:
                    strongest_by_role.append(strongest)
                    seen_roles.add(required_role)
            for index in matches:
                role = _quota_column_role(candidates[index])
                if role in seen_roles:
                    continue
                seen_roles.add(role)
                strongest_by_role.append(index)
            ordered = list(dict.fromkeys([*strongest_by_role, *matches]))
            period_reserved.extend(ordered[:per_period])
        period_reserved = list(dict.fromkeys(period_reserved))
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

    selected = list(
        dict.fromkeys(
            [*slot_reserved, *period_reserved, *role_reserved, *reserved]
        )
    )[:maximum]
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
    if not metric_qualifiers_compatible(expected_text, row):
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
    try:
        return execute_decimal_operation(operation, values)
    except UnsupportedOperationError as exc:
        raise ProgramDataError(f"Unsupported operation: {operation}") from exc


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
        try:
            minimum, maximum = dag_operation_arity_bounds(operation)
        except ProgramDataError:
            raise ProgramDataError(
                f"Unsupported derived operation: {candidate.derived_operation}"
            ) from None
        if not minimum <= len(candidate.derived_operands) <= maximum:
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
    if is_sum_operation(operation):
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
    for term in sorted(expected):
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
    # Use one tokenizer for labels and physical spans. Strip only a paired
    # outer border; an unbordered trailing pipe denotes an empty final cell.
    return [text for text, _span in _markdown_cells_with_spans(line)]


_DESCRIPTOR_HEADER_ANCHORS = {
    "tt",
    "stt",
    "chi tieu",
    "noi dung",
    "item",
    "indicator",
    "ma so",
    "don vi",
    "don vi tinh",
}


def _aligned_table_header_cells(
    cells: Sequence[str], row_width: int
) -> list[str]:
    """Align a MinerU header row without inventing a financial column.

    MinerU may omit the trailing blank of a merged header. A descriptor in
    the first physical column proves that the missing cell is on the right.
    Other short rows remain right-aligned because their missing cell is
    commonly the blank row-code corner on the left.
    """

    aligned = list(cells)
    if len(aligned) >= row_width:
        return aligned[-row_width:]
    missing = row_width - len(aligned)
    first_nonempty = next(
        (normalize_text(cell) for cell in aligned if cell.strip()),
        "",
    )
    if first_nonempty in _DESCRIPTOR_HEADER_ANCHORS or first_nonempty.startswith(
        ("don vi ", "don vi tinh ", "dvt ")
    ):
        return aligned + [""] * missing
    return [""] * missing + aligned


def _inherited_header_cell(cells: Sequence[str], column_index: int) -> str:
    """Resolve an explicit or left-inherited merged Markdown header cell."""

    if 0 <= column_index < len(cells) and cells[column_index].strip():
        return cells[column_index].strip()
    for previous in range(column_index - 1, -1, -1):
        if cells[previous].strip():
            return cells[previous].strip()
    return ""


def _is_markdown_separator(line: str) -> bool:
    cells = _markdown_cells(line)
    return bool(cells) and all(
        not cell or bool(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")))
        for cell in cells
    )


def _numbered_table_body_row(line: str) -> bool:
    cells = _markdown_cells(line)
    return bool(cells and re.match(r"^\d+\s*[-.)]\s*[^\W\d_]", cells[0], re.UNICODE))


def _is_table_header_or_separator(
    lines: Sequence[str], line_index: int
) -> bool:
    line = lines[line_index].strip()
    if _is_markdown_separator(line):
        return True
    if "|" not in line:
        return False
    if _numbered_table_body_row(line):
        return False
    normalized_cells = {normalize_text(cell) for cell in _markdown_cells(line)}
    if normalized_cells & {"chi tieu", "cac chi tieu", "column name"}:
        return True
    # A value row before a repeated header/separator remains a body row.
    # Require a financial descriptor and at least two scalar value cells;
    # maturity-band headers ("trên 1.000...") are not scalar observations.
    from .unit_defaults import monetary_label
    cells = _markdown_cells(line)
    scalar_values = 0
    for cell in cells:
        value_text = cell.rsplit(":", 1)[-1].strip()
        if re.fullmatch(r"[()+\-−]?\d[\d., ]*[%)]?", value_text) and not re.fullmatch(r"(?:19|20)\d{2}", value_text):
            scalar_values += 1
    # Percentage and ratio rows are observations too. A separator following
    # them is often the beginning of the next table in MinerU Markdown.
    # Require a non-header descriptor in the first cell and numeric values
    # in at least two later cells; year and maturity headers stay headers.
    first = normalize_text(cells[0]) if cells else ""
    descriptor = bool(re.search(r"[a-z]", first)) and first not in {
        "chi tieu", "chi so", "noi dung", "item", "indicator", "ma so",
        "don vi", "don vi tinh",
    } and not re.search(r"\b(?:19|20)\d{2}\b", first)
    # MinerU may repeat a table header immediately after a numbered body row.
    # A row code plus a descriptor and two scalar cells is still a body row.
    numbered_descriptor = (
        len(cells) >= 4
        and bool(re.fullmatch(r"\d{1,3}", cells[0].strip()))
        and bool(re.search(r"[a-z]", normalize_text(cells[1])))
        and not re.search(r"\b(?:19|20)\d{2}\b", normalize_text(cells[1]))
    )
    if scalar_values >= 2 and (descriptor or numbered_descriptor):
        return False
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

    # Consecutive MinerU tables can omit the blank line and separator before
    # a repeated "Chỉ số | year | year" header. Bind each following row to
    # the nearest explicit descriptor/year header, not an older table's values.
    for index in range(line_index-1,block_start-1,-1):
        cells=_markdown_cells(lines[index])
        if (cells and normalize_text(cells[0]) in {"chi so","chi tieu","cac chi tieu","item","indicator"}
                and sum(bool(re.search(r"\b(?:19|20)\d{2}\b",normalize_text(cell))) for cell in cells[1:])>=2):
            return [lines[index].strip()]
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
            if not cells or _financial_value_cell_indices(cells) or _numbered_table_body_row(candidate):
                break
            headers.append(candidate)
        if headers:
            return list(reversed(headers))
        # OCR may place a separator after numbered body rows. The first
        # explicit descriptor/year header still owns those rows.
        first = lines[block_start].strip()
        first_cells = _markdown_cells(first)
        if first_cells and normalize_text(first_cells[0]) in {"chi tieu", "noi dung", "item", "indicator"}:
            return [first]

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
                "chi so",
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
    # Accent folding maps both Vietnamese "cộng" and the proper noun
    # component in "công ty" to "cong". Remove the entity-kind phrase
    # before classifying arithmetic total markers.
    total_surface = re.sub(r"\bcong\s+ty\b", " ", normalized)
    if re.search(r"(?:^| )(?:tong|tong cong|cong)(?: |$)", total_surface):
        return "total"
    if re.match(r"^(?:I|II|III|IV|V|VI|VII|VIII|IX|X)\.[ \t]+", row_label.strip()):
        return "subtotal"
    if re.match(r"^[A-ZĐ]\.[ \t]+", row_label.strip()):
        return "total"
    if any(term in normalized for term in ("tieu tong", "cong nhom")):
        return "subtotal"
    return "detail"


def _column_period(column_label: str) -> str:
    from .column_context import normalize_period_text
    normalized = normalize_period_text(column_label)
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
    # In a two-cell key/value row, words spelling the amount are not a label.
    if (len(row_cells) == 2 and cell_index == 1
            and re.search(r"[^\W\d_]", row_cells[0], re.UNICODE)
            and not re.search(r"\d", row_cells[0])
            and not _inline_period_column((row_cells[0],), 0)):
        return row_cells[0]
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
    question: str = "",
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
            header_cells = _aligned_table_header_cells(
                _markdown_cells(header_line),
                len(row_cells),
            )
            for header_cell in header_cells:
                if _column_period(header_cell) and header_cell not in period_columns:
                    period_columns.append(header_cell)
            part = _inherited_header_cell(header_cells, cell_index)
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
        header_role = _resolved_table_header_role(column_parts)
        if header_role:
            parts.append(f"COLUMN_ROLE: {header_role}")
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

    from .report_context import undated_monetary_field
    report_field = undated_monetary_field(raw_line, start, end)
    if report_field is not None:
        provenance = {"text": raw_line, "source_name": source_name, "page": page,
                      "line_number": source_line_start + line_index}
        return "\n".join(("ROW: " + report_field["row_label"],
                          "REPORT_FIELD_SOURCE: " + json.dumps(provenance, ensure_ascii=False, sort_keys=True),
                          "CELL: " + marked))
    from .source_assertions import dated_amount_assertion
    period_context = []
    for previous_index in range(max(0, line_index - 3), line_index):
        previous = lines[previous_index]
        if re.fullmatch(r"nam (?:19|20)\d{2}", normalize_text(previous)):
            from .physical_coordinates import line_digest
            period_context.append({"text": previous, "source_name": source_name, "page": page,
                "line_number": source_line_start + previous_index, "sha256": line_digest(previous)})
    assertion = dated_amount_assertion(raw_line, start, end, period_context)
    if assertion is not None:
        # A sentence has a temporal qualifier, not a manufactured table header.
        # Keep its original text so verification can reconstruct every binding.
        provenance = {"text": raw_line, "source_name": source_name, "page": page,
                      "line_number": source_line_start + line_index}
        if period_context:
            provenance["period_context"] = period_context
        return "\n".join((
            "ROW: " + assertion["row_label"],
            "COLUMN: " + assertion["column_label"],
            "PERIOD: " + assertion["period"],
            "COLUMN_ROLE: " + assertion["column_role"],
            "SOURCE_ASSERTION: " + json.dumps(provenance, ensure_ascii=False, sort_keys=True),
            "CELL: " + marked,
        ))

    plaintext_context = _plaintext_financial_row_context(
        lines,
        line_index,
        start,
        end,
        marked,
        question=question,
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
    """Tokenize cells without losing explicit blank columns or provenance.

    MinerU's unbordered rows serialize a blank final field as a trailing pipe.
    In bordered rows remove exactly one outer pipe at each edge, preserving
    additional pipes that encode blank cells (including merged-header corners).
    """
    separators = [index for index, value in enumerate(line) if value == "|"]
    if not separators:
        return []
    bordered = not line[:separators[0]].strip()
    boundaries = list(separators) if bordered else [-1, *separators]
    if not (bordered and not line[separators[-1] + 1:].strip()):
        boundaries.append(len(line))
    output: list[tuple[str, tuple[int, int]]] = []
    for left, right in zip(boundaries, boundaries[1:]):
        raw_start = left + 1
        raw = line[raw_start:right]
        text = raw.strip()
        if text:
            start = raw_start + len(raw) - len(raw.lstrip())
            end = raw_start + len(raw.rstrip())
        else:
            start = end = raw_start
        output.append((text, (start, end)))
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
            cells = _aligned_table_header_cells(raw_cells, row_width)
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
        and all(len(cells) <= row_width for cells in header_cells)
        and all(
            len(_aligned_table_header_cells(cells, row_width)) == row_width
            for cells in header_cells
        )
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


_NUMBERED_SECTION_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?!19\d{2}\b|20\d{2}\b)"
    r"\d+(?:\.\d+)*\.?\s+\S"
)


def _scope_heading_text(raw_line: str) -> str:
    text = raw_line.strip().lstrip("#•*- ").strip()
    return re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", text).strip()


_SCOPE_NARRATIVE_MARKERS = (
    " trong nam ",
    " trong ky ",
    " da thuc hien ",
    " dang thuc hien ",
    " se thuc hien ",
    " theo bao cao ",
    " cho biet ",
    " khong co ",
    " ghi nhan ",
    " tang so voi ",
    " giam so voi ",
)


def _scope_line_is_narrative(text: str) -> bool:
    normalized = f" {normalize_text(text)} "
    return (
        len(normalized.split()) > 24
        or "," in text
        or ";" in text
        or any(
            marker in normalized for marker in _SCOPE_NARRATIVE_MARKERS
        )
        or bool(re.search(r"[.!?]\s*$", text.rstrip().rstrip(":")))
    )


def _statement_scope_heading(raw_line: str) -> str:
    text = _scope_heading_text(raw_line)
    if (
        not text
        or "|" in text
        or len(text) > 140
        or _scope_line_is_narrative(text)
    ):
        return ""
    normalized = normalize_text(text).strip(" :")
    statement_prefixes = (
        "bao cao",
        "bang can doi",
        "ket qua kinh doanh",
        "bao cao ket qua",
        "tinh hinh tai chinh",
    )
    if (
        "hop nhat" in normalized
        and normalized.startswith(statement_prefixes)
    ):
        return "consolidated"
    if (
        normalized
        in {"cong ty me", "bao cao rieng", "bao cao tai chinh rieng"}
        or normalized.startswith(
            ("bao cao rieng ", "bao cao tai chinh rieng ")
        )
        or (
            normalized.startswith(statement_prefixes)
            and "rieng cong ty me" in normalized
        )
    ):
        return "parent_company"
    return ""


def _short_entity_scope_heading(raw_line: str) -> str:
    text = _scope_heading_text(raw_line)
    if (
        not text
        or "|" in text
        or len(text) > 120
        or _scope_line_is_narrative(text)
    ):
        return ""
    normalized = normalize_text(text).strip(" :")
    legal_prefixes = (
        "cong ty co phan ",
        "cong ty cp ",
        "cong ty tnhh ",
        "cong ty trach nhiem huu han ",
        "tong cong ty ",
        "tap doan ",
    )
    subtable_prefixes = (
        "cong ty con",
        "cong ty lien ket",
        "cong ty lien doanh",
        "don vi lien ket",
        "don vi lien doanh",
        "chi nhanh",
    )
    # A legal-name heading may end in a colon. A colon followed by prose is a
    # narrative sentence, except for the common ``subsidiary: legal name`` form.
    if ":" in text and not text.rstrip().endswith(":"):
        prefix, suffix = (part.strip() for part in text.split(":", 1))
        prefix_normalized = normalize_text(prefix)
        suffix_normalized = normalize_text(suffix).strip(" :")
        if (
            prefix_normalized.startswith(subtable_prefixes)
            and suffix_normalized
            and not _scope_line_is_narrative(suffix)
        ):
            return suffix_normalized
        return ""
    if normalized.startswith(legal_prefixes):
        return normalized
    for prefix in subtable_prefixes:
        marker = f"{prefix} "
        if not normalized.startswith(marker):
            continue
        remainder = normalized[len(marker) :].strip()
        if remainder.startswith(legal_prefixes):
            return remainder
    return ""


def _scope_company_like_line(raw_line: str) -> bool:
    normalized = normalize_text(_scope_heading_text(raw_line)).strip(" :")
    return normalized.startswith(
        ("cong ty ", "tong cong ty ", "tap doan ", "chi nhanh ", "don vi ")
    )


def _table_scope(lines: Sequence[str], line_index: int) -> str:
    """Resolve only a nearby heading inside the latest numbered section."""

    start, _ = _table_block_bounds(lines, line_index)
    window_start = max(0, start - 24)
    for index in range(start - 1, window_start - 1, -1):
        if _NUMBERED_SECTION_RE.match(lines[index]):
            window_start = index
            break
    entity_blocked = False
    for raw_line in reversed(lines[window_start:start]):
        statement = _statement_scope_heading(raw_line)
        if statement:
            return statement
        if not entity_blocked:
            entity = _short_entity_scope_heading(raw_line)
            if entity:
                return f"entity:{entity}"
        if _scope_company_like_line(raw_line):
            entity_blocked = True
    return ""


def _plaintext_assertion_identity(
    question: str,
    line: str,
    start: int,
    end: int,
) -> tuple[str, str] | None:
    """Prove metric and period identity for one flat sentence assertion.

    The assertion must contain one non-metadata numeric value and an explicit
    predicate.  Dates and years are metadata, not competing values.  A period
    is bound only when the single year requested by the question is stated in
    the same source sentence.
    """

    if not question or "|" in line:
        return None
    values: list[re.Match[str]] = []
    for token in _NUMBER_RE.finditer(line):
        if (token.start() > 0 and line[token.start() - 1] == "/") or (
            token.end() < len(line) and line[token.end()] == "/"
        ):
            continue
        parsed = parse_decimal(token.group(0))
        suffix = normalize_text(line[token.end() : token.end() + 20])
        explicit_date_component = (
            parsed is not None
            and parsed == parsed.to_integral_value()
            and (
                (Decimal(1) <= parsed <= Decimal(31) and suffix.startswith("thang"))
                or (
                    Decimal(1) <= parsed <= Decimal(12)
                    and suffix.startswith("nam")
                )
            )
        )
        if (
            parsed is None
            or explicit_date_component
            or _is_answer_metadata_number(line, token, parsed)
        ):
            continue
        values.append(token)
    if (
        len(values) != 1
        or values[0].start() != start
        or values[0].end() != end
    ):
        return None

    assertion = re.search(
        r"(?:^|[,;:])\s*(?P<label>[^,;:]{2,200}?)\s+"
        r"(?:là|la|is|was|were|đạt|dat|có\s+giá\s+trị|"
        r"co\s+gia\s+tri)\s*$",
        line[:start],
        flags=re.IGNORECASE,
    )
    if assertion is None:
        return None
    label = assertion.group("label").strip(" .:-")
    expected = _metric_alignment_tokens(compile_question_plan(question).metric_text)
    observed = _metric_alignment_tokens(label)
    if (
        not label
        or not expected
        or _fuzzy_content_overlap(expected, observed) < 0.90
    ):
        return None

    plan = compile_question_plan(question)
    requested = tuple(dict.fromkeys(plan.requested_years))
    source_years = set(
        re.findall(r"\b(?:19|20)\d{2}\b", normalize_text(line))
    )
    period = (
        requested[0]
        if len(requested) == 1 and requested[0] in source_years
        else ""
    )
    return label, period


def _plaintext_financial_row_context(
    lines: Sequence[str],
    line_index: int,
    start: int,
    end: int,
    marked: str,
    *,
    question: str = "",
    source_name: str,
    page: int | None,
    source_line_start: int,
    extraction_method: str,
) -> str:
    """Recover source-proven row/period metadata from flattened Markdown."""

    line = lines[line_index]
    assertion_identity = _plaintext_assertion_identity(
        question, line, start, end
    )
    if assertion_identity is not None:
        label, period = assertion_identity
        section, scope = _nearby_plaintext_table_provenance(lines, line_index)
        parts = [f"ROW: {label}", f"ROW_TYPE: {_row_type(label)}"]
        if period:
            parts.append(f"PERIOD: {period}")
        if section:
            parts.append(f"SECTION: {section}")
        if scope:
            parts.append(f"SCOPE: {scope}")
        parts.append(f"CELL: {marked}")
        return "\n".join(parts)

    matches: list[tuple[re.Match[str], Decimal]] = []
    for token in _NUMBER_RE.finditer(line):
        parsed = parse_decimal(token.group(0))
        if parsed is not None and (
            abs(parsed) >= Decimal(1000) or bool(token.group("pct"))
        ):
            matches.append((token, parsed))
    # Sentence predicates belong around the value, not anywhere in a row
    # label: accent folding would otherwise confuse "đất" with "đạt".
    if matches:
        first = matches[0][0]
        before_value, after_value = line[:first.start()], line[first.end():]
        if (re.search(r"\b(?:là|la|đạt)\s*$", before_value, re.IGNORECASE)
                or re.search(r"\b(?:la|bang|dat|chiem|so voi|so ke hoach)\b", normalize_text(after_value))):
            return ""
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
    has_plan = any(
        token in normalized for token in ("ke hoach", " plan")
    ) or bool(re.search(r"(?:^|\W)kh(?:\W|$)", normalized))
    has_actual = any(
        token in normalized for token in ("thuc hien", "thuc te", "actual")
    ) or bool(re.search(r"(?:^|\W)th(?:\W|$)", normalized))
    if "%" in label or "ratio" in normalized or " so voi " in f" {normalized} ":
        if has_plan:
            return "ACTUAL_TO_PLAN_RATIO"
        if has_actual:
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
    if any(token in normalized for token in ("du bao", "du kien", "uoc tinh", "forecast")):
        return "FORECAST"
    if has_plan and has_actual:
        return ""
    if has_plan:
        return "PLAN"
    if has_actual:
        return "ACTUAL"
    return ""


def _resolved_table_header_role(column_parts: Sequence[str]) -> str:
    """Resolve only role vocabulary physically present in one header path."""

    path = tuple(part for part in column_parts if part.strip())
    if not path:
        return ""
    roles = {
        role
        for surface in (*path, " / ".join(path))
        if (role := _flattened_role_key(surface))
    }
    for specialization in (
        "ACTUAL_TO_PLAN_RATIO",
        "ACTUAL_TO_PRIOR_RATIO",
        "REVISED_PLAN",
        "FORECAST",
    ):
        if specialization in roles:
            return specialization.lower()
    ordinary = roles & {"ACTUAL", "PLAN"}
    return next(iter(ordinary)).lower() if len(ordinary) == 1 else ""


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
        r"thuc hien|thuc te|actual|ke hoach|plan|du bao|du kien|uoc tinh|forecast"
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

    window_start = max(0, line_index - 12)
    for previous in range(line_index - 1, window_start - 1, -1):
        if "|" in lines[previous]:
            window_start = previous + 1
            break
    raw_window = [line.strip() for line in lines[window_start:line_index] if line.strip()]
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
        if (len(explicit_labels) == expected_count
                and not re.search(r"\b(?:thuc hien|thuc te|ke hoach|actual|plan|du bao|du kien|forecast)\b", normalized_header)):
            return explicit_labels, raw_window

    def role_tokens(raw: str) -> list[str]:
        normalized = normalize_text(raw)
        pattern = re.compile(
            r"ke hoach dieu chinh|ke hoach sua doi|revised plan|adjusted plan|"
            r"thuc hien|thuc te|actual|ke hoach|plan|du bao|du kien|uoc tinh|forecast"
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
        if "|" in header:
            break
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


def _explicit_currency_units(text: str) -> set[str]:
    """Return currency units stated by text without guessing from magnitude."""

    normalized = normalize_text(
        text.replace("Đ", "D").replace("đ", "d")
    )
    scaled: set[str] = set()
    patterns = (
        (
            "tỷ đồng",
            r"\bty\s*(?:dong|d|vnd)\b",
            r"\bty\b",
        ),
        (
            "triệu đồng",
            r"\b(?:trieu|tr)\s*(?:dong|d|vnd)\b",
            r"\b(?:trieu|tr)\b",
        ),
        (
            "nghìn đồng",
            r"\b(?:nghin|ngan|ngh)\s*(?:dong|d|vnd)\b",
            r"\b(?:nghin|ngan|ngh)\b",
        ),
    )
    # Parentheses around arbitrary prose/legal names do not declare a unit.
    # Only a short denomination or an explicit unit declaration licenses a
    # bare scale token; "ty" inside "cong ty" is never a denomination.
    bare_contexts = [normalized] if len(normalized.split()) <= 2 else []
    bare_contexts.extend(
        normalize_text(part) for part in re.findall(r"\(([^()]*)\)", text)
        if re.fullmatch(r"(?:ty|trieu|tr|nghin|ngan|ngh|dong|vnd|usd)", normalize_text(part)))
    bare_contexts.extend(re.findall(
        r"\b(?:don v(?:i)?(?: tinh)?|dvt|units?)\s*:?\s*((?:ty|trieu|tr|nghin|ngan|ngh))\b",
        normalized))
    unit_context = any(
        re.fullmatch(r"(?:ty|trieu|tr|nghin|ngan|ngh|dong|vnd|usd)", part)
        for part in bare_contexts
    ) or bool(re.search(r"\b(?:don v(?:i)?(?: tinh)?|dvt|units?)\b", normalized))
    for unit, paired, bare in patterns:
        explicit_pair = bool(re.search(paired, normalized))
        explicit_bare = bool(
            any(re.fullmatch(bare, part.strip()) for part in bare_contexts)
        )
        if explicit_pair or explicit_bare:
            scaled.add(unit)
    output: set[str] = set(scaled)
    if not scaled and (
        re.search(r"\bvnd\b", normalized)
        or (
            re.search(r"\bdong\b", normalized)
            and (
                unit_context
                or bool(re.search(r"\d\s*dong\b", normalized))
            )
        )
    ):
        output.add("VND")
    if re.search(r"\b(?:usd|dollar)\b", normalized):
        output.add("USD")
    return output


def _unique_currency_unit(
    texts: Sequence[str],
) -> tuple[str, bool]:
    units: set[str] = set()
    for text in texts:
        units.update(_explicit_currency_units(text))
    if len(units) == 1:
        return next(iter(units)), False
    return "", len(units) > 1


def _table_column_currency_unit(
    lines: Sequence[str],
    line_index: int,
    cell_index: int,
    row_width: int,
) -> tuple[str, bool]:
    path: list[str] = []
    for header_line in _table_header_lines(lines, line_index):
        cells = _aligned_table_header_cells(
            _markdown_cells(header_line), row_width
        )
        part = _inherited_header_cell(cells, cell_index)
        if part:
            path.append(part)
    return _unique_currency_unit(path)


def _table_caption_currency_unit(
    lines: Sequence[str], line_index: int
) -> tuple[str, bool]:
    start, _ = _table_block_bounds(lines, line_index)
    window_start = max(0, start - 8)
    for index in range(start - 1, window_start - 1, -1):
        if "|" in lines[index]:
            window_start = index + 1
            break
    window = lines[window_start:start]
    units: set[str] = set()
    for offset, raw in enumerate(window):
        text = raw.strip()
        if not text or text.startswith(">"):
            continue
        normalized = normalize_text(
            text.replace("Đ", "D").replace("đ", "d")
        )
        declared = any(
            marker in normalized
            for marker in ("don vi", "don v tinh", "dvt", "unit")
        )
        caption = normalized.startswith(("bang ", "bieu ", "table "))
        current = _explicit_currency_units(text)
        if declared or caption:
            units.update(current)
        if declared and not current:
            for following in window[offset + 1 : offset + 3]:
                if not following.strip() or "|" in following:
                    break
                units.update(_explicit_currency_units(following))
        if (
            current
            and len(normalized.split()) <= 3
            and offset >= len(window) - 2
        ):
            units.update(current)
    if len(units) == 1:
        return next(iter(units)), False
    return "", len(units) > 1


def _table_candidate_unit(
    lines: Sequence[str],
    line_index: int,
    char_start: int,
    *,
    is_percent: bool,
) -> str:
    if is_percent:
        return "%"
    raw = lines[line_index]
    number = _NUMBER_RE.match(raw, char_start)
    local = _number_local_unit(raw, number.end()) if number else ""
    if local:
        return local
    cells = _markdown_cells(raw)
    cell_index = max(
        0,
        raw[:char_start].count("|")
        - (1 if raw.lstrip().startswith("|") else 0),
    )
    if 0 <= cell_index < len(cells):
        inline = _infer_unit(cells[cell_index], False)
        if inline:
            return inline
    row_unit = _infer_unit(raw, False)
    if row_unit:
        return row_unit
    column, conflict = _table_column_currency_unit(
        lines, line_index, cell_index, len(cells)
    )
    if conflict:
        return ""
    if column:
        return column
    caption, conflict = _table_caption_currency_unit(lines, line_index)
    return "" if conflict else caption


def _number_local_unit(line: str, number_end: int) -> str:
    """Prefer the unit attached to this number over row/page currency context.

    Keep quantities such as floor area, headcount and shares in their own
    dimensions even when nearby prose or a financial table declares currency.
    """
    if re.match(r"^\s*[đĐ₫](?!\w)", line[number_end:]):
        return "VND"
    suffix = normalize_text(line[number_end:].replace("²", "2").replace("³", "3"))
    units = (
        (r"m\s*2", "m²"), (r"m\s*3", "m³"), (r"ha", "ha"),
        (r"kg", "kg"), (r"tan", "tấn"), (r"nguoi", "người"),
        (r"co phieu|co phan|shares?", "shares"),
    )
    for pattern, unit in units:
        if re.match(r"^\s*(?:" + pattern + r")\b", suffix):
            return unit
    currency = re.match(
        r"^\s*((?:(?:ty|trieu|nghin|ngan)\s*(?:dong|vnd))|vnd|dong|usd)\b",
        suffix)
    if currency:
        unit = _infer_unit(currency.group(1), False)
        if "menh gia" in normalize_text(line) or re.search(
            r"(?:dong|vnd)\s*/\s*(?:co phieu|co phan|shares?)\b",
            "/".join(normalize_text(part) for part in line[number_end:].split("/")),
        ):
            return f"{unit}/share"
        return unit
    return ""


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
    currency_units = _explicit_currency_units(line)
    if "menh gia" in normalized:
        if len(currency_units) == 1:
            unit = next(iter(currency_units))
            return "VND/share" if unit == "VND" else f"{unit}/share"
        return ""
    if len(currency_units) == 1:
        return next(iter(currency_units))
    if len(currency_units) > 1:
        return ""
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



