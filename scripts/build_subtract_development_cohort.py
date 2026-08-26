#!/usr/bin/env python3
"""Build a provenance-locked, development-only temporal SUBTRACT cohort.

The builder consumes only a locked development page manifest and its reviewed
MinerU Markdown.  It emits at most one non-zero temporal subtraction per
company group.  Questions state the operand order explicitly; no example is
retained unless the shared Direction Contract proves the gold order and the
reversed pair as a valid hard negative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrm_finance_qa.direction_contract import (  # noqa: E402
    assess_program_direction,
    reversed_pair_is_proven_negative,
)
from hrm_finance_qa.evidence_schema import build_evidence_graph  # noqa: E402
from hrm_finance_qa.ontology import normalize_text  # noqa: E402
from hrm_finance_qa.program_data import (  # noqa: E402
    NumericCandidate,
    ProgramExample,
    ReasoningProgram,
    execute_program,
    extract_numeric_candidates,
    program_example_to_dict,
)
from hrm_finance_qa.question_planner import (  # noqa: E402
    DirectionIntent,
    compile_question_plan,
)
from hrm_finance_qa.v12b2_supervision import (  # noqa: E402
    supervision_integrity_issues,
)


PAGE_RE = re.compile(r"^## Page (\d+)\s*$")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
PROTECTED_MARKERS = ("blind", "holdout", "final_eval", "test.jsonl")
MOJIBAKE_MARKERS = ("Ãƒ", "Ã‚", "Ã„", "Ã¡Âº", "Ã¡Â»", "Ä‘", "áº", "á»")
FINANCIAL_ROW_TERMS = (
    "tai san",
    "no phai tra",
    "von chu so huu",
    "doanh thu",
    "loi nhuan",
    "chi phi",
    "hang ton kho",
    "phai thu",
    "phai tra",
    "tien va",
    "luu chuyen tien",
    "vay va no",
)
RATIO_ROW_TERMS = (
    "he so",
    "ty le",
    "ty suat",
    "bien loi",
    "vong quay",
    "roe",
    "roa",
    "ros",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_pages(path: Path) -> dict[int, str]:
    pages: dict[int, list[str]] = {}
    current: int | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        match = PAGE_RE.match(line.strip())
        if match:
            current = int(match.group(1))
            pages.setdefault(current, [])
        elif current is not None:
            pages[current].append(line)
    return {page: "\n".join(lines).strip() for page, lines in pages.items()}


def _mojibake_score(value: str) -> int:
    return sum(value.count(marker) for marker in MOJIBAKE_MARKERS)


def _repair_text(value: str) -> str:
    """Repair display text only; raw candidate spans remain unchanged."""

    repaired = value
    for _ in range(2):
        best = repaired
        for encoding in ("cp1252", "latin1"):
            try:
                candidate = repaired.encode(encoding).decode("utf-8")
            except UnicodeError:
                continue
            if _mojibake_score(candidate) < _mojibake_score(best):
                best = candidate
        if best == repaired:
            break
        repaired = best
    return re.sub(r"\s+", " ", repaired).strip()


def _period(candidate: NumericCandidate) -> str:
    return (candidate.period or candidate.column_label).strip()


def _period_order(value: str) -> tuple[int, int] | None:
    normalized = normalize_text(value)
    years = set(YEAR_RE.findall(normalized))
    if len(years) == 1:
        return (2, int(next(iter(years))))
    if any(term in normalized for term in ("cuoi ky", "cuoi nam", "ky nay", "nam nay")):
        return (1, 1)
    if any(term in normalized for term in ("dau ky", "dau nam", "ky truoc", "nam truoc")):
        return (1, 0)
    return None


def _display_period(candidate: NumericCandidate) -> str:
    """Return a planner-stable label without changing source provenance."""

    raw = _period(candidate)
    years = set(YEAR_RE.findall(normalize_text(raw)))
    if len(years) == 1:
        return next(iter(years))
    normalized = normalize_text(raw)
    if any(term in normalized for term in ("cuoi ky", "cuoi nam", "ky nay", "nam nay")):
        return "năm nay"
    if any(term in normalized for term in ("dau ky", "dau nam", "ky truoc", "nam truoc")):
        return "năm trước"
    return _repair_text(raw)


def _cell_value_follows_row_label(candidate: NumericCandidate) -> bool:
    """Reject horizontally merged rows whose value belongs to a prior table.

    Native PDF extraction can concatenate two side-by-side tables into one
    Markdown row.  In those failures the numeric cell appears before the row
    label assigned by the table parser.  A genuine row may have STT/note cells,
    but its value must still occur after its own row label.
    """

    cell_line = next(
        (
            line.removeprefix("CELL:").strip()
            for line in candidate.context.splitlines()
            if line.startswith("CELL:")
        ),
        "",
    )
    if not cell_line or "|" not in cell_line:
        return True
    fields = [field.strip() for field in cell_line.strip().strip("|").split("|")]
    value_positions = [index for index, field in enumerate(fields) if "<VALUE>" in field]
    row = normalize_text(candidate.row_label)
    row_positions = [
        index
        for index, field in enumerate(fields)
        if row and row in normalize_text(re.sub(r"</?VALUE>", "", field))
    ]
    if len(value_positions) != 1 or not row_positions:
        return True
    return value_positions[0] > min(row_positions)


def _ratio_value_scale_is_plausible(candidate: NumericCandidate) -> bool:
    """Reject obvious amount/ratio cross-table assignments."""

    row = normalize_text(candidate.row_label)
    if not any(term in row for term in RATIO_ROW_TERMS):
        return True
    # Disclosed ratios may legitimately exceed 100%, but multi-million or
    # billion values cannot be coefficients, percentages, or turnover counts.
    return abs(candidate.decimal) < 100_000


def _is_shifted_note_reference(candidate: NumericCandidate) -> bool:
    """Detect a disclosure-note code shifted into the first period column."""

    raw = candidate.raw.strip().strip("()")
    if not re.fullmatch(r"\d{1,2}\.\d{1,2}", raw) or abs(candidate.decimal) >= 100:
        return False
    cell_line = next(
        (
            line.removeprefix("CELL:").strip()
            for line in candidate.context.splitlines()
            if line.startswith("CELL:")
        ),
        "",
    )
    if not cell_line or "<VALUE>" not in cell_line:
        return False
    suffix = cell_line.split("</VALUE>", 1)[-1]
    following_financial_values = 0
    for match in re.finditer(r"[()+-]?\d[\d.,]*%?", suffix):
        token = match.group(0).strip().strip("()")
        if token.endswith("%"):
            continue
        compact = re.sub(r"\D", "", token)
        if len(compact) >= 4:
            following_financial_values += 1
    return following_financial_values >= 2


def _column_unit_contract_is_plausible(candidate: NumericCandidate) -> bool:
    """Require ratio-labelled columns to contain ratio-like cells."""

    column = normalize_text(candidate.column_label)
    if any(term in column for term in ("ty trong", "ty le", "%")):
        return candidate.is_percent or normalize_text(candidate.unit) in {
            "%",
            "x",
            "lan",
            "vong",
        }
    return True


def _complete_table_candidate(candidate: NumericCandidate) -> bool:
    row = candidate.row_label.strip()
    first_letter = next((char for char in row if char.isalpha()), "")
    return bool(
        row
        and candidate.column_label
        and candidate.table_id
        and candidate.entity_id
        and candidate.subtable_id
        and "HEADER:" in candidate.context
        and "CELL:" in candidate.context
        and _period_order(_period(candidate)) is not None
        and not candidate.is_percent
        and not candidate.is_year_like
        and first_letter
        and first_letter.isupper()
        and len(normalize_text(row).split()) <= 14
        and _cell_value_follows_row_label(candidate)
        and _ratio_value_scale_is_plausible(candidate)
        and not _is_shifted_note_reference(candidate)
        and _column_unit_contract_is_plausible(candidate)
    )


def _endpoint_key(candidate: NumericCandidate) -> tuple[str, ...]:
    return (
        normalize_text(candidate.row_label),
        normalize_text(candidate.table_id),
        normalize_text(candidate.subtable_id),
        normalize_text(candidate.entity_id),
        normalize_text(candidate.scope),
        normalize_text(candidate.unit),
    )


def _candidate_key(candidate: NumericCandidate) -> tuple[str, ...]:
    return (
        *_endpoint_key(candidate),
        normalize_text(_period(candidate)),
        candidate.value,
    )


def _pointer_indices(
    question: str,
    context: str,
    targets: tuple[NumericCandidate, NumericCandidate],
    *,
    source_name: str,
    page: int,
    extraction_method: str,
) -> tuple[tuple[NumericCandidate, ...], tuple[int, int]] | None:
    extracted = extract_numeric_candidates(
        question,
        context,
        max_candidates=64,
        source_name=source_name,
        page=page,
        extraction_method=extraction_method,
    )
    selected = [
        candidate for candidate in extracted if _complete_table_candidate(candidate)
    ]
    grounded = tuple(
        replace(candidate, index=index) for index, candidate in enumerate(selected)
    )
    indices: list[int] = []
    for target in targets:
        matches = [
            candidate.index
            for candidate in grounded
            if _candidate_key(candidate) == _candidate_key(target)
        ]
        if len(matches) != 1:
            return None
        indices.append(matches[0])
    if len(set(indices)) != 2:
        return None
    return grounded, (indices[0], indices[1])


def _operand_payload(candidate: NumericCandidate, role: str) -> dict[str, object]:
    return {
        "candidate_index": candidate.index,
        "role": role,
        "source_name": candidate.source_name,
        "page": candidate.page,
        "row": candidate.row_label,
        "row_type": candidate.row_type,
        "column": candidate.column_label,
        "period": candidate.period or _period(candidate),
        "table_id": candidate.table_id,
        "section": candidate.section,
        "scope": candidate.scope,
        "entity_id": candidate.entity_id,
        "subtable_id": candidate.subtable_id,
        "unit": candidate.unit,
        "unit_status": "explicit" if candidate.unit else "missing",
        "source_span": list(candidate.source_span or (candidate.char_start, candidate.char_end)),
    }


def _supervision(
    candidates: tuple[NumericCandidate, ...],
    operands: tuple[int, int],
    *,
    company_key: str,
    source_sha256: str,
) -> dict[str, object]:
    left, right = (candidates[index] for index in operands)
    return {
        "schema": "structured-operand-supervision-v1",
        "classification": "mechanically_derived_development",
        "review_status": "pending_visual_review",
        "company_key": company_key,
        "source_sha256": source_sha256,
        "direction_contract": {
            "intent": "NEW_MINUS_OLD",
            "proof": "explicit_new_period_minus_old_period",
        },
        "gold_operands": [
            _operand_payload(left, "minuend"),
            _operand_payload(right, "subtrahend"),
        ],
        "hard_negatives": [
            {
                "type": "reversed_operand_order",
                "operand_slot": None,
                "candidate_indices": [operands[1], operands[0]],
                "proof": "shared_direction_contract",
            }
        ],
    }


def _quality_score(current: NumericCandidate, prior: NumericCandidate) -> tuple[object, ...]:
    row = normalize_text(current.row_label)
    explicit_years = int(
        _period_order(_period(current))[0] == 2
        and _period_order(_period(prior))[0] == 2
    )
    known_financial_row = int(any(term in row for term in FINANCIAL_ROW_TERMS))
    same_explicit_unit = int(bool(current.unit) and current.unit == prior.unit)
    nonnegative = int(current.decimal >= 0 and prior.decimal >= 0)
    magnitude = max(abs(current.decimal), abs(prior.decimal))
    return (
        known_financial_row,
        explicit_years,
        same_explicit_unit,
        nonnegative,
        magnitude,
        current.source_name,
        -(current.page or 0),
        row,
    )


def _cross_period_scale_discontinuity(
    current: NumericCandidate, prior: NumericCandidate
) -> bool:
    """Reject ratio rows that visibly switch between fraction and percent scale."""

    unit = normalize_text(current.unit or prior.unit)
    if unit not in {"x", "lan", "vong"}:
        return False
    low = min(abs(current.decimal), abs(prior.decimal))
    high = max(abs(current.decimal), abs(prior.decimal))
    return low <= 1 and high >= 10


def _make_example(
    *,
    source: dict[str, object],
    page: int,
    context: str,
    current: NumericCandidate,
    prior: NumericCandidate,
    extraction_method: str,
) -> tuple[ProgramExample | None, str]:
    row = _repair_text(current.row_label)
    current_label = _display_period(current)
    prior_label = _display_period(prior)
    question = (
        f"{row} thay đổi bao nhiêu từ {prior_label} đến {current_label}? "
        f"Hãy tính {current_label} trừ {prior_label}."
    )
    pointed = _pointer_indices(
        question,
        context,
        (current, prior),
        source_name=Path(str(source["document"])).name,
        page=page,
        extraction_method=extraction_method,
    )
    if pointed is None:
        return None, "pointer_unavailable"
    candidates, operands = pointed
    program = ReasoningProgram("SUBTRACT", operands)
    plan = compile_question_plan(question)
    graph = build_evidence_graph(candidates)
    direction = assess_program_direction(plan, program, graph)
    if plan.direction_intent is not DirectionIntent.NEW_MINUS_OLD:
        return None, "planner_direction_not_new_minus_old"
    if direction.outcome != "direction_supported":
        return None, f"gold_{direction.outcome}"
    if not reversed_pair_is_proven_negative(plan, program, graph):
        return None, "reversed_pair_unproven"
    answer = execute_program(program, candidates)
    stem = Path(str(source["document"])).stem
    row_digest = hashlib.blake2b(row.encode("utf-8"), digest_size=5).hexdigest()
    example = ProgramExample(
        example_id=f"subtract-development-{stem}-p{page}-{row_digest}",
        question=question,
        context=context,
        answer_text=(
            f"{current_label} trừ {prior_label}: {candidates[operands[0]].raw} "
            f"- {candidates[operands[1]].raw} = {answer}."
        ),
        candidates=candidates,
        program=program,
        answer_value=str(answer),
        task="QA",
        difficulty="DevelopmentDocumentDirection",
        supervision=_supervision(
            candidates,
            operands,
            company_key=str(source["company_key"]),
            source_sha256=str(source["sha256"]),
        ),
    )
    issues = supervision_integrity_issues(example, graph=graph, plan=plan)
    if issues:
        return None, "integrity:" + ",".join(issues)
    return example, "accepted"


def build_cohort(
    manifest: dict[str, object], markdown_dir: Path
) -> tuple[list[ProgramExample], dict[str, object]]:
    if manifest.get("classification") != "development_only":
        raise ValueError("SUBTRACT cohort requires development_only input")
    documents = manifest.get("documents")
    if not isinstance(documents, list):
        raise ValueError("Page manifest documents must be a list")
    extraction_method = "mineru" if manifest.get("ocr_performed") is True else "native"
    per_company: defaultdict[str, list[tuple[tuple[object, ...], ProgramExample]]] = defaultdict(list)
    rejections: Counter[str] = Counter()
    pages_scanned = pairs_considered = 0
    for source in documents:
        if not isinstance(source, dict):
            raise ValueError("Invalid page-manifest source")
        source_path = Path(str(source.get("document", "")))
        normalized_path = str(source_path).lower()
        if any(marker in normalized_path for marker in PROTECTED_MARKERS):
            raise ValueError(f"Refusing protected source path: {source_path}")
        if len(str(source.get("sha256", ""))) != 64 or not source.get("company_key"):
            raise ValueError(f"Incomplete source provenance: {source_path.name}")
        markdown_path = markdown_dir / f"{source_path.stem}.md"
        pages = _read_pages(markdown_path)
        for page_info in source.get("pages", []):
            if not isinstance(page_info, dict) or page_info.get("selection") not in {"core", "note"}:
                continue
            page = int(page_info["page"])
            context = pages.get(page, "")
            if not context:
                rejections["missing_markdown_page"] += 1
                continue
            pages_scanned += 1
            discovery = extract_numeric_candidates(
                "Giá trị chỉ tiêu tài chính theo từng kỳ là bao nhiêu?",
                context,
                max_candidates=128,
                source_name=source_path.name,
                page=page,
                extraction_method=extraction_method,
            )
            grouped: defaultdict[tuple[str, ...], dict[str, list[NumericCandidate]]] = defaultdict(lambda: defaultdict(list))
            for candidate in discovery:
                if _complete_table_candidate(candidate):
                    grouped[_endpoint_key(candidate)][normalize_text(_period(candidate))].append(candidate)
            row_provenance_counts = Counter(key[0] for key in grouped)
            for endpoint_key, period_map in grouped.items():
                if row_provenance_counts[endpoint_key[0]] > 1:
                    rejections["ambiguous_row_provenance"] += 1
                    continue
                if any(len(items) != 1 for items in period_map.values()):
                    rejections["ambiguous_endpoint_period"] += 1
                    continue
                ordered = sorted(
                    (items[0] for items in period_map.values()),
                    key=lambda item: _period_order(_period(item)) or (0, 0),
                )
                if len(ordered) < 2:
                    continue
                prior, current = ordered[-2:]
                pairs_considered += 1
                if current.decimal == prior.decimal:
                    rejections["equal_values_direction_unobservable"] += 1
                    continue
                if _cross_period_scale_discontinuity(current, prior):
                    rejections["cross_period_scale_discontinuity"] += 1
                    continue
                if current.unit and prior.unit and current.unit != prior.unit:
                    rejections["unit_mismatch"] += 1
                    continue
                example, reason = _make_example(
                    source=source,
                    page=page,
                    context=context,
                    current=current,
                    prior=prior,
                    extraction_method=extraction_method,
                )
                if example is None:
                    rejections[reason] += 1
                    continue
                per_company[str(source["company_key"])].append(
                    (_quality_score(current, prior), example)
                )
    examples = [
        max(items, key=lambda item: item[0])[1]
        for _, items in sorted(per_company.items())
        if items
    ]
    examples.sort(key=lambda item: item.example_id)
    direction_outcomes: Counter[str] = Counter()
    integrity_issues: Counter[str] = Counter()
    for example in examples:
        graph = build_evidence_graph(example.candidates)
        plan = compile_question_plan(example.question)
        direction_outcomes[assess_program_direction(plan, example.program, graph).outcome] += 1
        integrity_issues.update(supervision_integrity_issues(example, graph=graph, plan=plan))
    report = {
        "schema": "subtract-development-cohort-audit-v1",
        "classification": "development_only_new_company_candidates",
        "blind_test_used": False,
        "global_validation_loaded": False,
        "model_invoked": False,
        "fine_tuning_performed": False,
        "threshold_changed": False,
        "ready_for_phase_c": False,
        "review_status": "pending_visual_review",
        "metrics": {
            "documents_scanned": len(documents),
            "pages_scanned": pages_scanned,
            "temporal_pairs_considered": pairs_considered,
            "examples": len(examples),
            "company_groups": len(per_company),
            "direction_outcomes": dict(sorted(direction_outcomes.items())),
            "integrity_issue_counts": dict(sorted(integrity_issues.items())),
            "rejection_counts": dict(sorted(rejections.items())),
        },
    }
    return examples, report


def _review_markdown(examples: list[ProgramExample], report: dict[str, object]) -> str:
    metrics = report["metrics"]
    lines = [
        "# SUBTRACT new-company development review",
        "",
        f"- Candidates: {metrics['examples']}",
        f"- Company groups: {metrics['company_groups']}",
        "- Status: pending visual review; not Phase-C eligible.",
        "- Review rule: verify both cells are on the same financial row/table/unit and the newer period is the minuend.",
        "",
    ]
    for example in examples:
        left, right = (example.candidates[index] for index in example.program.operands)
        lines.extend(
            [
                f"## {example.example_id}",
                "",
                f"- Source: `{left.source_name}`, page {left.page}",
                f"- Company group: `{example.supervision['company_key']}`",
                f"- Question: {example.question}",
                f"- Minuend: `{left.row_label}` / `{left.column_label}` = `{left.raw}`",
                f"- Subtrahend: `{right.row_label}` / `{right.column_label}` = `{right.raw}`",
                f"- Signed answer: `{example.answer_value}`",
                "- Review: [ ] pass  [ ] reject",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-manifest", type=Path, required=True)
    parser.add_argument("--markdown-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--review-markdown", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.report, args.review_markdown):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
    manifest = json.loads(args.page_manifest.read_text(encoding="utf-8"))
    examples, report = build_cohort(manifest, args.markdown_dir)
    report["inputs"] = {
        "page_manifest": str(args.page_manifest.resolve()),
        "page_manifest_sha256": _sha256(args.page_manifest),
        "markdown_dir": str(args.markdown_dir.resolve()),
    }
    for path in (args.output, args.report, args.review_markdown):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(program_example_to_dict(item), ensure_ascii=False) + "\n" for item in examples),
        encoding="utf-8",
    )
    report["output_sha256"] = _sha256(args.output)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.review_markdown.write_text(_review_markdown(examples, report), encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
