"""Source-cell ambiguity checks, independent of question, gold and issuer identity.

This guard does not recover lost page geometry. Mixed percentage and untyped
numbers inside the same serialized Markdown cell need source review. Separate
cells and unstructured prose are outside this narrow contract.
"""
from functools import lru_cache


@lru_cache(maxsize=8192)
def _context_binding_issues(context: str) -> tuple[str, ...]:
    # Imported lazily so extraction and downstream consumers can share this
    # check without a program_data import cycle or a second number grammar.
    from .program_data import _NUMBER_RE, parse_decimal

    for line in context.splitlines():
        if not line.startswith("CELL:") or "|" not in line:
            continue
        start, end = line.find("<VALUE>"), line.find("</VALUE>")
        if start < 0 or end < start:
            continue
        left = line.rfind("|", 0, start)
        right = line.find("|", end + len("</VALUE>"))
        if left < 0 or right < 0:
            continue
        cell = line[left + 1:right].replace("<VALUE>", "").replace("</VALUE>", "")
        tokens = [m for m in _NUMBER_RE.finditer(cell) if parse_decimal(m.group(0)) is not None]
        # Prose can explicitly distinguish measures (e.g. "up 5% from 100").
        # This narrow guard only handles unlabeled numeric fragments. It does
        # not certify the semantics of labeled multi-value cells.
        remainder = _NUMBER_RE.sub("", cell)
        if any(character.isalpha() for character in remainder):
            continue
        if any(m.group("pct") for m in tokens) and any(not m.group("pct") for m in tokens):
            return ("source_cell_binding_ambiguous",)
    return ()


def candidate_binding_issues(candidate) -> tuple[str, ...]:
    return _context_binding_issues(getattr(candidate, "context", ""))


def program_binding_issues(program, candidates, program_dag=None) -> tuple[str, ...]:
    pending = list(program.operands)
    if program_dag is not None:
        pending.extend(program_dag.leaf_operands)
    visited = set()
    issues = []
    while pending:
        index = pending.pop()
        if index in visited or not 0 <= index < len(candidates):
            continue
        visited.add(index)
        candidate = candidates[index]
        issues.extend(candidate_binding_issues(candidate))
        pending.extend(getattr(candidate, "derived_operands", ()))
    return tuple(dict.fromkeys(issues))
