from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .document_search import (
    DocumentSearchError,
    SearchHit,
)
from .ontology import VARIABLES, FormulaSpec, VariableSpec, normalize_text, resolve_metric


@dataclass(frozen=True)
class RetrievalRound:
    number: int
    query: str
    document_id: str | None
    hit_ids: tuple[str, ...]
    newly_covered: tuple[str, ...]
    missing_after: tuple[str, ...]


@dataclass(frozen=True)
class DocumentEvidenceResult:
    question: str
    metric_key: str | None
    formula_expression: str | None
    selected_document_id: str | None
    requested_year: int | None
    reporting_year: int | None
    required_variables: tuple[str, ...]
    covered_variables: tuple[str, ...]
    missing_variables: tuple[str, ...]
    rounds: tuple[RetrievalRound, ...]
    hits: tuple[SearchHit, ...]
    variable_evidence: tuple[tuple[str, tuple[str, ...]], ...]

    @property
    def complete(self) -> bool:
        return (
            bool(self.required_variables)
            and not self.missing_variables
            and not self.year_mismatch
        )

    @property
    def year_mismatch(self) -> bool:
        return (
            self.requested_year is not None
            and self.reporting_year is not None
            and self.requested_year != self.reporting_year
        )


class DocumentEvidenceRetriever:
    """Expansion, retrieval and evidence checks for one financial report."""

    def __init__(self, index: Any) -> None:
        self.index = index

    def retrieve(
        self,
        question: str,
        *,
        document_id: str | None = None,
        top_k: int = 15,
        max_rounds: int = 5,
    ) -> DocumentEvidenceResult:
        if max_rounds < 1:
            raise DocumentSearchError("max_rounds phai lon hon hoac bang 1")
        formula = resolve_metric(question)
        required = formula.variables if formula else ()
        selected_document = (
            self.index.resolve_document_id(document_id) if document_id else None
        )
        if formula and selected_document is None:
            discovery_index = getattr(self.index, "sparse_index", self.index)
            discovery_hits = discovery_index.search(
                question, top_k=max(15, top_k)
            )
            if discovery_hits:
                selected_document = self._select_discovery_document(
                    question, discovery_hits
                )
        current_query = self._initial_query(question, formula)
        merged_hits: dict[str, SearchHit] = {}
        rounds: list[RetrievalRound] = []
        covered: set[str] = set()

        for round_number in range(1, max_rounds + 1):
            hits = self.index.search(
                current_query,
                document_id=selected_document,
                top_k=top_k,
            )
            if formula and selected_document is None and hits:
                selected_document = self._select_document(hits, required)
                hits = tuple(
                    hit
                    for hit in hits
                    if hit.chunk.document_id == selected_document
                )

            before = set(covered)
            for hit in hits:
                merged_hits.setdefault(hit.chunk.chunk_id, hit)
            covered = self._covered_variables(tuple(merged_hits.values()), required)
            missing = tuple(item for item in required if item not in covered)
            rounds.append(
                RetrievalRound(
                    number=round_number,
                    query=current_query,
                    document_id=selected_document,
                    hit_ids=tuple(hit.chunk.chunk_id for hit in hits),
                    newly_covered=tuple(
                        item for item in required if item in covered - before
                    ),
                    missing_after=missing,
                )
            )

            if not formula or not missing:
                break
            next_query = self._followup_query(
                question, formula, missing, round_number + 1
            )
            if not hits and next_query == current_query:
                break
            current_query = next_query

        ordered_hits = tuple(
            sorted(
                merged_hits.values(),
                key=lambda hit: (
                    sum(
                        _hit_covers_variable(hit, VARIABLES[variable])
                        for variable in required
                    ),
                    hit.score,
                ),
                reverse=True,
            )
        )
        ordered_covered = tuple(item for item in required if item in covered)
        ordered_missing = tuple(item for item in required if item not in covered)
        evidence = tuple(
            (
                variable,
                tuple(
                    hit.chunk.chunk_id
                    for hit in ordered_hits
                    if _hit_covers_variable(hit, VARIABLES[variable])
                ),
            )
            for variable in required
        )
        reporting_year = (
            self.index.outline(selected_document).reporting_year
            if selected_document
            else None
        )
        year_match = re.search(r"\b(20\d{2})\b", question)
        return DocumentEvidenceResult(
            question=question,
            metric_key=formula.key if formula else None,
            formula_expression=formula.expression if formula else None,
            selected_document_id=selected_document,
            requested_year=int(year_match.group(1)) if year_match else None,
            reporting_year=reporting_year,
            required_variables=required,
            covered_variables=ordered_covered,
            missing_variables=ordered_missing,
            rounds=tuple(rounds),
            hits=ordered_hits,
            variable_evidence=evidence,
        )

    def _initial_query(self, question: str, formula: FormulaSpec | None) -> str:
        if formula is None:
            return question
        parts = [question, formula.label, formula.expression]
        for variable_key in formula.variables:
            variable = VARIABLES[variable_key]
            parts.extend((variable.label, *variable.aliases))
        return _unique_phrases(parts)

    def _followup_query(
        self,
        question: str,
        formula: FormulaSpec,
        missing: tuple[str, ...],
        round_number: int,
    ) -> str:
        parts = [question, formula.expression]
        for variable_key in missing:
            variable = VARIABLES[variable_key]
            parts.extend((variable.label, *variable.aliases, *variable.item_ids))
            if variable.statement:
                parts.append(variable.statement.replace("_", " "))
        if round_number > 2 and missing:
            focus = VARIABLES[missing[(round_number - 3) % len(missing)]]
            parts = [question, focus.label, *focus.aliases, *focus.item_ids]
            if focus.statement:
                parts.append(focus.statement.replace("_", " "))
        return _unique_phrases(parts)

    def _select_document(
        self, hits: tuple[SearchHit, ...], required: tuple[str, ...]
    ) -> str:
        grouped: dict[str, list[tuple[int, SearchHit]]] = {}
        for rank, hit in enumerate(hits):
            grouped.setdefault(hit.chunk.document_id, []).append((rank, hit))

        def document_score(
            item: tuple[str, list[tuple[int, SearchHit]]]
        ) -> tuple[int, float, int]:
            _, ranked_hits = item
            document_hits = tuple(hit for _, hit in ranked_hits)
            covered = self._covered_variables(document_hits, required)
            return (
                len(covered),
                sum(hit.score for _, hit in ranked_hits),
                -min(rank for rank, _ in ranked_hits),
            )

        return max(grouped.items(), key=document_score)[0]

    def _select_discovery_document(
        self, question: str, hits: tuple[SearchHit, ...]
    ) -> str:
        grouped: dict[str, list[SearchHit]] = {}
        for hit in hits:
            grouped.setdefault(hit.chunk.document_id, []).append(hit)
        question_tokens = {
            token
            for token in normalize_text(question).split()
            if len(token) > 2 and not token.isdigit()
        }

        def discovery_score(item: tuple[str, list[SearchHit]]) -> tuple[int, float]:
            document_id, document_hits = item
            source_name = self.index.outline(document_id).source_name
            name_tokens = set(normalize_text(f"{document_id} {source_name}").split())
            name_overlap = len(question_tokens.intersection(name_tokens))
            relevance = max(hit.score for hit in document_hits)
            relevance += sum(hit.score for hit in document_hits) * 0.1
            return name_overlap, relevance

        return max(grouped.items(), key=discovery_score)[0]

    def _covered_variables(
        self, hits: tuple[SearchHit, ...], required: tuple[str, ...]
    ) -> set[str]:
        return {
            variable_key
            for variable_key in required
            if any(
                _hit_covers_variable(hit, VARIABLES[variable_key]) for hit in hits
            )
        }


def _hit_covers_variable(hit: SearchHit, variable: VariableSpec) -> bool:
    raw_lines = tuple(line for line in hit.chunk.text.splitlines() if line.strip())
    normalized_lines = tuple(normalize_text(line) for line in raw_lines)
    aliases = (variable.label, *variable.aliases, *variable.item_ids)
    for raw_alias in aliases:
        alias = normalize_text(raw_alias)
        if not alias:
            continue
        alias_tokens = tuple(token for token in alias.split() if len(token) > 1)
        for position in range(len(normalized_lines)):
            combined = " ".join(normalized_lines[position : position + 2])
            matched = alias in combined
            if not matched and len(alias_tokens) >= 2:
                matched = set(alias_tokens).issubset(set(combined.split()))
            if not matched:
                continue
            context_start = max(0, position - 1)
            context_end = min(len(raw_lines), position + 3)
            if any(
                _line_has_financial_value(line)
                for line in raw_lines[context_start:context_end]
            ):
                return True
    return False


def _line_has_financial_value(value: str) -> bool:
    without_enumerator = re.sub(r"^\s*\d{1,2}[.)]\s*", "", value)
    number_tokens = re.findall(r"\(?-?\d[\d.,]*\)?", without_enumerator)
    if len(number_tokens) >= 2:
        return True
    if number_tokens:
        token = number_tokens[0]
        digits = re.sub(r"\D", "", token)
        if len(digits) == 4 and 2000 <= int(digits) <= 2035:
            return False
        if len(digits) >= 5 or (("," in token or "." in token) and len(digits) >= 3):
            return True
    return bool(number_tokens and re.search(r"(?:^|\s)-(?:\s|$)", value))


def _unique_phrases(parts: list[str]) -> str:
    seen: set[str] = set()
    unique: list[str] = []
    for part in parts:
        normalized = normalize_text(part)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(part.replace("_", " "))
    return " | ".join(unique)
