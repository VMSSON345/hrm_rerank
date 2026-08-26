from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .engine import Answer, FinancialQAEngine, QAError
from .neural import UNKNOWN_METRIC, fact_text
from .ontology import (
    FORMULAS,
    VARIABLES,
    FormulaSpec,
    VariableSpec,
    normalize_text,
    resolve_metric,
)
from .reasoning import HierarchicalReasoningBundle, ReasoningPrediction
from .repository import Fact, FinancialRepository


@dataclass(frozen=True)
class ReasoningExpansionPlan:
    formula: FormulaSpec
    source: str
    confidence: float
    margin: float
    segments: int
    halt_q: float
    continue_q: float


@dataclass(frozen=True)
class ReasoningRetrievalMatch:
    variable_key: str
    fact: Fact
    score: float
    reasoning_confidence: float
    method: str
    candidate_count: int
    predicted_statement: str
    statement_confidence: float
    statement_segments: int
    variable_segments: int


class ReasoningExpansionAgent:
    def __init__(
        self,
        model: HierarchicalReasoningBundle,
        *,
        min_confidence: float = 0.18,
        min_margin: float = 0.01,
    ) -> None:
        self.model = model
        self.min_confidence = min_confidence
        self.min_margin = min_margin

    def expand(
        self, question: str, explicit_metric: str | None = None
    ) -> ReasoningExpansionPlan:
        query = explicit_metric or question
        symbolic = resolve_metric(query)
        if symbolic is not None:
            return ReasoningExpansionPlan(
                symbolic, "symbolic-ontology", 1.0, 1.0, 0, 0.0, 0.0
            )
        prediction = self.model.predict("metric", query)
        if (
            prediction.label == UNKNOWN_METRIC
            or prediction.confidence < self.min_confidence
            or prediction.margin < self.min_margin
        ):
            supported = ", ".join(FORMULAS)
            raise QAError(
                "HRM khong du tu tin de chon cong thuc "
                f"(top={prediction.label}, confidence={prediction.confidence:.3f}, "
                f"margin={prediction.margin:.3f}, segments={prediction.segments}). "
                f"Ho tro: {supported}"
            )
        return ReasoningExpansionPlan(
            FORMULAS[prediction.label],
            "hierarchical-reasoning-H/L",
            prediction.confidence,
            prediction.margin,
            prediction.segments,
            prediction.halt_q,
            prediction.continue_q,
        )


class ReasoningHierarchicalRetriever:
    def __init__(
        self,
        repository: FinancialRepository,
        model: HierarchicalReasoningBundle,
        *,
        fallback_confidence: float = 0.30,
        fallback_lexical_threshold: float = 0.20,
    ) -> None:
        self.repository = repository
        self.model = model
        self.fallback_confidence = fallback_confidence
        self.fallback_lexical_threshold = fallback_lexical_threshold
        self._cache: dict[
            tuple[str, int, str], ReasoningRetrievalMatch | None
        ] = {}

    def retrieve(
        self, ticker: str, year: int, variable_key: str
    ) -> ReasoningRetrievalMatch | None:
        cache_key = (ticker, year, variable_key)
        if cache_key in self._cache:
            return self._cache[cache_key]
        spec = VARIABLES[variable_key]
        all_facts = self.repository.facts(ticker, year)
        if not all_facts:
            self._cache[cache_key] = None
            return None

        statement_prediction = self._predict_statement(spec)
        sections = {statement_prediction.label}
        if spec.statement:
            sections.add(spec.statement)
        candidates = [fact for fact in all_facts if fact.statement in sections]
        if not candidates:
            candidates = list(all_facts)

        texts = [fact_text(fact) for fact in candidates]
        predictions = self.model.predict_batch("variable", texts)
        label_index = self.model_label_index("variable", variable_key)
        scored: list[tuple[float, float, bool, Fact, ReasoningPrediction]] = []
        for fact, prediction in zip(candidates, predictions):
            exact_rank = self._exact_rank(fact, spec)
            is_exact = exact_rank is not None
            exact_score = (
                0.0 if exact_rank is None else max(0.90, 1.0 - 0.03 * exact_rank)
            )
            reasoning_score = prediction.probabilities[label_index]
            section_score = 1.0 if fact.statement in sections else 0.0
            lexical_score = self._lexical_score(fact, spec)
            score = (
                0.68 * exact_score
                + 0.22 * reasoning_score
                + 0.08 * section_score
                + 0.02 * lexical_score
            )
            scored.append((score, reasoning_score, is_exact, fact, prediction))
        scored.sort(
            key=lambda item: (item[0], item[1], -item[3].line_order), reverse=True
        )
        best_score, reasoning_score, is_exact, best_fact, best_prediction = scored[0]
        method = "exact+HRM" if is_exact else "HRM-fallback"
        if not is_exact:
            lexical_score = self._lexical_score(best_fact, spec)
            if (
                best_prediction.label != variable_key
                or reasoning_score < self.fallback_confidence
                or lexical_score < self.fallback_lexical_threshold
            ):
                self._cache[cache_key] = None
                return None

        match = ReasoningRetrievalMatch(
            variable_key=variable_key,
            fact=best_fact,
            score=best_score,
            reasoning_confidence=reasoning_score,
            method=method,
            candidate_count=len(candidates),
            predicted_statement=statement_prediction.label,
            statement_confidence=statement_prediction.confidence,
            statement_segments=statement_prediction.segments,
            variable_segments=best_prediction.segments,
        )
        self._cache[cache_key] = match
        return match

    @staticmethod
    def model_label_index(task: str, label: str) -> int:
        from .reasoning import LABELS

        return LABELS[task].index(label)

    def _predict_statement(self, spec: VariableSpec) -> ReasoningPrediction:
        query = " ".join((spec.label, *spec.aliases, *spec.item_ids))
        return self.model.predict("statement", query)

    @staticmethod
    def _exact_rank(fact: Fact, spec: VariableSpec) -> int | None:
        try:
            return spec.item_ids.index(fact.item_id)
        except ValueError:
            return None

    @staticmethod
    def _lexical_score(fact: Fact, spec: VariableSpec) -> float:
        fact_tokens = set(normalize_text(fact_text(fact)).split())
        query_tokens = set(
            normalize_text(" ".join((spec.label, *spec.aliases, *spec.item_ids))).split()
        )
        if not fact_tokens or not query_tokens:
            return 0.0
        return len(fact_tokens & query_tokens) / len(fact_tokens | query_tokens)


class HierarchicalReasoningQAEngine:
    """Actual H/L recurrent HRM for planning/ranking + symbolic execution."""

    def __init__(
        self,
        repository: FinancialRepository,
        model: HierarchicalReasoningBundle,
    ) -> None:
        self.repository = repository
        self.model = model
        self.symbolic_engine = FinancialQAEngine(repository)
        self.expansion_agent = ReasoningExpansionAgent(model)
        self.retriever = ReasoningHierarchicalRetriever(repository, model)

    def answer(
        self,
        question: str = "",
        *,
        ticker: str | None = None,
        year: int | None = None,
        metric: str | None = None,
        adjustments: tuple[str, ...] = (),
    ) -> Answer:
        plan = self.expansion_agent.expand(question, metric)
        ticker, ticker_inferred = self.symbolic_engine._resolve_ticker(question, ticker)
        year, year_inferred = self.symbolic_engine._resolve_year(question, ticker, year)
        trace = [
            f"Document scope: selected {ticker} and reporting year {year}.",
            (
                "HRM high-level plan: "
                f"metric={plan.formula.key}, source={plan.source}, "
                f"confidence={plan.confidence:.3f}, margin={plan.margin:.3f}, "
                f"segments={plan.segments}."
            ),
            f"Formula plan: {plan.formula.expression}.",
        ]
        if plan.segments:
            trace.append(
                "ACT decision: "
                f"halt_q={plan.halt_q:.3f}, continue_q={plan.continue_q:.3f}."
            )
        if ticker_inferred:
            trace.append(f"Ticker parsed from question: {ticker}.")
        if year_inferred:
            trace.append(f"No year supplied; selected latest available year {year}.")

        facts: dict[str, Fact] = {}
        matches: dict[str, ReasoningRetrievalMatch] = {}
        for variable_key in plan.formula.variables:
            match = self.retriever.retrieve(ticker, year, variable_key)
            if match is None:
                trace.append(
                    f"HRM evidence step: no candidate passed checks for {variable_key}."
                )
                continue
            matches[variable_key] = match
            facts[variable_key] = match.fact
            trace.append(
                "H-module/statement plan: "
                f"{variable_key} -> {match.predicted_statement} "
                f"(p={match.statement_confidence:.3f}, "
                f"segments={match.statement_segments})."
            )
            trace.append(
                "L-module/cell refinement: "
                f"ranked {match.candidate_count} candidates; selected "
                f"{match.fact.item_id} by {match.method} "
                f"(score={match.score:.3f}, HRM={match.reasoning_confidence:.3f}, "
                f"segments={match.variable_segments})."
            )
            trace.append(
                "Evidence validation: "
                f"{match.fact.ticker}/{match.fact.year}, line={match.fact.line_order}."
            )

        values = {key: fact.value for key, fact in facts.items()}
        applied = FinancialQAEngine._apply_adjustments(values, adjustments)
        value = FinancialQAEngine._calculate(plan.formula.key, values)
        checks = self.symbolic_engine.validate(ticker, year)
        warnings = list(
            self.symbolic_engine._plausibility_warnings(
                plan.formula.key, value, values
            )
        )
        failed = [check for check in checks if check.status == "fail"]
        if failed:
            warnings.append(
                f"Evaluation found {len(failed)} failed accounting constraint(s)."
            )
        fallbacks = [
            match for match in matches.values() if match.method == "HRM-fallback"
        ]
        if fallbacks:
            warnings.append(
                "Some evidence was selected by HRM fallback because no exact item_id matched."
            )
        trace.append("Symbolic executor: calculated with Decimal arithmetic.")
        trace.append("Evaluation: checked evidence completeness and accounting constraints.")
        trace.append(f"HRM trainable parameters: {self.model.parameter_count:,}.")
        return Answer(
            ticker=ticker,
            year=year,
            metric=plan.formula,
            value=value,
            facts=facts,
            trace=tuple(trace),
            checks=checks,
            adjustments=applied,
            warnings=tuple(warnings),
        )

    def validate(
        self, ticker: str, year: int, tolerance: Decimal = Decimal("0.01")
    ) -> tuple[object, ...]:
        return self.symbolic_engine.validate(ticker, year, tolerance)
