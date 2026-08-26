from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .engine import Answer, FinancialQAEngine, QAError
from .neural import (
    UNKNOWN_METRIC,
    NeuralModelBundle,
    Prediction,
    fact_text,
)
from .ontology import (
    FORMULAS,
    VARIABLES,
    FormulaSpec,
    VariableSpec,
    normalize_text,
    resolve_metric,
)
from .repository import Fact, FinancialRepository


@dataclass(frozen=True)
class ExpansionPlan:
    formula: FormulaSpec
    source: str
    confidence: float
    margin: float


@dataclass(frozen=True)
class RetrievalMatch:
    variable_key: str
    fact: Fact
    score: float
    neural_confidence: float
    method: str
    candidate_count: int
    predicted_statement: str
    statement_confidence: float


class ExpansionAgent:
    def __init__(
        self,
        model: NeuralModelBundle,
        *,
        min_confidence: float = 0.18,
        min_margin: float = 0.015,
    ) -> None:
        self.model = model
        self.min_confidence = min_confidence
        self.min_margin = min_margin

    def expand(self, question: str, explicit_metric: str | None = None) -> ExpansionPlan:
        query = explicit_metric or question
        symbolic = resolve_metric(query)
        if symbolic is not None:
            return ExpansionPlan(symbolic, "symbolic-ontology", 1.0, 1.0)
        prediction = self.model.metric_model.predict(query)
        if (
            prediction.label == UNKNOWN_METRIC
            or prediction.confidence < self.min_confidence
            or prediction.margin < self.min_margin
        ):
            supported = ", ".join(FORMULAS)
            raise QAError(
                "Neural Expansion Agent khong du tu tin de chon cong thuc "
                f"(top={prediction.label}, confidence={prediction.confidence:.3f}, "
                f"margin={prediction.margin:.3f}). Ho tro: {supported}"
            )
        return ExpansionPlan(
            FORMULAS[prediction.label],
            "neural-softmax",
            prediction.confidence,
            prediction.margin,
        )


class HierarchicalRetriever:
    def __init__(
        self,
        repository: FinancialRepository,
        model: NeuralModelBundle,
        *,
        neural_fallback_threshold: float = 0.45,
        neural_fallback_lexical_threshold: float = 0.28,
    ) -> None:
        self.repository = repository
        self.model = model
        self.neural_fallback_threshold = neural_fallback_threshold
        self.neural_fallback_lexical_threshold = neural_fallback_lexical_threshold
        self._cache: dict[tuple[str, int, str], RetrievalMatch | None] = {}

    def retrieve(self, ticker: str, year: int, variable_key: str) -> RetrievalMatch | None:
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

        scored: list[tuple[float, float, bool, Fact]] = []
        for fact in candidates:
            exact_rank = self._exact_rank(fact, spec)
            is_exact = exact_rank is not None
            exact_score = 0.0 if exact_rank is None else max(0.90, 1.0 - 0.03 * exact_rank)
            neural_score = self.model.variable_model.probability(
                fact_text(fact), variable_key
            )
            section_score = 1.0 if fact.statement in sections else 0.0
            lexical_score = self._lexical_score(fact, spec)
            score = (
                0.70 * exact_score
                + 0.20 * neural_score
                + 0.08 * section_score
                + 0.02 * lexical_score
            )
            scored.append((score, neural_score, is_exact, fact))
        scored.sort(
            key=lambda item: (item[0], item[1], -item[3].line_order),
            reverse=True,
        )
        best_score, neural_score, is_exact, best_fact = scored[0]
        method = "hybrid-exact" if is_exact else "neural-fallback"
        if not is_exact:
            prediction = self.model.variable_model.predict(fact_text(best_fact))
            lexical_score = self._lexical_score(best_fact, spec)
            if (
                prediction.label != variable_key
                or neural_score < self.neural_fallback_threshold
                or lexical_score < self.neural_fallback_lexical_threshold
            ):
                self._cache[cache_key] = None
                return None
        match = RetrievalMatch(
            variable_key=variable_key,
            fact=best_fact,
            score=best_score,
            neural_confidence=neural_score,
            method=method,
            candidate_count=len(candidates),
            predicted_statement=statement_prediction.label,
            statement_confidence=statement_prediction.confidence,
        )
        self._cache[cache_key] = match
        return match

    def _predict_statement(self, spec: VariableSpec) -> Prediction:
        query = " ".join((spec.label, *spec.aliases, *spec.item_ids))
        return self.model.statement_model.predict(query)

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


class SolvingAgent:
    def solve(
        self,
        formula: FormulaSpec,
        facts: dict[str, Fact],
        adjustments: tuple[str, ...],
    ) -> tuple[Decimal, dict[str, Decimal], tuple[str, ...]]:
        values = {key: fact.value for key, fact in facts.items()}
        applied = FinancialQAEngine._apply_adjustments(values, adjustments)
        value = FinancialQAEngine._calculate(formula.key, values)
        return value, values, applied


class EvaluationAgent:
    def __init__(self, symbolic_engine: FinancialQAEngine) -> None:
        self.symbolic_engine = symbolic_engine

    def evaluate(
        self,
        ticker: str,
        year: int,
        formula: FormulaSpec,
        value: Decimal,
        values: dict[str, Decimal],
        matches: dict[str, RetrievalMatch],
    ) -> tuple[tuple[object, ...], tuple[str, ...]]:
        checks = self.symbolic_engine.validate(ticker, year)
        warnings = list(
            self.symbolic_engine._plausibility_warnings(formula.key, value, values)
        )
        failed = [check for check in checks if check.status == "fail"]
        if failed:
            warnings.append(
                f"Evaluation Agent found {len(failed)} failed accounting constraint(s)."
            )
        neural_fallbacks = [
            match for match in matches.values() if match.method == "neural-fallback"
        ]
        if neural_fallbacks:
            warnings.append(
                "Some evidence was selected by neural fallback because no exact item_id was available."
            )
        return checks, tuple(warnings)


class NeuroSymbolicHRMEngine:
    """Legacy Hierarchical Retrieval Model kept for backward compatibility.

    This class is not the recurrent Hierarchical Reasoning Model of Wang et al.
    New deployments should use ``HierarchicalReasoningQAEngine``.
    """

    def __init__(
        self,
        repository: FinancialRepository,
        model: NeuralModelBundle,
    ) -> None:
        self.repository = repository
        self.model = model
        self.symbolic_engine = FinancialQAEngine(repository)
        self.expansion_agent = ExpansionAgent(model)
        self.retriever = HierarchicalRetriever(repository, model)
        self.solving_agent = SolvingAgent()
        self.evaluation_agent = EvaluationAgent(self.symbolic_engine)

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
            f"Level 1/document: selected {ticker} report and year {year}.",
            (
                "Expansion Agent: "
                f"metric={plan.formula.key}, source={plan.source}, "
                f"confidence={plan.confidence:.3f}, margin={plan.margin:.3f}."
            ),
            f"Formula plan: {plan.formula.expression}.",
        ]
        if ticker_inferred:
            trace.append(f"Ticker was parsed from the question: {ticker}.")
        if year_inferred:
            trace.append(f"No year was supplied; selected latest available year {year}.")

        facts: dict[str, Fact] = {}
        matches: dict[str, RetrievalMatch] = {}
        for variable_key in plan.formula.variables:
            match = self.retriever.retrieve(ticker, year, variable_key)
            if match is None:
                trace.append(
                    f"Level 2-4: no evidence passed confidence checks for {variable_key}."
                )
                continue
            matches[variable_key] = match
            facts[variable_key] = match.fact
            trace.append(
                "Level 2/section: "
                f"{variable_key} -> {match.predicted_statement} "
                f"(p={match.statement_confidence:.3f})."
            )
            trace.append(
                "Level 3/table: "
                f"ranked {match.candidate_count} candidates; selected "
                f"{match.fact.item_id} by {match.method} (score={match.score:.3f}, "
                f"neural={match.neural_confidence:.3f})."
            )
            trace.append(
                "Level 4/cell: "
                f"validated numeric value for {match.fact.ticker}/{match.fact.year}, "
                f"line={match.fact.line_order}."
            )

        value, values, applied = self.solving_agent.solve(
            plan.formula, facts, adjustments
        )
        checks, warnings = self.evaluation_agent.evaluate(
            ticker, year, plan.formula, value, values, matches
        )
        trace.append(
            "Solving Agent: executed the formula with Decimal arithmetic."
        )
        trace.append(
            "Evaluation Agent: checked completeness, plausibility and accounting constraints."
        )
        trace.append(
            f"Neural bundle parameters: {self.model.parameter_count:,}."
        )
        return Answer(
            ticker=ticker,
            year=year,
            metric=plan.formula,
            value=value,
            facts=facts,
            trace=tuple(trace),
            checks=checks,  # type: ignore[arg-type]
            adjustments=applied,
            warnings=warnings,
        )

    def validate(
        self, ticker: str, year: int, tolerance: Decimal = Decimal("0.01")
    ) -> tuple[object, ...]:
        return self.symbolic_engine.validate(ticker, year, tolerance)
