from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hrm_finance_qa.document_hrm import (
    DocumentHRMBundle,
    DocumentHRMError,
    ProgramCandidateTrace,
    _abstention_threshold_for_prediction,
    _preserve_safe_residual_reference,
    _program_candidate_training_state,
)
from hrm_finance_qa.program_data import (
    NumericCandidate,
    ProgramExample,
    ReasoningProgram,
)
from hrm_finance_qa.program_reranker import (
    FEATURE_INDEX,
    LEGACY_PROGRAM_RERANKER_FEATURE_SCHEMA,
    PROGRAM_RERANKER_FEATURE_SCHEMA,
    PROGRAM_RERANKER_FEATURES,
    ProgramRerankerBundle,
    ProgramRerankerConfig,
    ProgramRerankerModel,
    RerankerCandidateRecord,
    RerankerExampleRecord,
    calibrate_override_margin,
    train_program_reranker,
)
from hrm_finance_qa.question_planner import QuestionProofContract
from hrm_finance_qa.v11_pipeline import (
    build_reranker_records,
    checkpoint_sha256,
    read_reranker_records,
    split_reranker_development,
    upgrade_reranker_records_to_supported_tristate,
    write_reranker_records,
)
from hrm_finance_qa.symbolic_verifier import (
    ConstraintResult,
    VerificationResult,
    VerificationState,
)
from scripts.train_document_hrm_v12b_reranker import _generate_sharded_records


def _features(
    neural: float,
    signal: float,
    *,
    legacy_rank: int = 1,
    legacy_winner: bool = True,
    legacy_score: float | None = None,
) -> tuple[float, ...]:
    values = [0.0] * len(PROGRAM_RERANKER_FEATURES)
    values[FEATURE_INDEX["neural_log_score"]] = neural
    values[FEATURE_INDEX["legacy_rank_score"]] = (
        float(legacy_score)
        if legacy_score is not None
        else (1.0 if legacy_winner else 0.0)
    )
    values[FEATURE_INDEX["legacy_winner"]] = float(legacy_winner)
    values[FEATURE_INDEX["legacy_reciprocal_rank"]] = 1.0 / legacy_rank
    values[FEATURE_INDEX["verifier_proof_score"]] = signal
    values[FEATURE_INDEX["verifier_state_valid"]] = float(signal > 0)
    values[FEATURE_INDEX["verifier_state_unknown"]] = float(signal == 0)
    values[FEATURE_INDEX["metric_match"]] = signal
    return tuple(values)


def _example(example_id: str, company: str, operation: str = "SELECT") -> ProgramExample:
    candidates = (
        NumericCandidate(
            index=0,
            value="10",
            raw="10",
            context="ROW: Doanh thu\nCOLUMN: 2023\nPERIOD: 2023",
            line_number=1,
            char_start=0,
            char_end=2,
            unit="VND",
            is_percent=False,
            is_year_like=False,
            relevance=1.0,
        ),
        NumericCandidate(
            index=1,
            value="20",
            raw="20",
            context="ROW: Doanh thu\nCOLUMN: 2022\nPERIOD: 2022",
            line_number=1,
            char_start=3,
            char_end=5,
            unit="VND",
            is_percent=False,
            is_year_like=False,
            relevance=0.5,
        ),
    )
    return ProgramExample(
        example_id=example_id,
        question="Doanh thu năm 2023 là bao nhiêu?",
        context=f"Công ty: {company}\nBáo cáo tài chính",
        answer_text="10",
        candidates=candidates,
        program=ReasoningProgram(operation, (0,)),
        answer_value="10",
    )


class ProgramRerankerTests(unittest.TestCase):
    def test_sidecar_without_override_preserves_frozen_reference(self) -> None:
        reference_program = ReasoningProgram("AVERAGE", (0, 1))
        alternative_program = ReasoningProgram("AVERAGE", (2, 3))
        reference = SimpleNamespace(program=reference_program)
        reference_row = SimpleNamespace(program=reference_program)
        alternative_row = SimpleNamespace(program=alternative_program)

        chosen, preserved = _preserve_safe_residual_reference(
            reference,
            [reference_row, alternative_row],
            alternative_row,
            reranker_overridden=False,
        )
        self.assertIs(chosen, reference_row)
        self.assertTrue(preserved)

        chosen, preserved = _preserve_safe_residual_reference(
            reference,
            [reference_row, alternative_row],
            alternative_row,
            reranker_overridden=True,
        )
        self.assertIs(chosen, alternative_row)
        self.assertFalse(preserved)

    def test_v12b_sharded_cache_resumes_only_verified_shards(self) -> None:
        examples = [SimpleNamespace(example_id=f"shard-{index}") for index in range(4)]

        def fake_build(_bundle, current, *, top_k, batch_size):
            self.assertEqual(top_k, 8)
            self.assertEqual(batch_size, 2)
            records = [
                RerankerExampleRecord(
                    item.example_id,
                    "SELECT",
                    (
                        RerankerCandidateRecord(
                            _features(-0.1, 1.0),
                            "SELECT:0",
                            True,
                            True,
                        ),
                        RerankerCandidateRecord(
                            _features(
                                -0.2,
                                0.0,
                                legacy_rank=2,
                                legacy_winner=False,
                            ),
                            "SELECT:1",
                            False,
                            False,
                        ),
                    ),
                )
                for item in current
            ]
            count = len(current)
            audit = {
                "examples": count,
                "records": count,
                "candidate_sets": count,
                "top_k": top_k,
                "no_trace": 0,
                "gold_program_recall_at_k": 1.0,
                "execution_recall_at_k": 1.0,
                "rerankable_gold_program_recall_at_k": 1.0,
                "rerankable_execution_recall_at_k": 1.0,
                "invalid_candidates_retained": 0,
                "hard_invalid_candidates_dropped": 0,
                "soft_unknown_candidates_retained": 0,
                "pre_prune_candidates": count * 2,
                "post_prune_candidates": count * 2,
                "training_trace_sets": count,
                "pre_prune_gold_program_recall": 1.0,
                "post_prune_gold_program_recall": 1.0,
                "pruning_gold_loss_count": 0,
            }
            return records, audit

        with TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            with patch(
                "scripts.train_document_hrm_v12b_reranker.build_reranker_records",
                side_effect=fake_build,
            ) as builder:
                first_records, first_audit, first_shards = _generate_sharded_records(
                    label="train",
                    bundle=SimpleNamespace(),
                    examples=examples,
                    cache_dir=cache_dir,
                    cache_identity="frozen-inputs",
                    top_k=8,
                    batch_size=2,
                    shard_size=2,
                    regenerate=False,
                )
            self.assertEqual(builder.call_count, 2)
            self.assertEqual(len(first_records), 4)
            self.assertEqual(first_audit["pruning_gold_loss_count"], 0)
            self.assertTrue(all(not item["reused"] for item in first_shards))

            with patch(
                "scripts.train_document_hrm_v12b_reranker.build_reranker_records",
                side_effect=AssertionError("verified shards must be reused"),
            ) as builder:
                second_records, second_audit, second_shards = _generate_sharded_records(
                    label="train",
                    bundle=SimpleNamespace(),
                    examples=examples,
                    cache_dir=cache_dir,
                    cache_identity="frozen-inputs",
                    top_k=8,
                    batch_size=2,
                    shard_size=2,
                    regenerate=False,
                )
            self.assertEqual(builder.call_count, 0)
            self.assertEqual(second_records, first_records)
            self.assertEqual(second_audit, first_audit)
            self.assertTrue(all(item["reused"] for item in second_shards))

    def test_positive_taxonomy_never_promotes_execution_only(self) -> None:
        canonical = RerankerCandidateRecord(
            _features(-0.2, 1.0),
            "ADD:0,1",
            False,
            True,
            positive_class="canonical_exact",
        )
        coincidental = RerankerCandidateRecord(
            _features(-0.1, 0.0),
            "ADD:2,3",
            False,
            True,
            positive_class="execution_only",
        )
        record = RerankerExampleRecord(
            example_id="taxonomy",
            operation="ADD",
            candidates=(canonical, coincidental),
        )

        self.assertTrue(record.has_canonical_positive)
        self.assertTrue(canonical.rank_positive)
        self.assertFalse(coincidental.rank_positive)

    def test_sidecar_round_trip_preserves_scores(self) -> None:
        torch.manual_seed(3)
        model = ProgramRerankerModel(ProgramRerankerConfig(hidden_size=8, dropout=0.0))
        bundle = ProgramRerankerBundle(
            model,
            metadata={"feature_schema": PROGRAM_RERANKER_FEATURE_SCHEMA},
        )
        rows = [
            _features(-1.0, 1.0),
            _features(
                -2.0,
                -1.0,
                legacy_rank=2,
                legacy_winner=False,
            ),
        ]
        expected = bundle.score_many(rows)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "reranker.pt"
            bundle.save(path)
            loaded = ProgramRerankerBundle.load(path)
            actual = loaded.score_many(rows)
        self.assertEqual(expected, actual)

    def test_legacy_v2_sidecar_remains_loadable(self) -> None:
        model = ProgramRerankerModel(
            ProgramRerankerConfig(hidden_size=8, dropout=0.0)
        )
        bundle = ProgramRerankerBundle(
            model,
            metadata={
                "feature_schema": LEGACY_PROGRAM_RERANKER_FEATURE_SCHEMA
            },
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-v2.pt"
            bundle.save(path)
            loaded = ProgramRerankerBundle.load(path)
        self.assertEqual(
            loaded.metadata["feature_schema"],
            LEGACY_PROGRAM_RERANKER_FEATURE_SCHEMA,
        )

    def test_candidate_cache_keeps_unknown_and_drops_invalid(self) -> None:
        example = _example("example-1", "A")
        unknown = ProgramCandidateTrace(
            program=ReasoningProgram("SELECT", (0,)),
            neural_score=-0.1,
            neural_rank=1,
            verification_state="UNKNOWN",
            proof_score=0.5,
            eligible=False,
            rejection_reasons=("operand_role_mismatch",),
            unknown_reasons=("same_table",),
            reranker_features=_features(-0.1, 1.0),
            legacy_rank=1,
            legacy_rank_score=1.0,
            legacy_winner=True,
            training_state="UNKNOWN",
            required_valid_fraction=0.5,
            required_unknown_fraction=0.5,
            optional_support_fraction=0.25,
            proof_obligations=("required:period_coverage:UNKNOWN",),
        )
        valid = ProgramCandidateTrace(
            program=ReasoningProgram("SELECT", (1,)),
            neural_score=-0.15,
            neural_rank=2,
            verification_state="VALID",
            proof_score=0.75,
            eligible=True,
            rejection_reasons=(),
            unknown_reasons=(),
            reranker_features=_features(
                -0.15,
                0.75,
                legacy_rank=2,
                legacy_winner=False,
            ),
            legacy_rank=2,
            legacy_rank_score=0.0,
            legacy_winner=False,
            required_valid_fraction=1.0,
            required_unknown_fraction=0.0,
            optional_support_fraction=0.75,
            proof_obligations=("required:period_coverage:VALID",),
        )
        invalid = ProgramCandidateTrace(
            program=ReasoningProgram("ADD", (0, 1)),
            neural_score=-0.2,
            neural_rank=3,
            verification_state="INVALID",
            proof_score=-1.0,
            eligible=False,
            rejection_reasons=("missing_requested_period",),
            unknown_reasons=(),
            reranker_features=_features(
                -0.2,
                -1.0,
                legacy_rank=3,
                legacy_winner=False,
            ),
        )
        fake_bundle = SimpleNamespace(
            program_reranker=None,
            predict_many=lambda examples: [
                SimpleNamespace(program_search_trace=(unknown, valid, invalid))
                for _ in examples
            ],
        )
        records, audit = build_reranker_records(fake_bundle, [example], top_k=2)
        self.assertEqual(len(records), 1)
        self.assertEqual(len(records[0].candidates), 2)
        self.assertEqual(records[0].candidates[0].program_key, "SELECT:0")
        self.assertEqual(audit["invalid_candidates_retained"], 0)
        self.assertFalse(audit["unknown_as_negative"])
        self.assertEqual(audit["soft_unknown_candidates_retained"], 1)
        self.assertEqual(
            records[0].candidates[0].training_state, "UNKNOWN"
        )
        self.assertFalse(records[0].candidates[0].runtime_eligible)
        self.assertEqual(audit["mean_required_valid_fraction"], 0.75)
        self.assertEqual(audit["mean_required_unknown_fraction"], 0.25)
        self.assertEqual(audit["mean_optional_support_fraction"], 0.5)
        self.assertEqual(
            audit["proof_obligation_counts"][
                "required:period_coverage:UNKNOWN"
            ],
            1,
        )
        self.assertEqual(
            audit["proof_contract_state_counts"],
            {"UNKNOWN": 1, "VALID": 1, "INVALID": 1},
        )

    def test_candidate_cache_retains_only_proven_directional_probe(self) -> None:
        candidates = (
            NumericCandidate(
                index=0,
                value="20",
                raw="20",
                context="ROW: Doanh thu\nCOLUMN: 2023\nPERIOD: 2023",
                line_number=1,
                char_start=0,
                char_end=2,
                unit="VND",
                is_percent=False,
                is_year_like=False,
                relevance=1.0,
            ),
            NumericCandidate(
                index=1,
                value="20",
                raw="20",
                context="ROW: Doanh thu\nCOLUMN: 2022\nPERIOD: 2022",
                line_number=1,
                char_start=3,
                char_end=5,
                unit="VND",
                is_percent=False,
                is_year_like=False,
                relevance=1.0,
            ),
        )
        example = ProgramExample(
            example_id="proven-directional-probe",
            question="Doanh thu thay đổi bao nhiêu từ 2022 đến 2023? Tính 2023 trừ 2022.",
            context="",
            answer_text="0",
            candidates=candidates,
            program=ReasoningProgram("SUBTRACT", (0, 1)),
            answer_value="0",
        )
        direct = ProgramCandidateTrace(
            program=example.program,
            neural_score=-0.1,
            neural_rank=1,
            verification_state="VALID",
            proof_score=1.0,
            eligible=True,
            rejection_reasons=(),
            unknown_reasons=(),
            reranker_features=_features(-0.1, 1.0),
            legacy_rank=1,
            legacy_rank_score=1.0,
            legacy_winner=True,
            generation_sources=("typed_slot_synthesis",),
            training_state="VALID",
            proof_contract_state="VALID",
        )
        reversal = ProgramCandidateTrace(
            program=ReasoningProgram("SUBTRACT", (1, 0)),
            neural_score=-0.2,
            neural_rank=2,
            verification_state="INVALID",
            proof_score=0.0,
            eligible=False,
            rejection_reasons=("temporal_direction",),
            unknown_reasons=(),
            reranker_features=_features(
                -0.2, -1.0, legacy_rank=2, legacy_winner=False
            ),
            legacy_rank=2,
            legacy_rank_score=0.0,
            legacy_winner=False,
            generation_sources=("directional_reversal_probe",),
            training_state="INVALID",
            proof_contract_state="INVALID",
        )
        fake_bundle = SimpleNamespace(
            program_reranker=None,
            predict_many=lambda examples: [
                SimpleNamespace(program_search_trace=(direct, reversal))
                for _ in examples
            ],
        )

        records, audit = build_reranker_records(fake_bundle, [example], top_k=64)

        self.assertEqual(len(records), 1)
        self.assertEqual(len(records[0].candidates), 2)
        retained = records[0].candidates[1]
        self.assertEqual(retained.program_key, "SUBTRACT:1,0")
        self.assertEqual(retained.training_state, "INVALID")
        self.assertEqual(retained.positive_class, "execution_only")
        self.assertTrue(retained.supervision_only)
        self.assertFalse(retained.runtime_eligible)
        self.assertEqual(audit["invalid_candidates_retained"], 0)
        self.assertEqual(
            audit["proven_supervision_invalid_candidates_retained"], 1
        )

    def test_training_state_relaxes_only_missing_or_fuzzy_metadata(self) -> None:
        unknown = VerificationResult(
            VerificationState.UNKNOWN,
            (ConstraintResult("same_table", VerificationState.UNKNOWN),),
        )
        valid = VerificationResult(
            VerificationState.VALID,
            (ConstraintResult("operation_match", VerificationState.VALID),),
        )
        invalid = VerificationResult(
            VerificationState.INVALID,
            (ConstraintResult("unit_algebra", VerificationState.INVALID),),
        )

        self.assertEqual(
            _program_candidate_training_state(
                unknown, ("operand_role_mismatch",)
            ),
            "UNKNOWN",
        )
        self.assertEqual(
            _program_candidate_training_state(
                valid, ("missing_requested_operand_roles",)
            ),
            "UNKNOWN",
        )
        self.assertEqual(
            _program_candidate_training_state(valid, ("operation_mismatch",)),
            "INVALID",
        )
        self.assertEqual(
            _program_candidate_training_state(
                invalid, ("operand_role_mismatch",)
            ),
            "INVALID",
        )

    def test_training_support_validates_explicit_period_proof(self) -> None:
        contract = QuestionProofContract(
            required=(
                "execution_valid",
                "operation_match",
                "operand_arity_intent",
                "period_coverage",
            ),
            optional=("same_row",),
            not_applicable=(),
        )
        supported = VerificationResult(
            VerificationState.UNKNOWN,
            (
                ConstraintResult("execution_valid", VerificationState.VALID),
                ConstraintResult("operation_match", VerificationState.VALID),
                ConstraintResult("operand_arity_intent", VerificationState.VALID),
                ConstraintResult("period_coverage", VerificationState.VALID),
                ConstraintResult("same_row", VerificationState.UNKNOWN),
            ),
            proof_contract=contract,
        )
        weak = VerificationResult(
            VerificationState.UNKNOWN,
            (
                ConstraintResult("execution_valid", VerificationState.VALID),
                ConstraintResult("operation_match", VerificationState.VALID),
                ConstraintResult("operand_arity_intent", VerificationState.VALID),
                ConstraintResult("same_row", VerificationState.UNKNOWN),
            ),
            proof_contract=contract,
        )

        self.assertEqual(
            supported.training_support_state, VerificationState.VALID
        )
        self.assertEqual(
            _program_candidate_training_state(supported, ()), "VALID"
        )
        self.assertEqual(
            weak.training_support_state, VerificationState.UNKNOWN
        )

    def test_candidate_cache_round_trip_preserves_tri_state_metadata(self) -> None:
        record = RerankerExampleRecord(
            example_id="cache-row",
            operation="SELECT",
            candidates=(
                RerankerCandidateRecord(
                    _features(-0.1, 0.0),
                    "SELECT:0",
                    True,
                    True,
                    positive_class="canonical_exact",
                    generation_sources=("typed_slot_synthesis",),
                    runtime_eligible=False,
                    training_state="UNKNOWN",
                    proof_contract_state="UNKNOWN",
                    rejection_reasons=("unary_grounding_mismatch",),
                    unknown_reasons=("period_coverage",),
                    required_valid_fraction=0.75,
                    required_unknown_fraction=0.25,
                    optional_support_fraction=0.5,
                    proof_obligations=(
                        "required:period_coverage:UNKNOWN",
                        "not_applicable:unit_algebra:NOT_APPLICABLE",
                    ),
                ),
            ),
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cache.jsonl"
            write_reranker_records(path, [record])
            loaded = read_reranker_records(path)

        self.assertEqual(loaded, [record])

    def test_legacy_cache_is_upgraded_without_rerunning_v8(self) -> None:
        features = list(_features(-0.1, 0.0))
        features[FEATURE_INDEX["execution_valid"]] = 1.0
        features[FEATURE_INDEX["operation_match"]] = 1.0
        features[FEATURE_INDEX["period_coverage"]] = 1.0
        record = RerankerExampleRecord(
            "legacy-cache",
            "SELECT",
            (
                RerankerCandidateRecord(
                    tuple(features),
                    "SELECT:0",
                    True,
                    True,
                    training_state="UNKNOWN",
                ),
            ),
        )

        upgraded, audit = upgrade_reranker_records_to_supported_tristate(
            [record]
        )

        candidate = upgraded[0].candidates[0]
        self.assertEqual(candidate.training_state, "VALID")
        self.assertEqual(
            candidate.features[FEATURE_INDEX["verifier_state_valid"]], 1.0
        )
        self.assertEqual(audit["changed_candidates"], 1)

    def test_single_candidate_uses_legacy_fallback_without_training_record(self) -> None:
        example = _example("example-single", "A")
        only = ProgramCandidateTrace(
            program=ReasoningProgram("SELECT", (0,)),
            neural_score=-0.1,
            neural_rank=1,
            verification_state="UNKNOWN",
            proof_score=0.5,
            eligible=True,
            rejection_reasons=(),
            unknown_reasons=("same_table",),
            reranker_features=_features(-0.1, 0.0),
            legacy_rank=1,
            legacy_rank_score=1.0,
            legacy_winner=True,
        )
        fake_bundle = SimpleNamespace(
            program_reranker=None,
            predict_many=lambda examples: [
                SimpleNamespace(program_search_trace=(only,)) for _ in examples
            ],
        )
        records, audit = build_reranker_records(fake_bundle, [example], top_k=2)
        self.assertEqual(records, [])
        self.assertEqual(
            audit["per_operation"]["SELECT"][
                "legacy_fallback_single_candidate"
            ],
            1,
        )
        self.assertEqual(
            audit["candidate_failure_modes"]["single_candidate_correct"],
            1,
        )

    def test_grouped_split_excludes_validation_and_has_no_group_overlap(self) -> None:
        examples = [
            _example(f"{company}-{index}", company)
            for company in ("A", "B", "C", "D")
            for index in range(4)
        ]
        validation = [examples[0]]
        train, calibration, audit = split_reranker_development(
            examples, validation, calibration_fraction=0.25, seed=5
        )
        self.assertNotIn(examples[0], train)
        self.assertNotIn(examples[0], calibration)
        self.assertFalse(
            any(item.context.startswith("Công ty: A") for item in [*train, *calibration])
        )
        train_companies = {item.context.splitlines()[0] for item in train}
        calibration_companies = {
            item.context.splitlines()[0] for item in calibration
        }
        self.assertFalse(train_companies & calibration_companies)
        self.assertEqual(audit["group_overlap"], 0)

    def test_training_uses_example_level_records(self) -> None:
        def record(index: int) -> RerankerExampleRecord:
            return RerankerExampleRecord(
                example_id=f"r-{index}",
                operation="SELECT",
                candidates=(
                    RerankerCandidateRecord(
                        _features(-1.0, 1.0), "SELECT:0", True, True
                    ),
                    RerankerCandidateRecord(
                        _features(
                            -0.1,
                            -1.0,
                            legacy_rank=2,
                            legacy_winner=False,
                        ),
                        "SELECT:1",
                        False,
                        False,
                    ),
                ),
            )

        train = [record(index) for index in range(8)]
        calibration = [record(index + 20) for index in range(4)]
        bundle, report = train_program_reranker(
            train,
            calibration,
            epochs=4,
            learning_rate=0.02,
            hidden_size=8,
            dropout=0.0,
            device="cpu",
        )
        self.assertGreater(bundle.parameter_count, 0)
        self.assertGreaterEqual(report.selected_epoch, 0)
        self.assertTrue(report.ranking_regression_guard_passed)
        self.assertFalse(report.confidence_target_met)
        self.assertFalse(report.regression_guard_passed)
        self.assertTrue(
            bundle.metadata["unknown_as_automatic_negative"] is False
        )

    def test_zero_residual_preserves_legacy_symbolic_winner(self) -> None:
        model = ProgramRerankerModel(
            ProgramRerankerConfig(hidden_size=8, dropout=0.0)
        )
        bundle = ProgramRerankerBundle(
            model,
            metadata={
                "feature_schema": PROGRAM_RERANKER_FEATURE_SCHEMA,
                "override_margin": 0.0,
                "residual_scale": 0.25,
            },
        )
        rows = [
            _features(-5.0, 1.0),
            _features(
                5.0,
                -1.0,
                legacy_rank=2,
                legacy_winner=False,
            ),
        ]
        winner, _, _, overridden, _ = bundle.select(rows)
        self.assertEqual(winner, 0)
        self.assertFalse(overridden)

    def test_override_margin_blocks_unvalidated_correction(self) -> None:
        model = ProgramRerankerModel(
            ProgramRerankerConfig(hidden_size=8, dropout=0.0)
        )
        bundle = ProgramRerankerBundle(
            model,
            metadata={
                "feature_schema": PROGRAM_RERANKER_FEATURE_SCHEMA,
                "override_margin": float("inf"),
                "residual_scale": 1.0,
            },
        )
        rows = [
            _features(-1.0, 1.0),
            _features(
                1.0,
                -1.0,
                legacy_rank=2,
                legacy_winner=False,
            ),
        ]
        bundle.score_many = lambda _: ((0.0, 10.0), (0.5, 0.5))  # type: ignore[method-assign]
        winner, _, _, overridden, _ = bundle.select(rows)
        self.assertEqual(winner, 0)
        self.assertFalse(overridden)

    def test_override_margin_is_operation_specific(self) -> None:
        model = ProgramRerankerModel(
            ProgramRerankerConfig(hidden_size=8, dropout=0.0)
        )
        bundle = ProgramRerankerBundle(
            model,
            metadata={
                "feature_schema": PROGRAM_RERANKER_FEATURE_SCHEMA,
                "override_margin": 0.0,
                "override_margins_by_operation": {
                    "SELECT": 1_000_000_000.0,
                    "ADD": 0.0,
                },
                "residual_scale": 1.0,
            },
        )
        rows = [
            _features(-1.0, 1.0),
            _features(
                1.0,
                -1.0,
                legacy_rank=2,
                legacy_winner=False,
            ),
        ]
        bundle.score_many = lambda _: ((0.0, 10.0), (0.5, 0.5))  # type: ignore[method-assign]
        select_winner, _, _, select_overridden, _ = bundle.select(
            rows, operation="SELECT"
        )
        add_winner, _, _, add_overridden, _ = bundle.select(
            rows, operation="ADD"
        )
        self.assertEqual(select_winner, 0)
        self.assertFalse(select_overridden)
        self.assertEqual(add_winner, 1)
        self.assertTrue(add_overridden)

    def test_override_calibration_cannot_trade_valid_coverage_for_unknown(self) -> None:
        legacy = RerankerCandidateRecord(
            _features(-1.0, 1.0),
            "SELECT:0",
            False,
            True,
            runtime_eligible=True,
            training_state="VALID",
        )
        unknown = RerankerCandidateRecord(
            _features(
                1.0,
                0.0,
                legacy_rank=2,
                legacy_winner=False,
            ),
            "SELECT:1",
            False,
            True,
            positive_class="execution_only",
            runtime_eligible=False,
            training_state="UNKNOWN",
        )
        record = RerankerExampleRecord("coverage-guard", "SELECT", (legacy, unknown))
        bundle = ProgramRerankerBundle(
            ProgramRerankerModel(
                ProgramRerankerConfig(hidden_size=8, dropout=0.0)
            ),
            metadata={
                "feature_schema": PROGRAM_RERANKER_FEATURE_SCHEMA,
                "residual_scale": 1.0,
            },
        )
        bundle.score_many = lambda _: ((0.0, 10.0), (0.5, 0.9))  # type: ignore[method-assign]

        margin, _, _, overrides = calibrate_override_margin(bundle, [record])

        self.assertGreaterEqual(margin, 1_000_000_000.0)
        self.assertEqual(overrides, 0)

    def test_sidecar_threshold_does_not_replace_v8_fallback_threshold(self) -> None:
        model = ProgramRerankerModel(
            ProgramRerankerConfig(hidden_size=8, dropout=0.0)
        )
        reranker = ProgramRerankerBundle(
            model,
            metadata={
                "feature_schema": PROGRAM_RERANKER_FEATURE_SCHEMA,
                "confidence_thresholds_by_operation": {"SELECT": 0.20},
            },
        )
        fake_bundle = SimpleNamespace(
            metadata={
                "abstention_threshold": 0.75,
                "abstention_thresholds_by_operation": {"SELECT": 0.85},
            },
            program_reranker=None,
        )
        DocumentHRMBundle.attach_program_reranker(fake_bundle, reranker)
        fallback_prediction = SimpleNamespace(
            program=ReasoningProgram("SELECT", (0,)),
            reranker_confidence=None,
        )
        sidecar_prediction = SimpleNamespace(
            program=ReasoningProgram("SELECT", (0,)),
            reranker_confidence=0.5,
        )
        self.assertEqual(
            _abstention_threshold_for_prediction(
                fake_bundle, fallback_prediction
            ),
            0.85,
        )
        self.assertEqual(
            _abstention_threshold_for_prediction(
                fake_bundle, sidecar_prediction
            ),
            0.20,
        )

    def test_v12b_sidecar_requires_matching_candidate_runtime(self) -> None:
        reranker = ProgramRerankerBundle(
            ProgramRerankerModel(
                ProgramRerankerConfig(hidden_size=8, dropout=0.0)
            ),
            metadata={
                "feature_schema": PROGRAM_RERANKER_FEATURE_SCHEMA,
                "typed_program_synthesis_enabled": True,
                "financial_identity_catalog_enabled": True,
                "candidate_recall_v12_4_enabled": True,
            },
        )
        fake_bundle = SimpleNamespace(metadata={}, program_reranker=None)

        with self.assertRaises(DocumentHRMError):
            DocumentHRMBundle.attach_program_reranker(fake_bundle, reranker)

        fake_bundle.metadata = {
            "typed_program_synthesis_enabled": True,
            "financial_identity_catalog_enabled": True,
            "candidate_recall_v12_4_enabled": True,
        }
        DocumentHRMBundle.attach_program_reranker(fake_bundle, reranker)
        self.assertIs(fake_bundle.program_reranker, reranker)

    def test_checkpoint_hash_detects_no_read_only_change(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "v8.pt"
            path.write_bytes(b"frozen-v8")
            before = checkpoint_sha256(path)
            _ = path.read_bytes()
            after = checkpoint_sha256(path)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
