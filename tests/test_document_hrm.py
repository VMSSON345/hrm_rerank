from __future__ import annotations

import unittest
import threading
import random
from dataclasses import asdict, replace
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import numpy as np

from hrm_finance_qa.document_hrm import (
    PAIR_FEATURE_SIZE,
    DocumentHRMConfig,
    DocumentHRMError,
    DocumentHRMForward,
    DocumentHRMBundle,
    DocumentHRMState,
    DocumentHierarchicalReasoningModel,
    EncodedBatch,
    SemanticEmbeddingCache,
    _abstention_threshold_for_operation,
    _alignment_terms,
    _align_directed_operands,
    _add_select_contrastive_loss,
    _apply_symbolic_operation_constraint,
    _explicit_multi_operand_parts,
    _explicit_multi_operand_indices,
    _build_staged_pair_optimizer,
    _build_staged_pointer_optimizer,
    _build_pointer_heads_only_optimizer,
    _build_select_pointer_optimizer,
    calibrate_abstention_threshold,
    calibrate_operation_abstention_thresholds,
    _candidate_semantic_text,
    _candidate_period_role,
    _candidate_period_years,
    _candidate_pair_features,
    _combine_unary_grounding_constraints,
    _combine_segment_losses,
    _commutative_pointer_set_loss,
    _compile_symbolic_query_spec,
    _configure_pair_training_stage,
    _configure_pointer_heads_only_training,
    _configure_select_pointer_training,
    _configure_unary_pointer_training_stage,
    _hard_negative_cross_entropy,
    _load_compatible_training_bundle,
    _grounding_gate_reasons,
    _pair_loss_scale,
    _pair_program_loss,
    _precision_survives_errors,
    _targeted_pair_margin_loss,
    _targeted_unary_margin_loss,
    _verifier_guided_invalid_masks,
    _verifier_guided_pair_margin_loss,
    _training_segment_loss,
    _training_pointer_predictions,
    _training_pointer_slot_logits,
    _metric_compatible_indices,
    _multi_operand_set_ranking_loss,
    _numeric_features,
    _narrow_unary_period_by_row_order,
    _operation_pointer_logits,
    _operation_beam_for_search,
    _operand_role_parts,
    _period_compatible_indices,
    _permute_candidates,
    _prefer_reporting_entity_indices,
    _program_search_rejection_reasons,
    _question_features,
    _role_compatible_pair_indices,
    _search_symbolic_programs,
    _candidate_programs_for_search,
    _split_from_validation_ids,
    _temporal_role_compatible_indices,
    _structured_negative_indices,
    _unary_structural_negative_mask,
    evaluate_document_hrm,
    train_document_hrm,
)
from hrm_finance_qa.program_data import NumericCandidate, OPERATION_TO_ID, ProgramExample
from hrm_finance_qa.document_page_index import (
    PageIndexedChunks,
    PageIndexStats,
    PageRecord,
    _has_compatible_high_quality_ocr,
)
from hrm_finance_qa.document_sources import EvidenceChunk, OCRConfig
from hrm_finance_qa.program_data import (
    ReasoningProgram,
    execute_program_dag,
    materialize_program_dag,
)
from hrm_finance_qa.ontology import normalize_text
from hrm_finance_qa.document_reasoning_engine import (
    DocumentHRMReasoningEngine,
    _augment_financial_identity_candidates,
    _debt_share_role,
    _expanded_retrieval_question,
    _missing_requested_periods,
    _missing_requested_period_qualifiers,
    _native_structured_evidence_gap,
    _predicted_add_requires_ocr_fallback,
    _uses_derived_table_formula,
)


def _small_config(**overrides: object) -> DocumentHRMConfig:
    values: dict[str, object] = {
        "vocab_size": 1024,
        "embedding_dim": 32,
        "hidden_size": 64,
        "max_question_tokens": 6,
        "max_candidate_tokens": 5,
        "max_candidates": 8,
        "attention_heads": 4,
        "feed_forward_multiplier": 2,
        "high_cycles": 1,
        "low_steps": 1,
        "max_segments": 1,
        "minimum_halt_segments": 1,
        "input_encoder_mode": "hash",
    }
    values.update(overrides)
    return DocumentHRMConfig(**values)


def _numeric_candidate(
    index: int,
    value: str,
    row: str,
    period: str = "",
    extraction_method: str = "",
) -> NumericCandidate:
    context = f"ROW: {row}\nROW_TYPE: detail"
    if period:
        context += f"\nCOLUMN: Năm {period}\nPERIOD: {period}"
    return NumericCandidate(
        index=index,
        value=value,
        raw=value,
        context=context,
        line_number=index + 1,
        char_start=0,
        char_end=len(value),
        unit="",
        is_percent=False,
        is_year_like=False,
        relevance=1.0,
        extraction_method=extraction_method,
    )


class FixedValidationSplitTests(unittest.TestCase):
    @staticmethod
    def _example(example_id: str) -> ProgramExample:
        candidate = _numeric_candidate(0, "100", "Doanh thu", "2024")
        return ProgramExample(
            example_id=example_id,
            question="Doanh thu nam 2024 la bao nhieu?",
            context="",
            answer_text="100",
            candidates=(candidate,),
            program=ReasoningProgram("SELECT", (0,)),
            answer_value="100",
        )

    def test_explicit_validation_can_include_source_derived_synthetic_ids(self) -> None:
        train_example = self._example("synthetic-document-train")
        validation_example = self._example("synthetic-document-validation")

        train, validation = _split_from_validation_ids(
            [train_example, validation_example],
            {validation_example.example_id},
            seed=7,
            allow_synthetic_validation=True,
        )

        self.assertEqual([item.example_id for item in train], [train_example.example_id])
        self.assertEqual(
            [item.example_id for item in validation],
            [validation_example.example_id],
        )

    def test_checkpoint_validation_keeps_rejecting_synthetic_ids(self) -> None:
        example = self._example("synthetic-document-validation")

        with self.assertRaisesRegex(
            DocumentHRMError, "Fixed validation IDs are missing"
        ):
            _split_from_validation_ids(
                [example],
                {example.example_id},
                seed=7,
            )


class LazyMinerUFallbackTests(unittest.TestCase):
    def test_high_quality_tesseract_page_index_record_is_reusable(self) -> None:
        record = PageRecord(
            document_fingerprint="fingerprint",
            source_path="report.pdf",
            source_name="report.pdf",
            source_size=1,
            source_mtime_ns=1,
            page_number=1,
            native_usable=False,
            native_score=0.0,
            native_reason="scan",
            native_text="",
            extraction_method="ocr",
            searchable_text="financial table",
            ocr_state="high_quality",
        )

        self.assertTrue(
            _has_compatible_high_quality_ocr(record, OCRConfig(enabled=True))
        )

    def test_unavailable_ocr_page_index_record_is_not_reusable(self) -> None:
        record = PageRecord(
            document_fingerprint="fingerprint",
            source_path="report.pdf",
            source_name="report.pdf",
            source_size=1,
            source_mtime_ns=1,
            page_number=1,
            native_usable=False,
            native_score=0.0,
            native_reason="scan",
            native_text="",
            extraction_method="ocr",
            searchable_text="",
            ocr_state="unavailable",
        )

        self.assertFalse(
            _has_compatible_high_quality_ocr(record, OCRConfig(enabled=True))
        )

    @staticmethod
    def _chunk(method: str = "native") -> EvidenceChunk:
        return EvidenceChunk(
            chunk_id=f"report:{method}:1",
            source_path="report.pdf",
            source_name="report.pdf",
            page_start=1,
            page_end=1,
            line_start=1,
            line_end=2,
            text="Doanh thu thuần | Năm 2024 | 100",
            score=1.0,
            extraction_methods=(method,),
        )

    @staticmethod
    def _engine(confidence: float = 0.90) -> tuple[DocumentHRMReasoningEngine, Mock]:
        prediction = SimpleNamespace(
            program=ReasoningProgram("SELECT", (0,)),
            confidence=confidence,
            operation_confidence=0.90,
            segments=1,
            halt_q=1.0,
            continue_q=0.0,
            grounding_constraints_applied=(),
            neural_operation_confidence=0.90,
        )
        predict = Mock(return_value=prediction)
        bundle = SimpleNamespace(
            model=SimpleNamespace(
                config=SimpleNamespace(
                    max_candidates=8,
                    hierarchical_input_mode="strict",
                )
            ),
            metadata={"abstention_threshold": 0.80},
            predict=predict,
        )
        return DocumentHRMReasoningEngine(bundle), predict

    @staticmethod
    def _indexed(method: str) -> PageIndexedChunks:
        return PageIndexedChunks(
            chunks=(LazyMinerUFallbackTests._chunk(method),),
            stats=PageIndexStats(total_pages=2, native_pages=1),
        )

    def test_native_answer_skips_mineru_even_when_confidence_is_low(self) -> None:
        engine, predict = self._engine(confidence=0.30)
        candidate = _numeric_candidate(
            0, "100", "Doanh thu thuần", "2024", "native"
        )
        with patch(
            "hrm_finance_qa.document_reasoning_engine.discover_documents",
            return_value=[Path("report.pdf")],
        ), patch(
            "hrm_finance_qa.document_reasoning_engine.build_page_indexed_chunks",
            return_value=self._indexed("native"),
        ) as build_index, patch(
            "hrm_finance_qa.document_reasoning_engine.rank_chunks",
            side_effect=lambda _question, chunks, **_kwargs: list(chunks),
        ), patch(
            "hrm_finance_qa.document_reasoning_engine.expand_neighbor_chunks",
            side_effect=lambda ranked, _chunks, **_kwargs: list(ranked),
        ), patch(
            "hrm_finance_qa.document_reasoning_engine.merge_candidates",
            return_value=[candidate],
        ), patch(
            "hrm_finance_qa.document_reasoning_engine.execute_program",
            return_value=Decimal("100"),
        ):
            answer = engine.answer("Doanh thu thuần năm 2024?", "report.pdf")

        self.assertEqual(build_index.call_count, 1)
        self.assertFalse(build_index.call_args.kwargs["ocr_config"].enabled)
        self.assertEqual(predict.call_count, 1)
        self.assertTrue(answer.abstained)
        self.assertFalse(any("MinerU-enabled" in item for item in answer.warnings))

    def test_wrong_native_metric_retries_with_mineru(self) -> None:
        engine, predict = self._engine(confidence=0.90)
        wrong_metric = _numeric_candidate(
            0, "1174500", "Doanh thu hoạt động", "2025", "native"
        )
        requested_metric = _numeric_candidate(
            0, "173171512356", "Lợi nhuận kế toán sau thuế TNDN", "2025", "ocr"
        )
        with patch(
            "hrm_finance_qa.document_reasoning_engine.discover_documents",
            return_value=[Path("report.pdf")],
        ), patch(
            "hrm_finance_qa.document_reasoning_engine.build_page_indexed_chunks",
            side_effect=[self._indexed("native"), self._indexed("mineru")],
        ) as build_index, patch(
            "hrm_finance_qa.document_reasoning_engine.rank_chunks",
            side_effect=lambda _question, chunks, **_kwargs: list(chunks),
        ), patch(
            "hrm_finance_qa.document_reasoning_engine.expand_neighbor_chunks",
            side_effect=lambda ranked, _chunks, **_kwargs: list(ranked),
        ), patch(
            "hrm_finance_qa.document_reasoning_engine.merge_candidates",
            side_effect=[[wrong_metric], [requested_metric]],
        ), patch(
            "hrm_finance_qa.document_reasoning_engine.execute_program",
            return_value=Decimal("173171512356"),
        ):
            answer = engine.answer(
                "Lợi nhuận kế toán sau thuế TNDN năm 2025 là bao nhiêu?",
                "report.pdf",
            )

        self.assertEqual(build_index.call_count, 2)
        self.assertEqual(predict.call_count, 1)
        self.assertTrue(any("requested metric" in item for item in answer.warnings))

    def test_missing_native_candidates_retries_with_mineru(self) -> None:
        engine, predict = self._engine()
        candidate = _numeric_candidate(
            0, "100", "Doanh thu thuần", "2024", "mineru"
        )
        with patch(
            "hrm_finance_qa.document_reasoning_engine.discover_documents",
            return_value=[Path("report.pdf")],
        ), patch(
            "hrm_finance_qa.document_reasoning_engine.build_page_indexed_chunks",
            side_effect=[self._indexed("native"), self._indexed("mineru")],
        ) as build_index, patch(
            "hrm_finance_qa.document_reasoning_engine.rank_chunks",
            side_effect=lambda _question, chunks, **_kwargs: list(chunks),
        ), patch(
            "hrm_finance_qa.document_reasoning_engine.expand_neighbor_chunks",
            side_effect=lambda ranked, _chunks, **_kwargs: list(ranked),
        ), patch(
            "hrm_finance_qa.document_reasoning_engine.merge_candidates",
            side_effect=[[], [candidate]],
        ), patch(
            "hrm_finance_qa.document_reasoning_engine.execute_program",
            return_value=Decimal("100"),
        ):
            answer = engine.answer("Doanh thu thuần năm 2024?", "report.pdf")

        self.assertEqual(build_index.call_count, 2)
        configs = [call.kwargs["ocr_config"] for call in build_index.call_args_list]
        self.assertFalse(configs[0].enabled)
        self.assertTrue(configs[1].enabled)
        self.assertFalse(
            build_index.call_args_list[0].kwargs[
                "prefer_mineru_for_selected_pages"
            ]
        )
        self.assertTrue(
            build_index.call_args_list[1].kwargs[
                "prefer_mineru_for_selected_pages"
            ]
        )
        self.assertEqual(predict.call_count, 1)
        self.assertTrue(any("MinerU-enabled" in item for item in answer.warnings))

    def test_missing_native_period_retries_with_mineru(self) -> None:
        engine, predict = self._engine()
        wrong_period = _numeric_candidate(0, "90", "Doanh thu thuần", "2023")
        requested_period = _numeric_candidate(
            0, "100", "Doanh thu thuần", "2024", "mineru"
        )
        with patch(
            "hrm_finance_qa.document_reasoning_engine.discover_documents",
            return_value=[Path("report.pdf")],
        ), patch(
            "hrm_finance_qa.document_reasoning_engine.build_page_indexed_chunks",
            side_effect=[self._indexed("native"), self._indexed("mineru")],
        ) as build_index, patch(
            "hrm_finance_qa.document_reasoning_engine.rank_chunks",
            side_effect=lambda _question, chunks, **_kwargs: list(chunks),
        ), patch(
            "hrm_finance_qa.document_reasoning_engine.expand_neighbor_chunks",
            side_effect=lambda ranked, _chunks, **_kwargs: list(ranked),
        ), patch(
            "hrm_finance_qa.document_reasoning_engine.merge_candidates",
            side_effect=[[wrong_period], [requested_period]],
        ), patch(
            "hrm_finance_qa.document_reasoning_engine.execute_program",
            return_value=Decimal("100"),
        ):
            answer = engine.answer("Doanh thu thuần năm 2024?", "report.pdf")

        self.assertEqual(build_index.call_count, 2)
        self.assertEqual(predict.call_count, 1)
        self.assertFalse(answer.abstained)
        self.assertTrue(any("requested period" in item for item in answer.warnings))

    def test_page_index_extracts_from_all_selected_page_chunks(self) -> None:
        engine, _predict = self._engine()
        ranked_chunk = self._chunk("native")
        sibling_chunk = replace(
            ranked_chunk,
            chunk_id="report:native:2",
            line_start=3,
            line_end=4,
            text="Tổng tài sản | Năm 2024 | 200",
        )
        indexed = PageIndexedChunks(
            chunks=(ranked_chunk, sibling_chunk),
            stats=PageIndexStats(total_pages=1, native_pages=1),
        )
        candidate = _numeric_candidate(
            0, "200", "Tổng tài sản", "2024", "native"
        )

        with patch(
            "hrm_finance_qa.document_reasoning_engine.discover_documents",
            return_value=[Path("report.pdf")],
        ), patch(
            "hrm_finance_qa.document_reasoning_engine.build_page_indexed_chunks",
            return_value=indexed,
        ), patch(
            "hrm_finance_qa.document_reasoning_engine.rank_chunks",
            return_value=[ranked_chunk],
        ), patch(
            "hrm_finance_qa.document_reasoning_engine.expand_neighbor_chunks",
            return_value=[ranked_chunk],
        ), patch(
            "hrm_finance_qa.document_reasoning_engine.merge_candidates",
            return_value=[candidate],
        ) as merge, patch(
            "hrm_finance_qa.document_reasoning_engine.execute_program",
            return_value=Decimal("200"),
        ):
            answer = engine.answer(
                "Tổng tài sản năm 2024?",
                "report.pdf",
                ocr=False,
            )

        extracted_chunks = merge.call_args.args[1]
        self.assertEqual(
            {item.chunk_id for item in extracted_chunks},
            {ranked_chunk.chunk_id, sibling_chunk.chunk_id},
        )
        self.assertEqual(
            [item.chunk_id for item in answer.retrieved_chunks],
            [ranked_chunk.chunk_id],
        )


class StructuredCandidateEncodingTests(unittest.TestCase):
    def test_candidate_features_include_provenance_and_exact_row_match(self) -> None:
        candidate = replace(
            _numeric_candidate(0, "100", "Doanh thu thuần", "2024"),
            page=12,
            table_id="table-3-8",
            section="Kết quả kinh doanh",
            scope="consolidated",
        )

        semantic = _candidate_semantic_text(candidate, mode="context")
        features = _numeric_features(
            candidate,
            0,
            64,
            question="Doanh thu thuần năm 2024 là bao nhiêu?",
            candidates=(candidate,),
            feature_size=18,
        )

        self.assertIn("PAGE: 12", semantic)
        self.assertIn("TABLE: table-3-8", semantic)
        self.assertIn("SECTION: Kết quả kinh doanh", semantic)
        self.assertIn("SCOPE: consolidated", semantic)
        self.assertEqual(features[15], 1.0)

    def test_structured_semantic_text_keeps_fields_and_drops_cell_noise(self) -> None:
        candidate = replace(
            _numeric_candidate(0, "100", "Doanh thu thuần", "2025"),
            context=(
                "ROW: Doanh thu thuần\n"
                "ROW_TYPE: total\n"
                "COLUMN: Năm 2025\n"
                "PERIOD: 2025\n"
                "HEADER: Doanh thu dịch vụ | Năm 2025 | Năm 2024\n"
                "CELL: Doanh thu thuần | <VALUE> 100 </VALUE> | 999"
            ),
        )

        text = _candidate_semantic_text(candidate, mode="structured")

        self.assertIn("Chỉ tiêu: Doanh thu thuần", text)
        self.assertIn("Cột: Năm 2025", text)
        self.assertIn("Kỳ: 2025", text)
        self.assertNotIn("Doanh thu dịch vụ", text)
        self.assertNotIn("999", text)

    def test_structured_semantic_text_preserves_unstructured_context(self) -> None:
        candidate = replace(
            _numeric_candidate(0, "100", "unused"),
            context="CONTEXT: Doanh thu thuần\nVALUE_LINE: Giá trị <VALUE> 100 </VALUE>",
        )

        self.assertEqual(
            _candidate_semantic_text(candidate, mode="structured"),
            candidate.context,
        )

    def test_pointer_hard_negative_loss_penalizes_stronger_wrong_candidate(self) -> None:
        scores = torch.tensor([[1.0, 2.0, 0.0]])
        targets = torch.tensor([0])

        baseline = _hard_negative_cross_entropy(
            scores, targets, margin=0.5, weight=0.0
        )
        with_margin = _hard_negative_cross_entropy(
            scores, targets, margin=0.5, weight=0.25
        )

        self.assertGreater(float(with_margin), float(baseline))

    def test_verifier_guided_masks_use_invalid_but_not_unknown(self) -> None:
        candidates = (
            _numeric_candidate(0, "120", "No dai han", "2023"),
            _numeric_candidate(1, "100", "No dai han", "2022"),
            _numeric_candidate(2, "80", "Tai san dai han", "2022"),
            _numeric_candidate(3, "90", "No dai han", ""),
        )
        example = ProgramExample(
            example_id="verifier-guided-mask",
            question=(
                "No dai han thay doi bao nhieu phan tram "
                "tu 2022 den 2023?"
            ),
            context="",
            answer_text="20",
            candidates=candidates,
            program=ReasoningProgram("PERCENT_CHANGE", (0, 1)),
            answer_value="20",
        )

        _, pair_mask, gold_supported = _verifier_guided_invalid_masks(example)

        self.assertTrue(gold_supported)
        self.assertTrue(pair_mask[1][0])
        self.assertTrue(pair_mask[0][2])
        self.assertFalse(pair_mask[0][3])
        self.assertFalse(pair_mask[0][1])

    def test_verifier_guided_masks_skip_ambiguous_direction_example(self) -> None:
        candidates = (
            _numeric_candidate(0, "10", "Doanh thu", "2025"),
            _numeric_candidate(1, "5", "Chi phi", "2025"),
        )
        example = ProgramExample(
            example_id="verifier-guided-ambiguous-subtract",
            question="Chênh lệch giữa Doanh thu và Chi phí là bao nhiêu?",
            context="",
            answer_text="5",
            candidates=candidates,
            program=ReasoningProgram("SUBTRACT", (0, 1)),
            answer_value="5",
        )

        _, pair_mask, gold_supported = _verifier_guided_invalid_masks(example)

        self.assertFalse(gold_supported)
        self.assertFalse(any(any(row) for row in pair_mask))

    def test_verifier_guided_pair_loss_ignores_sum3(self) -> None:
        scores = torch.zeros(2, 9)
        scores[0, 0 * 3 + 1] = 1.0
        scores[0, 0 * 3 + 2] = 2.0
        scores[1, 0 * 3 + 2] = 2.0
        pointers = torch.tensor(
            [[0, 1, -100, -100], [0, 1, 2, -100]]
        )
        operations = torch.tensor(
            [
                OPERATION_TO_ID["SUBTRACT"],
                OPERATION_TO_ID["SUM3"],
            ]
        )
        invalid = torch.zeros(2, 3, 3, dtype=torch.bool)
        invalid[0, 0, 2] = True
        invalid[1, 0, 2] = True

        combined = _verifier_guided_pair_margin_loss(
            scores,
            pointers,
            operations,
            invalid,
            margin=0.75,
        )
        binary_only = _verifier_guided_pair_margin_loss(
            scores[:1],
            pointers[:1],
            operations[:1],
            invalid[:1],
            margin=0.75,
        )

        self.assertGreater(float(combined), 0.0)
        self.assertAlmostEqual(float(combined), float(binary_only), places=6)

    def test_commutative_set_loss_is_invariant_to_pointer_slot_order(self) -> None:
        logits = torch.tensor(
            [
                [
                    [4.0, 0.0, 3.0, 0.0, 2.0],
                    [3.0, 0.0, 4.0, 0.0, 2.0],
                    [2.0, 0.0, 3.0, 0.0, 4.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ]
        )
        operations = torch.tensor([OPERATION_TO_ID["SUM3"]])
        pointers = torch.tensor([[0, 2, 4, -1]])
        mask = torch.ones(1, 5)

        original = _commutative_pointer_set_loss(
            logits, operations, pointers, mask
        )
        permuted = _commutative_pointer_set_loss(
            logits[:, [2, 0, 1, 3]], operations, pointers, mask
        )

        self.assertAlmostEqual(float(original), float(permuted), places=6)

    def test_multi_operand_set_loss_requires_every_gold_member(self) -> None:
        gold = torch.tensor([0, 2, 4])
        mask = torch.ones(5)
        complete = torch.tensor([4.0, 0.0, 4.0, 0.0, 4.0])
        missing_one = torch.tensor([4.0, 5.0, 4.0, 0.0, 0.0])

        complete_loss = _multi_operand_set_ranking_loss(complete, gold, mask)
        incomplete_loss = _multi_operand_set_ranking_loss(
            missing_one, gold, mask
        )

        self.assertLess(float(complete_loss), float(incomplete_loss))

    def test_pair_program_loss_accepts_any_gold_sum_pair(self) -> None:
        operations = torch.tensor([OPERATION_TO_ID["SUM4"]])
        pointers = torch.tensor([[0, 1, 2, 3]])
        valid_pair = torch.zeros(1, 25)
        valid_pair[0, 2 * 5 + 3] = 6.0
        wrong_pair = torch.zeros(1, 25)
        wrong_pair[0, 3 * 5 + 4] = 6.0

        valid_loss = _pair_program_loss(
            valid_pair,
            pointers,
            operations,
            hard_negative_weight=0.0,
        )
        wrong_loss = _pair_program_loss(
            wrong_pair,
            pointers,
            operations,
            hard_negative_weight=0.0,
        )

        self.assertLess(float(valid_loss), float(wrong_loss))

    def test_commutative_training_prediction_recovers_full_sum_set(self) -> None:
        pointer_logits = torch.zeros(1, 4, 5)
        pointer_logits[:, :, 0] = 5.0
        pointer_logits[:, :, 2] = 4.0
        pointer_logits[:, :, 4] = 3.0
        pair_logits = torch.full((1, 10, 5, 5), -10.0)
        pair_logits[0, OPERATION_TO_ID["SUM3"], 1, 3] = 10.0
        output = DocumentHRMForward(
            state=DocumentHRMState(
                high=torch.zeros(1, 1, 64),
                low=torch.zeros(1, 1, 64),
            ),
            operation_logits=torch.zeros(1, 10),
            pointer_logits=pointer_logits,
            q_values=torch.zeros(1, 2),
            pair_logits=pair_logits,
        )

        predicted = _training_pointer_predictions(
            output,
            torch.tensor([OPERATION_TO_ID["SUM3"]]),
            torch.tensor([[0, 2, 4, -1]]),
        )

        self.assertEqual(predicted[0, :3].tolist(), [0, 2, 4])

    def test_explicit_sum_prefers_exact_row_over_overlapping_component(self) -> None:
        candidates = (
            _numeric_candidate(
                0, "100", "Tiền và các khoản tương đương tiền", "cuoi nam"
            ),
            _numeric_candidate(1, "40", "Tiền", "cuoi nam"),
            _numeric_candidate(
                2, "60", "Các khoản tương đương tiền", "cuoi nam"
            ),
            _numeric_candidate(
                3, "80", "III.Các khoản phải thu ngắn hạn", "cuoi nam"
            ),
        )

        selected = _explicit_multi_operand_indices(
            (
                "Tính tổng số học của Tiền và các khoản tương đương tiền, "
                "Tiền và III.Các khoản phải thu ngắn hạn tại cuoi nam là bao nhiêu?"
            ),
            "SUM3",
            candidates,
        )

        self.assertEqual(selected, (0, 1, 3))


class SymbolicGroundingConstraintTests(unittest.TestCase):
    def test_operation_beam_keeps_bounded_neural_top_k(self) -> None:
        logits = torch.full((1, 10), -8.0)
        logits[0, OPERATION_TO_ID["SELECT"]] = 5.0
        logits[0, OPERATION_TO_ID["SUBTRACT"]] = 4.8
        logits[0, OPERATION_TO_ID["DIVIDE"]] = 4.4
        output = DocumentHRMForward(
            state=DocumentHRMState(
                high=torch.zeros(1, 1, 64), low=torch.zeros(1, 1, 64)
            ),
            operation_logits=logits,
            pointer_logits=torch.zeros(1, 4, 2),
            q_values=torch.zeros(1, 2),
        )

        operations = _operation_beam_for_search(
            output, 0, "SELECT", limit=3
        )

        self.assertEqual(
            operations, ("SELECT", "SUBTRACT", "DIVIDE")
        )

    def test_pair_generation_unions_joint_and_slot_pointer_beams(self) -> None:
        count = 12
        operation_logits = torch.full((1, 10), -5.0)
        operation_logits[0, OPERATION_TO_ID["ADD"]] = 5.0
        pointer_logits = torch.full((1, 4, count), -5.0)
        pointer_logits[0, 0, 10] = 9.0
        pointer_logits[0, 1, 11] = 9.0
        pair_logits = torch.full((1, 10, count, count), -10.0)
        pair_logits[0, OPERATION_TO_ID["ADD"]].flatten()[:64] = 8.0
        output = DocumentHRMForward(
            state=DocumentHRMState(
                high=torch.zeros(1, 1, 64), low=torch.zeros(1, 1, 64)
            ),
            operation_logits=operation_logits,
            pointer_logits=pointer_logits,
            q_values=torch.zeros(1, 2),
            pair_logits=pair_logits,
        )
        candidates = tuple(
            _numeric_candidate(
                index, str(index + 1), f"Chi tieu {index}", "2024", "native"
            )
            for index in range(count)
        )

        programs = _candidate_programs_for_search(
            output,
            0,
            "ADD",
            "Tinh tong",
            candidates,
            candidate_recall_v12_4_enabled=True,
        )

        self.assertIn(ReasoningProgram("ADD", (10, 11)), programs)

    def test_unanchored_rejected_baseline_returns_audit_trace(self) -> None:
        operation_logits = torch.full((1, 10), -5.0)
        operation_logits[0, OPERATION_TO_ID["ADD"]] = 5.0
        output = DocumentHRMForward(
            state=DocumentHRMState(
                high=torch.zeros(1, 1, 64), low=torch.zeros(1, 1, 64)
            ),
            operation_logits=operation_logits,
            pointer_logits=torch.zeros(1, 4, 2),
            q_values=torch.zeros(1, 2),
            pair_logits=torch.zeros(1, 10, 2, 2),
        )
        candidates = (
            _numeric_candidate(0, "60", "No ngan han", "2024", "native"),
            _numeric_candidate(1, "40", "No dai han", "2024", "native"),
        )

        result = _search_symbolic_programs(
            output,
            0,
            "Ket qua la bao nhieu?",
            candidates,
            ReasoningProgram("ADD", (0, 1)),
            typed_synthesis_enabled=True,
            financial_identity_catalog_enabled=False,
            candidate_recall_v12_4_enabled=True,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.program, ReasoningProgram("ADD", (0, 1)))
        self.assertIn("trace_complete_fallback", result.constraints_applied)
        self.assertTrue(result.trace)
        self.assertFalse(result.trace[0].eligible)
        self.assertIn(
            "missing_explicit_addition_intent",
            result.trace[0].rejection_reasons,
        )

    def test_semantic_cache_loads_only_requested_vectors(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = SemanticEmbeddingCache(
                "test-model", 384, cache_dir=root
            )
            wanted_text = "Doanh thu thuần năm 2025"
            wanted_key = first._cache_key(wanted_text)
            unused_key = first._cache_key("unused")
            wanted = np.arange(384, dtype=np.float32)
            unused = np.ones(384, dtype=np.float32)
            first._database.executemany(
                "INSERT INTO embeddings(model, cache_key, vector) VALUES (?, ?, ?)",
                (
                    ("test-model", wanted_key, wanted.tobytes()),
                    ("test-model", unused_key, unused.tobytes()),
                ),
            )
            first._database.commit()
            first._database.close()

            cache = SemanticEmbeddingCache("test-model", 384, cache_dir=root)
            self.assertEqual(cache._vectors, {})

            encoded = cache.encode((wanted_text,))

            self.assertEqual(tuple(encoded.shape), (1, 384))
            self.assertIn(wanted_key, cache._vectors)
            self.assertNotIn(unused_key, cache._vectors)
            cache._database.close()

    def test_semantic_cache_can_read_from_an_inference_thread(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cache = SemanticEmbeddingCache("test-model", 384, cache_dir=root)
            text = "Doanh thu thuần năm 2024"
            key = cache._cache_key(text)
            vector = np.arange(384, dtype=np.float32)
            cache._database.execute(
                "INSERT INTO embeddings(model, cache_key, vector) VALUES (?, ?, ?)",
                ("test-model", key, vector.tobytes()),
            )
            cache._database.commit()
            result: list[np.ndarray] = []
            errors: list[BaseException] = []

            def encode_in_thread() -> None:
                try:
                    result.append(cache.encode((text,)))
                except BaseException as exc:  # pragma: no cover - asserted below
                    errors.append(exc)

            worker = threading.Thread(target=encode_in_thread)
            worker.start()
            worker.join(timeout=5.0)

            self.assertFalse(worker.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(tuple(result[0].shape), (1, 384))
            cache._database.close()

    def test_grounded_confidence_is_versioned_and_recalibratable(self) -> None:
        model = DocumentHierarchicalReasoningModel(
            _small_config(pointer_conditioning_mode="joint_pair")
        )
        current = _numeric_candidate(
            0, "200", "Doanh thu thuần", "2025", "mineru"
        )
        prior = _numeric_candidate(
            1, "100", "Doanh thu thuần", "2024", "mineru"
        )
        operation_logits = torch.full((1, 10), -5.0)
        operation_logits[0, 0] = 5.0  # Neural prediction: SELECT.
        pair_logits = torch.zeros(1, 10, 2, 2)
        output = DocumentHRMForward(
            state=DocumentHRMState(
                high=torch.zeros(1, 1, 64), low=torch.zeros(1, 1, 64)
            ),
            operation_logits=operation_logits,
            pointer_logits=torch.zeros(1, 4, 2),
            q_values=torch.zeros(1, 2),
            pair_logits=pair_logits,
        )
        question = (
            "Doanh thu thuần thay đổi bao nhiêu phần trăm "
            "từ năm 2024 đến năm 2025?"
        )

        legacy = DocumentHRMBundle(model, metadata={})._decode_prediction(
            output, 0, 1, question=question, candidates=(current, prior)
        )
        grounded = DocumentHRMBundle(
            model, metadata={"confidence_schema": "grounded-constraints-v2"}
        )._decode_prediction(
            output, 0, 1, question=question, candidates=(current, prior)
        )

        self.assertEqual(legacy.program.operation, "PERCENT_CHANGE")
        self.assertEqual(grounded.program.operation, "PERCENT_CHANGE")
        self.assertLess(legacy.operation_confidence, 0.01)
        self.assertEqual(grounded.operation_confidence, 1.0)
        self.assertGreater(grounded.confidence, legacy.confidence)
        self.assertIn(
            "operation_wording", grounded.grounding_constraints_applied
        )

    def test_question_features_separate_percent_change_from_ratio(self) -> None:
        change = _question_features(
            "Doanh thu thay đổi bao nhiêu phần trăm từ năm 2024 đến năm 2025?"
        )
        ratio = _question_features(
            "Nợ phải trả chiếm bao nhiêu phần trăm tổng tài sản năm 2025?"
        )

        self.assertEqual(change[:2], [0.0, 1.0])
        self.assertEqual(ratio[:2], [1.0, 0.0])

    def test_question_features_separate_add_from_total_asset_select(self) -> None:
        addition = _question_features(
            "Tính tổng Tổng tài sản và Nợ phải trả tại năm 2024 là bao nhiêu?"
        )
        selection = _question_features(
            "Tổng tài sản tại năm 2024 là bao nhiêu?"
        )
        company_selection = _question_features(
            "Doanh thu của Tổng Công ty ABC trong năm 2024 là bao nhiêu?"
        )
        disclosed_total_selection = _question_features(
            "Tính tổng giá trị các khoản ký cược cuối năm 2024."
        )
        canonical_addition = _question_features(
            "Tổng Doanh thu thuần và Giá vốn hàng bán tại 2024 là bao nhiêu?"
        )
        single_metric_with_and = _question_features(
            "Tổng doanh thu bán hàng và cung cấp dịch vụ trong năm 2024 là bao nhiêu?"
        )

        self.assertEqual(addition[2], 1.0)
        self.assertEqual(addition[7], 0.0)
        self.assertEqual(selection[2], 0.0)
        self.assertEqual(selection[7], 1.0)
        self.assertEqual(company_selection[2], 0.0)
        self.assertEqual(disclosed_total_selection[2], 0.0)
        self.assertEqual(canonical_addition[2], 1.0)
        self.assertEqual(single_metric_with_and[2], 0.0)

    def test_explicit_add_frame_overrides_select_operation(self) -> None:
        question = (
            "Tổng Doanh thu thuần và Giá vốn hàng bán tại 2024 là bao nhiêu?"
        )
        ambiguous_single_metric = (
            "Tổng doanh thu bán hàng và cung cấp dịch vụ trong năm 2024 "
            "là bao nhiêu?"
        )

        self.assertEqual(
            _apply_symbolic_operation_constraint(question, "SELECT"), "ADD"
        )
        self.assertEqual(
            _apply_symbolic_operation_constraint(
                ambiguous_single_metric, "SELECT"
            ),
            "SELECT",
        )

    def test_single_metric_single_year_lookup_overrides_neural_add(self) -> None:
        variants = (
            "Doanh thu thuần năm 2023",
            "Doanh thu thuần năm 2023 là bao nhiêu?",
            "Cho biết doanh thu thuần 2023.",
        )

        for question in variants:
            with self.subTest(question=question):
                self.assertEqual(
                    _apply_symbolic_operation_constraint(question, "ADD"),
                    "SELECT",
                )

    def test_absolute_change_planner_sets_subtract_and_from_to_order(self) -> None:
        question = (
            "No dai han thay doi bao nhieu tu nam 2022 den nam 2023?"
        )
        current = _numeric_candidate(
            0, "1707170000", "No dai han", "2023", "native"
        )
        prior = _numeric_candidate(
            1, "0", "No dai han", "2022", "native"
        )

        self.assertEqual(
            _apply_symbolic_operation_constraint(
                question,
                "PERCENT_CHANGE",
            ),
            "SUBTRACT",
        )
        self.assertEqual(
            _align_directed_operands(
                question,
                "SUBTRACT",
                (1, 0),
                (current, prior),
            ),
            (0, 1),
        )

    def test_pair_decoder_never_reuses_same_candidate_pointer(self) -> None:
        model = DocumentHierarchicalReasoningModel(_small_config())
        operation_logits = torch.full((1, len(OPERATION_TO_ID)), -5.0)
        operation_logits[0, OPERATION_TO_ID["PERCENT_CHANGE"]] = 5.0
        pair_logits = torch.zeros(1, len(OPERATION_TO_ID), 2, 2)
        pair_logits[
            0,
            OPERATION_TO_ID["PERCENT_CHANGE"],
            0,
            0,
        ] = 20.0
        output = DocumentHRMForward(
            state=DocumentHRMState(
                high=torch.zeros(1, 1, 64),
                low=torch.zeros(1, 1, 64),
            ),
            operation_logits=operation_logits,
            pointer_logits=torch.zeros(1, 4, 2),
            q_values=torch.zeros(1, 2),
            pair_logits=pair_logits,
        )
        candidates = (
            _numeric_candidate(
                0, "295481188173", "Doanh thu thuan", "2023", "native"
            ),
            _numeric_candidate(
                1, "313255998102", "Doanh thu thuan", "2022", "native"
            ),
        )

        prediction = DocumentHRMBundle(model, metadata={})._decode_prediction(
            output,
            0,
            1,
            candidates=candidates,
        )

        self.assertEqual(prediction.program.operation, "PERCENT_CHANGE")
        self.assertEqual(len(set(prediction.program.operands)), 2)

    def test_single_metric_lookup_decodes_one_period_grounded_operand(self) -> None:
        model = DocumentHierarchicalReasoningModel(_small_config())
        operation_logits = torch.full((1, 10), -5.0)
        operation_logits[0, OPERATION_TO_ID["ADD"]] = 5.0
        pointer_logits = torch.zeros(1, 4, 2)
        pointer_logits[0, 0, 1] = 8.0
        output = DocumentHRMForward(
            state=DocumentHRMState(
                high=torch.zeros(1, 1, 64), low=torch.zeros(1, 1, 64)
            ),
            operation_logits=operation_logits,
            pointer_logits=pointer_logits,
            q_values=torch.zeros(1, 2),
        )
        current = _numeric_candidate(
            0, "5407553257", "Doanh thu thuần", "2023", "native"
        )
        prior = _numeric_candidate(
            1, "73112", "Doanh thu thuần", "2022", "native"
        )

        prediction = DocumentHRMBundle(
            model, metadata={"confidence_schema": "grounded-constraints-v2"}
        )._decode_prediction(
            output,
            0,
            1,
            question="Doanh thu thuần năm 2023",
            candidates=(current, prior),
        )

        self.assertEqual(prediction.program, ReasoningProgram("SELECT", (0,)))
        self.assertIn(
            "operation_wording", prediction.grounding_constraints_applied
        )
        self.assertIn("period", prediction.grounding_constraints_applied)

    def test_symbolic_program_search_reranks_exact_metric_over_neural_top1(
        self,
    ) -> None:
        operation_logits = torch.full((1, 10), -5.0)
        operation_logits[0, OPERATION_TO_ID["SELECT"]] = 5.0
        pointer_logits = torch.zeros(1, 4, 3)
        pointer_logits[0, 0, 0] = 9.0
        pointer_logits[0, 0, 1] = 1.0
        output = DocumentHRMForward(
            state=DocumentHRMState(
                high=torch.zeros(1, 1, 64), low=torch.zeros(1, 1, 64)
            ),
            operation_logits=operation_logits,
            pointer_logits=pointer_logits,
            q_values=torch.zeros(1, 2),
        )
        candidates = (
            _numeric_candidate(
                0,
                "900",
                "Doanh thu thuan tu ban hang va cung cap dich vu",
                "2023",
                "native",
            ),
            _numeric_candidate(
                1, "800", "Doanh thu thuan", "2023", "native"
            ),
            _numeric_candidate(
                2, "700", "Doanh thu thuan", "2022", "native"
            ),
        )
        question = "Doanh thu thuan nam 2023 la bao nhieu?"

        result = _search_symbolic_programs(
            output,
            0,
            question,
            candidates,
            ReasoningProgram("SELECT", (0,)),
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.program, ReasoningProgram("SELECT", (1,)))
        self.assertGreater(result.candidates_considered, 1)
        self.assertIn("symbolic_program_search", result.constraints_applied)

    def test_symbolic_program_search_reranks_pair_roles_before_execution(
        self,
    ) -> None:
        operation_logits = torch.full((1, 10), -5.0)
        operation_logits[0, OPERATION_TO_ID["ADD"]] = 5.0
        pointer_logits = torch.zeros(1, 4, 4)
        pair_logits = torch.zeros(1, 10, 4, 4)
        add_id = OPERATION_TO_ID["ADD"]
        pair_logits[0, add_id, 0, 2] = 9.0
        pair_logits[0, add_id, 0, 1] = 1.0
        output = DocumentHRMForward(
            state=DocumentHRMState(
                high=torch.zeros(1, 1, 64), low=torch.zeros(1, 1, 64)
            ),
            operation_logits=operation_logits,
            pointer_logits=pointer_logits,
            q_values=torch.zeros(1, 2),
            pair_logits=pair_logits,
        )
        candidates = (
            _numeric_candidate(
                0, "1000", "Doanh thu thuan", "2023", "native"
            ),
            _numeric_candidate(
                1, "600", "Gia von hang ban", "2023", "native"
            ),
            _numeric_candidate(
                2,
                "650",
                "Gia von hang ban va chi phi quan ly",
                "2023",
                "native",
            ),
            _numeric_candidate(
                3, "400", "Loi nhuan gop", "2023", "native"
            ),
        )
        question = (
            "Tong Doanh thu thuan va Gia von hang ban tai nam 2023 "
            "la bao nhieu?"
        )

        result = _search_symbolic_programs(
            output,
            0,
            question,
            candidates,
            ReasoningProgram("ADD", (0, 2)),
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.program, ReasoningProgram("ADD", (0, 1)))

    def test_financial_identity_recovers_total_when_direct_row_is_missing(self) -> None:
        operation_logits = torch.full((1, 10), -5.0)
        operation_logits[0, OPERATION_TO_ID["SELECT"]] = 5.0
        pointer_logits = torch.zeros(1, 4, 3)
        pointer_logits[0, 0, 2] = 9.0
        pair_logits = torch.zeros(1, 10, 3, 3)
        pair_logits[0, OPERATION_TO_ID["ADD"], 1, 2] = 9.0
        output = DocumentHRMForward(
            state=DocumentHRMState(
                high=torch.zeros(1, 1, 64), low=torch.zeros(1, 1, 64)
            ),
            operation_logits=operation_logits,
            pointer_logits=pointer_logits,
            q_values=torch.zeros(1, 2),
            pair_logits=pair_logits,
        )
        candidates = (
            _numeric_candidate(0, "60", "No ngan han", "2024", "native"),
            _numeric_candidate(1, "40", "No dai han", "2024", "native"),
            _numeric_candidate(2, "900", "Doanh thu thuan", "2024", "native"),
        )

        result = _search_symbolic_programs(
            output,
            0,
            "Tong no phai tra nam 2024 la bao nhieu?",
            candidates,
            ReasoningProgram("SELECT", (2,)),
            financial_identity_catalog_enabled=True,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.program, ReasoningProgram("ADD", (0, 1)))
        self.assertIn("financial_identity_catalog", result.constraints_applied)

    def test_financial_identity_prefers_direct_disclosed_total(self) -> None:
        operation_logits = torch.full((1, 10), -5.0)
        operation_logits[0, OPERATION_TO_ID["ADD"]] = 5.0
        pointer_logits = torch.zeros(1, 4, 3)
        pointer_logits[0, 0, 0] = 9.0
        pair_logits = torch.zeros(1, 10, 3, 3)
        pair_logits[0, OPERATION_TO_ID["ADD"], 1, 2] = 9.0
        output = DocumentHRMForward(
            state=DocumentHRMState(
                high=torch.zeros(1, 1, 64), low=torch.zeros(1, 1, 64)
            ),
            operation_logits=operation_logits,
            pointer_logits=pointer_logits,
            q_values=torch.zeros(1, 2),
            pair_logits=pair_logits,
        )
        candidates = (
            _numeric_candidate(0, "100", "Tong no phai tra", "2024", "native"),
            _numeric_candidate(1, "60", "No ngan han", "2024", "native"),
            _numeric_candidate(2, "40", "No dai han", "2024", "native"),
        )

        result = _search_symbolic_programs(
            output,
            0,
            "Tong no phai tra nam 2024 la bao nhieu?",
            candidates,
            ReasoningProgram("ADD", (1, 2)),
            financial_identity_catalog_enabled=True,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.program, ReasoningProgram("SELECT", (0,)))
        baseline_trace = next(
            item
            for item in result.trace
            if item.program == ReasoningProgram("ADD", (1, 2))
        )
        self.assertNotIn(
            "financial_identity_mismatch",
            baseline_trace.rejection_reasons,
        )

    def test_financial_identity_component_check_requires_addition(self) -> None:
        operation_logits = torch.full((1, 10), -5.0)
        operation_logits[0, OPERATION_TO_ID["SELECT"]] = 5.0
        pointer_logits = torch.zeros(1, 4, 4)
        pointer_logits[0, 0, 3] = 9.0
        pair_logits = torch.zeros(1, 10, 4, 4)
        pair_logits[0, OPERATION_TO_ID["ADD"], 1, 2] = 9.0
        output = DocumentHRMForward(
            state=DocumentHRMState(
                high=torch.zeros(1, 1, 64), low=torch.zeros(1, 1, 64)
            ),
            operation_logits=operation_logits,
            pointer_logits=pointer_logits,
            q_values=torch.zeros(1, 2),
            pair_logits=pair_logits,
        )
        candidates = (
            _numeric_candidate(
                0, "100", "Tong no phai tra", "2026", "native"
            ),
            _numeric_candidate(1, "60", "No ngan han", "2026", "native"),
            _numeric_candidate(2, "40", "No dai han", "2026", "native"),
            _numeric_candidate(
                3, "900", "Doanh thu thuan", "2026", "native"
            ),
        )

        questions = (
            (
                "Tong no phai tra nam 2026 la bao nhieu? "
                "Hay liet ke no ngan han va no dai han de kiem tra."
            ),
            (
                "Tong no phai tra nam 2026 bao gom no ngan han va no dai han "
                "la bao nhieu?"
            ),
        )
        for question in questions:
            with self.subTest(question=question):
                result = _search_symbolic_programs(
                    output,
                    0,
                    question,
                    candidates,
                    ReasoningProgram("SELECT", (3,)),
                    financial_identity_catalog_enabled=True,
                    candidate_recall_v12_4_enabled=True,
                )

                self.assertIsNotNone(result)
                assert result is not None
                self.assertEqual(
                    result.program,
                    ReasoningProgram("ADD", (1, 2)),
                    msg=result,
                )
                selected_trace = next(
                    item
                    for item in result.trace
                    if item.program == result.program
                )
                self.assertIn(
                    "financial_identity:balance.total_liabilities",
                    selected_trace.generation_sources,
                )

    def test_financial_identity_current_ratio_requires_division(self) -> None:
        operation_logits = torch.full((1, 10), -5.0)
        operation_logits[0, OPERATION_TO_ID["SELECT"]] = 5.0
        pointer_logits = torch.zeros(1, 4, 4)
        pointer_logits[0, 0, 0] = 9.0
        pair_logits = torch.zeros(1, 10, 4, 4)
        pair_logits[0, OPERATION_TO_ID["DIVIDE"], 1, 2] = 9.0
        output = DocumentHRMForward(
            state=DocumentHRMState(
                high=torch.zeros(1, 1, 64), low=torch.zeros(1, 1, 64)
            ),
            operation_logits=operation_logits,
            pointer_logits=pointer_logits,
            q_values=torch.zeros(1, 2),
            pair_logits=pair_logits,
        )
        candidates = (
            replace(
                _numeric_candidate(
                    0,
                    "3",
                    "He so thanh toan hien hanh",
                    "2026",
                    "native",
                ),
                unit="x",
            ),
            replace(
                _numeric_candidate(
                    1, "600", "Tai san ngan han", "2026", "native"
                ),
                unit="VND",
            ),
            replace(
                _numeric_candidate(
                    2, "200", "No ngan han", "2026", "native"
                ),
                unit="VND",
            ),
            _numeric_candidate(
                3, "900", "Doanh thu thuan", "2026", "native"
            ),
        )

        result = _search_symbolic_programs(
            output,
            0,
            (
                "Tinh he so thanh toan hien hanh (Current Ratio) nam 2026 "
                "va cho biet doanh nghiep co du kha nang tra no ngan han khong?"
            ),
            candidates,
            ReasoningProgram("SELECT", (0,)),
            financial_identity_catalog_enabled=True,
            candidate_recall_v12_4_enabled=True,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(
            result.program, ReasoningProgram("DIVIDE", (1, 2)), msg=result
        )
        selected_trace = next(
            item for item in result.trace if item.program == result.program
        )
        self.assertIn(
            "financial_identity:ratio.current",
            selected_trace.generation_sources,
        )

    def test_same_period_balance_difference_uses_two_requested_roles(self) -> None:
        operation_logits = torch.full((1, 10), -5.0)
        operation_logits[0, OPERATION_TO_ID["SUBTRACT"]] = 5.0
        pointer_logits = torch.zeros(1, 4, 4)
        pair_logits = torch.zeros(1, 10, 4, 4)
        pair_logits[0, OPERATION_TO_ID["SUBTRACT"], 0, 2] = 9.0
        output = DocumentHRMForward(
            state=DocumentHRMState(
                high=torch.zeros(1, 1, 64), low=torch.zeros(1, 1, 64)
            ),
            operation_logits=operation_logits,
            pointer_logits=pointer_logits,
            q_values=torch.zeros(1, 2),
            pair_logits=pair_logits,
        )
        candidates = (
            _numeric_candidate(0, "1000", "Tong tai san", "2026", "native"),
            _numeric_candidate(
                1, "1000", "Tong nguon von", "2026", "native"
            ),
            _numeric_candidate(2, "900", "Tong tai san", "2025", "native"),
            _numeric_candidate(
                3, "900", "Tong nguon von", "2025", "native"
            ),
        )

        result = _search_symbolic_programs(
            output,
            0,
            (
                "Tong tai san tinh den ngay 31/12/2026 co bang Tong nguon von "
                "khong? Neu lech thi chenh lech la bao nhieu?"
            ),
            candidates,
            ReasoningProgram("SUBTRACT", (0, 2)),
            candidate_recall_v12_4_enabled=True,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(
            result.program, ReasoningProgram("SUBTRACT", (0, 1)), msg=result
        )

    def test_same_period_balance_rejects_equal_change_column_distractors(self) -> None:
        operation_logits = torch.full((1, 10), -5.0)
        operation_logits[0, OPERATION_TO_ID["SUBTRACT"]] = 5.0
        pointer_logits = torch.zeros(1, 4, 4)
        pair_logits = torch.zeros(1, 10, 4, 4)
        pair_logits[0, OPERATION_TO_ID["SUBTRACT"], 2, 3] = 12.0
        output = DocumentHRMForward(
            state=DocumentHRMState(
                high=torch.zeros(1, 1, 64), low=torch.zeros(1, 1, 64)
            ),
            operation_logits=operation_logits,
            pointer_logits=pointer_logits,
            q_values=torch.zeros(1, 2),
            pair_logits=pair_logits,
        )
        candidates = (
            replace(
                _numeric_candidate(
                    0, "1000", "Tong tai san", "2026", "native"
                ),
                unit="VND",
            ),
            replace(
                _numeric_candidate(
                    1, "1000", "Tong nguon von", "2026", "native"
                ),
                unit="VND",
            ),
            replace(
                _numeric_candidate(
                    2, "-0.7", "Tong tai san", "", "native"
                ),
                unit="%",
                is_percent=True,
                column_label="% tang/(giam)",
            ),
            replace(
                _numeric_candidate(
                    3, "-0.7", "Tong nguon von", "", "native"
                ),
                unit="%",
                is_percent=True,
                column_label="% tang/(giam)",
            ),
        )

        result = _search_symbolic_programs(
            output,
            0,
            (
                "Tong tai san nam 2026 co bang Tong nguon von khong? "
                "Neu lech thi chenh lech la bao nhieu?"
            ),
            candidates,
            ReasoningProgram("SUBTRACT", (2, 3)),
            candidate_recall_v12_4_enabled=True,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(
            result.program, ReasoningProgram("SUBTRACT", (0, 1)), msg=result
        )
        distractor = next(
            item
            for item in result.trace
            if item.program == ReasoningProgram("SUBTRACT", (2, 3))
        )
        self.assertIn("operand_role_mismatch", distractor.rejection_reasons)

    def test_symbolic_query_spec_is_structural_not_company_specific(self) -> None:
        candidates = (
            _numeric_candidate(
                0, "100", "Chi phi tai chinh", "2024", "native"
            ),
        )

        spec = _compile_symbolic_query_spec(
            "Chi phi tai chinh nam 2024 la bao nhieu?",
            candidates,
        )

        self.assertEqual(spec.required_operation, "SELECT")
        self.assertEqual(spec.requested_years, ("2024",))
        self.assertTrue(spec.has_structured_evidence)

    def test_typed_synthesis_recovers_unseen_extraction_layout(self) -> None:
        operation_logits = torch.full((1, 10), -5.0)
        operation_logits[0, OPERATION_TO_ID["SELECT"]] = 5.0
        pointer_logits = torch.zeros(1, 4, 2)
        pointer_logits[0, 0, 0] = 8.0
        output = DocumentHRMForward(
            state=DocumentHRMState(
                high=torch.zeros(1, 1, 64), low=torch.zeros(1, 1, 64)
            ),
            operation_logits=operation_logits,
            pointer_logits=pointer_logits,
            q_values=torch.zeros(1, 2),
        )
        candidates = (
            replace(
                _numeric_candidate(
                    0, "49", "Doanh thu thuan", "2022", "native"
                ),
                extraction_method="unseen-layout-parser",
            ),
            replace(
                _numeric_candidate(
                    1, "40", "Doanh thu thuan", "2023", "native"
                ),
                extraction_method="unseen-layout-parser",
            ),
        )
        question = "Doanh thu thuan nam 2023 la bao nhieu?"

        legacy = _search_symbolic_programs(
            output,
            0,
            question,
            candidates,
            ReasoningProgram("SELECT", (0,)),
            typed_synthesis_enabled=False,
        )
        synthesized = _search_symbolic_programs(
            output,
            0,
            question,
            candidates,
            ReasoningProgram("SELECT", (0,)),
            typed_synthesis_enabled=True,
        )

        self.assertIsNone(legacy)
        self.assertIsNotNone(synthesized)
        assert synthesized is not None
        self.assertEqual(
            synthesized.program, ReasoningProgram("SELECT", (1,))
        )
        gold_trace = next(
            item
            for item in synthesized.trace
            if item.program == ReasoningProgram("SELECT", (1,))
        )
        self.assertIn("typed_slot_synthesis", gold_trace.generation_sources)
        self.assertFalse(
            any(
                "neural_structural_beam" in item.generation_sources
                for item in synthesized.trace
            )
        )

    def test_alignment_ignores_serialized_context_field_names(self) -> None:
        terms = _alignment_terms(
            "CONTEXT: Ha Dong VALUE_LINE: Du an so 89 Phung Hung"
        )

        self.assertNotIn("context", terms)
        self.assertNotIn("value", terms)
        self.assertNotIn("line", terms)
        self.assertIn("dong", terms)
        self.assertIn("phung", terms)

    def test_average_search_keeps_both_periods_on_the_same_metric_row(
        self,
    ) -> None:
        operation_logits = torch.full((1, 10), -5.0)
        operation_logits[0, OPERATION_TO_ID["AVERAGE"]] = 5.0
        pointer_logits = torch.zeros(1, 4, 3)
        pair_logits = torch.zeros(1, 10, 3, 3)
        average_id = OPERATION_TO_ID["AVERAGE"]
        pair_logits[0, average_id, 0, 1] = 9.0
        pair_logits[0, average_id, 1, 2] = 1.0
        output = DocumentHRMForward(
            state=DocumentHRMState(
                high=torch.zeros(1, 1, 64), low=torch.zeros(1, 1, 64)
            ),
            operation_logits=operation_logits,
            pointer_logits=pointer_logits,
            q_values=torch.zeros(1, 2),
            pair_logits=pair_logits,
        )
        candidates = (
            replace(
                _numeric_candidate(
                    0,
                    "100",
                    "Tien va cac khoan tuong duong tien",
                    "",
                    "native",
                ),
                period="dau nam",
                column_label="So dau nam",
                unit="",
            ),
            replace(
                _numeric_candidate(
                    1,
                    "200",
                    "2. Cac khoan tuong duong tien",
                    "",
                    "native",
                ),
                period="cuoi nam",
                column_label="So cuoi nam",
                unit="VND",
            ),
            replace(
                _numeric_candidate(
                    2,
                    "80",
                    "2. Cac khoan tuong duong tien",
                    "",
                    "native",
                ),
                period="dau nam",
                column_label="So dau nam",
                unit="VND",
            ),
        )
        question = (
            "Gia tri binh quan cua 2. Cac khoan tuong duong tien "
            "giua So cuoi nam va So dau nam la bao nhieu?"
        )

        result = _search_symbolic_programs(
            output,
            0,
            question,
            candidates,
            ReasoningProgram("AVERAGE", (1, 2)),
        )
        expanded = _search_symbolic_programs(
            output,
            0,
            question,
            candidates,
            ReasoningProgram("AVERAGE", (1, 2)),
            candidate_recall_v12_4_enabled=True,
        )

        self.assertIsNotNone(result)
        self.assertIsNotNone(expanded)
        assert result is not None
        assert expanded is not None
        self.assertEqual(
            result.program,
            ReasoningProgram("AVERAGE", (1, 2)),
        )
        self.assertEqual(expanded.program, result.program)
        reference_trace = next(
            item for item in expanded.trace if item.program == result.program
        )
        self.assertIn(
            "v12_3_reference_winner", reference_trace.generation_sources
        )

    def test_explicit_sum_arity_is_not_collapsed_to_add(self) -> None:
        sum3 = (
            'Tính tổng số học của "Chi phí tài chính", "Chi phí quản lý" '
            'và "Chi phí khác" tại 2024 là bao nhiêu?'
        )
        sum4 = (
            'Tính tổng số học của "Giá vốn", "Chi phí tài chính", '
            '"Chi phí quản lý" và "Chi phí khác" tại 2024 là bao nhiêu?'
        )
        unquoted_sum3 = (
            "Tổng Nợ ngắn hạn, Nợ dài hạn và Vốn chủ sở hữu "
            "tại SỐ CUỐI NĂM là bao nhiêu?"
        )

        self.assertEqual(
            _apply_symbolic_operation_constraint(sum3, "ADD"), "SUM3"
        )
        self.assertEqual(
            _apply_symbolic_operation_constraint(sum4, "ADD"), "SUM4"
        )
        self.assertEqual(
            _apply_symbolic_operation_constraint(unquoted_sum3, "SUBTRACT"),
            "SUM3",
        )

    def test_binary_add_with_comma_inside_metric_stays_add(self) -> None:
        question = (
            "Tổng Các khoản tương đương tiền và Dự thu cổ tức, tiền lãi "
            "chưa đến ngày nhận tại cuối năm là bao nhiêu?"
        )

        self.assertEqual(
            _apply_symbolic_operation_constraint(question, "SELECT"), "ADD"
        )

    def test_explicit_average_and_multiply_wording_override_lookup(self) -> None:
        average = (
            'Giá trị bình quân của "Nợ phải trả" giữa 2024 và 2025 '
            "là bao nhiêu?"
        )
        multiply = (
            "Giá trị theo mệnh giá của số cổ phiếu đang lưu hành "
            "tại 31/12/2025 là bao nhiêu?"
        )

        self.assertEqual(
            _apply_symbolic_operation_constraint(average, "SELECT"), "AVERAGE"
        )
        self.assertEqual(
            _apply_symbolic_operation_constraint(multiply, "SELECT"),
            "MULTIPLY",
        )

    def test_difference_cue_beats_average_inside_metric_name(self) -> None:
        question = (
            "Chênh lệch (Doanh thu thuần/Tổng tài sản bình quân) "
            "giữa năm 2025 và năm 2024 là bao nhiêu?"
        )

        self.assertEqual(
            _apply_symbolic_operation_constraint(question, "AVERAGE"),
            "SUBTRACT",
        )

    def test_percent_difference_cue_remains_percent_change(self) -> None:
        question = (
            "Chênh lệch doanh thu giữa năm 2025 và năm 2024 "
            "là bao nhiêu phần trăm?"
        )

        self.assertEqual(
            _apply_symbolic_operation_constraint(question, "SUBTRACT"),
            "PERCENT_CHANGE",
        )

    def test_average_metric_punctuation_is_not_misread_as_sum(self) -> None:
        questions = (
            "Gia tri binh quan cua Tang, giam cac khoan phai tra "
            "giua Nam 2025 va Nam 2024 la bao nhieu?",
            "Gia tri binh quan cua Tien thu tu phat hanh co phieu, nhan "
            "giua Nam nay va Nam truoc la bao nhieu?",
        )

        for question in questions:
            with self.subTest(question=question):
                self.assertEqual(
                    _apply_symbolic_operation_constraint(question, "SELECT"),
                    "AVERAGE",
                )

    def test_explicit_multi_operand_parts_extracts_quoted_and_plain_lists(
        self,
    ) -> None:
        quoted = (
            'Tính tổng "Nợ ngắn hạn", "Nợ dài hạn" và '
            '"Vốn chủ sở hữu" tại cuối năm.'
        )
        plain = (
            "Tổng bốn khoản phải thu dài hạn, tài sản cố định, "
            "tài sản dở dang dài hạn và tài sản dài hạn khác "
            "tại cuối năm là bao nhiêu?"
        )

        self.assertEqual(
            _explicit_multi_operand_parts(quoted, 3),
            ("Nợ ngắn hạn", "Nợ dài hạn", "Vốn chủ sở hữu"),
        )
        self.assertEqual(
            _explicit_multi_operand_parts(plain, 4),
            (
                "khoản phải thu dài hạn",
                "tài sản cố định",
                "tài sản dở dang dài hạn",
                "tài sản dài hạn khác",
            ),
        )

    def test_par_value_multiply_exposes_both_operand_roles(self) -> None:
        question = (
            "Giá trị theo mệnh giá của số cổ phiếu đang lưu hành "
            "tại 31/12/2025 là bao nhiêu?"
        )

        self.assertEqual(
            _operand_role_parts(question, "MULTIPLY"),
            ("so co phieu dang luu hanh", "menh gia"),
        )

    def test_add_exposes_two_explicit_operand_roles(self) -> None:
        question = (
            "Tính tổng Nợ ngắn hạn và Phải trả người bán ngắn hạn "
            "tại cuối năm 2024."
        )

        self.assertEqual(
            _operand_role_parts(question, "ADD"),
            ("no ngan han", "phai tra nguoi ban ngan han"),
        )

    def test_add_period_suffix_does_not_split_tai_san_metric(self) -> None:
        self.assertEqual(
            _operand_role_parts(
                "Tong Tien mat, vang va Tai san co dinh tai cuoi nam la bao nhieu?",
                "ADD",
            ),
            ("tien mat vang", "tai san co dinh"),
        )

    def test_binary_row_anchors_override_comma_based_sum3_guess(self) -> None:
        candidates = (
            _numeric_candidate(
                0, "16", "Tien gui va vay Chinh phu, NHNN", "2025", "native"
            ),
            _numeric_candidate(
                1, "148", "Tien gui va vay cac TCTD khac", "2025", "native"
            ),
            _numeric_candidate(
                2, "91", "Vay cac TCTD khac", "2025", "native"
            ),
        )

        spec = _compile_symbolic_query_spec(
            "Tong Tien gui va vay Chinh phu, NHNN va "
            "Tien gui va vay cac TCTD khac tai cuoi nam la bao nhieu?",
            candidates,
        )

        self.assertEqual(spec.required_operation, "ADD")
        self.assertEqual(spec.allowed_operations, ("ADD",))

    def test_add_does_not_invent_roles_for_aggregate_metric(self) -> None:
        self.assertIsNone(
            _operand_role_parts(
                "Tính tổng khấu hao tài sản cố định trong năm 2024.",
                "ADD",
            )
        )

    def test_add_role_constraint_matches_named_document_rows(self) -> None:
        candidates = (
            _numeric_candidate(0, "100", "Nợ ngắn hạn", "2024", "native"),
            _numeric_candidate(
                1, "40", "Phải trả người bán ngắn hạn", "2024", "native"
            ),
            _numeric_candidate(2, "60", "Nợ dài hạn", "2024", "native"),
        )

        compatible = _role_compatible_pair_indices(
            "Tổng Nợ ngắn hạn và Phải trả người bán ngắn hạn "
            "tại năm 2024 là bao nhiêu?",
            "ADD",
            candidates,
            {0, 1, 2},
        )

        self.assertEqual(compatible, ({0}, {1}))

    def test_add_role_constraint_prefers_exact_child_over_parent_subtotal(self) -> None:
        candidates = (
            _numeric_candidate(
                0, "100", "2. Tài sản cố định vô hình", "2025", "native"
            ),
            _numeric_candidate(
                1, "300", "Các khoản phải thu ngắn hạn", "2025", "native"
            ),
            _numeric_candidate(
                2,
                "40",
                "5. Dự phòng phải thu ngắn hạn khó đòi",
                "2025",
                "native",
            ),
        )

        compatible = _role_compatible_pair_indices(
            "Tổng 2. Tài sản cố định vô hình và "
            "5. Dự phòng phải thu ngắn hạn khó đòi năm 2025 là bao nhiêu?",
            "ADD",
            candidates,
            {0, 1, 2},
        )

        self.assertEqual(compatible, ({0}, {2}))

    def test_role_constraint_prefers_requested_year_over_change_columns(self) -> None:
        candidates = (
            replace(
                _numeric_candidate(
                    0, "1000", "Tong tai san", "2026", "native"
                ),
                unit="VND",
            ),
            replace(
                _numeric_candidate(
                    1, "1000", "Tong nguon von", "2026", "native"
                ),
                unit="VND",
            ),
            replace(
                _numeric_candidate(
                    2, "-0.7", "Tong tai san", "", "native"
                ),
                unit="%",
                is_percent=True,
                column_label="% tang/(giam)",
            ),
            replace(
                _numeric_candidate(
                    3, "-0.7", "Tong nguon von", "", "native"
                ),
                unit="%",
                is_percent=True,
                column_label="% tang/(giam)",
            ),
        )

        compatible = _role_compatible_pair_indices(
            "Tong tai san nam 2026 co bang Tong nguon von khong? "
            "Neu lech thi chenh lech la bao nhieu?",
            "SUBTRACT",
            candidates,
            set(range(len(candidates))),
        )

        self.assertEqual(compatible, ({0}, {1}))

    def test_role_constraint_keeps_unknown_period_when_no_match_exists(self) -> None:
        candidates = (
            _numeric_candidate(0, "1000", "Tong tai san", "", "native"),
            _numeric_candidate(1, "1000", "Tong nguon von", "", "native"),
        )

        compatible = _role_compatible_pair_indices(
            "Tong tai san nam 2026 co bang Tong nguon von khong? "
            "Neu lech thi chenh lech la bao nhieu?",
            "SUBTRACT",
            candidates,
            set(range(len(candidates))),
        )

        self.assertEqual(compatible, ({0}, {1}))

    def test_named_add_roles_reject_arbitrary_neural_pair(self) -> None:
        candidates = (
            _numeric_candidate(0, "100", "Doanh thu thuan", "2024", "native"),
            _numeric_candidate(
                1, "40", "So luong co phan da phat hanh", "2024", "native"
            ),
        )

        reasons = _grounding_gate_reasons(
            "Tong no ngan han va no dai han nam 2024 la bao nhieu?",
            ReasoningProgram("ADD", (0, 1)),
            candidates,
        )

        self.assertIn("missing_requested_operand_roles", reasons)

    def test_direct_coefficient_select_requires_direct_metric_row(self) -> None:
        candidates = (
            _numeric_candidate(0, "202353239327", "Tong tai san", "2024", "native"),
        )
        question = "He so no tren tong tai san nam 2024 la bao nhieu?"
        spec = _compile_symbolic_query_spec(question, candidates)

        reasons = _program_search_rejection_reasons(
            spec,
            question,
            ReasoningProgram("SELECT", (0,)),
            candidates,
        )

        self.assertIn("missing_direct_metric_evidence", reasons)

    def test_pair_features_distinguish_table_and_scope(self) -> None:
        left = replace(
            _numeric_candidate(0, "100", "Doanh thu thuần", "2024"),
            page=5,
            table_id="table-1",
            section="Kết quả kinh doanh",
            scope="consolidated",
            entity_id="entity:abc",
            subtable_id="table-1/subtable-1",
        )
        same = replace(left, index=1, raw="80", value="80")
        other = replace(
            left,
            index=2,
            raw="70",
            value="70",
            table_id="table-2",
            scope="parent_company",
            entity_id="entity:xyz",
            subtable_id="table-2/subtable-2",
        )

        same_features = _candidate_pair_features(left, same, 0, 1)
        other_features = _candidate_pair_features(left, other, 0, 2)

        self.assertEqual(len(same_features), PAIR_FEATURE_SIZE)
        self.assertEqual(
            same_features[10:],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        )
        self.assertEqual(
            other_features[10:],
            [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        )

    def test_reporting_entity_is_preferred_unless_question_names_embed(self) -> None:
        reporting = replace(
            _numeric_candidate(0, "100", "Tổng tài sản", "2024"),
            scope="",
            section="Tình hình hoạt động",
        )
        embedded = replace(
            reporting,
            index=1,
            raw="200",
            value="200",
            scope="entity:btp",
            section="Công ty Cổ phần Nhiệt điện Bà Rịa (BTP)",
        )
        candidates = (reporting, embedded)

        self.assertEqual(
            _prefer_reporting_entity_indices(
                "Tổng tài sản năm 2024 là bao nhiêu?", candidates, {0, 1}
            ),
            {0},
        )
        self.assertEqual(
            _prefer_reporting_entity_indices(
                "Tổng tài sản của BTP năm 2024 là bao nhiêu?", candidates, {0, 1}
            ),
            {0, 1},
        )

    def test_program_search_abstains_when_temporal_pair_lacks_required_unit_proof(self) -> None:
        reporting_current = replace(
            _numeric_candidate(
                0, "295481188173", "Doanh thu thuan", "2023", "native"
            ),
            scope="",
            section="4.1. Tinh hinh tai chinh",
        )
        reporting_prior = replace(
            _numeric_candidate(
                1, "313255998102", "Doanh thu thuan", "2022", "native"
            ),
            scope="",
            section="4.1. Tinh hinh tai chinh",
        )
        embedded_current = replace(
            _numeric_candidate(
                2, "40494083152", "Doanh thu thuan", "2023", "native"
            ),
            scope="entity:cong ty lien ket",
            section="Cong ty lien ket",
        )
        embedded_prior = replace(
            _numeric_candidate(
                3, "49276104876", "Doanh thu thuan", "2022", "native"
            ),
            scope="entity:cong ty lien ket",
            section="Cong ty lien ket",
        )
        candidates = (
            reporting_current,
            reporting_prior,
            embedded_current,
            embedded_prior,
        )
        operation_logits = torch.zeros(1, len(OPERATION_TO_ID))
        operation_logits[0, OPERATION_TO_ID["PERCENT_CHANGE"]] = 8.0
        pair_logits = torch.zeros(1, len(OPERATION_TO_ID), 4, 4)
        pair_logits[0, OPERATION_TO_ID["PERCENT_CHANGE"], 2, 3] = 9.0
        pair_logits[0, OPERATION_TO_ID["PERCENT_CHANGE"], 0, 1] = 2.0
        output = DocumentHRMForward(
            state=DocumentHRMState(
                high=torch.zeros(1, 1, 64),
                low=torch.zeros(1, 1, 64),
            ),
            operation_logits=operation_logits,
            pointer_logits=torch.zeros(1, 4, 4),
            q_values=torch.zeros(1, 2),
            pair_logits=pair_logits,
        )

        result = _search_symbolic_programs(
            output,
            0,
            "Doanh thu thuan thay doi bao nhieu phan tram "
            "tu nam 2022 den nam 2023?",
            candidates,
            ReasoningProgram("PERCENT_CHANGE", (2, 3)),
        )

        # The reporting-entity pair is structurally preferable, but these
        # legacy fixtures carry no unit metadata. QP3 requires unit proof for
        # percent change, so UNKNOWN must abstain rather than select either
        # the reporting or embedded pair.
        self.assertIsNone(result)

    def test_temporal_change_rejects_two_different_metric_rows(self) -> None:
        candidates = (
            _numeric_candidate(0, "1707170000", "II. No dai han", "2023"),
            _numeric_candidate(
                1,
                "941820740",
                "2. Tai san do dang dai han",
                "2022",
            ),
        )

        reasons = _grounding_gate_reasons(
            "No dai han thay doi bao nhieu tu nam 2022 den nam 2023?",
            ReasoningProgram("PERCENT_CHANGE", (0, 1)),
            candidates,
        )

        self.assertIn("temporal_metric_mismatch", reasons)

    def test_add_select_contrastive_loss_penalizes_confused_logits(self) -> None:
        logits = torch.zeros(2, len(OPERATION_TO_ID))
        targets = torch.tensor(
            [OPERATION_TO_ID["ADD"], OPERATION_TO_ID["SELECT"]]
        )
        logits[0, OPERATION_TO_ID["SELECT"]] = 1.0
        logits[1, OPERATION_TO_ID["ADD"]] = 1.0

        confused = _add_select_contrastive_loss(logits, targets)
        logits[0, OPERATION_TO_ID["ADD"]] = 2.0
        logits[1, OPERATION_TO_ID["SELECT"]] = 2.0
        separated = _add_select_contrastive_loss(logits, targets)

        self.assertGreater(confused.item(), separated.item())
    def test_reasoning_engine_keeps_calibration_methods(self) -> None:
        self.assertTrue(hasattr(DocumentHRMReasoningEngine, "_confidence_threshold"))
        self.assertTrue(hasattr(DocumentHRMReasoningEngine, "_uses_calibration_fallback"))

    def test_derived_table_formula_evidence_requires_review(self) -> None:
        candidate = _numeric_candidate(
            0, "62848794351367", "Doanh thu thuần", "2024", "mineru"
        )
        candidate = replace(
            candidate,
            context=candidate.context
            + "\nVALUE_ORIGIN: DERIVED_FROM_TABLE_FORMULA"
            + "\nCELL: 62.848.794.351.367",
        )

        self.assertTrue(
            _uses_derived_table_formula((SimpleNamespace(candidate=candidate),))
        )

    def test_par_value_identity_materializes_a_typed_intermediate(self) -> None:
        capital = replace(
            _numeric_candidate(
                0, "123464110000", "Von dieu le", "", "native"
            ),
            unit="VND",
            source_name="issuer-report.pdf",
            page=2,
        )
        issued = replace(
            _numeric_candidate(
                1,
                "12346411",
                "So luong co phan da phat hanh",
                "",
                "native",
            ),
            unit="shares",
            source_name="issuer-report.pdf",
            page=9,
        )
        outstanding = replace(
            _numeric_candidate(
                2,
                "10451182",
                "So luong co phan dang luu hanh",
                "",
                "native",
            ),
            unit="shares",
            source_name="issuer-report.pdf",
            page=9,
        )

        augmented = _augment_financial_identity_candidates(
            "Gia tri theo menh gia cua so co phieu dang luu hanh la bao nhieu?",
            (capital, issued, outstanding),
            max_candidates=8,
        )

        derived = next(
            item for item in augmented if item.extraction_method == "symbolic-derived"
        )
        self.assertEqual(derived.value, "10000")
        self.assertEqual(derived.unit, "VND/share")
        self.assertIn(
            "VALUE_ORIGIN: DERIVED_FROM_EXPLICIT_FINANCIAL_IDENTITY",
            derived.context,
        )

    def test_interest_bearing_debt_share_materializes_verified_dag(self) -> None:
        base = (
            ("80", "Vay va no thue tai chinh ngan han"),
            ("20", "Vay va no thue tai chinh dai han"),
            ("1000", "Tong nguon von"),
        )
        candidates = tuple(
            replace(
                _numeric_candidate(index, value, row, "2025", "native"),
                unit="VND",
                source_name="issuer-report.pdf",
                page=7,
                table_id="balance-sheet",
                scope="reporting",
            )
            for index, (value, row) in enumerate(base)
        )
        question = (
            "Tong no vay tai chinh chiu lai gom vay ngan han va dai han "
            "chiem bao nhieu phan tram Tong nguon von nam 2025?"
        )

        augmented = _augment_financial_identity_candidates(
            question, candidates, max_candidates=8
        )
        derived = augmented[-1]
        flat = ReasoningProgram("RATIO_PERCENT", (derived.index, 2))
        dag = materialize_program_dag(flat, augmented)

        self.assertEqual(len(augmented), 4)
        self.assertEqual(derived.value, "100")
        self.assertEqual(derived.derived_operation, "ADD")
        self.assertEqual(derived.derived_operands, (0, 1))
        self.assertEqual(derived.derivation_id, "interest_bearing_debt")
        assert dag is not None
        self.assertEqual(
            tuple(node.operation for node in dag.nodes),
            ("ADD", "SHARE_OF_TOTAL"),
        )
        self.assertEqual(dag.leaf_operands, (0, 1, 2))
        self.assertEqual(execute_program_dag(dag, augmented), Decimal("10"))
        self.assertEqual(_grounding_gate_reasons(question, flat, augmented), ())

    def test_interest_bearing_debt_share_rejects_ambiguous_or_mixed_evidence(
        self,
    ) -> None:
        question = (
            "Tong no vay tai chinh chiu lai gom vay ngan han va dai han "
            "chiem bao nhieu phan tram Tong nguon von nam 2025?"
        )

        def candidate(
            index: int, value: str, row: str, year: str, unit: str = "VND"
        ) -> NumericCandidate:
            return replace(
                _numeric_candidate(index, value, row, year, "native"),
                unit=unit,
                source_name="issuer-report.pdf",
                page=7,
                table_id="balance-sheet",
                scope="reporting",
            )

        wrong_period = (
            candidate(0, "80", "Vay va no thue tai chinh ngan han", "2025"),
            candidate(1, "20", "Vay va no thue tai chinh dai han", "2024"),
            candidate(2, "1000", "Tong nguon von", "2025"),
        )
        mixed_scale = (
            candidate(0, "80", "Vay va no thue tai chinh ngan han", "2025"),
            candidate(
                1,
                "20",
                "Vay va no thue tai chinh dai han",
                "2025",
                "trieu dong",
            ),
            candidate(2, "1000", "Tong nguon von", "2025"),
        )
        ambiguous = (
            candidate(0, "80", "Vay va no thue tai chinh ngan han", "2025"),
            candidate(1, "81", "No vay ngan han", "2025"),
            candidate(2, "20", "Vay va no thue tai chinh dai han", "2025"),
            candidate(3, "1000", "Tong nguon von", "2025"),
        )

        for case in (wrong_period, mixed_scale, ambiguous):
            with self.subTest(case=case):
                augmented = _augment_financial_identity_candidates(
                    question, case, max_candidates=8
                )
                self.assertFalse(
                    any(item.derivation_id == "interest_bearing_debt" for item in augmented)
                )

    def test_interest_bearing_debt_share_rejects_liability_semantic_confounds(
        self,
    ) -> None:
        question = (
            "Tong no vay tai chinh chiu lai gom vay ngan han va dai han "
            "chiem bao nhieu phan tram Tong nguon von nam 2025?"
        )

        def candidate(index: int, value: str, row: str) -> NumericCandidate:
            return replace(
                _numeric_candidate(index, value, row, "2025", "native"),
                unit="VND",
                source_name="issuer-report.pdf",
                page=7,
                table_id="balance-sheet",
                scope="reporting",
            )

        liability_rows = (
            candidate(0, "80", "Nợ ngắn hạn"),
            candidate(1, "20", "Nợ dài hạn"),
            candidate(2, "1000", "Tổng nguồn vốn"),
        )
        lending_asset_rows = (
            candidate(0, "80", "Phải thu về cho vay ngắn hạn"),
            candidate(1, "20", "Phải thu về cho vay dài hạn"),
            candidate(2, "1000", "Tổng nguồn vốn"),
        )
        for case in (liability_rows, lending_asset_rows):
            with self.subTest(case=case):
                augmented = _augment_financial_identity_candidates(
                    question, case, max_candidates=8
                )
                self.assertFalse(
                    any(item.derivation_id == "interest_bearing_debt" for item in augmented)
                )

        self.assertIsNone(_debt_share_role(liability_rows[0]))
        self.assertIsNone(_debt_share_role(lending_asset_rows[0]))

    def test_interest_bearing_debt_share_role_anchor_accepts_common_borrowing_word_orders(
        self,
    ) -> None:
        rows = (
            "Vay ngắn hạn",
            "Nợ vay ngắn hạn",
            "Vay và nợ thuê tài chính ngắn hạn",
        )
        for row in rows:
            with self.subTest(row=row):
                value = replace(
                    _numeric_candidate(0, "80", row, "2025", "native"),
                    unit="VND",
                )
                self.assertEqual(_debt_share_role(value), "short_term_borrowings")

    def test_interest_bearing_debt_share_retrieval_expands_three_raw_roles(
        self,
    ) -> None:
        question = (
            "Tong no vay tai chinh chiu lai gom vay ngan han va dai han "
            "chiem bao nhieu phan tram Tong nguon von nam 2025?"
        )

        expanded = normalize_text(_expanded_retrieval_question(question))

        self.assertIn("vay va no thue tai chinh ngan han", expanded)
        self.assertIn("vay va no thue tai chinh dai han", expanded)
        self.assertIn("tong nguon von", expanded)

    def test_interest_bearing_debt_share_search_overrides_wrong_neural_pair(
        self,
    ) -> None:
        question = (
            "Tong no vay tai chinh chiu lai gom vay ngan han va dai han "
            "chiem bao nhieu phan tram Tong nguon von nam 2025?"
        )
        raw = tuple(
            replace(
                _numeric_candidate(index, value, row, "2025", "native"),
                unit="VND",
                source_name="issuer-report.pdf",
                page=7,
                table_id="balance-sheet",
                scope="reporting",
            )
            for index, (value, row) in enumerate(
                (
                    ("80", "Vay va no thue tai chinh ngan han"),
                    ("20", "Vay va no thue tai chinh dai han"),
                    ("1000", "Tong nguon von"),
                )
            )
        )
        candidates = _augment_financial_identity_candidates(
            question, raw, max_candidates=8
        )
        operation_logits = torch.full((1, 10), -5.0)
        operation_logits[0, OPERATION_TO_ID["RATIO_PERCENT"]] = 5.0
        pair_logits = torch.zeros(1, 10, 4, 4)
        pair_logits[0, OPERATION_TO_ID["RATIO_PERCENT"], 0, 1] = 12.0
        output = DocumentHRMForward(
            state=DocumentHRMState(
                high=torch.zeros(1, 1, 64), low=torch.zeros(1, 1, 64)
            ),
            operation_logits=operation_logits,
            pointer_logits=torch.zeros(1, 4, 4),
            q_values=torch.zeros(1, 2),
            pair_logits=pair_logits,
        )

        result = _search_symbolic_programs(
            output,
            0,
            question,
            candidates,
            ReasoningProgram("RATIO_PERCENT", (0, 1)),
            typed_synthesis_enabled=True,
            financial_identity_catalog_enabled=True,
            candidate_recall_v12_4_enabled=True,
        )

        assert result is not None
        self.assertEqual(
            result.program,
            ReasoningProgram("RATIO_PERCENT", (3, 2)),
            msg=result,
        )
        selected = next(
            item for item in result.trace if item.program == result.program
        )
        self.assertIn("typed_slot_synthesis", selected.generation_sources)

    def test_interest_bearing_debt_share_grounding_rejects_tampered_sum(
        self,
    ) -> None:
        question = (
            "Tong no vay tai chinh chiu lai gom vay ngan han va dai han "
            "chiem bao nhieu phan tram Tong nguon von nam 2025?"
        )
        raw = tuple(
            replace(
                _numeric_candidate(index, value, row, "2025", "native"),
                unit="VND",
                source_name="issuer-report.pdf",
                page=7,
                table_id="balance-sheet",
                scope="reporting",
            )
            for index, (value, row) in enumerate(
                (
                    ("80", "Vay va no thue tai chinh ngan han"),
                    ("20", "Vay va no thue tai chinh dai han"),
                    ("1000", "Tong nguon von"),
                )
            )
        )
        augmented = _augment_financial_identity_candidates(
            question, raw, max_candidates=8
        )
        tampered = (*augmented[:-1], replace(augmented[-1], value="999"))

        reasons = _grounding_gate_reasons(
            question,
            ReasoningProgram("RATIO_PERCENT", (3, 2)),
            tampered,
        )

        self.assertIn("derived_value_mismatch", reasons)

    def test_interest_bearing_debt_share_is_year_order_and_quota_independent(
        self,
    ) -> None:
        role_rows = {
            "short": ("80", "Vay va no thue tai chinh ngan han", "VND"),
            "long": ("20", "Vay va no thue tai chinh dai han", "dong"),
            "total": ("1000", "Tong nguon von", "VND"),
            "noise": ("7", "Doanh thu thuan", "VND"),
        }
        for year, order in (
            ("2024", ("short", "long", "total", "noise")),
            ("2027", ("noise", "long", "total", "short")),
        ):
            with self.subTest(year=year, order=order):
                candidates = tuple(
                    replace(
                        _numeric_candidate(
                            index,
                            role_rows[key][0],
                            role_rows[key][1],
                            year,
                            "native",
                        ),
                        unit=role_rows[key][2],
                        relevance=0.01 if key == "noise" else 1.0,
                        source_name="issuer-report.pdf",
                        page=7,
                        table_id="balance-sheet",
                        scope="reporting",
                    )
                    for index, key in enumerate(order)
                )
                question = (
                    "Tong no vay tai chinh chiu lai gom vay ngan han va dai han "
                    "chiem bao nhieu phan tram Tong nguon von nam "
                    f"{year}?"
                )

                augmented = _augment_financial_identity_candidates(
                    question, candidates, max_candidates=4
                )
                derived = next(
                    item
                    for item in augmented
                    if item.derivation_id == "interest_bearing_debt"
                )
                total_index = next(
                    item.index
                    for item in augmented
                    if normalize_text(item.row_label) == "tong nguon von"
                )
                dag = materialize_program_dag(
                    ReasoningProgram(
                        "RATIO_PERCENT", (derived.index, total_index)
                    ),
                    augmented,
                )

                self.assertEqual(len(augmented), 4)
                self.assertFalse(
                    any(normalize_text(item.row_label) == "doanh thu thuan" for item in augmented)
                )
                assert dag is not None
                self.assertEqual(execute_program_dag(dag, augmented), Decimal("10"))
                self.assertEqual(
                    _grounding_gate_reasons(
                        question,
                        ReasoningProgram(
                            "RATIO_PERCENT", (derived.index, total_index)
                        ),
                        augmented,
                    ),
                    (),
                )

    def test_par_value_retrieval_expands_only_the_required_identity(self) -> None:
        expanded = _expanded_retrieval_question(
            "Gia tri theo menh gia cua so co phieu dang luu hanh la bao nhieu?"
        )
        ordinary = _expanded_retrieval_question(
            "Doanh thu thuan nam 2024 la bao nhieu?"
        )

        self.assertIn("von dieu le", normalize_text(expanded))
        self.assertIn("co phan da phat hanh", normalize_text(expanded))
        self.assertEqual(ordinary, "Doanh thu thuan nam 2024 la bao nhieu?")

    def test_missing_requested_period_is_reported(self) -> None:
        candidates = (
            _numeric_candidate(0, "1455150771623", "Tổng tài sản", "2024"),
        )
        missing = _missing_requested_periods(
            "Tổng tài sản năm 2024 tăng bao nhiêu so với năm 2023?",
            candidates,
        )
        self.assertEqual(missing, ("2023",))

    def test_missing_relative_period_qualifiers_are_reported(self) -> None:
        candidates = (
            _numeric_candidate(0, "200", "Tổng tài sản", "2025", "native"),
            _numeric_candidate(1, "100", "Tổng tài sản", "2024", "native"),
        )

        missing = _missing_requested_period_qualifiers(
            "Tổng tài sản cuối năm tăng bao nhiêu so với đầu năm?", candidates
        )

        self.assertEqual(missing, ("cuoi nam", "dau nam"))

    def test_native_gap_requires_both_ratio_roles(self) -> None:
        candidates = (
            _numeric_candidate(0, "219541462526", "Lợi nhuận trước thuế", "2025", "native"),
            _numeric_candidate(1, "4714032275888", "Tổng tài sản", "2025", "native"),
        )

        gap = _native_structured_evidence_gap(
            "Nợ phải trả chiếm bao nhiêu phần trăm tổng tài sản năm 2025?",
            candidates,
        )

        self.assertIn("operand roles", gap or "")

    def test_native_gap_requires_both_explicit_add_roles(self) -> None:
        candidates = (
            replace(
                _numeric_candidate(0, "11", "", "", "native"),
                context="",
                row_label="",
            ),
            _numeric_candidate(1, "2024.04", "Date", "", "native"),
        )

        gap = _native_structured_evidence_gap(
            "Tá»•ng GiÃ¡ vá»‘n hÃ ng bÃ¡n vÃ  Chi phÃ­ tÃ i chÃ­nh táº¡i nÄƒm nay lÃ  bao nhiÃªu?",
            candidates,
        )

        with patch(
            "hrm_finance_qa.document_reasoning_engine."
            "_apply_symbolic_operation_constraint",
            return_value="ADD",
        ):
            gap = _native_structured_evidence_gap(
                "Cong Gia von hang ban voi Chi phi tai chinh tai nam nay la bao nhieu?",
                candidates,
            )
        self.assertIn("operand roles", gap or "")

    def test_predicted_add_retries_ocr_only_below_structured_arity(self) -> None:
        one_structured = (
            replace(
                _numeric_candidate(0, "11", "", "", "native"),
                context="",
                row_label="",
            ),
            _numeric_candidate(1, "2024.04", "Date", "", "native"),
        )
        two_structured = (
            _numeric_candidate(0, "60", "Chi phi ban hang", "", "native"),
            _numeric_candidate(
                1,
                "20",
                "Chi phi quan ly doanh nghiep",
                "",
                "native",
            ),
        )

        self.assertTrue(
            _predicted_add_requires_ocr_fallback("ADD", one_structured)
        )
        self.assertFalse(
            _predicted_add_requires_ocr_fallback("ADD", two_structured)
        )
        self.assertFalse(
            _predicted_add_requires_ocr_fallback("SELECT", one_structured)
        )

    def test_relative_periods_cover_requested_years(self) -> None:
        current = replace(
            _numeric_candidate(0, "200", "Doanh thu thuần", extraction_method="ocr"),
            context=(
                "ROW: Doanh thu thuần\nROW_TYPE: detail\n"
                "COLUMN: Năm nay\nPERIOD: nam nay"
            ),
        )
        prior = replace(
            _numeric_candidate(1, "100", "Doanh thu thuần", extraction_method="ocr"),
            context=(
                "ROW: Doanh thu thuần\nROW_TYPE: detail\n"
                "COLUMN: Năm trước\nPERIOD: nam truoc"
            ),
        )
        question = "Doanh thu thay đổi bao nhiêu phần trăm từ 2024 đến 2025?"
        self.assertEqual(_missing_requested_periods(question, (current, prior)), ())
        self.assertEqual(
            _period_compatible_indices(
                question, "PERCENT_CHANGE", (current, prior), 2
            ),
            {0, 1},
        )

    def test_percent_change_word_order_beats_plain_subtract_marker(self) -> None:
        operation = _apply_symbolic_operation_constraint(
            "Doanh thu năm 2025 giảm bao nhiêu phần trăm so với năm 2024?",
            "SUBTRACT",
        )
        self.assertEqual(operation, "PERCENT_CHANGE")

    def test_period_comparison_is_subtract(self) -> None:
        operation = _apply_symbolic_operation_constraint(
            "So sánh tiền mặt cuối kỳ 2024 so với đầu kỳ 2024.",
            "SELECT",
        )
        self.assertEqual(operation, "SUBTRACT")

    def test_ty_trong_trong_tong_is_ratio(self) -> None:
        operation = _apply_symbolic_operation_constraint(
            "Tính tỷ trọng tiền trong tổng tài sản cuối kỳ.",
            "SUM4",
        )
        self.assertEqual(operation, "RATIO_PERCENT")

    def test_chiem_bao_nhieu_phan_tram_is_ratio(self) -> None:
        operation = _apply_symbolic_operation_constraint(
            "Nợ phải trả chiếm bao nhiêu phần trăm tổng tài sản năm 2024?",
            "SELECT",
        )
        self.assertEqual(operation, "RATIO_PERCENT")

    def test_explicit_chia_cho_selects_division_or_percent_ratio(self) -> None:
        division = _apply_symbolic_operation_constraint(
            "NỢ PHẢI TRẢ chia cho Nợ ngắn hạn tại cuối năm bằng bao nhiêu?",
            "SUBTRACT",
        )
        percent_ratio = _apply_symbolic_operation_constraint(
            "NỢ PHẢI TRẢ chia cho Tổng tài sản bằng bao nhiêu phần trăm?",
            "SUBTRACT",
        )
        self.assertEqual(division, "DIVIDE")
        self.assertEqual(percent_ratio, "RATIO_PERCENT")

    def test_period_constraint_excludes_unscoped_distractor(self) -> None:
        candidates = (
            _numeric_candidate(0, "1455150771623", "Tồng giá trị tài sn", "2024"),
            _numeric_candidate(1, "263030387825", "N phi trå", "2024"),
            _numeric_candidate(2, "500000", "Phó Tổng"),
        )
        compatible = _period_compatible_indices(
            "Nợ phải trả chiếm bao nhiêu phần trăm tổng tài sản năm 2024?",
            "RATIO_PERCENT",
            candidates,
            2,
        )
        self.assertEqual(compatible, {0, 1})

    def test_select_period_constraint_excludes_statement_code(self) -> None:
        amount = _numeric_candidate(
            0, "173171512356", "Lợi nhuận sau thuế", "2025", "mineru"
        )
        code = _numeric_candidate(
            1, "200", "Lợi nhuận sau thuế", extraction_method="mineru"
        )

        compatible = _period_compatible_indices(
            "Lợi nhuận sau thuế năm 2025 là bao nhiêu?",
            "SELECT",
            (amount, code),
            1,
        )

        self.assertEqual(compatible, {0})

    def test_ratio_alignment_handles_ocr_damaged_row_labels(self) -> None:
        candidates = (
            _numeric_candidate(0, "1455150771623", "Tồng giá trị tài sn", "2024"),
            _numeric_candidate(1, "263030387825", "N phi trå", "2024"),
        )
        operands = _align_directed_operands(
            "Nợ phải trả chiếm bao nhiêu phần trăm tổng tài sản năm 2024?",
            "RATIO_PERCENT",
            (0, 1),
            candidates,
        )
        self.assertEqual(operands, (1, 0))

    def test_ratio_role_constraint_rejects_duplicate_denominator_fact(self) -> None:
        candidates = (
            _numeric_candidate(0, "1455150771623", "Tồng giá trị tài sn", "2024", "ocr"),
            _numeric_candidate(1, "263030387825", "N phi trå", "2024", "ocr"),
            _numeric_candidate(2, "1455150771623", "Tổng giá trị tài sản", "2024", "ocr"),
        )
        compatible = _role_compatible_pair_indices(
            "Nợ phải trả chiếm bao nhiêu phần trăm tổng tài sản năm 2024?",
            "RATIO_PERCENT",
            candidates,
            {0, 1, 2},
        )
        self.assertIsNotNone(compatible)
        assert compatible is not None
        numerator, denominator = compatible
        self.assertEqual(numerator, {1})
        self.assertEqual(denominator, {0, 2})

    def test_ratio_role_constraint_handles_squashed_ocr_labels(self) -> None:
        candidates = (
            _numeric_candidate(0, "3117771571701", "NPHẢITRẢ", "2025", "ocr"),
            _numeric_candidate(1, "4714032275888", "TỔNGCỘNGTÀISẢN", "2025", "ocr"),
            _numeric_candidate(
                2, "42", "NGUYÊN TẮC GHI NHẬN NỢ PHẢI TRẢ TÀI CHÍNH", "2025", "ocr"
            ),
        )
        compatible = _role_compatible_pair_indices(
            "Theo báo cáo tình hình tài chính, nợ phải trả chiếm bao nhiêu "
            "phần trăm tổng cộng tài sản năm 2025?",
            "RATIO_PERCENT",
            candidates,
            {0, 1, 2},
        )
        self.assertEqual(compatible, ({0}, {1}))

    def test_select_metric_constraint_uses_question_after_report_qualifier(self) -> None:
        candidates = (
            _numeric_candidate(0, "100", "TỔNGCỘNGTÀISẢN", "2025", "ocr"),
            _numeric_candidate(1, "100", "TỔNG CỘNG NGUỒN VỐN", "2025", "ocr"),
        )
        allowed = _metric_compatible_indices(
            "Theo bảng cân đối kế toán, tổng cộng tài sản năm 2025 là bao nhiêu?",
            candidates,
            {0, 1},
        )
        self.assertEqual(allowed, {0})

    def test_select_metric_constraint_applies_to_structured_native_rows(self) -> None:
        candidates = (
            _numeric_candidate(0, "2529125816261", "công ty mẹ", "", "native"),
            _numeric_candidate(1, "2210570390648", "TỔNG CỘNG", "", "native"),
        )

        allowed = _metric_compatible_indices(
            "Lợi nhuận sau thuế của công ty mẹ REE năm 2025 là bao nhiêu?",
            candidates,
            {0, 1},
        )

        self.assertEqual(allowed, {0})

    def test_select_parent_company_anchor_rejects_total_profit_row(self) -> None:
        candidates = (
            _numeric_candidate(
                0, "3150404939011", "Tong loi nhuan sau thue", "2025", "native"
            ),
            _numeric_candidate(
                1, "2529125816261", "Cong ty me", "2025", "native"
            ),
        )

        allowed = _metric_compatible_indices(
            "Loi nhuan sau thue cua cong ty me REE nam 2025 la bao nhieu?",
            candidates,
            {0, 1},
        )

        self.assertEqual(allowed, {1})

    def test_unary_absolute_year_does_not_infer_period_from_filename(self) -> None:
        current = replace(
            _numeric_candidate(0, "2529125816261", "Cong ty me", "", "native"),
            source_name="report_25CN.pdf",
            page=53,
            line_number=61,
            char_start=30,
            char_end=43,
        )
        prior = replace(
            _numeric_candidate(1, "1993385852649", "Cong ty me", "", "native"),
            source_name="report_25CN.pdf",
            page=53,
            line_number=61,
            char_start=46,
            char_end=59,
        )

        self.assertEqual(
            _narrow_unary_period_by_row_order(
                "Loi nhuan cua cong ty me nam 2025 la bao nhieu?",
                (current, prior),
                {0, 1},
            ),
            {0, 1},
        )

    def test_unary_period_metadata_wins_over_column_order(self) -> None:
        prior = replace(
            _numeric_candidate(0, "90", "Doanh thu", "dau nam", "native"),
            source_name="report.pdf",
            page=1,
            line_number=10,
            char_start=20,
        )
        current = replace(
            _numeric_candidate(1, "100", "Doanh thu", "cuoi nam", "native"),
            source_name="report.pdf",
            page=1,
            line_number=10,
            char_start=40,
        )

        narrowed = _narrow_unary_period_by_row_order(
            "Doanh thu tại cuối năm là bao nhiêu?",
            (prior, current),
            {0, 1},
        )

        self.assertEqual(narrowed, {1})

    def test_select_metric_constraint_rejects_unstructured_prose_fuzzy_match(self) -> None:
        target = _numeric_candidate(
            0, "284331609643", "TỔNG CỘNG TÀI SẢN", "cuoi nam", "ocr"
        )
        prose = replace(
            _numeric_candidate(1, "26203512622", "unused", "", "ocr"),
            context=(
                "CONTEXT: Tài sản ngắn hạn giảm, tương đương mức giảm.\n"
                "VALUE_LINE: Tài sản dài hạn giảm <VALUE> 26.203.512.622 </VALUE> VND"
            ),
        )

        allowed = _metric_compatible_indices(
            "Theo bảng cân đối kế toán, tổng cộng tài sản của ABR cuối năm 2025 là bao nhiêu VND?",
            (target, prose),
            {0, 1},
        )

        self.assertEqual(allowed, {0})

    def test_select_metric_constraint_rejects_before_after_tax_conflict(self) -> None:
        candidates = (
            _numeric_candidate(
                0,
                "219541462526",
                "Lợi nhuận trước Thuế Thu nhập doanh nghiệp",
                "2025",
                "ocr",
            ),
            _numeric_candidate(
                1,
                "173171512356",
                "LỢI NHUẬN KẾ TOÁN SAU THUẾ TNDN",
                "2025",
                "ocr",
            ),
        )

        allowed = _metric_compatible_indices(
            "Lợi nhuận kế toán sau thuế thu nhập doanh nghiệp năm 2025 là bao nhiêu?",
            candidates,
            {0, 1},
        )

        self.assertEqual(allowed, {1})

    def test_select_prefers_metric_when_absolute_period_mask_is_disjoint(self) -> None:
        allowed, metric_applied, period_applied = (
            _combine_unary_grounding_constraints({0, 1}, {8, 9})
        )

        self.assertEqual(allowed, {0, 1})
        self.assertTrue(metric_applied)
        self.assertFalse(period_applied)

    def test_select_intersects_compatible_metric_and_period_masks(self) -> None:
        allowed, metric_applied, period_applied = (
            _combine_unary_grounding_constraints({0, 1}, {0, 8})
        )

        self.assertEqual(allowed, {0})
        self.assertTrue(metric_applied)
        self.assertTrue(period_applied)

    def test_ratio_role_constraint_does_not_override_clean_context(self) -> None:
        candidates = (
            _numeric_candidate(0, "100", "Chi phí nhân viên", "2024"),
            _numeric_candidate(1, "200", "Tổng chi phí hoạt động", "2024"),
        )
        compatible = _role_compatible_pair_indices(
            "Tỷ lệ chi phí nhân viên trên tổng chi phí hoạt động năm 2024?",
            "RATIO_PERCENT",
            candidates,
            {0, 1},
        )
        self.assertIsNone(compatible)

    def test_ratio_role_constraint_combines_structured_ocr_and_native_facts(self) -> None:
        candidates = (
            _numeric_candidate(0, "100", "N phi trå", "2024", "ocr"),
            _numeric_candidate(1, "200", "Tổng tài sản", "2024", "native"),
        )
        compatible = _role_compatible_pair_indices(
            "Nợ phải trả chiếm bao nhiêu phần trăm tổng tài sản năm 2024?",
            "RATIO_PERCENT",
            candidates,
            {0, 1},
        )
        self.assertEqual(compatible, ({0}, {1}))

    def test_ratio_role_constraint_applies_to_structured_native_rows(self) -> None:
        candidates = (
            _numeric_candidate(0, "15278313579883", "C. NỢ PHẢI TRẢ", "", "native"),
            _numeric_candidate(
                1, "40074851708537", "TỔNG CỘNG TÀI SẢN", "", "native"
            ),
            _numeric_candidate(2, "26373366190770", "TÀI SẢN DÀI HẠN", "", "native"),
        )

        compatible = _role_compatible_pair_indices(
            "Nợ phải trả chiếm bao nhiêu phần trăm tổng cộng tài sản năm 2025?",
            "RATIO_PERCENT",
            candidates,
            {0, 1, 2},
        )

        self.assertEqual(compatible, ({0}, {1}))

    def test_ratio_role_constraint_rejects_weak_relative_matches(self) -> None:
        candidates = (
            _numeric_candidate(
                0, "107672573473", "Cac khoan phai thu dai han", "", "native"
            ),
            _numeric_candidate(
                1, "2478048502242", "Bat dong san dau tu", "", "native"
            ),
        )

        compatible = _role_compatible_pair_indices(
            "No phai tra chiem bao nhieu phan tram tong cong tai san cuoi nam 2025?",
            "RATIO_PERCENT",
            candidates,
            {0, 1},
        )

        self.assertIsNone(compatible)

    def test_ratio_role_constraint_uses_current_same_row_values(self) -> None:
        candidates = (
            replace(
                _numeric_candidate(0, "15278313579883", "No phai tra", "", "native"),
                source_name="REE_25CN_BCTN.pdf",
                page=52,
                line_number=300,
                char_start=30,
            ),
            replace(
                _numeric_candidate(1, "13907555789461", "No phai tra", "", "native"),
                source_name="REE_25CN_BCTN.pdf",
                page=52,
                line_number=300,
                char_start=46,
            ),
            replace(
                _numeric_candidate(2, "40074851708537", "Tong cong tai san", "", "native"),
                source_name="REE_25CN_BCTN.pdf",
                page=51,
                line_number=270,
                char_start=30,
            ),
            replace(
                _numeric_candidate(3, "36362339883577", "Tong cong tai san", "", "native"),
                source_name="REE_25CN_BCTN.pdf",
                page=51,
                line_number=270,
                char_start=46,
            ),
        )

        compatible = _role_compatible_pair_indices(
            "No phai tra chiem bao nhieu phan tram tong cong tai san cuoi nam 2025?",
            "RATIO_PERCENT",
            candidates,
            {0, 1, 2, 3},
        )

        self.assertEqual(compatible, ({0}, {2}))

    def test_temporal_role_constraint_keeps_question_metric(self) -> None:
        candidates = (
            _numeric_candidate(0, "200", "Doanh thu thuần", "2025", "ocr"),
            _numeric_candidate(1, "100", "Doanh thu thuần", "2024", "ocr"),
            _numeric_candidate(2, "80", "Chi phí quản lý", "2025", "ocr"),
            _numeric_candidate(3, "70", "Chi phí quản lý", "2024", "ocr"),
        )
        allowed = _temporal_role_compatible_indices(
            "Doanh thu thuần thay đổi bao nhiêu phần trăm từ 2024 đến 2025?",
            "PERCENT_CHANGE",
            candidates,
            {0, 1, 2, 3},
        )
        self.assertEqual(allowed, {0, 1})

    def test_temporal_role_constraint_applies_to_structured_native_rows(self) -> None:
        candidates = (
            _numeric_candidate(0, "40074851708537", "TỔNG CỘNG TÀI SẢN", "", "native"),
            _numeric_candidate(1, "36362339883577", "TỔNG CỘNG TÀI SẢN", "", "native"),
            _numeric_candidate(2, "2210570390648", "TỔNG CỘNG", "", "native"),
            _numeric_candidate(3, "7564703325", "Dự phòng công trình", "", "native"),
        )

        allowed = _temporal_role_compatible_indices(
            "Tổng cộng tài sản cuối năm tăng bao nhiêu so với đầu năm?",
            "SUBTRACT",
            candidates,
            {0, 1, 2, 3},
        )

        self.assertEqual(allowed, {0, 1})

    def test_temporal_total_asset_anchor_rejects_investment_property(self) -> None:
        candidates = (
            _numeric_candidate(
                0, "2478048502242", "Bat dong san dau tu", "cuoi nam", "native"
            ),
            _numeric_candidate(
                1, "2542405763957", "Bat dong san dau tu", "dau nam", "native"
            ),
            _numeric_candidate(
                2, "40074851708537", "Tong cong tai san", "cuoi nam", "native"
            ),
            _numeric_candidate(
                3, "36362339883577", "Tong cong tai san", "dau nam", "native"
            ),
        )

        allowed = _temporal_role_compatible_indices(
            "Tong cong tai san cuoi nam 2025 tang bao nhieu so voi dau nam 2025?",
            "SUBTRACT",
            candidates,
            {0, 1, 2, 3},
        )

        self.assertEqual(allowed, {2, 3})

    def test_subtract_decrease_orients_baseline_before_current(self) -> None:
        candidates = (
            _numeric_candidate(0, "1452678263060", "Tổng tài sản", "2025"),
            _numeric_candidate(1, "1455150771623", "Tổng tài sản", "2024"),
        )
        operands = _align_directed_operands(
            "Tổng tài sản năm 2025 giảm bao nhiêu so với năm 2024?",
            "SUBTRACT",
            (0, 1),
            candidates,
        )
        self.assertEqual(operands, (1, 0))

    def test_subtract_comparison_preserves_signed_period_direction(self) -> None:
        candidates = (
            _numeric_candidate(0, "106604863", "Doanh thu", "2024"),
            _numeric_candidate(1, "45459982069", "Doanh thu", "2023"),
        )
        operands = _align_directed_operands(
            "So sánh doanh thu năm 2024 và năm 2023. Thay đổi như thế nào?",
            "SUBTRACT",
            (0, 1),
            candidates,
        )
        self.assertEqual(operands, (0, 1))

    def test_explicit_temporal_difference_preserves_signed_period_order(self) -> None:
        candidates = (
            _numeric_candidate(0, "933.75", "Doanh thu thuần", "2023"),
            _numeric_candidate(1, "3475.71", "Doanh thu thuần", "2022"),
        )
        operands = _align_directed_operands(
            "Chênh lệch Doanh thu thuần giữa 2023 và 2022 là bao nhiêu?",
            "SUBTRACT",
            (1, 0),
            candidates,
        )
        self.assertEqual(operands, (0, 1))

    def test_from_to_absolute_change_preserves_signed_period_order(self) -> None:
        candidates = (
            _numeric_candidate(0, "-6295683732", "Loi nhuan sau thue", "2024"),
            _numeric_candidate(1, "703150885", "Loi nhuan sau thue", "2023"),
        )
        operands = _align_directed_operands(
            "Loi nhuan sau thue thay doi bao nhieu tu nam 2023 den nam 2024?",
            "SUBTRACT",
            (1, 0),
            candidates,
        )

        self.assertEqual(operands, (0, 1))

    def test_explicit_relative_difference_preserves_current_prior_order(self) -> None:
        current = _numeric_candidate(
            0, "90", "Nợ phải trả", "cuoi nam", "native"
        )
        prior = _numeric_candidate(
            1, "100", "Nợ phải trả", "dau nam", "native"
        )
        operands = _align_directed_operands(
            "Chênh lệch Nợ phải trả giữa Số cuối năm và Số đầu năm là bao nhiêu?",
            "SUBTRACT",
            (1, 0),
            (current, prior),
        )
        self.assertEqual(operands, (0, 1))

    def test_exact_opening_date_keeps_calendar_year_without_double_shift(self) -> None:
        opening = _numeric_candidate(
            0, "100", "Tong nguon von", "01/01/2025", "native"
        )
        closing = _numeric_candidate(
            1, "120", "Tong nguon von", "31/12/2025", "native"
        )

        self.assertEqual(_candidate_period_role(opening), "prior")
        self.assertEqual(_candidate_period_role(closing), "current")
        self.assertEqual(_candidate_period_years(opening), {"2025"})
        self.assertEqual(_candidate_period_years(closing), {"2025"})
        self.assertEqual(
            _candidate_period_years(
                opening,
                "Tong nguon von tai 01/01/2025 la bao nhieu?",
            ),
            {"2025"},
        )

    def test_relative_start_end_headers_keep_legacy_role_mapping(self) -> None:
        opening = _numeric_candidate(
            0, "100", "Tong nguon von", "dau nam 2025", "native"
        )
        closing = _numeric_candidate(
            1, "120", "Tong nguon von", "cuoi nam 2025", "native"
        )

        self.assertEqual(_candidate_period_role(opening), "prior")
        self.assertEqual(_candidate_period_role(closing), "current")
        self.assertEqual(_candidate_period_years(opening), {"2024"})
        self.assertEqual(_candidate_period_years(closing), {"2025"})

    def test_ambiguous_negative_balances_are_not_oriented_by_magnitude(self) -> None:
        current = _numeric_candidate(
            0, "-164724941920", "Giá trị hao mòn lũy kế", "2024"
        )
        prior = _numeric_candidate(
            1, "-163588262992", "Giá trị hao mòn lũy kế", "2023"
        )
        operands = _align_directed_operands(
            "So sánh giá trị hao mòn lũy kế giữa hai kỳ.",
            "SUBTRACT",
            (0, 1),
            (current, prior),
        )
        self.assertEqual(operands, (0, 1))

    def test_percent_change_from_to_orients_new_before_baseline(self) -> None:
        candidates = (
            _numeric_candidate(0, "1172121547172", "Doanh thu thuần", "2024"),
            _numeric_candidate(1, "844346681090", "Doanh thu thuần", "2025"),
        )
        operands = _align_directed_operands(
            "Doanh thu thay đổi bao nhiêu phần trăm từ năm 2024 sang năm 2025?",
            "PERCENT_CHANGE",
            (0, 1),
            candidates,
        )
        self.assertEqual(operands, (1, 0))

    def test_percent_change_from_to_with_den_orients_new_before_baseline(self) -> None:
        candidates = (
            _numeric_candidate(0, "100", "Doanh thu thuần", "2024"),
            _numeric_candidate(1, "200", "Doanh thu thuần", "2025"),
        )
        operands = _align_directed_operands(
            "Doanh thu thay đổi bao nhiêu phần trăm từ năm 2024 đến năm 2025?",
            "PERCENT_CHANGE",
            (0, 1),
            candidates,
        )
        self.assertEqual(operands, (1, 0))

    def test_percent_change_from_to_with_parenthetical_units_orients_new_first(self) -> None:
        candidates = (
            _numeric_candidate(0, "100", "Doanh thu thuần", "2024"),
            _numeric_candidate(1, "200", "Doanh thu thuần", "2025"),
        )
        operands = _align_directed_operands(
            "Doanh thu thay đổi bao nhiêu phần trăm từ "
            "Năm 2024 (VND) đến Năm 2025 (VND)?",
            "PERCENT_CHANGE",
            (0, 1),
            candidates,
        )

        self.assertEqual(operands, (1, 0))

    def test_start_end_increase_orients_current_before_prior(self) -> None:
        current = replace(
            _numeric_candidate(0, "200", "Tổng tài sản", extraction_method="ocr"),
            context=(
                "ROW: Tổng tài sản\nROW_TYPE: total\n"
                "COLUMN: Cuối năm 2025\nPERIOD: cuoi nam 2025"
            ),
        )
        prior = replace(
            _numeric_candidate(1, "100", "Tổng tài sản", extraction_method="ocr"),
            context=(
                "ROW: Tổng tài sản\nROW_TYPE: total\n"
                "COLUMN: Đầu năm 2025\nPERIOD: dau nam 2025"
            ),
        )
        operands = _align_directed_operands(
            "Tổng tài sản cuối năm 2025 tăng bao nhiêu so với đầu năm 2025?",
            "SUBTRACT",
            (1, 0),
            (current, prior),
        )
        self.assertEqual(operands, (0, 1))

    def test_start_end_same_year_keeps_both_relative_periods(self) -> None:
        current = replace(
            _numeric_candidate(0, "200", "Tổng tài sản", extraction_method="ocr"),
            context=(
                "ROW: Tổng tài sản\nROW_TYPE: total\n"
                "COLUMN: Số cuối năm\nPERIOD: cuoi nam"
            ),
        )
        prior = replace(
            _numeric_candidate(1, "100", "Tổng tài sản", extraction_method="ocr"),
            context=(
                "ROW: Tổng tài sản\nROW_TYPE: total\n"
                "COLUMN: Số đầu năm\nPERIOD: dau nam"
            ),
        )

        compatible = _period_compatible_indices(
            "Tổng tài sản cuối năm 2025 tăng bao nhiêu so với đầu năm 2025?",
            "SUBTRACT",
            (current, prior),
            2,
        )

        self.assertEqual(compatible, {0, 1})

    def test_same_row_without_header_preserves_unproven_neural_order(self) -> None:
        current = replace(
            _numeric_candidate(0, "4714032275888", "Tổng cộng tài sản"),
            source_name="ABS.pdf",
            page=31,
            line_number=100,
            char_start=32,
            char_end=45,
        )
        prior = replace(
            _numeric_candidate(1, "3390042913247", "Tổng cộng tài sản"),
            source_name="ABS.pdf",
            page=31,
            line_number=100,
            char_start=48,
            char_end=61,
        )
        candidates = (current, prior)

        operands = _align_directed_operands(
            "Tổng cộng tài sản cuối năm 2025 tăng bao nhiêu so với đầu năm 2025?",
            "SUBTRACT",
            (1, 0),
            candidates,
        )

        self.assertEqual(operands, (1, 0))

    def test_same_row_decrease_without_header_preserves_unproven_order(self) -> None:
        current = replace(
            _numeric_candidate(0, "284331609643", "Tổng cộng tài sản"),
            source_name="ABR.pdf",
            page=33,
            line_number=100,
            char_start=32,
            char_end=44,
        )
        prior = replace(
            _numeric_candidate(1, "354781425043", "Tổng cộng tài sản"),
            source_name="ABR.pdf",
            page=33,
            line_number=100,
            char_start=47,
            char_end=59,
        )
        candidates = (current, prior)

        operands = _align_directed_operands(
            "Tổng cộng tài sản cuối năm 2025 giảm bao nhiêu so với đầu năm 2025?",
            "SUBTRACT",
            (0, 1),
            candidates,
        )

        self.assertEqual(operands, (0, 1))


def _batch() -> EncodedBatch:
    batch_size = 2
    candidates = 4
    return EncodedBatch(
        question_ids=torch.randint(1, 100, (batch_size, 6)),
        question_mask=torch.ones(batch_size, 6),
        question_features=torch.zeros(batch_size, 12),
        candidate_ids=torch.randint(1, 100, (batch_size, candidates, 5)),
        candidate_token_mask=torch.ones(batch_size, candidates, 5),
        numeric_features=torch.zeros(batch_size, candidates, 18),
        candidate_mask=torch.ones(batch_size, candidates),
        operation_targets=torch.tensor([2, 5]),
        pointer_targets=torch.tensor([[0, 1, -1, -1], [2, 3, -1, -1]]),
    )


class SequentialPointerTests(unittest.TestCase):
    def test_pointer_heads_only_validates_safe_training_contract(self) -> None:
        joint_config = _small_config(pointer_conditioning_mode="joint_pair")
        with self.assertRaisesRegex(DocumentHRMError, "requires initialize_from"):
            train_document_hrm(
                "missing.jsonl",
                config=joint_config,
                pointer_heads_only=True,
                preserve_initial_threshold=True,
            )
        with self.assertRaisesRegex(DocumentHRMError, "joint_pair conditioning"):
            train_document_hrm(
                "missing.jsonl",
                config=_small_config(pointer_conditioning_mode="independent"),
                initialize_from="missing.pt",
                pointer_heads_only=True,
                preserve_initial_threshold=True,
            )
        with self.assertRaisesRegex(
            DocumentHRMError, "requires preserve_initial_threshold"
        ):
            train_document_hrm(
                "missing.jsonl",
                config=joint_config,
                initialize_from="missing.pt",
                pointer_heads_only=True,
            )

    def test_pointer_heads_only_loss_excludes_operation_and_q(self) -> None:
        operation_loss = torch.tensor(1.0, requires_grad=True)
        pointer_loss = torch.tensor(2.0, requires_grad=True)
        pair_loss = torch.tensor(3.0, requires_grad=True)
        q_loss = torch.tensor(4.0, requires_grad=True)

        loss = _training_segment_loss(
            operation_loss=operation_loss,
            pointer_loss=pointer_loss,
            pair_loss=pair_loss,
            q_loss=q_loss,
            pair_loss_scale=0.75,
            pointer_heads_only=True,
        )
        loss.backward()

        self.assertIsNone(operation_loss.grad)
        self.assertEqual(pointer_loss.grad.item(), 2.0)
        self.assertEqual(pair_loss.grad.item(), 0.75)
        self.assertIsNone(q_loss.grad)

    def test_deep_supervision_favors_later_reasoning_segments(self) -> None:
        segment_losses = [
            torch.tensor(1.0, requires_grad=True),
            torch.tensor(2.0, requires_grad=True),
            torch.tensor(3.0, requires_grad=True),
        ]

        loss = _combine_segment_losses(segment_losses)
        loss.backward()

        self.assertAlmostEqual(float(loss.detach()), 14.0 / 6.0, places=6)
        self.assertAlmostEqual(segment_losses[0].grad.item(), 1.0 / 6.0)
        self.assertAlmostEqual(segment_losses[1].grad.item(), 2.0 / 6.0)
        self.assertAlmostEqual(segment_losses[2].grad.item(), 3.0 / 6.0)

    def test_pointer_heads_only_stage_isolates_unary_and_pair_parameters(self) -> None:
        model = DocumentHierarchicalReasoningModel(
            _small_config(pointer_conditioning_mode="joint_pair")
        )

        _configure_pointer_heads_only_training(model)

        for name, parameter in model.named_parameters():
            expected = (
                name == "pointer_relevance_scale"
                or name.startswith("pointer_")
                or name.startswith("pair_")
            )
            self.assertEqual(parameter.requires_grad, expected, name)

    def test_pointer_heads_only_stage_requires_joint_pair_parameters(self) -> None:
        model = DocumentHierarchicalReasoningModel(
            _small_config(pointer_conditioning_mode="independent")
        )

        with self.assertRaisesRegex(DocumentHRMError, "joint-pair parameters"):
            _configure_pointer_heads_only_training(model)

    def test_pointer_heads_only_optimizer_contains_only_heads_without_decay(self) -> None:
        model = DocumentHierarchicalReasoningModel(
            _small_config(pointer_conditioning_mode="joint_pair")
        )
        _configure_pointer_heads_only_training(model)

        optimizer = _build_pointer_heads_only_optimizer(
            model,
            learning_rate=6e-4,
        )

        optimized = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        expected = {
            id(parameter)
            for parameter in model.parameters()
            if parameter.requires_grad
        }
        self.assertEqual(optimized, expected)
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [6e-4])
        self.assertEqual([group["weight_decay"] for group in optimizer.param_groups], [0.0])

    def test_pair_head_only_stage_freezes_and_restores_backbone(self) -> None:
        model = DocumentHierarchicalReasoningModel(
            _small_config(pointer_conditioning_mode="joint_pair")
        )

        _configure_pair_training_stage(model, pair_head_only=True)

        self.assertTrue(
            all(
                parameter.requires_grad
                for name, parameter in model.named_parameters()
                if name.startswith("pair_")
            )
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for name, parameter in model.named_parameters()
                if not name.startswith("pair_")
            )
        )

        _configure_pair_training_stage(model, pair_head_only=False)

        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

    def test_pair_head_only_stage_rejects_model_without_pair_head(self) -> None:
        model = DocumentHierarchicalReasoningModel(
            _small_config(pointer_conditioning_mode="independent")
        )

        with self.assertRaises(DocumentHRMError):
            _configure_pair_training_stage(model, pair_head_only=True)

    def test_staged_training_can_keep_operation_head_frozen(self) -> None:
        model = DocumentHierarchicalReasoningModel(
            _small_config(pointer_conditioning_mode="joint_pair")
        )

        _configure_pair_training_stage(
            model,
            pair_head_only=False,
            freeze_operation_head=True,
        )

        self.assertTrue(
            all(
                not parameter.requires_grad
                for name, parameter in model.named_parameters()
                if name.startswith("operation_head.")
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for name, parameter in model.named_parameters()
                if not name.startswith("operation_head.")
            )
        )

    def test_staged_optimizer_uses_lower_backbone_learning_rate(self) -> None:
        model = DocumentHierarchicalReasoningModel(
            _small_config(pointer_conditioning_mode="joint_pair")
        )

        optimizer = _build_staged_pair_optimizer(
            model,
            learning_rate=6e-4,
            backbone_learning_rate=6e-5,
        )

        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups],
            [6e-5, 6e-4],
        )

    def test_pointer_head_only_stage_isolates_unary_pointer_parameters(self) -> None:
        model = DocumentHierarchicalReasoningModel(
            _small_config(pointer_conditioning_mode="joint_pair")
        )

        _configure_unary_pointer_training_stage(
            model,
            pointer_head_only=True,
            freeze_operation_head=True,
            freeze_pair_head=True,
        )

        for name, parameter in model.named_parameters():
            expected = name == "pointer_relevance_scale" or name.startswith(
                "pointer_"
            )
            self.assertEqual(parameter.requires_grad, expected, name)

    def test_pointer_stage_keeps_operation_and_pair_heads_frozen(self) -> None:
        model = DocumentHierarchicalReasoningModel(
            _small_config(pointer_conditioning_mode="joint_pair")
        )

        _configure_unary_pointer_training_stage(
            model,
            pointer_head_only=False,
            freeze_operation_head=True,
            freeze_pair_head=True,
        )

        for name, parameter in model.named_parameters():
            expected = not (
                name.startswith("operation_head.") or name.startswith("pair_")
            )
            self.assertEqual(parameter.requires_grad, expected, name)

    def test_staged_pointer_optimizer_uses_lower_backbone_learning_rate(self) -> None:
        model = DocumentHierarchicalReasoningModel(
            _small_config(pointer_conditioning_mode="joint_pair")
        )

        optimizer = _build_staged_pointer_optimizer(
            model,
            learning_rate=6e-4,
            backbone_learning_rate=6e-5,
        )

        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups],
            [6e-5, 6e-4],
        )

    def test_select_pointer_training_only_opens_existing_select_bilinear(self) -> None:
        model = DocumentHierarchicalReasoningModel(
            _small_config(pointer_conditioning_mode="joint_pair")
        )

        _configure_select_pointer_training(model)

        for name, parameter in model.named_parameters():
            self.assertEqual(
                parameter.requires_grad,
                name == "pair_operation_bilinear",
                name,
            )

    def test_select_pointer_optimizer_has_no_weight_decay(self) -> None:
        model = DocumentHierarchicalReasoningModel(
            _small_config(pointer_conditioning_mode="joint_pair")
        )

        optimizer = _build_select_pointer_optimizer(
            model,
            learning_rate=6e-4,
        )

        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.param_groups[0]["lr"], 6e-4)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.0)

    def test_training_initialization_requires_identical_architecture(self) -> None:
        config = _small_config(pointer_conditioning_mode="joint_pair")
        model = DocumentHierarchicalReasoningModel(config)
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "base.pt"
            DocumentHRMBundle(model).save(checkpoint)

            loaded = _load_compatible_training_bundle(checkpoint, config, "cpu")
            self.assertEqual(loaded.model.config, config)

            with self.assertRaises(DocumentHRMError):
                _load_compatible_training_bundle(
                    checkpoint,
                    replace(config, hidden_size=128),
                    "cpu",
                )

    def test_forward_emits_all_pointer_slots(self) -> None:
        model = DocumentHierarchicalReasoningModel(
            _small_config(
                pointer_conditioning_mode="operation_sequential",
                pointer_teacher_forcing=0.5,
            )
        )
        output, _ = model.forward_segment(_batch())

        self.assertEqual(tuple(output.operation_logits.shape), (2, 10))
        self.assertEqual(tuple(output.pointer_logits.shape), (2, 4, 4))
        self.assertIsNotNone(model.pointer_operation_embedding)

    def test_independent_mode_preserves_legacy_parameter_layout(self) -> None:
        model = DocumentHierarchicalReasoningModel(
            _small_config(pointer_conditioning_mode="independent")
        )

        self.assertIsNone(model.pointer_operation_embedding)
        self.assertFalse(
            any(name.startswith("pointer_conditioner") for name in model.state_dict())
        )

    def test_second_pointer_uses_first_pointer_context_during_training(self) -> None:
        model = DocumentHierarchicalReasoningModel(
            _small_config(
                dropout=0.0,
                pointer_conditioning_mode="operation_sequential",
                pointer_teacher_forcing=1.0,
            )
        )
        original = _batch()
        changed = replace(
            original,
            pointer_targets=torch.tensor([[3, 1, -1, -1], [0, 3, -1, -1]]),
        )

        original_output, _ = model.forward_segment(original)
        changed_output, _ = model.forward_segment(changed)

        self.assertFalse(
            torch.allclose(
                original_output.pointer_logits[:, 1],
                changed_output.pointer_logits[:, 1],
            )
        )

    def test_later_sequential_keeps_first_pointer_independent(self) -> None:
        model = DocumentHierarchicalReasoningModel(
            _small_config(
                dropout=0.0,
                pointer_conditioning_mode="later_sequential",
                pointer_teacher_forcing=1.0,
            )
        )
        original = _batch()
        changed = replace(
            original,
            pointer_targets=torch.tensor([[3, 1, -1, -1], [0, 3, -1, -1]]),
        )

        original_output, _ = model.forward_segment(original)
        changed_output, _ = model.forward_segment(changed)

        self.assertTrue(
            torch.allclose(
                original_output.pointer_logits[:, 0],
                changed_output.pointer_logits[:, 0],
            )
        )
        self.assertFalse(
            torch.allclose(
                original_output.pointer_logits[:, 1],
                changed_output.pointer_logits[:, 1],
            )
        )

    def test_joint_pair_reranker_is_detached_from_unary_pointer(self) -> None:
        model = DocumentHierarchicalReasoningModel(
            _small_config(
                dropout=0.0,
                pointer_conditioning_mode="joint_pair",
            )
        )
        batch = replace(
            _batch(),
            pair_features=torch.zeros(2, 4, 4, PAIR_FEATURE_SIZE),
        )

        output, _ = model.forward_segment(batch)

        self.assertIsNotNone(output.pair_logits)
        assert output.pair_logits is not None
        self.assertEqual(tuple(output.pair_logits.shape), (2, 10, 4, 4))
        self.assertLess(float(output.pair_logits[0, 2, 0, 0].detach()), -1e8)
        output.pair_logits[0, 2, 0, 1].backward()
        self.assertIsNotNone(model.pair_operation_bilinear.grad)
        self.assertIsNone(model.pointer_queries[0].weight.grad)

    def test_end_to_end_joint_pair_trains_unary_pointer(self) -> None:
        model = DocumentHierarchicalReasoningModel(
            _small_config(
                dropout=0.0,
                pointer_conditioning_mode="joint_pair",
                pair_backbone_mode="end_to_end",
            )
        )
        batch = replace(
            _batch(),
            pair_features=torch.zeros(2, 4, 4, PAIR_FEATURE_SIZE),
        )

        output, _ = model.forward_segment(batch)

        assert output.pair_logits is not None
        output.pair_logits[0, 2, 0, 1].backward()
        self.assertIsNotNone(model.pointer_queries[0].weight.grad)

    def test_normalized_end_to_end_pair_relation_is_bounded(self) -> None:
        model = DocumentHierarchicalReasoningModel(
            _small_config(
                dropout=0.0,
                pointer_conditioning_mode="joint_pair",
                pair_backbone_mode="normalized_end_to_end",
            )
        )
        batch = replace(
            _batch(),
            pair_features=torch.zeros(2, 4, 4, PAIR_FEATURE_SIZE),
        )
        with torch.no_grad():
            for query in model.pointer_queries:
                query.weight.zero_()
                query.bias.zero_()
            assert model.pair_operation_bilinear is not None
            assert model.pair_feature_weights is not None
            assert model.pair_relation_scale is not None
            model.pair_operation_bilinear.fill_(1000.0)
            model.pair_feature_weights.zero_()
            model.pair_relation_scale.fill_(20.0)

        output, _ = model.forward_segment(batch)

        assert output.pair_logits is not None
        valid = output.pair_logits[0, 2][~torch.eye(4, dtype=torch.bool)]
        self.assertLessEqual(float(valid.abs().max().detach()), 1.00001)

    def test_normalized_detached_pair_does_not_train_unary_pointer(self) -> None:
        model = DocumentHierarchicalReasoningModel(
            _small_config(
                dropout=0.0,
                pointer_conditioning_mode="joint_pair",
                pair_backbone_mode="normalized_detached",
            )
        )
        batch = replace(
            _batch(),
            pair_features=torch.zeros(2, 4, 4, PAIR_FEATURE_SIZE),
        )

        output, _ = model.forward_segment(batch)

        assert output.pair_logits is not None
        output.pair_logits[0, 2, 0, 1].backward()
        self.assertIsNotNone(model.pair_operation_bilinear.grad)
        self.assertIsNone(model.pointer_queries[0].weight.grad)

    def test_warmup_pair_switch_controls_backbone_gradient(self) -> None:
        model = DocumentHierarchicalReasoningModel(
            _small_config(
                dropout=0.0,
                pointer_conditioning_mode="joint_pair",
                pair_backbone_mode="warmup_end_to_end",
                pair_backbone_warmup_epochs=1,
            )
        )
        batch = replace(
            _batch(),
            pair_features=torch.zeros(2, 4, 4, PAIR_FEATURE_SIZE),
        )

        warmup_output, _ = model.forward_segment(batch)
        assert warmup_output.pair_logits is not None
        warmup_output.pair_logits[0, 2, 0, 1].backward()
        self.assertIsNone(model.pointer_queries[0].weight.grad)

        model.zero_grad(set_to_none=True)
        model.pair_backbone_detached = False
        joint_output, _ = model.forward_segment(batch)
        assert joint_output.pair_logits is not None
        joint_output.pair_logits[0, 2, 0, 1].backward()
        self.assertIsNotNone(model.pointer_queries[0].weight.grad)

    def test_pair_loss_scale_ramps_to_configured_weight(self) -> None:
        config = _small_config(
            pair_loss_weight=0.2,
            pair_loss_warmup_epochs=4,
        )

        self.assertEqual(_pair_loss_scale(config, 0), 0.0)
        self.assertAlmostEqual(_pair_loss_scale(config, 1), 0.05)
        self.assertAlmostEqual(_pair_loss_scale(config, 4), 0.2)
        self.assertAlmostEqual(_pair_loss_scale(config, 8), 0.2)

    def test_calibration_requires_one_error_precision_buffer(self) -> None:
        self.assertFalse(
            _precision_survives_errors(
                12,
                15,
                target_precision=0.8,
                stability_errors=1,
            )
        )
        self.assertTrue(
            _precision_survives_errors(
                13,
                15,
                target_precision=0.8,
                stability_errors=1,
            )
        )
        self.assertTrue(
            _precision_survives_errors(
                12,
                15,
                target_precision=0.8,
                stability_errors=0,
            )
        )

    def test_operation_calibration_maximizes_safe_global_coverage(self) -> None:
        first = replace(
            _numeric_candidate(0, "100", "Doanh thu"),
            unit="VND",
        )
        second = replace(
            _numeric_candidate(1, "200", "Chi phÃ­"),
            unit="VND",
        )
        select_examples = [
            ProgramExample(
                example_id=f"select-calibration-{index}",
                question="Doanh thu lÃ  bao nhiÃªu?",
                context="",
                answer_text="100",
                candidates=(first, second),
                program=ReasoningProgram("SELECT", (0,)),
                answer_value="100",
            )
            for index in range(2)
        ]
        add_examples = [
            ProgramExample(
                example_id=f"add-calibration-{index}",
                question="Tá»•ng doanh thu vÃ  chi phÃ­ lÃ  bao nhiÃªu?",
                context="",
                answer_text="300",
                candidates=(first, second),
                program=ReasoningProgram("ADD", (0, 1)),
                answer_value="300",
            )
            for index in range(2)
        ]
        predictions = [
            SimpleNamespace(
                confidence=0.90,
                program=ReasoningProgram("SELECT", (1,)),
            ),
            SimpleNamespace(
                confidence=0.80,
                program=ReasoningProgram("SELECT", (0,)),
            ),
            SimpleNamespace(
                confidence=0.70,
                program=ReasoningProgram("ADD", (0, 1)),
            ),
            SimpleNamespace(
                confidence=0.60,
                program=ReasoningProgram("ADD", (0, 1)),
            ),
        ]
        bundle = SimpleNamespace(
            predict_many=Mock(return_value=predictions)
        )

        thresholds, accepted, precision, target_met = (
            calibrate_operation_abstention_thresholds(
                bundle,
                [*select_examples, *add_examples],
                target_precision=0.80,
                minimum_accepted=2,
                stability_errors=0,
            )
        )

        self.assertEqual(accepted, 3)
        self.assertEqual(precision, 1.0)
        self.assertTrue(target_met)
        self.assertEqual(thresholds["ADD"], 0.60)
        self.assertEqual(thresholds["SELECT"], 0.80)

    def test_operation_threshold_overrides_global_threshold(self) -> None:
        bundle = SimpleNamespace(
            metadata={
                "abstention_threshold": 0.90,
                "abstention_thresholds_by_operation": {
                    "ADD": 0.55,
                },
            }
        )

        self.assertEqual(
            _abstention_threshold_for_operation(bundle, "ADD"), 0.55
        )
        self.assertEqual(
            _abstention_threshold_for_operation(bundle, "SELECT"), 0.90
        )

    def test_grounding_gate_rejects_selected_wrong_period(self) -> None:
        candidate = _numeric_candidate(
            0, "100", "Doanh thu thuần", "2024", extraction_method="native"
        )

        reasons = _grounding_gate_reasons(
            "Doanh thu thuần năm 2025 là bao nhiêu?",
            ReasoningProgram("SELECT", (0,)),
            (candidate,),
        )

        self.assertIn("missing_requested_period", reasons)

    def test_grounding_gate_rejects_reversed_ratio_roles(self) -> None:
        debt = _numeric_candidate(
            0, "40", "Nợ phải trả", "2025", extraction_method="native"
        )
        assets = _numeric_candidate(
            1, "100", "Tổng tài sản", "2025", extraction_method="native"
        )

        reasons = _grounding_gate_reasons(
            "Nợ phải trả chiếm bao nhiêu phần trăm tổng tài sản năm 2025?",
            ReasoningProgram("RATIO_PERCENT", (1, 0)),
            (debt, assets),
        )

        self.assertIn("operand_role_mismatch", reasons)

    def test_grounding_gate_rejects_incompatible_additive_units(self) -> None:
        cash = replace(_numeric_candidate(0, "100", "Tiền"), unit="VND")
        shares = replace(_numeric_candidate(1, "20", "Cổ phiếu"), unit="shares")

        reasons = _grounding_gate_reasons(
            "Tính tổng tiền và số cổ phiếu.",
            ReasoningProgram("ADD", (0, 1)),
            (cash, shares),
        )

        self.assertIn("incompatible_operand_units", reasons)

    def test_grounding_gate_rejects_implicit_add_and_extra_period(self) -> None:
        current = _numeric_candidate(
            0, "5407553257", "Doanh thu thuần", "2023", "native"
        )
        prior = _numeric_candidate(
            1, "73112", "Doanh thu thuần", "2022", "native"
        )

        reasons = _grounding_gate_reasons(
            "Doanh thu thuần năm 2023",
            ReasoningProgram("ADD", (0, 1)),
            (current, prior),
        )

        self.assertIn("missing_explicit_addition_intent", reasons)
        self.assertIn("operand_outside_requested_period", reasons)

    def test_grounding_gate_accepts_explicit_same_period_add(self) -> None:
        revenue = _numeric_candidate(
            0, "100", "Doanh thu thuần", "2023", "native"
        )
        cost = _numeric_candidate(
            1, "40", "Giá vốn hàng bán", "2023", "native"
        )

        reasons = _grounding_gate_reasons(
            "Tính tổng Doanh thu thuần và Giá vốn hàng bán tại năm 2023.",
            ReasoningProgram("ADD", (0, 1)),
            (revenue, cost),
        )

        self.assertNotIn("missing_explicit_addition_intent", reasons)
        self.assertNotIn("operand_outside_requested_period", reasons)

    def test_grounding_gate_accepts_well_grounded_ratio(self) -> None:
        debt = replace(
            _numeric_candidate(
                0, "40", "Nợ phải trả", "2025", extraction_method="native"
            ),
            unit="VND",
        )
        assets = replace(
            _numeric_candidate(
                1, "100", "Tổng tài sản", "2025", extraction_method="native"
            ),
            unit="VND",
        )

        reasons = _grounding_gate_reasons(
            "Nợ phải trả chiếm bao nhiêu phần trăm tổng tài sản năm 2025?",
            ReasoningProgram("RATIO_PERCENT", (0, 1)),
            (debt, assets),
        )

        self.assertEqual(reasons, ())

    def test_targeted_pair_margin_penalizes_structural_near_miss(self) -> None:
        scores = torch.zeros(1, 9)
        scores[0, 1] = 1.0
        scores[0, 2] = 2.0
        targets = torch.tensor([1])
        features = torch.zeros(1, 3, 3, 10)
        features[0, 0, 2, 0] = 1.0

        loss = _targeted_pair_margin_loss(scores, targets, features)

        self.assertGreater(float(loss), 0.0)

    def test_targeted_pair_margin_consumes_explicit_structured_negative(self) -> None:
        scores = torch.zeros(1, 9)
        scores[0, 1] = 1.0
        scores[0, 2] = 2.0
        targets = torch.tensor([1])
        features = torch.zeros(1, 3, 3, 10)
        explicit = torch.zeros(1, 3, 3, dtype=torch.bool)
        explicit[0, 0, 2] = True

        loss = _targeted_pair_margin_loss(scores, targets, features, explicit)

        self.assertGreater(float(loss), 0.0)

    def test_structured_negative_indices_and_permutation_preserve_targets(self) -> None:
        candidates = tuple(
            _numeric_candidate(index, str(100 - index), f"row-{index}", "2024")
            for index in range(4)
        )
        example = ProgramExample(
            example_id="structured-negative",
            question="A tru B la bao nhieu?",
            context="",
            answer_text="1",
            candidates=candidates,
            program=ReasoningProgram("SUBTRACT", (0, 1)),
            answer_value="1",
            supervision={
                "gold_operands": [
                    {"candidate_index": 0},
                    {"candidate_index": 1},
                ],
                "hard_negatives": [
                    {
                        "type": "same_period_wrong_row",
                        "operand_slot": 0,
                        "candidate_indices": [2],
                    },
                    {
                        "type": "reversed_operand_order",
                        "operand_slot": None,
                        "candidate_indices": [1, 0],
                    },
                ],
            },
        )

        unary, pairs = _structured_negative_indices(example)
        self.assertEqual(unary[0], (2,))
        self.assertEqual(set(pairs), {(1, 0), (2, 1)})

        permuted = _permute_candidates(example, random.Random(7))
        assert permuted.supervision is not None
        gold = permuted.supervision["gold_operands"]
        negatives = permuted.supervision["hard_negatives"]
        self.assertEqual(
            [item["candidate_index"] for item in gold],
            list(permuted.program.operands),
        )
        self.assertEqual(
            negatives[1]["candidate_indices"],
            [permuted.program.operands[1], permuted.program.operands[0]],
        )

    def test_unary_structural_negatives_cover_row_and_period_near_misses(self) -> None:
        candidates = (
            _numeric_candidate(0, "100", "Doanh thu thuan", "2024"),
            _numeric_candidate(1, "90", "Doanh thu thuan", "2023"),
            _numeric_candidate(2, "500", "Tong tai san", "2024"),
            _numeric_candidate(3, "12", "Ma so"),
        )

        targeted = _unary_structural_negative_mask(candidates, 0)

        self.assertEqual(targeted, [False, True, True, False])

    def test_unary_structural_negatives_do_not_guess_missing_metadata(self) -> None:
        candidates = (
            _numeric_candidate(0, "100", "Doanh thu thuan", "2024"),
            _numeric_candidate(1, "90", "Doanh thu thuan"),
        )

        targeted = _unary_structural_negative_mask(candidates, 0)

        self.assertEqual(targeted, [False, False])

    def test_targeted_unary_margin_penalizes_only_structural_near_miss(self) -> None:
        scores = torch.tensor([[1.0, 2.0, 9.0]])
        targets = torch.tensor([0])
        targeted = torch.tensor([[False, True, False]])

        loss = _targeted_unary_margin_loss(
            scores, targets, targeted, margin=0.5
        )

        self.assertAlmostEqual(float(loss), 1.5, places=6)

    def test_calibration_floor_does_not_admit_unprovable_grounding(self) -> None:
        candidate = _numeric_candidate(0, "100", "Doanh thu")
        examples = [
            ProgramExample(
                example_id=f"floor-{index}",
                question="Gia tri la bao nhieu?",
                context="",
                answer_text="100",
                candidates=(candidate,),
                program=ReasoningProgram("SELECT", (0,)),
                answer_value="100",
            )
            for index in range(5)
        ]
        predictions = [
            SimpleNamespace(
                confidence=confidence,
                program=ReasoningProgram("SELECT", (0,)),
            )
            for confidence in (0.60, 0.70, 0.80, 0.90, 1.00)
        ]
        bundle = SimpleNamespace(predict_many=Mock(return_value=predictions))

        threshold, accepted, precision, target_met = (
            calibrate_abstention_threshold(
                bundle,
                examples,
                minimum_accepted=1,
                stability_errors=0,
                minimum_threshold=0.85,
            )
        )

        self.assertEqual(threshold, 0.85)
        self.assertEqual(accepted, 0)
        self.assertEqual(precision, 0.0)
        self.assertFalse(target_met)

    def test_select_residual_is_zero_initialized_and_isolated_from_base_pointer(self) -> None:
        model = DocumentHierarchicalReasoningModel(
            _small_config(
                dropout=0.0,
                pointer_conditioning_mode="joint_pair",
            )
        )
        batch = replace(
            _batch(),
            pair_features=torch.zeros(2, 4, 4, PAIR_FEATURE_SIZE),
        )

        initial, _ = model.forward_segment(batch)
        assert initial.select_pointer_logits is not None
        self.assertTrue(
            torch.allclose(initial.select_pointer_logits, initial.pointer_logits[:, 0])
        )

        with torch.no_grad():
            assert model.pair_left is not None and model.pair_right is not None
            assert model.pair_operation_bilinear is not None
            assert model.pair_relation_scale is not None
            model.pair_left.weight.zero_()
            model.pair_left.bias.fill_(1.0)
            model.pair_right.weight.zero_()
            model.pair_right.bias.fill_(1.0)
            model.pair_operation_bilinear[OPERATION_TO_ID["SELECT"]].copy_(
                torch.eye(model.pair_operation_bilinear.shape[-1])
            )
            model.pair_relation_scale.fill_(1.0)

        updated, _ = model.forward_segment(batch)
        assert updated.select_pointer_logits is not None
        assert initial.pair_logits is not None and updated.pair_logits is not None
        self.assertTrue(torch.allclose(updated.pointer_logits, initial.pointer_logits))
        self.assertTrue(
            torch.allclose(
                updated.pair_logits[:, OPERATION_TO_ID["SUBTRACT"]],
                initial.pair_logits[:, OPERATION_TO_ID["SUBTRACT"]],
            )
        )
        self.assertFalse(
            torch.allclose(updated.select_pointer_logits, updated.pointer_logits[:, 0])
        )

    def test_training_pointer_logits_apply_select_residual_only_to_select_rows(self) -> None:
        pointer_logits = torch.zeros(2, 4, 3)
        pointer_logits[1, 0, 2] = 4.0
        select_pointer_logits = torch.zeros(2, 3)
        select_pointer_logits[0, 1] = 5.0
        output = DocumentHRMForward(
            state=DocumentHRMState(
                torch.zeros(2, 3, 64),
                torch.zeros(2, 3, 64),
            ),
            operation_logits=torch.zeros(2, 10),
            pointer_logits=pointer_logits,
            q_values=torch.zeros(2, 2),
            select_pointer_logits=select_pointer_logits,
        )
        operation_targets = torch.tensor(
            [OPERATION_TO_ID["SELECT"], OPERATION_TO_ID["SUBTRACT"]]
        )

        logits = _training_pointer_slot_logits(output, operation_targets, 0)

        self.assertEqual(logits[0].argmax().item(), 1)
        self.assertEqual(logits[1].argmax().item(), 2)
        self.assertTrue(
            torch.equal(
                _operation_pointer_logits(output, 0, 0, "SELECT"),
                output.select_pointer_logits[0],
            )
        )

    def test_training_pointer_predictions_use_joint_pair(self) -> None:
        pointer_logits = torch.zeros(1, 4, 3)
        pointer_logits[0, 0, 2] = 5.0
        pointer_logits[0, 1, 2] = 5.0
        pair_logits = torch.zeros(1, 10, 3, 3)
        pair_logits[0, 2, 0, 1] = 9.0
        output = DocumentHRMForward(
            state=DocumentHRMState(
                torch.zeros(1, 3, 64),
                torch.zeros(1, 3, 64),
            ),
            operation_logits=torch.zeros(1, 10),
            pointer_logits=pointer_logits,
            q_values=torch.zeros(1, 2),
            pair_logits=pair_logits,
        )

        predicted = _training_pointer_predictions(
            output,
            torch.tensor([2]),
            torch.tensor([[0, 1, -1, -1]]),
        )

        self.assertEqual(predicted[0, :2].tolist(), [0, 1])

    def test_training_pointer_predictions_keep_legacy_unary_when_requested(self) -> None:
        pointer_logits = torch.zeros(1, 4, 3)
        pointer_logits[0, 0, 2] = 5.0
        pointer_logits[0, 1, 2] = 5.0
        pair_logits = torch.zeros(1, 10, 3, 3)
        pair_logits[0, 2, 0, 1] = 9.0
        output = DocumentHRMForward(
            state=DocumentHRMState(
                torch.zeros(1, 3, 64),
                torch.zeros(1, 3, 64),
            ),
            operation_logits=torch.zeros(1, 10),
            pointer_logits=pointer_logits,
            q_values=torch.zeros(1, 2),
            pair_logits=pair_logits,
        )

        predicted = _training_pointer_predictions(
            output,
            torch.tensor([2]),
            torch.tensor([[0, 1, -1, -1]]),
            use_pair=False,
        )

        self.assertEqual(predicted[0, :2].tolist(), [2, 2])

    def test_joint_pair_preserves_independent_parameter_initialization(self) -> None:
        independent = DocumentHierarchicalReasoningModel(
            _small_config(pointer_conditioning_mode="independent")
        )
        joint = DocumentHierarchicalReasoningModel(
            _small_config(pointer_conditioning_mode="joint_pair")
        )

        joint_state = joint.state_dict()
        for name, value in independent.state_dict().items():
            self.assertTrue(torch.equal(value, joint_state[name]), name)

    def test_loads_schema_6_checkpoint_as_independent_pointer(self) -> None:
        config = _small_config(pointer_conditioning_mode="independent")
        model = DocumentHierarchicalReasoningModel(config)
        raw_config = asdict(config)
        raw_config.pop("pointer_conditioning_mode")
        raw_config.pop("pointer_teacher_forcing")
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "schema6.pt"
            torch.save(
                {
                    "schema_version": 6,
                    "config": raw_config,
                    "metadata": {},
                    "state_dict": model.state_dict(),
                },
                checkpoint,
            )

            loaded = DocumentHRMBundle.load(checkpoint)

        self.assertEqual(
            loaded.model.config.pointer_conditioning_mode, "independent"
        )

    def test_loads_schema_7_joint_pair_as_detached_backbone(self) -> None:
        config = _small_config(pointer_conditioning_mode="joint_pair")
        model = DocumentHierarchicalReasoningModel(config)
        raw_config = asdict(config)
        raw_config.pop("pair_backbone_mode")
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "schema7.pt"
            torch.save(
                {
                    "schema_version": 7,
                    "config": raw_config,
                    "metadata": {},
                    "state_dict": model.state_dict(),
                },
                checkpoint,
            )

            loaded = DocumentHRMBundle.load(checkpoint)

        self.assertEqual(loaded.model.config.pair_backbone_mode, "detached")

    def test_loads_schema_8_with_legacy_pair_loss_defaults(self) -> None:
        config = _small_config(pointer_conditioning_mode="joint_pair")
        model = DocumentHierarchicalReasoningModel(config)
        raw_config = asdict(config)
        raw_config.pop("pair_loss_weight")
        raw_config.pop("pair_loss_warmup_epochs")
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "schema8.pt"
            torch.save(
                {
                    "schema_version": 8,
                    "config": raw_config,
                    "metadata": {},
                    "state_dict": model.state_dict(),
                },
                checkpoint,
            )

            loaded = DocumentHRMBundle.load(checkpoint)

        self.assertEqual(loaded.model.config.pair_loss_weight, 1.0)
        self.assertEqual(loaded.model.config.pair_loss_warmup_epochs, 0)

    def test_loads_schema_9_with_warmup_defaults(self) -> None:
        config = _small_config(pointer_conditioning_mode="joint_pair")
        model = DocumentHierarchicalReasoningModel(config)
        raw_config = asdict(config)
        raw_config.pop("pair_backbone_warmup_epochs")
        raw_config.pop("pair_targeted_negative_weight")
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "schema9.pt"
            torch.save(
                {
                    "schema_version": 9,
                    "config": raw_config,
                    "metadata": {},
                    "state_dict": model.state_dict(),
                },
                checkpoint,
            )

            loaded = DocumentHRMBundle.load(checkpoint)

        self.assertEqual(loaded.model.config.pair_backbone_warmup_epochs, 0)
        self.assertEqual(loaded.model.config.pair_targeted_negative_weight, 0.0)

    def test_rejects_invalid_teacher_forcing(self) -> None:
        with self.assertRaises(DocumentHRMError):
            _small_config(pointer_teacher_forcing=1.1).validate()

    def test_rejects_end_to_end_pair_without_joint_conditioning(self) -> None:
        with self.assertRaises(DocumentHRMError):
            _small_config(pair_backbone_mode="end_to_end").validate()

    def test_rejects_negative_pair_loss_schedule(self) -> None:
        with self.assertRaises(DocumentHRMError):
            _small_config(pair_loss_weight=-0.1).validate()
        with self.assertRaises(DocumentHRMError):
            _small_config(pair_loss_warmup_epochs=-1).validate()
        with self.assertRaises(DocumentHRMError):
            _small_config(pair_backbone_warmup_epochs=-1).validate()
        with self.assertRaises(DocumentHRMError):
            _small_config(pair_targeted_negative_weight=-0.1).validate()


class DocumentHRMEvaluationMetricTests(unittest.TestCase):
    def test_commutative_pointer_order_is_canonical_but_raw_is_reported(self) -> None:
        candidates = tuple(
            _numeric_candidate(index, value, label, "2024")
            for index, (value, label) in enumerate(
                (("10", "Chi tieu dau ky"), ("20", "Chi tieu cuoi ky"))
            )
        )
        example = ProgramExample(
            example_id="average-reversed-eval",
            question="Gia tri binh quan dau ky va cuoi ky la bao nhieu?",
            context="",
            answer_text="15",
            candidates=candidates,
            program=ReasoningProgram("AVERAGE", (1, 0)),
            answer_value="15",
        )
        prediction = SimpleNamespace(
            program=ReasoningProgram("AVERAGE", (0, 1)),
            confidence=1.0,
            segments=3,
        )
        bundle = SimpleNamespace(
            model=Mock(),
            metadata={"abstention_threshold": 0.0},
            predict_many=Mock(return_value=[prediction]),
        )

        metrics = evaluate_document_hrm(bundle, [example])

        self.assertEqual(metrics["program_exact_accuracy"], 1.0)
        self.assertEqual(metrics["pointer_pair_accuracy"], 1.0)
        self.assertEqual(metrics["raw_program_exact_accuracy"], 0.0)
        self.assertEqual(metrics["raw_pointer_pair_accuracy"], 0.0)
        self.assertEqual(metrics["grounded_semantic_program_accuracy"], 1.0)
        self.assertEqual(
            metrics["program_operation_average_program_exact_accuracy"], 1.0
        )
        self.assertEqual(
            metrics["program_operation_average_raw_program_exact_accuracy"], 0.0
        )

    def test_per_operation_quality_is_not_operation_classification_f1(self) -> None:
        candidates = tuple(
            _numeric_candidate(index, value, label, "2024")
            for index, (value, label) in enumerate(
                (("10", "Chi tieu A"), ("5", "Chi tieu B"), ("1", "Chi tieu C"))
            )
        )
        examples = [
            ProgramExample(
                example_id=f"add-eval-{index}",
                question="Tong cua chi tieu A va chi tieu B la bao nhieu?",
                context="",
                answer_text="15",
                candidates=candidates,
                program=ReasoningProgram("ADD", (0, 1)),
                answer_value="15",
            )
            for index in range(2)
        ]
        predictions = [
            SimpleNamespace(
                program=ReasoningProgram("ADD", (0, 2)),
                confidence=1.0,
                segments=3,
            ),
            SimpleNamespace(
                program=ReasoningProgram("ADD", (0, 1)),
                confidence=1.0,
                segments=3,
            ),
        ]
        bundle = SimpleNamespace(
            model=Mock(),
            metadata={"abstention_threshold": 0.0},
            predict_many=Mock(return_value=predictions),
        )

        metrics = evaluate_document_hrm(bundle, examples)

        self.assertEqual(metrics["operation_class_add_f1"], 1.0)
        self.assertEqual(metrics["program_operation_add_support"], 2.0)
        self.assertEqual(metrics["program_operation_add_operation_accuracy"], 1.0)
        self.assertEqual(
            metrics["program_operation_add_program_exact_accuracy"], 0.5
        )
        self.assertEqual(metrics["program_operation_add_execution_accuracy"], 0.5)
        self.assertEqual(metrics["program_operation_add_pointer_pair_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
