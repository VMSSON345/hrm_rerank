# FinHRM QP3-CORE diverse development ADD cohort

- Reviewed cohort gate: **PASS**.
- Coverage: `8` examples / `8` companies; `4 NATIVE_TEXT_TABLE` + `4 OCR_OR_HYBRID_TABLE`; one positive per company and company-disjoint across strata.
- Visual review: `9` candidates inspected, `8` approved, `1` rejected for a cross-region row/value provenance collision and excluded.
- Explicit wrong-period counterfactual false accepts: `4 -> 0` across the development audit.
- Frozen replay: gold execution `363/363`, permutation `363/363`, counterfactual false accepts `0/141`.

## Qualification standard

A positive requires explicit two-role ADD syntax, exactly two distinct logical and physical cells, same explicit period/source/entity/unit, complete page/table/row/column/span provenance, exact execution, a generic hard negative, independent visual approval, permutation invariance, and a semantic counterfactual that is not ANSWER.

## Remaining locked gate

The cohort is ready for frozen end-to-end replay, but it does not by itself close QP3-CORE: chunk retrieval and neural operation/operand proposal have not yet been replayed from PDF+question on these eight reviewed records. QP3 answer presentation, QP4 and Phase D remain locked.
