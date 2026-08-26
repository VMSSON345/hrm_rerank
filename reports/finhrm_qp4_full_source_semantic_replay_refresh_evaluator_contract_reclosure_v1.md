# QP4 evaluator contract reclosure

Status: **PASS**

Step: `QP4-FULL-SOURCE-SEMANTIC-REPLAY-REFRESH-EVALUATOR-CONTRACT-RECLOSURE`

## Execution boundary

This is the artifact-only evaluator transform authorized by the locked precommit. All parent sizes and SHA-256 values, all 11 frozen runtime hashes, and the frozen checkpoint size/hash matched before reclassification. No runtime, model, source loader, OCR, threshold, checkpoint, or test execution was changed or invoked.

The corrected diagnostic was recomputed in this same execution from the frozen observed `question_plan` and primary/permutation run fields. Legacy `question_plan_passed`, `exact_pass`, and `first_failing_layer` fields were not reused as the new oracle or as fallback attribution.

## Reclosure gates

| Gate | Corrected result |
|---|---:|
| Typed question plan exact | 71/90 |
| Typed question plan fail | 19/90 |
| Primary exact pass v2 | 12/90 |
| Permutation exact pass v2 | 15/90 |
| Canonical row-manifest SHA-256 | `6d16d35c8abe43f94845949bd1dab354452035677cf8b6decd0a27ad84a52237` |
| Source errors | 0 |
| Accepted wrong | 0 |
| Old-winner false accepts | 0 |
| Semantic-control false accepts | 0 |
| Hard-negative answers | 0 |

All expected counts and the expected canonical hash match the precommit exactly.

## Corrected residual

Primary residual is **78/90**; permutation residual is **75/90**. The 60 positive targets remain **0/60 ANSWER**, so the corrected positive-target residual is honestly **60/60**:

| Earliest corrected layer | Positive targets |
|---|---:|
| QUESTION_PLAN | 19 |
| CANDIDATE_POPULATION_OR_RETRIEVAL | 24 |
| SEMANTIC_BINDING_OR_PROGRAM_SYNTHESIS | 17 |
| SYMBOLIC_PROOF_OR_ACTION | 0 |
| PASS | 0 |

Primary safe-control pass is **12/30** and safe-control residual is **18/30**. The corrected primary distribution across all 90 rows is QUESTION_PLAN 19, SOURCE 0, CANDIDATE 29, BINDING 20, PROOF/ACTION 10, PASS 12.

Six failure strata meet the locked generic coverage rule: base candidate (16), role-swap candidate (4), base question-plan (6), mutation question-plan (13), ambiguity proof/action (4), and year-shift proof/action (6). This coverage is measured across real companies, sources, document/table layout families, and extraction families; no synthetic layout is counted.

The question-plan inventory contains 19 failures: three exact typed signatures cover 18 rows across 6 companies, 6 sources, 3 real document layouts, 6 table layouts, and 2 extraction families. The remaining single `PERCENT_CHANGE` mutation row is non-generic and stays fail-closed. Evaluator correction alone does not prove a runtime-symbol cause.

## Decision

Evaluator reclosure is closed. QP4 is not complete; QP5 and Phase D remain locked.

Next allowed action: `QP4-FULL-SOURCE-SEMANTIC-REPLAY-REFRESH-QUESTION-PLAN-ROOT-DECOMPOSITION-PRECOMMIT`.

Its scope is the 19 positive-target QUESTION_PLAN failures. Any later generic runtime work is limited to the three exact typed signatures covering 18 rows; the one non-generic row remains fail-closed. Runtime editing is not automatically authorized by this artifact.
