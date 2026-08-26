# QP4 post-source-protocol diagnostic review precommit

Status: **PRECOMMIT PASS** (development-only, artifact-only, no runtime/source execution).

## Honest reading of the current result

The system is not failing at source protocol or unsafe acceptance: materialization and both source replays completed `90/90`, accepted-wrong and semantic-control false accepts are `0`, and the frozen core replay remains execution/permutation `363/363` with counterfactual false accepts `0/141`.

The utility gap is nevertheless material. Only `8/90` rows match their exact oracle, `0/60` main targets and `0/72` rows expected to ANSWER are proof-complete. All `12` canonical positives fail, so the problem cannot be dismissed as only long-question, no-diacritics, or paraphrase noise.

Negative controls must be interpreted separately: all `18/18` remain fail-closed and none answer, but only `8/18` match the exact action contract. Ten missing-proof controls return `REJECT/INVALID` instead of required `ABSTAIN/UNKNOWN`. This is safe against false acceptance, but it is still a proof-semantics mismatch.

## What is worth diagnosing

1. **P0 — proof/safety contract:** the generic ambiguity and year-shift strata where missing proof becomes rejection. Existing zero false-accept/source-protocol counts are frozen no-regression gates, not reasons for new code.
2. **P1 — verified-answer utility:** canonical base/reference positives first; then positive role-swap/operation-contrast controls; then generic clause-reorder/reference-paraphrase planner failures. A later repair is admissible only for the earliest layer and one coherent category-specific root across multiple companies and real layouts.
3. **P2 — dependent metrics:** `0/12` surface groups and source permutation exact `37/90` remain reported, but cannot independently authorize a patch. They are re-measured after an admitted P0/P1 repair.
4. **P3 — optional/non-generic:** single-company/single-layout exceptions, score/rank/confidence shifts, and cosmetic reporting differences remain in the denominator but do not justify runtime work.

This policy prevents chasing every failure merely to improve a percentage. It also prevents hiding the important fact that canonical verified-answer coverage is currently zero.

## Frozen next review

The next review reads only the frozen 90 logical rows and their nested permutations. It does not read PDFs/OCR content, run OCR/source/model replay, generate new questions, or modify runtime. Every row must receive exactly one value/root-cause disposition, with earliest-layer attribution and no cross-category pooling.

Any later runtime edit requires a separate implementation precommit with the smallest generic symbol whitelist and full regression, candidate-permutation, counterfactual, safe-replay, and source-backed replay obligations. Missing proof remains `UNKNOWN → ABSTAIN`; contradiction remains `INVALID → REJECT`.

Authorized next step after lock: `QP4-POST-QUESTION-SURFACE-POST-CACHE-RESIDUAL-COVERAGE-POST-SOURCE-PROTOCOL-DIAGNOSTIC-REVIEW`.

QP4 is not complete. QP5 and Phase D remain locked.
