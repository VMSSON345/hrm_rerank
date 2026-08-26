# QP4 post-UNIT_SCALE development baseline refresh v1

- Refresh: `COMPLETE`; frozen gates: `FAIL`.
- Compiler: canonical `13/13`, ambiguity `3/3`, surface `73/91`, counterfactual `57/57`.
- ADD source: 5 companies / 5 PDFs / 3 real layouts; surface `20/35`.
- Safe replay: actions `{'ABSTAIN': 92, 'ANSWER': 257, 'REJECT': 14}`, execution `363/363`, permutation `363/363`, counterfactual false accepts `0`.
- Compatible regression: `348/348`.

| ACTUAL/PLAN dimension | Paired gold | Old-winner false accepts | Gate |
|---|---:|---:|---|
| COLUMN_ROLE | 0/4 | 0 | FAIL |
| DIRECTION | 0/4 | 0 | FAIL |
| REFERENCE_BASE | 0/4 | 0 | FAIL |

Failed gates: `actual_plan_paired_gold:COLUMN_ROLE`, `actual_plan_paired_gold:DIRECTION`, `actual_plan_paired_gold:REFERENCE_BASE`, `add_source_surface_winner_invariance`, `compiler_surface_semantic_invariance`.

No runtime, threshold, checkpoint or model change was made. Safe abstention is not counted as an answer; synthetic layouts are not confirmation.

Next: QP4-POST-UNIT-SCALE-BASELINE-DIAGNOSTIC-REVIEW-PRECOMMIT: review only generic residual groups and isolate the first failing layer; no runtime change is authorized.
