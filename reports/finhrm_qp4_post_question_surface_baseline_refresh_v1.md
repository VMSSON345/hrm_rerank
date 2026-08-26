# QP4 post-question-surface baseline refresh v1

- Refresh: `COMPLETE`; QP4 semantic gates: `FAIL`.
- Compiler: canonical `13/13`, ambiguity `3/3`, frozen surface `91/91`, corrected hold-ahead `78/78`, minimal pairs `57/57`.
- ADD source: `20/35` surface invariance over `5` companies / `5` PDFs / `3` real layouts; prior passes retained `20/20`.
- Source execution errors retained in denominator: `12`.
- Safe replay: actions `{'ABSTAIN': 92, 'ANSWER': 257, 'REJECT': 14}`, execution `363/363`, permutation `363/363`, counterfactual false accepts `0`.
- Compatible regression: `354/354`.

| ACTUAL/PLAN dimension | Paired gold | Old-winner false accepts | Gate |
|---|---:|---:|---|
| COLUMN_ROLE | 0/4 | 0 | FAIL |
| DIRECTION | 0/4 | 0 | FAIL |
| REFERENCE_BASE | 0/4 | 0 | FAIL |

Failed semantic gates: `actual_plan_paired_gold:COLUMN_ROLE`, `actual_plan_paired_gold:DIRECTION`, `actual_plan_paired_gold:REFERENCE_BASE`, `add_source_surface_winner_invariance`.

The same frozen contract was applied to every eligible development record. Errors, timeouts and abstentions were not removed or counted as answers. This is not unseen or universal-report confirmation.

No runtime, threshold, checkpoint or model change was made. QP4 is incomplete; QP5 and Phase D remain locked.

Next: QP4-POST-QUESTION-SURFACE-BASELINE-DIAGNOSTIC-REVIEW-PRECOMMIT: freeze a report-agnostic first-failing-layer review for the measured residual groups; no runtime change is authorized.
