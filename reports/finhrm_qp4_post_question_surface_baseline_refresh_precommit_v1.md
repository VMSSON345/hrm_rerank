# QP4 post-question-surface baseline refresh precommit v1

- Gate: `PASS`.
- This is a measurement-only precommit. The refresh has not run and no runtime file changed.
- Generalization boundary: one frozen contract over every eligible development record; no report filtering, no report-specific rules, and no unseen/universal claim.
- Compiler inventory: `13` answer families, `3` ambiguity controls, `91` frozen surfaces, `78` corrected hold-ahead surfaces, `57` semantic minimal pairs.
- ADD source inventory: `50` queries over `5` companies / `5` PDFs / `3` real layouts. Current surface result is `20/35`; all `15` residuals remain in the denominator.
- ACTUAL/PLAN inventory: `24` queries over `4` companies / `4` PDFs / `3` document layouts / `4` table layouts.
- Measurement completion and QP4 semantic pass are separate statuses. Errors, timeouts and abstentions cannot be dropped or counted as answers.
- Frozen safe replay: actions `{'ANSWER': 257, 'ABSTAIN': 92, 'REJECT': 14}`, execution `363/363`, permutation `363/363`, counterfactual false accepts `0`.
- Missing proof remains ABSTAIN; contradiction remains REJECT. QP4 is incomplete; QP5 and Phase D stay locked.

Next: QP4-POST-QUESTION-SURFACE-BASELINE-REFRESH: run the exact frozen compiler, ADD-source, ACTUAL/PLAN-source, compatible regression and safe-replay measurements; make no runtime changes.
