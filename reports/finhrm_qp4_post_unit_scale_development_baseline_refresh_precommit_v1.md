# QP4 post-UNIT_SCALE development baseline refresh precommit v1

- Gate: `PASS`.
- Stage: measurement-only precommit; the baseline refresh has not run and runtime is unchanged.
- Compiler tier: `13` families, `3` ambiguity controls, `91` surface variants, `57` semantic pairs.
- ADD source tier: `5` anchors / `5` companies / `5` PDFs / `3` real layouts, `50` queries.
- ACTUAL/PLAN source tier: `4` anchors / `4` companies / `4` PDFs / `3` document layouts / `4` table layouts, `24` queries.
- Combined source queries: `74`; per-query timeout `15s`; existing development cache only; no OCR auto-generation.
- Gold candidate/program, rank, score and confidence are forbidden as semantic proof. Safe abstention is not counted as answer; synthetic layouts are not confirmation.
- Historical baselines remain immutable. QP4 is incomplete; QP5 and Phase D remain locked.

Next: QP4-POST-UNIT-SCALE-DEVELOPMENT-BASELINE-REFRESH: run the frozen compiler, ADD-source and ACTUAL/PLAN-source tiers under this exact contract; make no runtime changes.
