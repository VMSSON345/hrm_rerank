# QP4 post-UNIT_SCALE baseline diagnostic-review precommit v1

- Gate: `PASS`.
- Stage: diagnostic measurement precommit only; no diagnostic source replay or runtime edit occurred.
- Compiler residuals: `18` across three generic surface groups.
- ADD surface residuals: `15` queries / `5` companies / `5` PDFs / `3` real layouts.
- ACTUAL/PLAN residuals: `24` queries / `4` companies / `4` PDFs / `3` document layouts.
- Source-error stratum: `11/39`; non-error fail-closed stratum: `28/39`.
- Earliest-layer order is frozen: question plan -> source load/extraction -> candidate population/retrieval -> semantic binding/synthesis -> symbolic proof/action.
- Gold is post-hoc diagnostic evidence only and is never passed into the engine. Rank, score, confidence and safe abstention are not semantic proof.
- Runtime whitelist is empty. Any later implementation requires a new precommit after generic multi-company/layout attribution.
- QP4 is incomplete; QP5 and Phase D remain locked.

Next: QP4-POST-UNIT-SCALE-BASELINE-DIAGNOSTIC-REVIEW: run the frozen 18 compiler residual and 39 source diagnostic queries, assign only the earliest failing layer using post-hoc gold comparison, and make no runtime changes.
