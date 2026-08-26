# QP4 post-cache source-error diagnostic-review precommit v1

- Gate: `PASS`; runtime unchanged, OCR/source replay not rerun, and runtime whitelist empty.
- Source residual: `2` queries over `1` company / `1` PDF / `1` real layout. Repeatable within one source, but not generic.
- Other residual coverage: candidate/retrieval `2/2/2`; semantic binding/program synthesis `3/3/2` (companies/PDFs/real layouts). Neither reaches the frozen `3/3/3` minimum.
- Decision: authorize only an artifact-only diagnostic review from the final parent report and lock. No OCR, PDF read, source replay, runtime edit, threshold/model/checkpoint change, or report-specific rule is authorized.
- Frozen parent controls remain: compiler `242/242`, regression `360/360`, execution `363/363`, permutation `363/363`, and `141` counterfactuals with zero false accepts.
- QP4 remains incomplete; QP5 and Phase D remain locked.
