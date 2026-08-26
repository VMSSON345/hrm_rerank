# QP4 residual coverage cohort build v1

- Gate: `PASS` for `QP4-POST-QUESTION-SURFACE-POST-CACHE-RESIDUAL-COVERAGE-COHORT-BUILD`.
- Audited: `169` catalog PDFs / `150` company groups; admitted `167`, excluded `2` protected-marker sources, hash mismatches `0`, read errors `0`.
- Selected: `6` anchors / `6` companies / `6` source hashes, at most one anchor per company and source.
- Real diversity: `3` document layout families, `6` table layout families, `6` layout-instance fingerprints, `4` normalized metric identities and `2` extraction families.
- Extraction families are existing native vector text and hybrid-scan PDF text layer. MinerU/new OCR was not run.
- Every anchor has explicit current ACTUAL, same-period PLAN, prior-period ACTUAL, period, unit, entity scope and physical page/table/row/column provenance; all six pages were independently rendered and visually reviewed.
- Two visible operand/printed-ratio contradictions were rejected. One broader diagnosis-company membership was conservatively excluded. No contradiction was repaired to make the cohort pass.
- PA03 supplies the target `RATIO_PERCENT` surfaces. PA07 is retained only as an operation-contrast control and is not counted as target-stratum coverage.
- Per target semantic program: `5` meaning-preserving surfaces (`canonical`, `paraphrase_1`, `long_clause`, `clause_reordered`, `no_diacritics`) plus reference-base mutation, role swap, year shift, missing-role, ambiguity and PA07 operation controls.
- Runtime, checkpoint, model and thresholds are unchanged. Frozen regression/permutation/counterfactual/safe-replay results are retained by exact hashes and were not rerun because this build forbids the reasoning engine and requires zero source replay.
- QP4 is not complete. Runtime edits and source replay remain locked.
- Next allowed action: `QP4-POST-QUESTION-SURFACE-POST-CACHE-RESIDUAL-COVERAGE-SOURCE-REPLAY-PRECOMMIT`.
