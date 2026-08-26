# QP4 post-UNIT_SCALE baseline diagnostic review v1

- Diagnostic gate: `PASS`; runtime unchanged.
- Compiler residuals: `18`.
- Source residual queries: `39`; observed source errors `11`.
- First failing layers: `{'CANDIDATE_POPULATION_OR_RETRIEVAL': 2, 'QUESTION_PLAN': 19, 'SEMANTIC_BINDING_OR_PROGRAM_SYNTHESIS': 7, 'SOURCE_LOAD_OR_EXTRACTION': 8, 'SYMBOLIC_PROOF_OR_ACTION': 3}`.

| Source residual group | Queries | Companies | PDFs | Real layouts | First-layer strata |
|---|---:|---:|---:|---:|---|
| ACTUAL_PLAN:COLUMN_ROLE | 8 | 4 | 4 | 3 | {'CANDIDATE_POPULATION_OR_RETRIEVAL': 2, 'SEMANTIC_BINDING_OR_PROGRAM_SYNTHESIS': 4, 'SYMBOLIC_PROOF_OR_ACTION': 2} |
| ACTUAL_PLAN:DIRECTION | 8 | 4 | 4 | 3 | {'SEMANTIC_BINDING_OR_PROGRAM_SYNTHESIS': 3, 'SOURCE_LOAD_OR_EXTRACTION': 4, 'SYMBOLIC_PROOF_OR_ACTION': 1} |
| ACTUAL_PLAN:REFERENCE_BASE | 8 | 4 | 4 | 3 | {'QUESTION_PLAN': 4, 'SOURCE_LOAD_OR_EXTRACTION': 4} |
| ADD_SURFACE:CLAUSE_PERMUTATION | 5 | 5 | 5 | 3 | {'QUESTION_PLAN': 5} |
| ADD_SURFACE:DISCOURSE_PREFIX_SUFFIX | 5 | 5 | 5 | 3 | {'QUESTION_PLAN': 5} |
| ADD_SURFACE:SHORT_LONG_EQUIVALENT | 5 | 5 | 5 | 3 | {'QUESTION_PLAN': 5} |

Gold was used only after observation. DocumentSourceError and timeout are failures; safe abstention is not an answer. Rank, score and confidence are not proof.
Runtime edit remains unauthorized. QP4 is incomplete; QP5 and Phase D remain locked.

Next: QP4-QUESTION-SURFACE-IMPLEMENTATION-PRECOMMIT: freeze the smallest generic question_planner symbol scope for the isolated compiler surface groups; no runtime change is authorized by this diagnostic. Source/retrieval strata remain separately locked.
