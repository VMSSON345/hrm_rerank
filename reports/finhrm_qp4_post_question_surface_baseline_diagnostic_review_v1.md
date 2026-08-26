# QP4 post-question-surface baseline diagnostic review v1

- Diagnostic gate: `PASS`; runtime unchanged.
- Compiler control: `242/242`.
- Source observations: `40`; source errors `12`.
- First failing layers: `{'CANDIDATE_POPULATION_OR_RETRIEVAL': 2, 'QUESTION_PLAN': 16, 'SEMANTIC_BINDING_OR_PROGRAM_SYNTHESIS': 7, 'SOURCE_LOAD_OR_EXTRACTION': 12, 'SYMBOLIC_PROOF_OR_ACTION': 3}`.
- Macro/worst group pass rate: `0.000/0.000`.

| Diagnostic group | Queries | Companies | PDFs | Real layouts | First-layer strata |
|---|---:|---:|---:|---:|---|
| ACTUAL_PLAN:COLUMN_ROLE | 8 | 4 | 4 | 3 | {'CANDIDATE_POPULATION_OR_RETRIEVAL': 2, 'SEMANTIC_BINDING_OR_PROGRAM_SYNTHESIS': 4, 'SYMBOLIC_PROOF_OR_ACTION': 2} |
| ACTUAL_PLAN:DIRECTION | 8 | 4 | 4 | 3 | {'SEMANTIC_BINDING_OR_PROGRAM_SYNTHESIS': 3, 'SOURCE_LOAD_OR_EXTRACTION': 4, 'SYMBOLIC_PROOF_OR_ACTION': 1} |
| ACTUAL_PLAN:REFERENCE_BASE | 8 | 4 | 4 | 3 | {'QUESTION_PLAN': 4, 'SOURCE_LOAD_OR_EXTRACTION': 4} |
| ADD_COUNTERFACTUAL_SOURCE_ERROR:OPERAND_YEAR | 1 | 1 | 1 | 1 | {'SOURCE_LOAD_OR_EXTRACTION': 1} |
| ADD_SURFACE:CLAUSE_PERMUTATION | 5 | 5 | 5 | 3 | {'QUESTION_PLAN': 4, 'SOURCE_LOAD_OR_EXTRACTION': 1} |
| ADD_SURFACE:DISCOURSE_PREFIX_SUFFIX | 5 | 5 | 5 | 3 | {'QUESTION_PLAN': 4, 'SOURCE_LOAD_OR_EXTRACTION': 1} |
| ADD_SURFACE:SHORT_LONG_EQUIVALENT | 5 | 5 | 5 | 3 | {'QUESTION_PLAN': 4, 'SOURCE_LOAD_OR_EXTRACTION': 1} |

Gold was used only after observation. Source errors are failures; safe abstention is not an answer. Rank, score and confidence are not proof.
Runtime edit remains unauthorized. QP4 is incomplete; QP5 and Phase D remain locked.

Next: QP4-POST-QUESTION-SURFACE-RESIDUAL-IMPLEMENTATION-PRECOMMIT: freeze only the generic earliest-layer strata and exact smallest runtime symbols; no runtime edit is authorized by this diagnostic.
