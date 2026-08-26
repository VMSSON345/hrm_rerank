# QP4 lazy development OCR-cache materialization and source replay v1

- Gate: `FAIL`; runtime unchanged.
- Selection: `8` question-triggered observations, `4` companies, `4` PDFs, `3` real layouts; whole-corpus OCR is false.
- Structured cache artifacts: `16`; artifact-only replay source errors: `2`.
- Replay earliest layers: `{'CANDIDATE_POPULATION_OR_RETRIEVAL': 2, 'SEMANTIC_BINDING_OR_PROGRAM_SYNTHESIS': 4, 'SOURCE_LOAD_OR_EXTRACTION': 2}`; paired PASS `0/4`; accepted-wrong `0`; false accepts `0`.
- Compiler `242/242`; regression `360/360`; frozen execution/permutation `363/363/363/363`; counterfactual false accepts `0`.
- Cache availability is not semantic proof. Missing proof remains ABSTAIN and contradiction remains REJECT. QP4 remains incomplete; QP5 and Phase D remain locked.
