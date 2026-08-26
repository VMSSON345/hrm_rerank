# QP4 post-question-surface residual implementation precommit v1

- Gate: `PASS`; runtime unchanged.
- Four generic `QUESTION_PLAN` strata are frozen: three ADD surface families and ACTUAL/PLAN reference base, each on 4 companies, 4 PDFs and 3 real layouts.
- Runtime scope is one file and three existing symbols; source, retrieval, synthesis, verifier, direction, threshold, checkpoint and model remain locked.
- Implementation must retain 242/242 compiler control, all source failures, zero accepted-wrong/false-accept, and 363/363 execution/permutation with 0/141 counterfactual false accepts.
