# QP4 UNIT_SCALE question-planner implementation precommit v1

- Gate: `PASS`
- Compiler evidence: `0/13` UNIT_SCALE pairs detected before implementation.
- Source evidence: `4/5` old-winner false accepts; failures span `4` companies and `3` real layouts.
- Future runtime whitelist: `src/hrm_finance_qa/question_planner.py` only.
- Typed scope: `QuestionPlan.requested_scale`, `_compile_requested_scale`, and wiring in `compile_question_plan`.
- Conversion, retrieval, binding, verifier, renderer and other QP4 dimensions remain frozen.
- This precommit does not change or authorize runtime in the same step; QP5 and Phase D remain locked.
