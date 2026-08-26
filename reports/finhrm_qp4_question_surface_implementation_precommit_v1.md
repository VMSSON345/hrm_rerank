# QP4 question-surface implementation precommit v1

- Gate: `PASS`.
- Isolated evidence: `18` compiler residuals; only `operand_slots[*].metric` differs.
- Frozen semantic breadth: `13` answer families, `3` ambiguity controls, `91` surface questions and `57` minimal pairs.
- Orthogonal hold-ahead: `78` questions across `13` families and `3` target surface categories; current exact `33/78` is diagnostic only.
- Future runtime scope: only `_strip_nonsemantic_metric_surface` and its wiring inside `compile_question_plan` in `question_planner.py`.
- Operation intent, ADD framing, retrieval, source extraction, synthesis, verifier, renderer, QP5 and Phase D remain locked.
- This precommit changes no runtime and makes no company/layout claim from synthetic question variants.
