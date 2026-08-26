# QP4 question-surface implementation v1

- Gate: `PASS`.
- Compiler: isolated `18/18`, frozen surface `91/91`, corrected hold-ahead `78/78`, minimal pairs `57/57`.
- Source-backed ADD surface: `20/35`; retained prior passes `20/20` across `5` companies and `3` real layouts.
- Compatible regression: `354/354`; immutable historical hash assertions are audited by their original locks, not rewritten.
- Safe replay: actions `{'ABSTAIN': 92, 'ANSWER': 257, 'REJECT': 14}`, execution `363/363`, permutation `363/363`, counterfactual false accepts `0`.
- Only question-planner operand-metric surface normalization changed; source/retrieval, synthesis, proof, QP5 and Phase D remain locked.
