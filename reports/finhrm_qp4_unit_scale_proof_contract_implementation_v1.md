# QP4 UNIT_SCALE proof-contract implementation v1

- Gate: `FAIL`.
- Source gate: `PASS`; old-winner false accepts `0`.
- Prior ANSWER to ABSTAIN: `4/4`; operation/value/unit/evidence retained `4/4`.
- Frozen replay actions: observed `{'ABSTAIN': 92, 'ANSWER': 257, 'REJECT': 14}`, expected `{'ANSWER': 259, 'ABSTAIN': 90, 'REJECT': 14}`.
- Execution `363/363`, permutation `363/363`, counterfactual false accepts `0`.
- No conversion, rescaling, retrieval, binding, synthesis, engine or renderer change was made.
- QP4 is incomplete; QP5 and Phase D remain locked.
- Frozen-action conflict: examples `159` and `621` explicitly request VND but their selected evidence has missing unit proof; UNKNOWN/ABSTAIN is required.

Next: QP4-UNIT-SCALE-FROZEN-ACTION-CONTRACT-REVIEW-PRECOMMIT must resolve the two missing-unit proof cases without forcing ANSWER or changing runtime.
