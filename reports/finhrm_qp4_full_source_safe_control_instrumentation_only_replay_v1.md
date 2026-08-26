# QP4 instrumentation-only replay — 8 frozen safe controls

- Overall replay status: `FAIL`; decision: `HOLD`.
- Scope: 8/8 frozen controls (Candidate 5, Binding/Provenance 3); Proof/Action replayed 0/10 and remains deferred.
- Inference: 8 uninstrumented baselines + 8 instrumented runs; no cohort expansion; MinerU auto-generation disabled.
- Output semantic envelopes unchanged: `8/8`; decisions unchanged: `8/8`; baselines matching frozen compact observations: `8/8`.
- Runtime unchanged: `true`; checkpoint unchanged: `true`.
- Physical cache manifest exact: `false`. No MinerU file changed and no page-index logical document/page row has a replay-day timestamp, but SQLite container bytes changed; integrity therefore remains conservatively failed.

## Independent cross-check against 60 positive targets

The partition is recomputed directly from each frozen anchor/category's typed operation, ordered roles, ordered years and reference base; evaluator-reclosure result/hash is not correctness evidence.

- Question Plan failures excluded first: `19/60`.
- Candidate-boundary overlap: `24` rows / `6` companies / `3` document layouts.
- Binding-boundary overlap: `17` rows / `4` companies / `2` document layouts.
- Candidate conclusion: `INSUFFICIENT_EVIDENCE`.
- Binding/Provenance conclusion: `INSUFFICIENT_EVIDENCE`.

Trace mới chứng minh boundary-condition overlap, chưa chứng minh cùng upstream code-level root cause trên positive-target surface. Không đề xuất repair; QP5 và Phase D vẫn khóa.
