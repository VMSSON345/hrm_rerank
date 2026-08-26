# QP4 actual/plan ratio-direction source cohort v1

- Cohort frozen: `true`
- Source evidence coverage complete: `true`
- Runtime gates passed: `false`
- Coverage: 4 companies / 4 PDFs / 3 document layouts / 4 table layouts

| Dimension | Base gold | Mutated gold | Paired gold | Old-winner false accepts | Runtime gate |
|---|---:|---:|---:|---:|---|
| COLUMN_ROLE | 0/4 | 0/4 | 0/4 | 0 | FAIL |
| DIRECTION | 0/4 | 0/4 | 0/4 | 0 | FAIL |
| REFERENCE_BASE | 0/4 | 0/4 | 0/4 | 0 | FAIL |

Source diversity is now confirmed independently of runtime behavior. Runtime remains frozen; safe abstention is not counted as an answer and synthetic layouts are not used.
