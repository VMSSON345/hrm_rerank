# QP3-CORE-GENERALIZATION-COVERAGE-DIAGNOSTIC-MATRIX

Gate: **PASS**.

| Track | Source examples | Companies | Source layouts | Projected | Matrix layouts | Controls |
|---|---:|---:|---:|---:|---:|---:|
| period_and_provenance | 74 | 73 | 2 | 3 | 3 | 308 |
| operation_and_add_grounding | 67 | 47 | 3 | 0 | 3 | 268 |
| query_metric | 33 | 26 | 6 | 0 | 6 | 132 |

Mỗi track có positive, hard-negative, explicit counterfactual và candidate-permutation controls.
Layout projection của period chỉ là diagnostic synthetic control, không phải unseen confirmation.
Runtime, threshold, checkpoint và QP4 vẫn khóa.
