# QP4 Question Plan diagnosis/instrumentation — new trust baseline

Status: **PASS**.

Governing baseline: `5da2ff00bf58496a36be7c08d5bbdb1959a9542e` (root/forward anchor only).

The cohort was rebuilt from the six-anchor development artifact and the raw 90-row development replay. Old precommits, corrected-evaluator selection, candidate/binding traces, and protected evaluation artifacts are not inputs. The raw `question_plan_passed` flag is ignored; correctness is recomputed from anchor programs and evidence roles/periods.

Independent result: 19 Question Plan divergences. No old root-cause hypothesis is carried forward.

Observed projection groups (descriptive only, not causal proof):

- `base_target/clause_reordered`: 6 rows, 6 companies, 6 sources, 3 real document layouts; causal claim: NONE.
- `reference_base_mutation_target/paraphrase_1`: 6 rows, 6 companies, 6 sources, 3 real document layouts; causal claim: NONE.
- `reference_base_mutation_target/clause_reordered`: 6 rows, 6 companies, 6 sources, 3 real document layouts; causal claim: NONE.
- `reference_base_mutation_target/long_clause`: 1 rows, 1 companies, 1 sources, 1 real document layouts; causal claim: NONE.

The replay harness is persistent and hash-locked before PASS. Precommit mode performs zero compiler calls. Replay mode is fail-closed unless this precommit is PASS, then performs exactly 19 baseline and 19 traced calls with per-case fsynced JSONL and exact noninterference checks.

Runtime/model/threshold/checkpoint changes are not authorized.
