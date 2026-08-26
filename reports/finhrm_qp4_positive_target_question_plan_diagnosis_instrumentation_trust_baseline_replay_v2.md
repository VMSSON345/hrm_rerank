# QP4 Question Plan diagnosis/instrumentation replay — new trust baseline

Status: **PASS**.

Executed 19 baseline and 19 instrumented compiler calls. Frozen baseline match: 19/19. Observer noninterference: 19/19.

Exact code-derived causal clusters:

- `_compile_operand_slots:1144` / `eba132638bf3`: 5 rows, 5 companies, 5 sources, 3 real layouts — GENERIC_ROOT_CAUSE.
- `compile_question_plan:575` / `3b833b370d54`: 6 rows, 6 companies, 6 sources, 3 real layouts — GENERIC_ROOT_CAUSE.
- `compile_question_plan:575` / `f84d0926d0c7`: 6 rows, 6 companies, 6 sources, 3 real layouts — GENERIC_ROOT_CAUSE.
- `_compile_operand_slots:1144` / `9060a31d0b18`: 1 rows, 1 companies, 1 sources, 1 real layouts — LOCAL_INSUFFICIENT_FOR_GENERICITY.
- `compile_question_plan:521` / `aeca041828bc`: 1 rows, 1 companies, 1 sources, 1 real layouts — LOCAL_INSUFFICIENT_FOR_GENERICITY.

No runtime, model, threshold, checkpoint, candidate, binding, PDF, OCR, or page-index change/call was authorized or performed.
