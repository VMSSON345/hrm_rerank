# QP4 Question Plan generic runtime-repair precommit

Status: **PASS**.

Governing commit: `bb191858f92731a3521296017c77acc21ac8967c`; direct parent: `5da2ff00bf58496a36be7c08d5bbdb1959a9542e`.

Authorized runtime surface: `src/hrm_finance_qa/question_planner.py` only, within `compile_question_plan` and `_compile_operand_slots`, limited to the three ordered hunks. The `compile_question_plan:521` singleton is frozen and excluded.

The repair predicate is typed question-side semantics only: explicit percent request, exactly two occurrences of the same typed role, and two distinct explicit periods. Company, source, layout, literal year, value, full question, and case ID are forbidden.

Post-repair: 90 canonical plus 90 deterministic order-permuted compiler calls. Exact gates are 18 repaired, 72 unchanged, 19/19 closure replay, 30/30 controls, 90/90 order invariance, and no introduced DAG error.

Candidate 21/24 and Binding/Provenance 15/17 remain OPEN. No downstream call or correctness claim is made.

Next action: Implement only the three ordered hunks, then run the locked evaluator.
