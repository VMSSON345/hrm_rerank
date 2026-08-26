# S4.2 — Development-only false-accept audit for SELECT/ADD

## Scope lock

- Source: the 1,246-example S4.1 development manifest and its five company/report-disjoint calibration folds.
- Scoped operations: gold `SELECT` and gold `ADD` only.
- Sealed/global validation loaded: no.
- Blind/test/holdout loaded: no.
- Fine-tuning performed: no.
- Threshold changed: no.
- Phase D remains locked.

The audit replays the current S4.1 winner over the frozen Top-64 v5 candidate cache. It applies the same supervision-integrity exclusions as S4.1, then inspects only accepted programs whose signed execution is wrong.

## Counts

| Gold operation | Accepted | Correct accepted | False accepts | Accepted precision |
|---|---:|---:|---:|---:|
| SELECT | 116 | 92 | 24 | 79.31% |
| ADD | 111 | 93 | 18 | 83.78% |
| Scoped total | 227 | 185 | 42 | 81.50% |

The 42 scoped false accepts account for 42/108 = 38.89% of all false accepts in the frozen S4.1 result. They occur on 35 distinct source-page pairs, so the failure is not isolated to one company, report, or page.

## Proof-obligation partition

The proof-contract state gives a clean, non-overlapping partition:

| Gold operation | Required proof VALID but answer wrong | Required proof UNKNOWN and answer accepted | Required proof INVALID and answer accepted |
|---|---:|---:|---:|
| SELECT | 22 | 2 | 0 |
| ADD | 4 | 14 | 0 |
| Total | 26 | 16 | 0 |

No false accept contains an explicit required contradiction. The current problem is therefore not an `INVALID` leak. It has two structural forms:

1. Required evidence is missing but runtime still accepts `UNKNOWN` (16 cases).
2. The available obligations are too weak or metadata is falsely specific, so every required obligation is marked `VALID` although the grounded value is wrong (26 cases).

### ADD

All 14 `UNKNOWN` ADD cases have the same proof failure: `operation_match=UNKNOWN` and `add_grounding=UNKNOWN`. For each of them the planner emits no operation hypothesis, no operand roles, and no proof operand roles even though the surface form is `Tổng <role A> và <role B> ...`. The causes are general:

- relative-period suffixes such as `tại đầu năm`, `tại năm trước`, or merged headers remain attached to the second role;
- the fixed financial-anchor vocabulary does not recognize many valid roles such as cash-flow labels, taxes payable, and specific receivable/payable rows;
- after the role compiler fails, an ADD candidate can still win under the tri-state runtime because missing required proof is `UNKNOWN`, not a contradiction.

The four `VALID` ADD false accepts contain three physical semantic-cell collisions and one period-role error. The period obligation is not required for normal ADD, so a `năm trước` question can select `năm nay` operands while `add_grounding` remains valid.

### SELECT

Twenty-two of 24 SELECT false accepts are marked proof-complete. `query_metric` is `VALID` in all 22, and `period_coverage` is `VALID` in the 15 cases where it is required. This exposes missing proof dimensions:

- `period_coverage` compares year sets, not exact date/relative-period/column intent; it cannot distinguish `01/01/2025` from `31/12/2025`, or `Kế hoạch 2025` from `Thực hiện 2025`;
- relative periods such as `Số đầu năm` do not enter the required contract unless a numeric year is also recovered;
- SELECT does not require requested unit compatibility; four wrong-unit selections are accepted;
- reporting entity/scope is optional unless a named entity is compiled; three wrong-scope selections are accepted with `reporting_entity=UNKNOWN`;
- metric proof alone cannot resolve repeated rows or parallel subtables carrying the same label and period.

One of the two `UNKNOWN` SELECT cases is a wrong-operation winner; the other has `query_metric=UNKNOWN`. Both remain runtime-eligible under the current tri-state policy.

## General structural error taxonomy

Tags below are multi-label and must not be added together.

| Structural symptom | SELECT | ADD |
|---|---:|---:|
| Wrong period or period role | 10 | 7 |
| Same current typed signature, conflicting values | 10 | 10 |
| Same full physical semantic cell, conflicting values | 5 | 9 |
| Same metric/period duplicated across subtables | 5 | 1 |
| Wrong metric or operand role | 4 | 2 |
| Wrong unit | 4 | 0 |
| Wrong entity/scope | 3 | 0 |
| ADD intent and roles unresolved | 0 | 14 |

The strongest extraction-level signal is value collision: 20/42 false accepts have the same current typed signature as a gold operand but a different numeric value. In 14/42, the collision remains even after including source, page, table, subtable, row ID, column ID, entity, period, and unit. A verifier or reranker cannot safely distinguish these cells because the evidence graph presents them as the same logical cell. This must be handled as ambiguity/abstention or fixed at table-cell reconstruction, not learned from company-specific examples.

## General fixes indicated by the audit (not implemented here)

1. Compile ADD intent before financial-vocabulary filtering, detach absolute and relative period suffixes from both operand roles, and require proof for both recovered roles. Candidate evidence may validate a role, but a fixed list of row-name anchors must not be the sole parser gate.
2. Add a logical-cell uniqueness obligation after metric/period/entity/unit narrowing. Conflicting values with the same physical semantic identity must become `UNKNOWN`/ambiguous rather than competing by neural score.
3. Represent SELECT column intent explicitly: exact date endpoint, relative period role, plan/actual status, and other header qualifiers. Numeric year overlap alone is insufficient period proof.
4. Require SELECT unit family and entity/scope when the question or evidence supplies those dimensions. Missing metadata remains `UNKNOWN`; explicit mismatch remains `INVALID`.
5. Preserve subtable and column lineage through deduplication. Parallel subtables and merged horizontal panels must not collapse into one typed cell solely because metric and year match.

These are proof/schema/parser changes, not threshold changes. They must be developed and audited only on company/report-disjoint development folds, followed by the same frozen S4 replay. No Phase-D training is justified by this audit alone.

## Artifacts

- Machine-readable audit: `reports/select_add_false_accepts_development_v1.json`
- Reproduction script: `scripts/analyze_select_add_false_accepts.py`
- Frozen S4.1 reference: `reports/subtract_direction_s4_development_folds_v2.json`

