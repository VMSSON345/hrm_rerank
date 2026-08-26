# QP4 PA07 root-decomposition cohort build

Status: **FAIL_CLOSED**. Build gate: **FAIL**. Runtime remains locked.

## Outcome

The frozen development-only procedure processed all 160/160 PDFs from 143 companies and verified 160/160 source hashes before selection. It scanned 8,020 physical pages, audited 183 generic candidate pages and examined 169 native table grids. No protected split, ABW, new OCR, cache answer, planner output, HRM prediction, rank, score or confidence was used for eligibility.

Only one of the three structural roots meets the independently frozen minimum of three companies, three sources, three real layout families and three layout instances:

| Structural root | Companies | Sources | Real layouts | Instances | Gate |
|---|---:|---:|---:|---:|---|
| Hierarchical multirow role/period header | 1 | 1 | 1 | 1 | FAIL |
| Flattened compact role/year header | 3 | 3 | 3 | 3 | PASS |
| Fragmented visual row label | 0 | 0 | 0 | 0 | FAIL |

Because all three gates are required and evidence cannot be pooled across roots, the overall cohort is not admitted.

## Verified evidence

The flattened root is supported by three company-disjoint, source-disjoint real layouts:

- SRA: landscape four-column table, same-cell role/year headers, explicit consolidated scope and explicit `đồng` unit.
- PCE: portrait six-column table, same-cell abbreviated role/year headers and explicit row unit `tỷ đồng`.
- TOT: landscape two-level repeated-period table, wrapped metric row and explicit `triệu đồng` unit.

The hierarchical root has one genuine source:

- ABB: period `2024` is a parent header over explicit child columns `Thực hiện` and `Kế hoạch`; the cited row and `tỷ đồng` unit are geometrically explicit.

The apparent fragmented PAN row was rejected. Although its label is split across two lines and both 2023 roles are explicit, the plan/actual table itself does not state a unit. The `tỷ đồng` label belongs to another table on the page. Reusing that label or matching values across tables would infer evidence from nearby context/value equality, which the frozen contract forbids.

Other material rejections include a false full-page table merge (ADP), different ACTUAL/PLAN periods (VCC), a source-level semantic contradiction (KSF), and an infographic rather than a physical table (ADS).

## Generalization assessment

This is not optimization for named reports. Identity and filenames were used only to locate hash-locked sources and enforce disjointness. Canonical family IDs exclude company, filename, hash, page, year, metric literal, numeric value, question literal and example ID. The accepted anchors use the same structural predicates and every incomplete/conflicting case is rejected or left unknown.

The result is deliberately not made prettier: flattened coverage is real, but hierarchical and fragmented coverage are insufficient after catalog exhaustion. Therefore the precommitted source-replay diagnostic step is not unlocked, no runtime edit is authorized, QP4 is not complete, and QP5/Phase D remain locked.

Detailed cell bounding boxes, formulas, replay values, family fingerprints and exact shortfalls are in the final JSON.
