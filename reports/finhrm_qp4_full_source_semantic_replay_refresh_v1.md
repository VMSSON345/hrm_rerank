# QP4 Full Source Semantic Replay Refresh

Kết quả: **FAIL_CLOSED**. QP4 chưa hoàn tất; QP5 và Phase D vẫn khóa.

Execution đã chạy đủ cohort development được precommit: `90/90` primary source observations và `90/90` deterministic candidate permutations trên `6` company, `6` PDF, `3` real document layout, `6` real table layout và `2` extraction family. Không thêm, bỏ, thay hoặc retry-select observation.

## Integrity và retained controls

- Primary source returned: `90/90`; permutation source returned: `90/90`; source error/timeout: `0`.
- Candidate population trước transform khớp primary: `90/90`.
- MinerU cache manifest: `377/377` file, hash trước/sau khớp; OCR generation mới: `0`.
- Compiler: `242/242`; regression: `360/360`.
- Frozen safe replay: actions `257 ANSWER / 92 ABSTAIN / 14 REJECT`; execution `363/363`; permutation `363/363`.
- Counterfactual: `141`; false accept: `0`.
- Runtime, checkpoint, model và threshold không đổi; không đọc protected evaluation và không dùng ABW.

## Release gates

| Gate | Required | Observed | Status |
|---|---:|---:|---:|
| Proof-complete exact target answers | 60/60 | 0/60 | FAIL |
| Base exact answers | 30/30 | 0/30 | FAIL |
| Reference-base mutation exact answers | 30/30 | 0/30 | FAIL |
| Surface-equivalence groups | 12/12 | 0/12 | FAIL |
| Source candidate permutation | 90/90 | 37/90 | FAIL |
| Accepted-wrong | 0 | 0 | PASS |
| Reference-base old-winner false accept | 0 | 0 | PASS |
| Semantic-control false accept | 0 | 0 | PASS |
| Hard-negative answers | 0 | 0 | PASS |

Permutation dùng đúng `FINHRM_CANDIDATE_REVERSE_REINDEX_REMAP_V1`. Candidate population của hai lượt khớp `90/90`, nên `37/90` là observation hợp lệ của transform hiện hành. Việc con số này tình cờ bằng measurement lịch sử không chứng minh transform lịch sử giống transform mới và không được gọi là reproduction, improvement hay regression.

Primary actions không có ANSWER: `62 REJECT`, `28 ABSTAIN`. Vì vậy không có accepted-wrong, nhưng safe abstention/rejection không được tính thành exact target answer.

## Earliest failing layers

- `CANDIDATE_POPULATION_OR_RETRIEVAL`: `46`.
- `SEMANTIC_BINDING_OR_PROGRAM_SYNTHESIS`: `25`.
- `SYMBOLIC_PROOF_OR_ACTION`: `11`.
- `PASS`: `8`, gồm `6` hard-negative missing-role và `2` ambiguity controls; không có target answer nào trong nhóm PASS.

Target failure không nằm ở một PDF riêng: cả `base_target` và `reference_base_mutation_target` đều có `30` observations trên đủ `6` company và `3` document layouts, với exact answer `0`. Vì vậy không được sửa riêng theo company, filename, page, year, value, example ID hoặc question literal.

## Bước tiếp theo

Chỉ mở `QP4-FULL-SOURCE-SEMANTIC-REPLAY-REFRESH-SEMANTIC-DIAGNOSTIC-REVIEW-PRECOMMIT` để khóa diagnostic theo earliest failing layer và generic coverage. Bước này chưa tự động cho phép sửa runtime. Source protocol không cần reclose vì source error `0`, candidate-population comparison đủ `90/90` và cache không đổi.

Chi tiết 90-row audit ledger nằm trong final JSON; không persist raw neural score/confidence trace.
