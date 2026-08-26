# QP4 full-source semantic diagnostic review v1

## Kết luận

Review đã hoàn tất nhưng phải **FAIL CLOSED vì evaluator contract không nhất quán**. Đây là nguyên nhân khiến các vòng chẩn đoán QP4 dễ đi sai hướng: ledger đang gán một số lỗi planner/control xuống candidate, binding hoặc proof rồi mới yêu cầu tìm runtime root cause.

Không có evidence để kết luận HRM yếu, OCR lỗi hoặc cần thêm báo cáo. Source đã trả đủ `90/90`, không có source error và không tạo OCR mới. Runtime vẫn an toàn nhưng chưa dùng được trên positive targets: `0/60` ANSWER, đồng thời false accept vẫn `0`.

## Hai defect đã được xác nhận

1. `PLAN_ORACLE_LAYER_CONFLICT`: `13` hàng `reference_base_mutation_target` không có planner operation `RATIO_PERCENT` như oracle bắt buộc. Có `12` hàng là `SELECT`, `1` hàng là `PERCENT_CHANGE`, nhưng cả `13` vẫn có `question_plan_passed=true`; `9` bị gán xuống candidate/retrieval và `4` xuống binding/synthesis thay vì QUESTION_PLAN. Defect phủ `6 company / 6 source / 3 real document layout / 6 table layout / 2 extraction family`.

2. `SAFE_ABSTENTION_MARKED_FAILURE`: `4` PA07 controls trả đúng safe outcome được contract cho phép là `ABSTAIN/UNKNOWN`, nhưng vẫn bị ghi `exact_pass=false`; `3` bị gán candidate/retrieval và `1` gán proof/action. Defect phủ `4 company / 4 source / 3 real document layout / 4 table layout / 2 extraction family`.

Các hàng lịch sử không bị sửa hoặc relabel trong review này. Chúng được quarantine khỏi runtime-cause diagnosis cho tới khi evaluator được reclose.

## Sau khi quarantine evaluator conflicts

Toàn bộ `90` hàng vẫn được giữ và có đúng một disposition:

- `17` evaluator/ledger inconsistency.
- `8` PASS controls.
- `35` generic failure-surface nhưng chưa có code-level root cause.
- `30` non-generic failures vẫn giữ fail-closed.

Chỉ còn bốn category×layer strata đạt minimum generic. Hai candidate strata còn phân mảnh thành nhiều symptom signature, nên chưa đủ chứng minh một root cause. Hai signature đồng nhất đáng chú ý là:

- `ambiguity_control × proof/action`: `4` hàng trên `4 company / 3 layout`, upstream đã exact nhưng runtime trả `REJECT/INVALID` thay vì `ABSTAIN/UNKNOWN`.
- `minimal_pair_year_shift × proof/action`: `6` hàng trên `6 company / 3 layout`, upstream đã exact nhưng runtime trả `REJECT/INVALID` thay vì `ABSTAIN/UNKNOWN` khi thiếu shifted-period proof.

Dù vậy, compact ledger không có proof-clause trace hoặc runtime-symbol trace nên chưa được phép sửa verifier.

## Candidate permutation

Permutation exact chỉ `37/90`; `53` semantic envelopes thay đổi. Trong đó observed value thay đổi `52`, provenance completeness thay đổi `15`, binding-exact thay đổi `2`, action thay đổi `3`, state thay đổi `7`; candidate support và operation không đổi. Đây là evidence mạnh về order sensitivity sau candidate availability, nhưng rank/score không được dùng làm semantic proof và review không đề cử runtime symbol.

## Bước tiếp theo ngắn nhất

Không mở thêm source/OCR/cohort và không root-decompose runtime trên ledger sai. Bước duy nhất được phép là:

`QP4-FULL-SOURCE-SEMANTIC-REPLAY-REFRESH-EVALUATOR-CONTRACT-RECLOSURE-PRECOMMIT`

Bước này phải reclose ý nghĩa của `question_plan`, `question_plan_passed`, earliest-layer attribution và safe-control `exact_pass`, ưu tiên reclassify artifact-only trên đúng `90` hàng hiện có. Không chạy lại PDF/OCR/model nếu các frozen fields đã đủ; không sửa runtime reasoning, model, threshold hoặc checkpoint.

QP4 chưa hoàn tất. QP5 và Phase D vẫn khóa.
