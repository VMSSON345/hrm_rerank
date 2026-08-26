# QP4 full-source evaluator-contract reclosure precommit v1

Precommit gate: **PASS**. Đây là contract-only step; chưa reclassify final ledger, không chạy PDF/OCR/source/model và không sửa runtime.

## Schema audit đã sửa phạm vi defect

Diagnostic trước tìm thấy `13` planner conflicts và `4` PA07 safe-abstention false failures. Audit đầy đủ mọi typed-plan dimension phát hiện thêm `6` `base_target × clause_reordered` rows có ordered roles `PLAN,ACTUAL` và reference base `actual` thay vì `ACTUAL,PLAN` và `plan`, nhưng vẫn bị ghi `question_plan_passed=true`.

Vì vậy inventory đúng được khóa là:

- `19` question-plan false-pass rows: `6` base clause-reordered và `13` reference-base mutation.
- `4` primary cùng `5` permutation PA07 outcomes là `ABSTAIN/UNKNOWN` được full-refresh oracle cho phép nhưng legacy `exact_pass` không công nhận.
- Defect đều nằm trên real development evidence, phủ tối đa `6 company / 6 source / 3 document layout / 6 table layout / 2 extraction family`; không phải lỗi một báo cáo.

## Reclosure thực thi được mà không replay

`FINHRM_QP4_FULL_SOURCE_EVALUATOR_RECLOSURE_V1` lấy đúng `90` rows trong full-source report và exact anchor oracle từ frozen cohort. `row.question_plan` được định nghĩa là observed runtime typed plan; legacy `question_plan_passed`, `exact_pass` và `first_failing_layer` không được dùng làm oracle mới.

Transform chỉ tính lại:

- `question_plan_exact_v2` từ operation, operation candidates, ordered roles, ordered periods, reference base và abstention flag.
- Primary/permutation `exact_pass_v2` theo category oracle.
- Primary/permutation earliest layer v2 theo thứ tự planner → source → safe/exact PASS → candidate → binding → proof/action.

Safe `ABSTAIN/UNKNOWN` của control có thể là control PASS nhưng tuyệt đối không tính positive ANSWER. `REJECT/INVALID` không được tự chuyển thành PASS khi compact ledger thiếu contradiction proof.

Schema probe đã materialize được toàn bộ `90/90` rows với digest khóa:

`6d16d35c8abe43f94845949bd1dab354452035677cf8b6decd0a27ad84a52237`

Kết quả bắt buộc khi execution:

- Question plan exact `71/90`; failure `19/90`.
- Primary layers: QUESTION_PLAN `19`, candidate `29`, binding `20`, proof/action `10`, PASS `12`.
- Permutation layers: QUESTION_PLAN `19`, candidate `28`, binding `20`, proof/action `8`, PASS `15`.
- Positive target metrics giữ nguyên `0/60`; permutation gate giữ nguyên `37/90`; không gọi reclassification là model improvement.

## Không thêm một vòng diagnostic review

Execution reclosure phải đồng thời recompute corrected generic strata. Nếu mọi count và digest khớp, bước sau đi thẳng vào:

`QP4-FULL-SOURCE-SEMANTIC-REPLAY-REFRESH-QUESTION-PLAN-ROOT-DECOMPOSITION-PRECOMMIT`

Scope là đủ `19` positive QUESTION_PLAN failures; generic runtime work chỉ được dùng ba exact typed signatures phủ `18` rows trên `6 company / 3 layout`. Một long-clause `PERCENT_CHANGE` đơn lẻ vẫn fail-closed. Runtime edit vẫn chưa được phép.

QP4 chưa hoàn tất; QP5 và Phase D vẫn khóa.
