# QP4 full-source semantic diagnostic-review precommit v1

Gate precommit: **PASS**. Đây là bước khóa hợp đồng chẩn đoán chỉ đọc; chưa chạy diagnostic review, không đọc lại PDF/OCR, không inference mới và không sửa runtime/model/threshold/checkpoint.

## Denominator được khóa

- Giữ đủ `90` primary rows và `90` permutation results. Primary có `82` failure và `8` PASS controls; không được bỏ hàng hoặc thêm control để làm đẹp tỷ lệ.
- Earliest-layer primary: candidate/retrieval `46`, binding/synthesis `25`, proof/action `11`, PASS `8`.
- Earliest-layer permutation: candidate/retrieval `46`, binding/synthesis `25`, proof/action `9`, PASS `10`.
- Hai hàng chuyển từ proof/action sang PASS khi permutation; `37/90` mới là invariance gate đã khóa, không được thay bằng số `10` permutation oracle passes.
- Các safety control vẫn nguyên: accepted-wrong `0`, old-winner false accept `0`, semantic-control false accept `0`, hard-negative ANSWER `0`; regression `360/360`, execution/permutation `363/363`, counterfactual `141` với false accept `0`.

## Coverage trung thực

Có đúng `6` failure strata đạt ngưỡng development-generic `>=3 company`, `>=3 source`, `>=3 real document layout`; candidate/retrieval còn phải có `>=2 extraction family` và các nhánh layout-relevant có `>=3 table layout`:

- Candidate/retrieval × `base_target`: `21` rows, `5/5/3/5/2` company/source/document-layout/table-layout/extraction.
- Candidate/retrieval × `minimal_pair_role_swap`: `4`, coverage `4/4/3/4/2`.
- Candidate/retrieval × `PA07_operation_contrast`: `4`, coverage `4/4/3/4/2`.
- Candidate/retrieval × `reference_base_mutation_target`: `17`, coverage `5/5/3/5/2`.
- Proof/action × `ambiguity_control`: `4`, coverage `4/4/3/4/2`.
- Proof/action × `minimal_pair_year_shift`: `6`, coverage `6/6/3/6/2`.

Có `5` failure strata chưa generic. Đặc biệt toàn bộ binding/synthesis tuy có `25` rows nhưng chỉ phủ `2` document-layout và `1` extraction-family; không được gọi là lỗi tổng quát hoặc lấy làm căn cứ sửa runtime. Một PA07 proof/action đơn lẻ cũng chỉ có `1/1/1/1/1`.

## Điều diagnostic review được phép kết luận

Review kế tiếp chỉ được tổng hợp layer/category/coverage, mismatch flags và transition primary↔permutation từ ledger đã khóa. Ledger đủ để xác nhận **generic failure surface**, nhưng không có đầy đủ candidate identity, operand trace, proof clause hoặc code-path trace để chứng minh runtime root cause.

Vì vậy, cùng một symptom signature chỉ được đề cử cho root-decomposition cohort. Không được quy lỗi cho HRM, OCR hay một symbol runtime chỉ dựa vào số failure. Không dùng company, filename, page, year, value, example/question literal, candidate rank, neural score hoặc confidence làm semantic proof.

Các 6 company, 6 source và 3 document-layout hiện tại đều đã tham gia diagnosis nên không được dùng làm confirmation độc lập cho một sửa đổi tương lai. Confirmation sau sửa phải development-safe và company/source/layout-instance-disjoint; nếu sửa layout thì phải layout-disjoint.

Runtime whitelist vẫn rỗng. QP4 chưa hoàn tất; QP5 và Phase D vẫn khóa.

Next allowed action: `QP4-FULL-SOURCE-SEMANTIC-REPLAY-REFRESH-SEMANTIC-DIAGNOSTIC-REVIEW`.
