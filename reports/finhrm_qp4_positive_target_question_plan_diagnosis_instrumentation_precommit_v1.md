# QP4 positive-target Question Plan diagnosis/instrumentation precommit

## Kết luận

Precommit **PASS**. Bước duy nhất được mở tiếp theo là `QP4-POSITIVE-TARGET-QUESTION-PLAN-DIAGNOSIS-INSTRUMENTATION-REPLAY`. Bước này chưa sửa runtime và chưa cho phép repair.

## Residual accounting đã khóa

Candidate positive-target:

- Đã instrument trực tiếp `3/24` case.
- HUB: `LOCAL_CONFIRMED`.
- VNL: `LOCAL_CONFIRMED_DIFFERENT_SIGNATURE`.
- ICT: `UNRESOLVED`.
- Generic exact-signature hypothesis bị `REJECTED` cho đúng frozen 3-case cohort.
- `21/24` case còn lại vẫn `OPEN`; không được báo là đã giải quyết.

Binding/Provenance positive-target:

- Đã xét `2/17` case.
- Diversity chỉ `2 company / 2 source / 2 real document layout`, thấp hơn governing gate `3/3/3`.
- Không được kết luận generic và không được thêm case post-hoc vào cohort 2-case đã khóa.
- Muốn xét thêm phải có precommit cohort mới.
- `15/17` case còn lại vẫn `OPEN`.

Candidate và Binding/Provenance đều không nằm trong replay kế tiếp.

## Question Plan surface đã khóa

Toàn bộ `19/19` corrected positive-target Question Plan residual được giữ theo đúng thứ tự row của corrected evaluator, không lấy mẫu và không có fallback:

- `6` base-target clause-reordered: observed role/reference-base reversal;
- `6` mutation paraphrase: observed `SELECT`, current/prior, thiếu reference base;
- `6` mutation clause-reordered: observed `SELECT`, prior/current, thiếu reference base;
- `1` mutation long-clause: observed `PERCENT_CHANGE` thay vì expected `RATIO_PERCENT`.

Mỗi nhóm 6-row phủ `6 company / 6 source / 3 real document layout / 6 real table layout / 2 extraction family` và có 4 question hash khác nhau. Singleton long-clause chỉ có `1/1/1`, nên không thể tự tạo generic verdict.

Corrected evaluator chỉ cung cấp membership của 19 residual. Correctness oracle được dựng độc lập từ frozen anchor program cùng evidence role/period:

- base: `RATIO_PERCENT(current_actual, current_plan)`, roles `ACTUAL, PLAN`, reference base `plan`;
- mutation: `RATIO_PERCENT(current_actual, prior_actual)`, roles `ACTUAL, ACTUAL`, reference base `actual_<prior period>`.

Gold/expected plan chỉ được dùng post-hoc, không truyền vào compiler.

## Replay contract

Replay kế tiếp chạy đúng:

- `19` unwrapped baseline `compile_question_plan` calls;
- `19` instrumented calls trên cùng câu hỏi, cùng frozen order;
- tổng cộng `38` compiler calls;
- `0` model inference, `0` Candidate/Binding replay;
- không đọc PDF, OCR, page-index hay checkpoint.

Instrumentation chỉ quan sát `compile_question_plan` và các helper question-side đã liệt kê trong JSON contract. Wrapper phải gọi original symbol đúng một lần với nguyên arguments, trả nguyên object/value và được restore kể cả khi lỗi. Semantic plan baseline/instrumented phải exact `19/19`; baseline cũng phải exact `19/19` với frozen parent projection.

Trace phải ghi trực tiếp JSONL qua file handle riêng, append từng case rồi `flush + os.fsync`; stdout chỉ ghi progress. Harness tạm phải xóa sau execution.

## Generic diagnosis gate

Không được suy root cause từ cùng failure layer, cùng surface label hoặc cùng observed projection. Chỉ được lập causal partition khi trace chứng minh cùng:

- module và first-divergence producer symbol;
- branch/predicate thực sự được chạy;
- causal state transition;
- downstream incorrect plan field do transition đó gây ra.

Mỗi generic partition phải đạt ít nhất `3 company / 3 source / 3 real document layout`. Tất cả 19 case phải có disposition; không được bỏ các row không khớp để làm metric đẹp. Nếu không đủ exact code-level evidence thì `HOLD`. Nếu đủ, vẫn phải mở một runtime-repair precommit riêng trước khi sửa code.

## Feasibility verification

- Parent hashes: PASS.
- Corrected Question Plan selection: `19/19`.
- Frozen evaluator/full-source ordinals và IDs: `19/19`.
- Question selectors và SHA-256: `19/19`.
- Observed signatures: `19/19`.
- Oracle materialization: executable từ frozen anchors.
- Runtime hashes: unchanged.
- Runtime whitelist: empty.

QP4 chưa complete; QP5 và Phase D vẫn khóa.
