# QP4 semantic-trace evaluator-contract reclosure

## Kết quả

Execution **PASS** với outcome `CONFIRMED_GENERIC_CAUSAL_ROOT`. Evaluator contract đã được đóng lại bằng `168` development-only source query trong hai process sạch; không sửa runtime, model, checkpoint, threshold và không tạo OCR mới.

Bare primary tái hiện historical parent `42/42`. Observed trace khớp bare baseline `42/42` primary và `42/42` permutation. Trace completeness cũng đạt `42/42 + 42/42`; source error, accepted-wrong, permutation false accept và hard-negative ANSWER đều bằng `0`.

Reverse-v1 permutation có exact `20/42`, tình cờ bằng con số historical. Hai con số không được coi là cùng transform vì recipe cũ không tồn tại; không có claim cải thiện, regression hay reproduction.

## Root generic duy nhất

Chỉ một trong `37` exact typed signature group đạt diversity contract:

- category: `PA07_operation_contrast`;
- expected action: `ANSWER`;
- earliest layer: `CANDIDATE_POPULATION_OR_RETRIEVAL`;
- `3` row / `3` company / `3` source / `2` real document layout / `3` real table layout / `2` extraction family;
- predictor input có `64` candidate nhưng không có candidate nào đồng thời mang explicit `ACTUAL` role và requested-period match, cũng không có candidate nào mang explicit `PLAN` role và requested-period match;
- observed actions: `1 REJECT + 2 ABSTAIN`, không có `ANSWER`;
- cả ba row invariant dưới reverse-v1 permutation.

Đây là missing typed candidate support trước neural proposal. Không được suy rằng candidate cap tự nó là nguyên nhân, không mở rộng root sang ba PA07 row còn lại và không quy failure cho model HRM 843.097 tham số.

Các layer khác không đủ confirmation: binding/program synthesis có maximum exact group `2`, symbolic proof/action maximum `2`. Chúng được giữ nguyên, không đề cử runtime work.

## Safety

- Gold execution: `363/363`.
- Candidate permutation: `363/363`.
- Counterfactual: `141`, false accept `0`.
- Regression: `360/360`, exit code `0`.
- Runtime/checkpoint hash giữ nguyên.

## Authorization

Chỉ mở một implementation precommit cho exact root trên:

`QP4-POST-QUESTION-SURFACE-POST-CACHE-RESIDUAL-COVERAGE-POST-SOURCE-PROTOCOL-SEMANTIC-TRACE-PA07-CANDIDATE-POPULATION-IMPLEMENTATION-PRECOMMIT`

Precommit đó phải chứng minh code location và minimum runtime symbol whitelist trước khi sửa. Hiện tại runtime edit vẫn chưa được phép. QP4 chưa hoàn tất; QP5 và Phase D vẫn khóa.
