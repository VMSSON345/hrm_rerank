# QP4 semantic-trace reclosure precommit

Status: **PRECOMMIT PASS**. Đây chỉ là contract pass; semantic trace chưa chạy, chưa có root cause mới, chưa sửa lỗi và không có metric nào được cải thiện.

## Mục đích trung thực

Trace được mở để trả lời một câu hỏi: candidate/binding/proof đang loại evidence vì một nguyên nhân generic có thể sửa an toàn, hay vì nhiều nguyên nhân khác nhau phải tiếp tục fail-closed? Không mặc định rằng runtime cần sửa.

Baseline vẫn giữ nguyên: target `0/60`, expected ANSWER `0/72`, canonical positive `0/12`, negative exact action `8/18`, surface groups `0/12`, source permutation exact `37/90`; accepted-wrong, semantic-control false accept và hard-negative ANSWER đều `0`.

## Phạm vi tối thiểu

Trace execution sau precommit chỉ gồm `42` logical observation và `42` candidate-permutation observation:

- `24` positive: toàn bộ canonical base/reference và toàn bộ role-swap/PA07 positive control;
- `18` negative: toàn bộ ambiguity, missing-role và year-shift control.

Mỗi category có đúng `6` row, tổng cộng `6` company / `6` source / `3` real document layout / `6` real table layout / `2` extraction family. Không chọn theo company, filename, page, year, value, ID, question literal, action hiện tại, rank, score hay confidence. `29` P2 và `5` P3 vẫn nằm trong frozen denominator nhưng không tốn trace hoặc tạo áp lực sửa metric phụ.

## Observer không được thay đổi hệ thống

Runtime file whitelist rỗng. Trace dùng các field hiện có (`program_search_trace`, typed synthesis, slot binding, DAG, proof obligations/reasons) và một wrapper tạm thời quanh final candidate augmentation. Wrapper phải gọi original đúng một lần, trả lại chính object original, không reorder/mutate/filter/append và phải bị xóa sau khi final artifact verify.

Mọi primary/permutation output phải khớp frozen parent. Baseline permutation exact phải tái hiện đúng `20/42`, không được “nâng” thành `42/42`. Chỉ một thay đổi action/program/evidence/candidate count do observer cũng làm trace run FAIL; không được báo đó là cải thiện.

Persisted trace chỉ chứa typed categorical projection, local pointer mapping, rejection/UNKNOWN reasons và proof states. Cấm lưu question/raw OCR/label/filename/page/year/value/span, neural/reranker score/rank/confidence hoặc warning prose. Thiếu field phải ghi `TRACE_INCOMPLETE` rồi fail, không tự suy từ question, nearby context hoặc absence.

## Quyết định sau trace

- Chỉ cùng category + earliest layer + exact typed rejection/proof signature, đạt tối thiểu company/source/layout contract, mới là generic causal root.
- Nếu root heterogeneous/non-generic: không sửa runtime.
- Nếu trace thiếu hoặc observer can thiệp: trace invalid, không sửa runtime.
- Nếu evaluator/oracle sai: reclose evaluator trước.
- Nếu có generic root thật: chỉ mở một implementation precommit riêng để khóa smallest symbol whitelist và toàn bộ safety/replay gates.

Không dùng trace để tối ưu metric, biến ABSTAIN/REJECT thành ANSWER, hoặc tuyên bố unseen/universal generalization.

Authorized next step after lock: `QP4-POST-QUESTION-SURFACE-POST-CACHE-RESIDUAL-COVERAGE-POST-SOURCE-PROTOCOL-SEMANTIC-TRACE-RECLOSURE`.

QP4 chưa complete. QP5 và Phase D vẫn khóa.
