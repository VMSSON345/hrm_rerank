# QP4 semantic-trace evaluator-contract reclosure precommit

## Kết quả

Precommit **PASS** và chỉ mở một execution step observability-only. Không có runtime, model, checkpoint, threshold hay OCR change.

Điểm khác quan trọng so với precommit lỗi trước: contract này đã được kiểm tra khả năng thực thi trước khi khóa. Runtime hiện có đủ answer/prediction/program-trace fields; probe tạm đã xác nhận full-reverse candidate transform và remap `derived_operands`, selected program, mọi program-search trace, DAG candidate leaf và slot binding đều round-trip được. Probe đã bị xóa.

## Xử lý baseline cũ

`20/42` vẫn được giữ nguyên như historical result của một permutation recipe không được lưu. Contract mới không giả định recipe đó, không yêu cầu tái hiện `20/42`, không so sánh metric mới với nó và không gọi khác biệt là cải thiện hay regression.

Permutation mới có ID `FINHRM_CANDIDATE_REVERSE_REINDEX_REMAP_V1`:

- candidate order được đảo toàn bộ theo `new[j] = old[N-1-j]`;
- candidate được reindex và mọi `derived_operands` được remap;
- model nhận đúng sequence đã đảo một lần;
- output được remap new-to-old ở selected program, toàn bộ search-trace program, DAG candidate reference và slot binding;
- trace rows, operand-slot order và mọi non-index field không bị sắp lại.

Metric primary/permutation của transform mới chỉ được báo mô tả trên denominator `42`; không có target để làm đẹp metric.

## Execution đã khóa

Execution tiếp theo gộp baseline materialization và observed trace để không tạo thêm chuỗi bước dư thừa:

1. Process A chạy `42` bare primary và `42` bare reverse-v1 permutation, không capture trace input.
2. Process B sạch chạy lại `42` observed primary và `42` observed permutation với identity-preserving capture.
3. `42/42` primary và `42/42` permutation phải khớp bare baseline trên action, state, program, DAG, evidence/result digest, obligations, reasons và acceptance basis.
4. Bare primary còn phải khớp `42/42` historical parent trên tất cả field mà parent thực sự có. Không suy các field parent còn thiếu.

Tổng cộng `168` source query, chỉ sáu development source có hash đã khóa, `mineru_auto_generate=false`. Timeout/source failure giữ nguyên denominator và làm execution FAIL.

## Scope và safety

Cohort giữ `42` câu thuộc bảy category, `6` company, `6` source, `3` real document layout, `6` real table layout và `2` extraction family. Không đọc protected split, không dùng ABW hoặc gold trong runtime.

Final trace không được lưu question/raw OCR/label/company/source/page/year/value/rank/score/confidence/warning prose hay opaque digest. Generic root vẫn cần exact typed signature trên ít nhất ba company, ba source và nhiều real layout; số tham số model không được coi là causal proof.

Sau execution phải chạy lại safe replay `363/363`, candidate permutation `363/363`, `141` counterfactual với false accept `0`, cùng relevant regression và source safety gates.

## Authorization

Bước duy nhất được mở là:

`QP4-POST-QUESTION-SURFACE-POST-CACHE-RESIDUAL-COVERAGE-POST-SOURCE-PROTOCOL-SEMANTIC-TRACE-EVALUATOR-CONTRACT-RECLOSURE`

Runtime whitelist rỗng. QP4 chưa hoàn tất; QP5 và Phase D vẫn khóa.
