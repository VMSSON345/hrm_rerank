# QP4 residual-coverage source-replay diagnostic review v1

- Diagnostic gate: `PASS`; QP4 gate vẫn `FAIL`. Review chỉ đọc `90` logical rows và `90` permutation results đã khóa; không đọc PDF/OCR cache, không chạy source/model và không sửa runtime.
- Primary earliest-layer partition giữ nguyên: question plan `19`, source `37`, candidate/retrieval `12`, binding/synthesis `7`, proof/action `6`, PASS control `9`.
- Row dispositions: `26` confirmed generic semantic root, `37` environment/cache availability, `18` non-generic fail-closed và `9` PASS controls.
- Năm generic semantic root-signature đã tự đạt tối thiểu `3 company / 3 source / 3 real document layout`: ba lỗi planner ở clause reordering/reference-base surfaces, một PA07 candidate-support gap và một ambiguity UNKNOWN/ABSTAIN bị biến thành INVALID/REJECT. Chưa root nào có exact runtime symbol đủ để cấp quyền sửa trực tiếp.
- `41` source errors cùng chữ ký thiếu compatible cache khi auto-generation tắt. Chỉ `37` là earliest source failure trên `4` company / `4` PDF / `3` document layout / `4` table layout / `2` extraction family; `4` lỗi còn lại đứng sau question-plan failure.
- Source evidence chỉ xác nhận source-protocol/cache availability blocker. `571` pre-ranking attempt và `1142` unselected request page-events không phải OCR generation: newly materialized `0`, unselected generated `0`, whole-document OCR `0`. Không có quyền kết luận hay sửa OCR/layout algorithm.
- Có `23/90` permutation mismatch. Hai coherent stratum đạt `3/3/3`, nhưng frozen report chưa chỉ ra exact responsible symbol; gate này tiếp tục fail-closed.
- `18` residual không có coherent generic root signature và không được gộp category/layer để tạo coverage.
- Frozen controls giữ nguyên: compiler `242/242`, regression `360/360`, execution/permutation `363/363`, `141` counterfactual với `0` false accept và safe replay `PASS`.
- Runtime whitelist rỗng. Bước tiếp theo duy nhất là `QP4-POST-QUESTION-SURFACE-POST-CACHE-RESIDUAL-COVERAGE-SOURCE-PROTOCOL-RECLOSURE-PRECOMMIT`.
- Sau review này còn tối thiểu `5` gated actions trước QP5 nếu không xuất hiện root mới; planning range thực tế `5–8`. QP5 chỉ mở sau khi full QP4 reclosure `PASS`.
