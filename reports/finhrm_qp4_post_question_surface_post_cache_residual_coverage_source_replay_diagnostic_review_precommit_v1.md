# QP4 residual-coverage source-replay diagnostic-review precommit v1

- Gate: `PASS`; đây là contract-only precommit. Không chạy lại source/PDF/OCR/MinerU, không inference mới và không sửa runtime.
- Frozen denominator: `90` logical primary rows cùng `90` permutation results đã có sẵn. Primary partition là question plan `19`, source `37`, candidate/retrieval `12`, binding/synthesis `7`, proof/action `6`, PASS control `9`.
- Có `41` source errors cùng chữ ký cache-miss khi auto-generation bị tắt. Chỉ `37` là earliest source failure; `4` hàng đã fail ở question plan trước đó và không được relabel thành source root cause.
- Tám stratum generic được giữ tách biệt theo đúng `category × earliest-layer`; mỗi stratum tự đạt tối thiểu `3` company / `3` source / `3` real document layout. Không gộp câu hỏi khác nghĩa, layer khác nhau hoặc PASS controls để tạo coverage.
- Source review phải phân biệt request/attempt với generation: replay ghi `571` pre-ranking attempt page-events và `1142` unselected requested page-events, nhưng `0` newly materialized và `0` unselected generated. Timeout/cache availability không tự chứng minh lỗi OCR/layout algorithm.
- Mỗi logical row phải nhận đúng một disposition có evidence. Generic root cause chỉ được xác nhận khi cùng category, cùng earliest layer và cùng exact root signature; group size một mình không đủ.
- Frozen controls phải giữ: compiler `242/242`, regression `360/360`, execution/permutation `363/363`, `141` counterfactual với `0` false accept và safe replay gate `PASS`.
- Runtime whitelist vẫn rỗng. QP4 chưa hoàn tất; QP5 và Phase D vẫn khóa.
- Next allowed action: `QP4-POST-QUESTION-SURFACE-POST-CACHE-RESIDUAL-COVERAGE-SOURCE-REPLAY-DIAGNOSTIC-REVIEW`, chỉ đọc final replay report/lock đã khóa và xuất đúng một JSON, một MD, một lock.
