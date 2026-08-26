# QP4 Full Source Semantic Replay Refresh Precommit

Kết quả precommit: **PASS**. Bước này chỉ khóa contract; chưa chạy source replay và không sửa runtime/model/checkpoint/threshold/gold.

## Cohort và denominator đã khóa

- Giữ nguyên toàn bộ `6` anchor: `6` company, `6` PDF, `3` real document layout, `6` real table layout và `2` extraction family.
- Inventory: `30` base target + `30` reference-base mutation target + `30` semantic control = `90` primary observations.
- Candidate permutation: đúng `90` observation tương ứng.
- Target gates: exact answer `60/60`, base `30/30`, reference-base mutation `30/30`, surface-equivalence `12/12`.
- Safety gates: accepted-wrong `0`, old-winner false accept `0`, semantic-control false accept `0`, hard-negative answer `0`.
- Không được drop, thay anchor, thêm câu hỏi hoặc chọn lại cohort sau khi thấy outcome.

## Tính executable

Schema audit đạt:

- `6/6` source tồn tại và khớp SHA-256.
- `6/6` anchor có đủ ba operand, program, expected Decimal và provenance để hậu kiểm.
- Đủ `90/90` primary oracle và `90/90` permutation oracle.
- Permutation dùng contract đã reclose `FINHRM_CANDIDATE_REVERSE_REINDEX_REMAP_V1`: reverse, reindex candidate và remap chính xác program/trace/DAG/slot binding. Không cần seed hoặc hidden state.
- Observation chỉ nhận question + development PDF ở public boundary; gold chỉ dùng sau khi runtime hoàn tất.

Con số lịch sử `37/90` dùng transform không được freeze nên không phải baseline của transform mới và không được gọi là cải thiện/regression. Gate hiện tại vẫn là `90/90`.

## Execution được phép

`QP4-FULL-SOURCE-SEMANTIC-REPLAY-REFRESH` sẽ chạy đúng 90 primary observations với `mineru_auto_generate=false`, sau đó chạy permutation transform cho từng candidate sequence. Source error, timeout và abstention vẫn nằm trong denominator; safe abstention không được tính thành answered.

Sau source replay phải chạy lại compiler `242/242`, regression `360/360`, frozen execution/permutation `363/363` và `141` counterfactual với false accept `0`.

Nếu mọi gate pass, chỉ được chạy lại `QP4-FINAL-GATE-RECLOSURE`; chưa được tự mở QP5. Nếu còn fail, chỉ được mở diagnostic-review precommit cho earliest-layer group generic quan sát được.

QP4 chưa hoàn tất; QP5 và Phase D vẫn khóa.
