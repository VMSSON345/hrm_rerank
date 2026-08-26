# QP4 post-source-protocol diagnostic review v1

Diagnostic gate: **PASS**. QP4 gate: **FAIL**. Review chỉ đọc `90` frozen logical rows và `90` nested permutation results; không đọc PDF/OCR cache, không chạy source/model/OCR và không sửa runtime.

## Kết luận trung thực

- Safety/source protocol vẫn đóng: source error `0`, accepted-wrong `0`, semantic-control false accept `0`, hard-negative ANSWER `0`; frozen execution/permutation giữ `363/363`, counterfactual false accept `0/141`.
- Utility lõi chưa đóng: target `0/60`, toàn bộ positive expected-ANSWER `0/72`, canonical positive `0/12`. Vì canonical cũng fail, đây không chỉ là lỗi câu dài/paraphrase/không dấu.
- Negative controls đều fail-closed `18/18`, nhưng exact action contract chỉ `8/18`; mười row dùng `INVALID/REJECT` thay cho `UNKNOWN/ABSTAIN`.

## Phân loại 90 row

- `8` exact PASS controls.
- `9` P0 generic contract signatures: `6` year-shift missing-proof bị biến thành REJECT và `3` ambiguity null-plan bị materialize thành ratio rồi REJECT. Một ambiguity khác có signature riêng nên không được gộp.
- `18` P1 generic planner signatures: base clause-reorder đảo attachment ACTUAL/PLAN (`6`); reference-base clause-reorder (`6`) và paraphrase (`6`) làm ratio intent rơi thành SELECT, mất reference base.
- `21` P1 core rows có generic earliest-layer signature nhưng **trace chưa đủ để quy causal runtime root**: canonical candidate/retrieval, canonical reference-base binding, role-swap candidate support và PA07 operation-contrast candidate support. Artifact chỉ lưu boolean support/binding, không lưu full typed candidate inventory, rejection reason, synthesized slots hay proof-obligation reason.
- `29` P2 surface rows phụ thuộc cùng candidate/binding failure của canonical core; không được mở patch riêng để nâng surface/permutation metric.
- `5` P3 single-company/single-layout hoặc heterogeneous exception; giữ trong denominator nhưng không sửa.

Permutation exact hiện `37/90`, mismatch `53/90`, nhưng false accept vẫn `0`. Đây là dependent robustness metric, không phải semantic proof và không tự cấp quyền sửa.

## Quyết định

Chưa mở implementation precommit. Một patch planner đơn lẻ không tạo ra canonical verified answer nào vì canonical planning vốn đã đúng và fail ở candidate/binding. Sửa candidate/binding ngay cũng là phỏng đoán vì artifact không giữ exact rejection reason hay smallest responsible symbol.

Bước tiếp theo chỉ khóa minimal semantic trace cho toàn bộ `24` canonical/positive-semantic-control observations và `18` negative controls (`42` primary + `42` permutation), trên đủ `6` company / `6` source / `3` real document layout. Không tạo cohort/câu hỏi mới và không sinh OCR mới. Trace chỉ được giữ typed plan, typed candidate admission/rejection, synthesized slots, verifier obligations và final action; cấm dùng rank/score/confidence làm proof hoặc persist raw OCR/question literal.

Authorized next step: `QP4-POST-QUESTION-SURFACE-POST-CACHE-RESIDUAL-COVERAGE-POST-SOURCE-PROTOCOL-SEMANTIC-TRACE-RECLOSURE-PRECOMMIT`.

QP4 chưa complete. QP5 và Phase D tiếp tục khóa.
