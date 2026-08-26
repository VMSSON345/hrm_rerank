# QP4 Final Gate Reclosure

Kết luận: **FAIL_CLOSED**. QP4 chưa hoàn tất; QP5 và Phase D vẫn khóa.

Đây là lượt tổng hợp chỉ đọc. Không sửa runtime, model, checkpoint, threshold hay gold; không đọc validation/blind/test/sealed/holdout/final; không dùng ABW hoặc synthetic layout làm confirmation.

## Kết quả chạy mới

- Compiler inventory: `242/242`; ADD surface projection `15/15`; reference-base projection `4/4`.
- Affected regression: `360/360`, exit code `0`.
- Frozen safe replay: actions `257 ANSWER / 92 ABSTAIN / 14 REJECT`; execution `363/363`; candidate permutation `363/363`.
- Counterfactual: `141`, false accept `0`.
- Checkpoint `models/document_hrm_V8_candidate.pt`: hash không đổi.
- Integrity: `8` terminal lock và `109` nested path/hash reference đều khớp; không thiếu file, không mismatch.

## Gate inventory

| Gate | Kết quả | Evidence |
|---|---:|---|
| Contract/artifact integrity | PASS | 8/8 lock, 109/109 reference hash |
| Question planner và surface compiler | PASS | 242/242 |
| Unit-scale proof contract | PASS | Stricter proof contract đã reclose; 2 ANSWER cũ chuyển ABSTAIN, không promote unsafe answer |
| Page-selective source protocol | PASS | 90 primary + 90 permutation, source error 0, cache reusable 90, accepted-wrong 0 |
| Regression và frozen safety | PASS | 360/360; execution/permutation 363/363; false accept 0/141 |
| Native/OCR citation carrier | PASS | coverage 3/3/3/3; citation 6/6; program 3/3; negative controls 15/15 |
| Full source exact target | **FAIL** | yêu cầu 60/60; locked full-cohort gần nhất 0/60 |
| Full source base exact | **FAIL** | yêu cầu 30/30; gần nhất 0/30 |
| Full source reference-base exact | **FAIL** | yêu cầu 30/30; gần nhất 0/30 |
| Full source surface equivalence | **FAIL** | yêu cầu 12/12; gần nhất 0/12 |
| Full source-cohort permutation | **FAIL** | yêu cầu 90/90; post-protocol observation gần nhất 37/90 |

`363/363` permutation là frozen safe replay, không phải source-cohort permutation `90/90`; hai gate không được tráo cho nhau. Citation cohort mới chỉ chứng minh carrier/citation/program trên ba real source-layout instances. Nó không chạy lại toàn bộ 60 target question và 90 source observations qua discovery → retrieval → planning → binding → proof/action, nên không thể thay thế các gate đang fail.

Các số `0/60`, `0/30`, `0/12`, `37/90` là full-cohort measurement được khóa gần nhất, nhưng có trước một số sửa chữa QP4 về sau. Điều đó không cho phép suy PASS; nó cho thấy hiện chưa có evidence full replay hiện hành đủ để đóng gate.

## Bước duy nhất được mở tiếp

`QP4-FULL-SOURCE-SEMANTIC-REPLAY-REFRESH-PRECOMMIT`: khóa một lượt **read-only rerun đúng cohort 90 primary/90 permutation hiện có**, giữ nguyên gold và mọi denominator. Không thêm example, không sửa runtime/model/threshold/checkpoint. Nếu refresh pass thì chạy lại final gate; nếu fail thì chỉ mở diagnostic/precommit cho failure group generic quan sát được.

Chi tiết máy đọc: `reports/finhrm_qp4_final_gate_reclosure_v1.json`.
