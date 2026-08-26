# QP4 Candidate replacement instrumentation replay v2

## Kết luận

**HOLD — INSUFFICIENT_EVIDENCE.** Replay hoàn tất đúng ba Candidate case đã precommit, nhưng không chứng minh được cùng `module/symbol + branch/predicate + causal signature` trên cả ba company/layout. Không mở Candidate runtime repair.

## Capture failure và cơ chế thay thế

Replay trước hoàn thành inference nhưng chỉ xuất payload trace dài qua stdout. Terminal transport giữ khoảng 3.000 trên khoảng 59.000 token rồi tiến trình kết thúc, nên phần bị cắt không còn buffer để phục hồi. Đây là lỗi vận chuyển output, không phải lỗi model/runtime.

Replay thay thế ghi trực tiếp vào `reports/finhrm_qp4_positive_target_candidate_replacement_instrumentation_replay_v2.jsonl` bằng file handle độc lập. Mỗi case được append, `flush` và `os.fsync` ngay sau khi ghi. File trace không dùng SQLite connection/handle, không truyền qua page-index layer và không phụ thuộc stdout.

## Execution boundary

- Candidate: đúng 3 baseline + 3 instrumented inference: HUB, ICT, VNL theo frozen order.
- Binding/Provenance: 0 inference; không replay vì chưa đạt invariant 3/3/3.
- Không thêm, thay, fallback case hay mở rộng cohort.
- Output/decision baseline so với instrumented: **3/3 unchanged**.
- Baseline khớp các observed field đã khóa: **3/3**.
- Runtime edit: 0. Model/threshold/checkpoint change: 0.
- Candidate diversity thực chạy: 3 case, 3 company, 3 source, 3 real document layout, 3 table layout, 2 extraction family.

## Root-cause evidence

1. `qp4-rcc-hub-parent-profit-after-tax:base_target:canonical`
   - Missing operand: `current_plan`.
   - Exact local cause được trace: module `hrm_finance_qa.program_data`, symbol `merge_candidates->_candidate_quota_indices`, predicate `len(output) > max_candidates`.
   - Causal signature: physical expected candidate có mặt trước quota, không nằm trong selected indices sau quota và vắng tại predictor boundary.

2. `qp4-rcc-ict-total-revenue:base_target:canonical`
   - Missing operand: `current_plan`.
   - Expected page đã vào merge input, nhưng không có traced quota event ở causal path cuối chứng minh operand này bị loại. Cause code-level vẫn chưa xác định.

3. `qp4-rcc-vnl-profit-before-tax:base_target:canonical`
   - Missing operands: `current_actual`, `current_plan`.
   - Trace quan sát exclusion tại cả `extract_numeric_candidates->_candidate_quota_indices` với `len(raw_candidates) > max_candidates` và `merge_candidates->_candidate_quota_indices` với `len(output) > max_candidates`.
   - Không có một symbol/predicate duy nhất giống HUB; không được gộp các failure cùng layer thành cùng root cause.

Exact case cause được chứng minh chỉ **1/3**. Vì vậy diversity denominator đạt không thể thay thế exact code-level causal identity; kết luận cluster là `INSUFFICIENT_EVIDENCE`, không phải `GENERIC_ROOT_CAUSE`.

## Page-index integrity

- `original_before`: `df8be07147d790b7ef18a917aca8f3d3c28ca7923b959a78c2dc721a11284fd7`
- `original_after`: `df8be07147d790b7ef18a917aca8f3d3c28ca7923b959a78c2dc721a11284fd7`
- `copy_before`: `df8be07147d790b7ef18a917aca8f3d3c28ca7923b959a78c2dc721a11284fd7`
- `copy_after`: `df6968e58157f6cca352ee049312ad24ff7563ead8858113f1984a95d5d6ab83`

Frozen original byte-identical trước/sau. Byte change chỉ xảy ra trên SQLite copy tạm theo cơ chế write-capable open đã được chẩn đoán trước; copy và sidecar đã xóa sau replay.

## Next allowed action

**HOLD. Không mở Candidate repair.** Mọi replay hoặc instrumentation bổ sung cần authorization mới; Binding/Provenance vẫn không được replay từ cohort 2/2/2 hiện tại.
