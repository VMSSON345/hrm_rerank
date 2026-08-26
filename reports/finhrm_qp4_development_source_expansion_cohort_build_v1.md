# QP4 development-source expansion cohort build

## Kết quả

Build **FAIL_CLOSED**. Đúng 50 PDF development đã hash-lock được xử lý, không thay source hậu kiểm. Không tìm được anchor mới đủ contract cho `hierarchical` hoặc `fragmented`; vì vậy runtime, source replay diagnostic, QP5, Phase D và publication pilot tiếp tục khóa.

Đây không phải kết luận rằng hệ thống thất bại trên 50 báo cáo hay rằng hai cấu trúc không tồn tại trong báo cáo Việt Nam. Build chỉ kiểm tra native text/word geometry cho hai signature PA07 rất hẹp. Hai mươi lăm source là image-dominant và contract cấm sinh OCR, nên chúng là evidence không khả dụng trong bước này, không phải negative prevalence proof.

## Frozen selection và execution

- eligible pool: 112 PDF / 112 company;
- deterministic sample: 50 PDF / 50 company;
- eligible-pool hash và selected-manifest hash đều exact;
- source processed/hash match: `50/50`;
- physical pages scanned: `2.425`;
- native/geometry non-empty pages: `1.541/1.541`;
- broad candidate pages: `19` trên `8` source;
- native table grids trên candidate pages: `13`;
- source read error: `0`;
- new OCR, question planner, HRM/ranker, answer evaluator và protected split read: `0`.

Terminal source disposition:

- `25` `REJECT_NO_NATIVE_TEXT_OR_GEOMETRY`: phần lớn non-cover page là image-only; không được OCR trong build này;
- `24` `REJECT_NO_TARGET_STRUCTURE`: native content không có một trong hai exact target signature đủ proof;
- `1` `REJECT_MISSING_EXPLICIT_SAME_PERIOD`: bảng thật có `Actual 2024` và `Plan 2025`, không phải same-period PA07;
- admitted target anchor: `0`.

## Visual QA material

Visual QA dùng PyMuPDF vì Poppler không có trong môi trường. Chỉ render tạm bốn structurally material pages và đã xóa sau verify.

- NHA trang vật lý 100: bảng có `Plan/Actual`, nhưng `2024` chỉ là section title; không có parent period header/spanner vật lý gắn năm với hai role column.
- NHA trang vật lý 108: one-row `Plan/Actual/% Planning/Change`; năm vẫn chỉ ở surrounding section title, không được suy vào cell provenance.
- HTI trang vật lý 73: row label có wrap và table header rõ, nhưng `Actual in 2024` đối chiếu `Plan in 2025`; phải reject vì khác kỳ.
- CMK trang vật lý 3: bảng thật có `Kế hoạch 2024 / Thực hiện 2024` và `Tr.đ`, nhưng đây là `flattened compact role-year header`. Nó là out-of-target control, không được dùng để lấp hierarchical/fragmented gate.

Không dùng section title, nearby prose, thiếu conflict, numeric equality, rank/score/confidence hoặc filename/page/year/value làm semantic proof.

## Exact gate

`PA07_ROOT_HIERARCHICAL_MULTIROW_ROLE_PERIOD_HEADER_V1`:

- frozen prior: `1 company / 1 source / 1 real layout / 1 instance`;
- new admitted: `0`;
- combined: `1/3`, shortfall `2` cho từng dimension;
- gate: **FAIL**.

`PA07_ROOT_FRAGMENTED_VISUAL_ROW_LABEL_V1`:

- frozen prior: `0`;
- new admitted: `0`;
- combined: `0/3`, shortfall `3` cho từng dimension;
- gate: **FAIL**.

`flattened` giữ frozen `PASS_3_3_3`; CMK không được cộng thêm vì không phải expansion target.

## Integrity và trạng thái tiếp theo

Runtime/test/model/checkpoint/threshold không đổi; không fine-tune, không ABW, không protected evaluation, không synthetic confirmation và không metric beautification. Vì không có runtime change, regression, permutation, counterfactual và safe replay không được chạy lại trong structural build này.

Theo precommit failure behavior, trạng thái bắt buộc là: **remain FAIL_CLOSED and report exact deterministic-sample shortfalls; no runtime edit**. Không tự tăng sample, thay source, nới proof hoặc mở source replay diagnostic để làm đẹp gate.
