# FinHRM reference review — neuro-symbolic strategy and answer presentation

Ngày rà soát: 20/08/2026. Phạm vi: bốn tài liệu Markdown trong `tham_khao/` và kiến trúc/runtime hiện có của FinHRM.

## 1. Kết luận

Không đổi research claim và không chuyển sang enterprise-KG/KG-completion. Hướng đúng vẫn là:

```text
financial report
  -> typed, provenance-aware evidence graph
  -> neural intent/retrieval proposals
  -> symbolic binding, execution, proof and abstention
  -> verified answer envelope
  -> deterministic user-facing rendering
```

Điều cần bổ sung là **answer contract**, không phải một reasoning model mới. Runtime hiện đã có value, program, operands,
unit, source name, page và evidence context nhưng `DocumentReasoningAnswer.render()` đang thiên về technical audit. Nó chưa có
user-facing claim, citation chuẩn, phép tính dễ đọc hoặc summary nhiều mục như sản phẩm yêu cầu.

## 2. Điều học được từ tài liệu tham khảo

### 2.1 Những điểm phù hợp và nên áp dụng

- VeNRA/UFL: truy xuất typed variables thay vì đưa text dài cho model tự tìm số và tự tính. `EvidenceCell`/evidence graph của
  FinHRM đã đi đúng hướng này.
- Double-Lock Grounding: số phải có physical alignment và semantic identity. FinHRM nên diễn đạt contract hiện hữu thành hai
  khóa rõ ràng: `source_span/page/table/row/column` và `metric/period/entity/role/unit`.
- Schema-driven cognitive offloading: `ACTUAL`, `PLAN`, `LIMIT`, `FORMULA`, `TOTAL`, `SUBTOTAL` phải là typed state; một formula
  hoặc limit không được dùng như scalar actual.
- Hybrid lexical–semantic retrieval: neural/semantic retrieval chỉ tạo candidate; lexical/typed constraints và proof quyết định
  acceptance. Không dùng similarity score làm semantic proof.
- Program-aided execution: model không tự tính; executor tạo kết quả và derivation có thể replay.
- Ecological adversarial controls: cần kiểm tra wrong year/neighbor column, variable swap, unit-scale swap, wrong entity,
  wrong citation và correct-result/wrong-provenance.
- Survey về KG nhấn mạnh open-world uncertainty: thiếu triple/evidence không đồng nghĩa false. Điều này củng cố chính sách
  `UNKNOWN -> ABSTAIN` và cấm suy period/provenance từ absence.
- Rule chỉ hữu ích khi bổ sung thông tin mà neural/data chưa mã hóa. Không thêm GNN/KGE/rule layer chỉ để stack model.
- “Interpretability” phải được đo bằng proof-trace fidelity, citation accuracy, derivation correctness và claim support; không
  chỉ đánh giá cảm tính câu trả lời dễ đọc.

### 2.2 Những điểm không áp dụng ở thesis core hiện tại

- Không thêm 3B Sentinel, fine-tune, confidence threshold hoặc LLM-as-judge trong QP3. Đây là subsystem mới và không giải quyết
  gate ADD layout hiện tại.
- Không dùng fallback kiểu “trust raw text over structured data” để bypass evidence graph. Nếu graph và text mâu thuẫn, FinHRM
  phải `ABSTAIN/CONFLICT`, không trích số ad hoc rồi vẫn trả lời.
- Không nhập threshold lexical/alignment từ paper khác. Mọi threshold phải có precommit/ablation riêng.
- Không chuyển sang KG completion, learned rule weights, online learning hoặc enterprise ontology. Bài toán hiện tại là grounded
  financial document QA, không phải dự đoán triple còn thiếu.
- Không coi các con số thực nghiệm của preprint enterprise-KG là bằng chứng cho FinHRM; tài liệu này chỉ dùng để tham khảo
  kiến trúc/giải thích.

## 3. Verified Answer Envelope v1 đề xuất

Renderer chỉ được nhận object đã qua executor/verifier:

```text
status: ANSWER | ABSTAIN | REJECT
answer_mode: FACT | COMPUTATION | VERIFIED_SUMMARY
claim: metric + period + entity/scope + value + unit + scale
canonical_value: exact unscaled numeric value
display_value: locale-aware value, never the source of computation
program: operation/DAG + operand references + intermediate results
citations[]: document title, statement/note, page, table, row, column, source span
proof: required obligation states + acceptance basis
summary_coverage: requested fields, verified fields, missing/conflicting fields
warnings: conversion, restatement, conflict or incomplete coverage
```

Không có `page` thì không được sinh “trang X”. Không có statement/note identity thì dùng citation thấp hơn đã được chứng minh
(document/page/table/row/column); không tự đặt tên báo cáo. Khi đổi VND sang triệu/tỷ, giữ cả `canonical_value`, `scale` và
`display_value` để tránh mơ hồ dấu chấm/dấu phẩy.

## 4. Ba kiểu trình bày bắt buộc

### 4.1 FACT — câu tra cứu một chỉ tiêu

```text
Doanh thu thuần năm 2024 là **12.450 tỷ đồng**
(giá trị gốc: 12.450.000.000.000 đồng).

Nguồn: Báo cáo kết quả hoạt động kinh doanh năm 2024, trang X,
dòng “Doanh thu thuần”, cột “Năm 2024”.
```

Chỉ hiển thị đúng mẫu trên khi metric, year, entity, unit/scale, logical cell và citation đều VALID.

### 4.2 COMPUTATION — câu cần tính

```text
Kết quả: Tổng A và B năm 2024 là **150 tỷ đồng**.

Phép tính:
- A = 100 tỷ đồng — nguồn: Báo cáo ..., trang X, dòng A, cột 2024.
- B = 50 tỷ đồng — nguồn: Báo cáo ..., trang Y, dòng B, cột 2024.
- 100 + 50 = 150 tỷ đồng.

Kết luận: ...
```

Mỗi operand và intermediate phải mang citation/lineage riêng. Renderer không được viết một formula khác program đã execute,
không được làm tròn trước khi tính và không được che operand UNKNOWN.

### 4.3 VERIFIED_SUMMARY — “cho biết nội dung chính của báo cáo”

Summary không phải một lần gọi LLM tự do. Query được compile thành một inventory các claim độc lập, ví dụ:

- tổng tài sản, tổng nợ phải trả, vốn chủ sở hữu;
- doanh thu thuần, lợi nhuận sau thuế;
- lưu chuyển tiền thuần từ hoạt động kinh doanh;
- thay đổi so với đầu kỳ/kỳ trước nếu cả hai endpoint được chứng minh;
- các note/điểm nổi bật chỉ khi có cited qualitative evidence.

Renderer tạo bảng:

| Nội dung | Số liệu/so sánh | Điểm nổi bật đã được chứng minh | Nguồn |
|---|---:|---|---|
| Bảng cân đối kế toán — Tổng tài sản | 89.816... VND; tăng 8% | Tài sản ngắn hạn ...; tài sản dài hạn ... | trang 2–4, row/column |
| Kết quả kinh doanh — Doanh thu thuần | ... | tăng/giảm ... | trang ..., row/column |
| Lưu chuyển tiền tệ — CFO | ... | ... | trang ..., row/column |

Mỗi row là một verified sub-answer. Nếu inventory yêu cầu `M` trường nhưng chỉ `N` trường VALID, hệ thống phải hoặc
`ABSTAIN` theo `answer_mode_complete`, hoặc ghi rõ **“Tóm tắt đã xác minh N/M mục; các mục còn lại thiếu/conflict evidence”**.
Không được trình bày partial output như một “tóm tắt đầy đủ”. “Điểm nổi bật” định lượng được derive từ verified values;
nguyên nhân/đánh giá định tính chỉ được viết khi report text cung cấp citation trực tiếp.

## 5. Gate và metric cần bổ sung khi được mở runtime

- numeric exact và unit/scale exact;
- citation document/page/table/row/column exact;
- operand citation coverage và derivation replay exact;
- answer claim entailment và unsupported-claim count = 0;
- summary field completeness/coverage;
- permutation invariance của content dù thứ tự candidate đổi;
- hard negatives: wrong page, wrong statement, adjacent column/year, swapped operand, scale ×1.000, missing summary field,
  qualitative cause không có evidence;
- company/layout-disjoint confirmation, không template riêng theo report;
- ABW không làm confirmation; không đọc protected split để prompt-tune.

## 6. Điều chỉnh thứ tự roadmap

Giữ nguyên bước duy nhất đang mở: đóng ADD diversity gate của
`QP3-CORE-GENERALIZATION-COVERAGE-IMPLEMENTATION`. Không sửa renderer trong scope đó.

Sau khi QP3 core đóng, thêm `QP3-ANSWER-PRESENTATION-PRECOMMIT` trước QP4 để khóa schema, renderer modes, citation and
summary-completeness gates. Bản đầu dùng deterministic templates; LLM explanation/synthesis vẫn là extension sau QP5 và mọi
claim của nó phải được kiểm lại từ Verified Answer Envelope.

