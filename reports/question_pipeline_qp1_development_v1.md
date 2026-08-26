# Báo cáo QP1 — Evidence schema và table-header reconstruction

Ngày khóa: 13/08/2026  
Phạm vi: development-only và ABW diagnosis-only  
Quyết định: QP1 hoàn tất; cho phép bắt đầu QP2 nhưng chưa cho phép bật strict verifier/acceptance.

## Mục tiêu

QP1 bổ sung bằng chứng cấu trúc để phân biệt các ô cùng metric và cùng năm nhưng khác vai trò cột,
đặc biệt là `actual`, `plan`, `revised_plan`, `forecast`, `% actual/plan` và
`% actual/prior`. Vòng này không thay đổi question compiler, neural score, threshold hay quyết định
`ACCEPT/ABSTAIN`.

## Đã triển khai

- `EvidenceCell` có thêm `column_role`, `column_role_source`, trạng thái
  `EXPLICIT/INHERITED/CONFLICT/UNKNOWN`, raw `header_path`, provenance từng tầng header,
  raw header, extraction method và source span.
- Reconstruct được Markdown header nhiều tầng, ô header gộp, repeated header gần nhất, panel song
  song và native-PDF header bị dàn thành nhiều dòng.
- Role chỉ được suy từ header rõ ràng; không suy `actual/plan` từ vị trí trái-phải hoặc giá trị số.
- Native và OCR bất đồng trên cùng logical cell bị đánh `CONFLICT`; neural score không được tự chọn.
- Cache identity nâng thành `v12b2-applicability-proof-candidate-cache-v5`, gắn
  `evidence-cell-header-role-v2` và fingerprint
  `de4c47be90bced3f4700a39aeebb53efdad081585a8169cfd547c77c6b3c89b0`.
  Cache cũ chưa được rebuild và không còn hợp lệ cho schema mới.

## Kết quả development-only

Audit trên 1.246 example và 37.781 frozen candidate:

- 1.766 candidate có role rõ; 36.015 candidate giữ `UNKNOWN`;
- role: 1.299 `actual`, 159 `plan`, 16 `forecast`, 86 `% actual/plan`,
  206 `% actual/prior`;
- 9 candidate thuộc 6 example có xung đột header và được giữ `CONFLICT`;
- 1.458/1.458 counterfactual đổi `actual ↔ plan` làm role đổi đúng, đạt 100%;
- chỉ 115 example trong cohort cũ có ít nhất một role plan/actual rõ. Vì cohort cũ chủ yếu chứa
  bảng theo kỳ, tỷ lệ `UNKNOWN` cao không được diễn giải là độ chính xác thấp của QP1;
- trong 5 gold example có câu hỏi yêu cầu riêng actual/plan, 3 example gắn đúng role ở gold operand
  và 2 giữ `UNKNOWN`; mẫu này quá nhỏ để kết luận release.

## ABW diagnosis-only

Đọc lại page-index local của `ABW_23CN_BCTN.pdf`, trang 10, QP1 tái dựng đúng năm cột:

1. `363.638.445.420` — `Thực hiện 2022` — `actual`;
2. `472.723.394.169` — `Kế hoạch 2023` — `plan`;
3. `304.912.639.828` — `Thực hiện 2023` — `actual`;
4. `-35,50%` — `% TH 2023 / KH 2023` — `actual_to_plan_ratio`;
5. `-16,15%` — `% TH 2023 / TH 2022` — `actual_to_prior_ratio`.

ABW vẫn mang nhãn diagnosis-only, không đủ điều kiện confirmation và không được dùng để chọn
threshold. Kết quả này chứng minh extraction đã mang được role đúng, chưa chứng minh runtime sẽ trả
đúng số vì QP2/QP3 chưa nối question role vào typed program và proof obligation.

## Kiểm thử

109/109 test liên quan đạt, gồm positive, missing-role, conflict, counterfactual, repeated header,
merged header, parallel panel, no-diacritic/OCR noise, candidate permutation và hồi quy
`program_data/question_planner/direction_contract`.

## Invariants

- Không đọc validation/blind/test trong các script QP1.
- Không fine-tune; checkpoint vẫn có SHA-256
  `3a33276ded0d2ff2dd3d200c4c774a59eec0f8cad7102bc273cb263d983944de`.
- Không thay threshold, neural score, runtime acceptance hoặc Phase D lock.
- Không replay S4 để tuyên bố metric mới vì QP1 chưa được nối vào acceptance path.

## Bước tiếp theo

Thực hiện QP2: mở rộng `QuestionPlan` và compiler để gắn `actual/plan/revised_plan/forecast`, year,
entity, unit và answer mode theo từng operand; compile PA01–PA12 và TY01–TY12; câu dài/multi-output
phải decompose hoặc abstain rõ. Chỉ sau khi compiler contract đạt mới sang QP3 để thêm
`column_role_match` vào proof obligation.

