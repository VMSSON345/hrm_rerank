# Báo cáo QP2/QP3 priority — nối column role vào runtime

Ngày: 13/08/2026  
Phạm vi: development-only; ABW diagnosis-only  
Quyết định: bật strict role grounding cho câu hỏi explicit `actual/plan`, nhưng không bật
proof-conditioned acceptance cho SELECT và không thay threshold.

## Mục tiêu ưu tiên

Giải quyết trực tiếp lỗi ABW sau QP1: evidence đã biết `actual/plan` nhưng question compiler,
typed synthesis và verifier chưa dùng role đó. Mục tiêu an toàn là winner phải đúng role hoặc
ABSTAIN; không được tiếp tục đưa ô kế hoạch sai ra làm candidate kỹ thuật.

## Đã triển khai

- `QuestionPlan.column_roles` biên dịch role theo operand ở dạng contract
  `ACTUAL/PLAN/REVISED_PLAN/FORECAST`.
- Câu lookup có role rõ được khóa operation `SELECT`; câu hoàn thành kế hoạch được biên dịch
  `RATIO_PERCENT(ACTUAL, PLAN)`.
- Sửa lỗi từ `hoạt động` bị hiểu nhầm token `đồng`, nhờ đó metric giữ đúng
  `doanh thu hoạt động` và không tự sinh currency constraint.
- Typed synthesis tạo slot `(metric, year, column_role, entity, unit)` và loại candidate
  role sai, thiếu role hoặc conflict khỏi explicit-role slot.
- Proof contract có required obligation `column_role_match`.
- Runtime grounding trả `column_role_mismatch`, `missing_requested_column_role` hoặc
  `column_role_conflict`; neural score không được vượt qua các lỗi này.
- Với role lặp qua hai năm, compiler giữ role theo từng operand thay vì gộp thành một role.

## Compiler audit

Catalog development-only 299 câu được giữ nguyên:

- compiler exact tăng từ `7/299 = 2,34%` lên `93/299 = 31,10%`;
- column-role pass tăng từ `70/299 = 23,41%` lên `212/299 = 70,90%`;
- operation pass tăng từ `161/299 = 53,85%` lên `213/299 = 71,24%`;
- plan/actual exact đạt `42/132`; two-year exact đạt `36/132`;
- PA01 và PA02 đạt `9/11`; hai failure còn lại mỗi family là hard-negative/ambiguity cần
  document-conditioned abstention, không thể kết luận chỉ từ compiler-only text;
- toàn bộ QP2 chưa hoàn tất: answer mode, multi-output, discourse và PA04–PA12/TY còn phải làm.

## ABW full-runtime diagnosis

Với câu `Doanh thu hoạt động thực hiện năm 2023 là bao nhiêu?`:

- compiled plan: `SELECT`, role `ACTUAL`, year `2023`, metric `doanh thu hoạt động`;
- winner đổi từ số kế hoạch sai `472.723.394.169` sang đúng ô thực hiện
  `304.912.639.828`;
- evidence: `COLUMN: Thực hiện 2023`, role `actual`, confidence state `EXPLICIT`;
- required proof đạt `100%`: `query_metric`, `column_role_match`, `period_coverage` và các
  obligation nền đều `VALID`;
- confidence vẫn là `0,516992 < 0,845693`, nên kết quả giữ `ABSTAIN` theo threshold hiện hành.

ABW vẫn là diagnosis-only, không được dùng làm confirmation hoặc chọn operating point.

## S4 audit-only counterfactual

Trong 227 accepted SELECT/ADD của frozen S4 chỉ có 2 accepted SELECT explicit role:

- strict role filter loại `F=1` false accept và mất `T=0` true accept;
- gold role retention `2/2`;
- conservative overall: 936 accepted, 829 đúng, 107 false accept;
- coverage `95,902%`, accepted precision `88,568%`, `9F−T=9`;
- fail precision gate 90%, fail `9F−T ≥ 143`, và cỡ mẫu explicit-role không đủ.

Do đó không mở proof-conditioned SELECT acceptance. Việc mở lane này lúc này sẽ là kết luận vượt
quá bằng chứng dù ABW riêng lẻ có complete proof.

## Kiểm thử và invariants

- 264/264 test liên quan đạt.
- Không fine-tune; checkpoint SHA-256 vẫn là
  `3a33276ded0d2ff2dd3d200c4c774a59eec0f8cad7102bc273cb263d983944de`.
- Không thay threshold, không đọc validation/blind/test, Phase D vẫn khóa.
- Cache cũ không được rebuild; QP1 cache schema v5 vẫn là contract hiện hành.

## Bước tiếp theo

Tiếp tục QP2 cho PA04–PA12/TY và tạo/reuse development-only company-disjoint explicit-role
positive + hard-negative cohort đủ lớn. Sau đó chạy counterfactual theo gate đã khóa; chỉ khi pass
mới cân nhắc proof-conditioned acceptance cho SELECT. Trong thời gian đó UI phải giữ hành vi:
đúng evidence hoặc ABSTAIN, không trả evidence sai role.

