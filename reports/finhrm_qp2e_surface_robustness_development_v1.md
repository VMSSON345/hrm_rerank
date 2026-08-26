# FinHRM QP2E — frozen surface robustness

Ngày hoàn tất: 13/08/2026  
Phân loại: development-only synthetic compiler repair  
Phạm vi: frozen 135-row compiler catalog; không đọc tài liệu, evidence hoặc model

## Kết luận

QP2E safe gate đạt. Trong 43 lỗi surface còn lại sau QP2D, 31 lỗi được sửa bằng hai thay đổi tổng quát:

- chiếu column role theo **use-site trong program DAG**, không nhân bản immutable operand slot;
- nhận diện câu tự mô tả ambiguity bằng tổ hợp context nhiều cột và việc không chỉ rõ role.

Raw frozen exact tăng `92/135 = 68,15%` lên `123/135 = 91,11%`, có 31 improvement và 0 regression.
Trên phần compiler-only khả định, kết quả là `123/123 = 100%`. Mười hai dòng còn lại đều là
`hard_negative_missing_role` cần quan sát evidence trước khi quyết định; chúng không được ép thành ABSTAIN bằng
question keyword.

## Tại sao không tối ưu theo một năm, metric hoặc câu hỏi

Implementation không dùng family ID, company, numeric value hay các năm cố định 2022–2024:

- năm được nhận bằng pattern năm và quan hệ kỳ gốc/kỳ đích;
- role dùng normalized lexicon `ACTUAL/PLAN`, `thực hiện/kế hoạch`;
- role output được suy ra từ input references của DAG;
- ambiguity chỉ được phát khi context nói rõ có nhiều cột/role và request không chọn role;
- một keyword đơn lẻ như `bảng` không đủ kích hoạt từ chối.

Counterfactual ngoài frozen years/metric đạt `7/7`, gồm lợi nhuận 2028, chi phí 2027/2026 và tài sản
2029/2027. Các case bao phủ multi-step shortfall/excess, two-year percentage change, missing role, missing base và
explicit ambiguity.

## Kết quả

| Kiểm tra | Kết quả |
|---|---:|
| Frozen raw exact trước QP2E | 92/135 |
| Frozen raw exact sau QP2E | 123/135 |
| Compiler-identifiable subset | 123/123 |
| Improvements | 31 |
| Regressions | 0 |
| Answer variants retention | 108/108 |
| Ambiguity controls | 12/12 |
| Column-role use-site projection | 135/135 |
| Strict canonical thesis core | 16/16 |
| Metric/year counterfactual | 7/7 |
| Regression tests | 216/216 |

## Program role projection

`operand_slots` tiếp tục là tập identity duy nhất. QP2E bổ sung projection bất biến
`program_column_roles`, duyệt input references theo DAG:

- PA03 role-swap: `RATIO_PERCENT(plan, actual)` → `PLAN, ACTUAL`;
- PA08: `SUBTRACT(plan, actual) → RATIO_PERCENT(shortfall, plan)` → `PLAN, ACTUAL, PLAN`;
- PA09: `SUBTRACT(actual, plan) → RATIO_PERCENT(excess, plan)` → `ACTUAL, PLAN, PLAN`.

Projection này sửa 19 lỗi role contract nhưng không đổi `QuestionPlan.column_roles` cũ, không tạo duplicate slot và
chưa được runtime consumer sử dụng.

## Ambiguity surface

Mười hai câu `ambiguity_control` nói rõ bảng có nhiều cột cùng năm nhưng câu hỏi không xác định vai trò cột. Chúng
được compile thành `ambiguous_metric_or_column_role`, không có DAG. Rule cần đồng thời:

1. context bảng/báo cáo có nhiều cột;
2. câu tự xác nhận không nêu/chỉ rõ role.

Do đó câu hợp lệ có tiền tố `căn cứ bảng số liệu` không bị từ chối.

## Mười hai dòng không thể sửa an toàn ở question compiler

Toàn bộ residual là variant `hard_negative_missing_role`. Frozen catalog có mâu thuẫn trực tiếp:

`Doanh thu năm 2023 tăng bao nhiêu phần trăm so với năm 2022?`

- `A05-canonical`: `ANSWER`;
- `TY02-hard_negative_missing_role`: `ABSTAIN / missing_required_column_role`.

Hai record có cùng normalized question nhưng behavior trái ngược. Không có deterministic question-only compiler nào
có thể đạt cả hai. Ép mọi câu thiếu từ `thực hiện/kế hoạch` thành ABSTAIN cũng làm hỏng các câu BCTC hợp lệ dùng
ACTUAL mặc định.

Quyết định an toàn: chuyển 12 dòng sang QP3 evidence-conditioned proof. Chỉ từ chối khi evidence thực tế chứng minh
có nhiều cell cùng metric/year/entity/unit nhưng column role khác nhau và câu hỏi không chỉ rõ role. Nếu evidence chỉ
có một logical cell phù hợp, không được từ chối chỉ vì thiếu keyword role.

## Ranh giới an toàn

- Không đọc PDF, content, validation, blind hoặc test data.
- Không load model/checkpoint và không fine-tune.
- Không đổi threshold hoặc acceptance policy.
- Không sửa synthesis, verifier, executor hoặc `program_data.py`.
- `program_column_roles`, `abstention_reason`, compiler DAG và `program_identity_ids` vẫn non-consuming.
- Phase D vẫn khóa tại accepted precision baseline `88,4739% < 90%`.

## Bước tiếp theo

QP3A phải được precommit trước khi sửa runtime: tích hợp column-role proof theo evidence cho đúng 12 deferred hard
negatives, gồm positive unique-role, hard-negative multiple-role và counterfactual role-swap. Không thay threshold.

Machine-readable source of truth:
`reports/finhrm_qp2e_surface_robustness_development_v1.json`.
