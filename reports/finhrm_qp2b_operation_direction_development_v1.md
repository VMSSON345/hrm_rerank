# FinHRM QP2B — operation và direction intent

Ngày hoàn tất: 13/08/2026  
Phân loại: development-only synthetic compiler repair  
Phạm vi: question planning; operand slots vẫn chưa được synthesis/verifier tiêu thụ

## Kết luận

QP2B đạt gate. Compiler hiện xác định đúng root operation và direction cho toàn bộ 108 frozen answer
variants thuộc thesis core, gồm sáu family mục tiêu PA04, PA06, PA07, PA08, PA09 và TY04.

QP2B thay đổi runtime planning behavior vì `operation` và `direction_intent` là các field runtime đã
tồn tại. Tuy nhiên không sửa source của program synthesis, symbolic verifier hoặc acceptance policy;
không thay threshold/checkpoint và không bật các typed operand slots mới làm runtime input.

## Kết quả

| Kiểm tra | Kết quả |
|---|---:|
| Tất cả frozen answer variants operation+direction exact | 108/108 |
| Sáu target family | 54/54 |
| Role-swap counterfactual | 12/12 |
| Year-shift counterfactual | 12/12 |
| Long/no-diacritic/clause-reordered | 36/36 |
| QP2A canonical slot retention | 13/13 |
| QP2A frozen slot retention | 108/108 |
| Legacy collateral regression | 0 |
| Regression tests | 131/131 |

Strict thesis-core exact tăng từ `5/16 = 31,25%` lên `7/16 = 43,75%`. Operation tăng `9/13 →
13/13`; direction tăng `8/13 → 13/13`. Legacy exact tăng `54/135 → 90/135`; có 36 improvement
trên PA04, PA06, PA07 và TY04, không có regression.

PA08/PA09 đã đúng root operation và direction nhưng chưa exact toàn contract vì QP2C chưa tạo
reference base và DAG đa bước. Vì vậy việc operation/direction đạt 100% không có nghĩa hai family này
đã executable hoặc được phép acceptance.

## Quy tắc tổng quát đã triển khai

### SUBTRACT

- Cú pháp `A trừ B` tạo `SUBTRACT + LEFT_MINUS_RIGHT` khi A/B có role rõ.
- `A vượt B bao nhiêu đồng` tạo `SUBTRACT + LEFT_MINUS_RIGHT`.
- Không hard-code ACTUAL−PLAN: role-swap `PLAN vượt ACTUAL` vẫn giữ đúng lexical order.

### PERCENT_CHANGE

- `A thay đổi bao nhiêu phần trăm so với B` tạo `PERCENT_CHANGE + LEFT_MINUS_RIGHT` khi A/B là hai
  semantic role khác nhau.
- Với hai role khác nhau qua hai năm, language/role order được ưu tiên hơn calendar recency. Vì vậy
  `PLAN 2024 cao hơn ACTUAL 2023` là LEFT−RIGHT, không bị đổi thành NEW−OLD.
- Câu temporal cùng role như `ACTUAL 2023 so với ACTUAL 2022` vẫn giữ `NEW_MINUS_OLD`.

### Multi-step ratio intent

- `A còn thiếu ... để đạt B` tạo root `RATIO_PERCENT` với direction theo target−observed:
  `PLAN_MINUS_ACTUAL` hoặc counterfactual `ACTUAL_MINUS_PLAN`.
- `A vượt B bao nhiêu phần trăm` tạo root `RATIO_PERCENT` với direction observed−base:
  `ACTUAL_MINUS_PLAN` hoặc counterfactual `PLAN_MINUS_ACTUAL`.
- QP2B chỉ định danh phép toán gốc/direction; không giả lập DAG. QP2C mới được phép tạo
  `SUBTRACT → RATIO_PERCENT` và reference base.

## Gap còn lại

- QP2A typed operand schema: đạt, không còn family bị ảnh hưởng.
- QP2B operation/direction: đạt, không còn family bị ảnh hưởng.
- QP2C reference base/DAG: còn PA03, PA07, PA08, PA09, TY02, TY04 và IBDS01.
- QP2D safe abstention: còn C14 và C15.
- Strict thesis gate 16/16 chưa đạt; runtime acceptance chưa được phép thay đổi.

## Invariant

- Không đọc PDF hoặc nội dung tài liệu.
- Không load model/checkpoint để inference.
- Không đọc global/sealed validation, blind hoặc test data.
- Không fine-tune; không đổi checkpoint/threshold/acceptance policy.
- `program_synthesis.py` và `symbolic_verifier.py` giữ nguyên hash.
- Phase D vẫn khóa tại accepted precision baseline `88,4739% < 90%`.

Machine-readable source of truth:
`reports/finhrm_qp2b_operation_direction_development_v1.json`.
