# FinHRM QP2C — reference base và compiler-side program DAG

Ngày hoàn tất: 13/08/2026  
Phân loại: development-only synthetic compiler repair  
Phạm vi: question planning; DAG và `program_identity_ids` chưa được synthesis, verifier, executor hoặc acceptance tiêu thụ

## Kết luận

QP2C đạt gate. Question compiler hiện tạo được `reference_base` và program DAG bất biến, có kiểu, cho toàn bộ
13 answer family của thesis core. Tất cả node chỉ tham chiếu typed operand slot ID hoặc output node đứng trước;
không mang numeric value, candidate index, page, table hay vị trí trái/phải từ tài liệu.

Thay đổi nằm trong runtime planning source, nhưng các field mới vẫn **non-consuming**: chưa được nối vào program
synthesis, symbolic verifier, executor hoặc acceptance policy. QP2C vì vậy chỉ chứng minh cấu trúc intent, chưa tuyên
bố câu hỏi đã executable và không làm thay đổi quyết định ACCEPT/ABSTAIN của hệ thống.

## Kết quả

| Kiểm tra | Kết quả |
|---|---:|
| Canonical answer contract exact | 13/13 |
| Frozen answer variants exact | 108/108 |
| ABSTAIN/ambiguity controls không sinh DAG | 27/27 |
| Role-swap counterfactual | 12/12 |
| Year-shift counterfactual | 12/12 |
| PA08/PA09 multi-step variants | 18/18 |
| IBDS01 canonical debt identity | đạt |
| QP2A operand-slot retention | 108/108 |
| QP2B operation/direction retention | 108/108 |
| DAG validation errors | 0 |
| Provenance violations | 0 |
| Legacy collateral regression | 0 |
| Regression tests | 193/193 |

Strict thesis-core exact tăng từ `7/16 = 43,75%` lên `14/16 = 87,50%`. Hai dimension
`reference_base` và `program_dag` đều đạt `13/13`. Legacy exact giữ nguyên `90/135`, không improvement ngoài
phạm vi và không regression. Hai case chưa exact là C14/C15, được dành riêng cho QP2D safe abstention.

## Cấu trúc đã triển khai

- `ProgramIntentNode` là dataclass bất biến gồm node ID, operation, input references, output unit family và
  `intent_sources`.
- `QuestionPlan.reference_base` tham chiếu trực tiếp một typed operand slot ID.
- `QuestionPlan.program_dag` là tuple topological; node cuối phải có ID `answer`.
- `QuestionPlan.program_identity_ids` tách khỏi `financial_identity_ids` cũ, tránh vô tình kích hoạt consumer
  runtime hiện hữu cho IBDS01.
- Validator fail-closed khi gặp arity sai, input treo, node/operand trùng, unit không tương thích, root operation
  không khớp hoặc denominator không đúng reference base.

## Program intent chính

- PA03: `RATIO_PERCENT(actual, plan)`, base là `plan`.
- PA07: `PERCENT_CHANGE(actual, plan)`, base là `plan`.
- PA08: `shortfall = SUBTRACT(plan, actual)` rồi `RATIO_PERCENT(shortfall, plan)`.
- PA09: `excess = SUBTRACT(actual, plan)` rồi `RATIO_PERCENT(excess, plan)`.
- TY02: `PERCENT_CHANGE(actual_2023, actual_2022)`, base là `actual_2022`.
- TY04: `PERCENT_CHANGE(plan_2024, actual_2023)`, base là `actual_2023`.
- IBDS01: `debt = ADD(short_term_debt, long_term_debt)` rồi
  `RATIO_PERCENT(debt, total_capital)`, identity compiler-side là `balance.financial_debt`.

Các family SELECT/SUBTRACT còn lại cũng được compile thành DAG một node để toàn bộ answer contract dùng chung
một schema, thay vì chỉ tạo DAG cho nhóm phần trăm.

## Kiểm tra phản ví dụ và giữ lại hành vi cũ

- Role-swap làm đổi operand/reference base theo semantic role, không hard-code ACTUAL/PLAN.
- Year-shift giữ denominator là kỳ gốc, không phụ thuộc năm cụ thể 2022/2023/2024.
- Câu thiếu role, role xung đột hoặc bị phân loại ABSTAIN không được phát DAG.
- PA08/PA09 bắt buộc hai node theo đúng thứ tự `SUBTRACT → RATIO_PERCENT`.
- IBDS01 dùng ba typed slot riêng và không ghi vào `financial_identity_ids` cũ.

## Ranh giới an toàn

- Không đọc PDF hoặc nội dung tài liệu.
- Không load model/checkpoint để inference.
- Không đọc global/sealed validation, blind hoặc test data.
- Không fine-tune; không đổi checkpoint, threshold hoặc acceptance policy.
- Không sửa `program_synthesis.py`, `symbolic_verifier.py`, executor hay `program_data.py`.
- Phase D vẫn khóa tại accepted precision baseline `88,4739% < 90%`.

## Bước tiếp theo được phép

QP2D: sửa safe abstention cho C14/C15 bằng conflict/underspecification contract tổng quát. Vẫn giữ compiler DAG
và `program_identity_ids` non-consuming; chưa nối vào runtime execution/acceptance, chưa thay threshold.

Machine-readable source of truth:
`reports/finhrm_qp2c_reference_dag_development_v1.json`.
