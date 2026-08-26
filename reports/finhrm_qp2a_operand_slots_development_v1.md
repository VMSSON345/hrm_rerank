# FinHRM QP2A — typed operand slots

Ngày hoàn tất: 13/08/2026  
Phân loại: development-only synthetic compiler repair  
Phạm vi: question compiler; chưa nối slot mới vào synthesis, verifier hoặc acceptance

## Kết luận

QP2A đạt gate đã precommit. `QuestionPlan` hiện phát ra immutable `OperandSlot` theo schema
`question-operand-slot-v1`, gồm:

- `id`;
- `metric`;
- `year`;
- `column_role`;
- `entity_scope`;
- `unit_family`;
- `intent_sources`.

Slot chỉ chứa intent lấy từ câu hỏi và ontology/financial-identity contract. Slot không chứa numeric
value, candidate index, page, table coordinate hoặc vị trí trái/phải của evidence. Các field mới chưa
được program synthesis hoặc verifier tiêu thụ nên QP2A chưa thay acceptance.

## Kết quả gate

| Kiểm tra | Kết quả |
|---|---:|
| Canonical answer family slot exact | 13/13 |
| Frozen answer variant slot exact | 108/108 |
| Missing-role không materialize slot | 12/12 |
| Duplicate slot ID | 0 |
| Numeric/document-conditioned provenance violation | 0 |
| Legacy collateral regression | 0 |
| Regression tests | 116/116 |

Strict thesis-core exact tăng từ `1/16 = 6,25%` lên `5/16 = 31,25%`. Typed operand dimensions tăng
từ `0/13` lên `13/13`. Bốn family mới exact toàn contract hiện hành là PA01, PA02, PA05 và TY01;
C13 tiếp tục exact. Các family còn lại fail vì operation/direction, reference base/DAG hoặc safe
abstention thuộc QP2B–QP2D, không phải vì operand slot.

Legacy exact tăng từ `53/135` lên `54/135`; improvement duy nhất là
`PA05-minimal_pair_role_swap`, không có regression. Legacy score không tăng nhiều vì scorer cũ không
chấm typed slots; nó được giữ để đo collateral trên field runtime cũ.

## Quy tắc biên dịch

### Role và year

- Plan/actual slot chỉ được tạo khi câu hỏi nêu role rõ.
- Một năm với nhiều role gắn cùng explicit year cho từng slot.
- Hai năm gắn role vào đúng năm được nêu; slot ID có dạng `actual_2023`, `actual_2022`.
- Minimal role/year swap phải đổi slot identity tương ứng.
- `A thấp hơn B` tạo operand order `B, A` từ lexical order, không hard-code PLAN−ACTUAL.
- Câu thiếu requested column role không được suy role từ year, vị trí cột hoặc dữ liệu số.

### Metric và unit

- Discourse prefix như “hãy cho biết”, “theo báo cáo”, clause dài và arithmetic suffix không đi vào
  metric identity.
- Requested percent/ratio là đơn vị của answer; leaf operand của phép tỷ lệ vẫn là currency.
- Direct lookup không ép đơn vị khi câu hỏi không nêu đơn vị, dùng `currency_or_source`.

### IBDS01

QP2A tạo đúng ba leaf intent:

1. `short_term_borrowings`;
2. `long_term_borrowings`;
3. `total_capital`.

Role `ACTUAL` của ba balance-sheet leaf mang provenance
`column_role:financial_statement_actual_default`, tách biệt với role lexical. QP2A chưa kích hoạt
financial identity và chưa tạo DAG `ADD → RATIO_PERCENT`; hai việc này thuộc QP2C.

## Thay đổi runtime và giới hạn kết luận

`question_planner.py` là runtime source nên source hash đã thay đổi. Tuy nhiên:

- `operand_slots` chưa được synthesis đọc;
- `operand_slots` chưa được verifier đọc;
- acceptance policy và threshold không đổi;
- không load PDF/model và không thực hiện inference trong audit;
- không fine-tune và không đọc global/sealed validation hoặc blind/test.

Do đó không được kết luận UI đã xử lý đầy đủ câu dài/multi-step sau QP2A. Phase D vẫn khóa ở baseline
accepted precision `88,4739% < 90%`.

## Bước tiếp theo

Thực hiện QP2B cho operation và direction trên sáu canonical family PA04, PA06, PA07, PA08, PA09,
TY04; đồng thời replay 108 answer variants và counterfactual role/year. Giữ `operand_slots` ở chế độ
non-consuming. QP2C mới thêm reference base/program DAG; QP3 mới xem xét nối proof vào acceptance.

Machine-readable source of truth:
`reports/finhrm_qp2a_operand_slots_development_v1.json`.
