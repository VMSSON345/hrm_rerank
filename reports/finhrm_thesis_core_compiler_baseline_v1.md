# FinHRM thesis-core compiler baseline v1

Ngày chạy: 13/08/2026  
Phân loại: development-only synthetic compiler audit  
Chế độ: compiler-only, không tài liệu, không PDF, không model/checkpoint inference, không execution

## Kết luận

Compiler hiện chưa đạt thesis-core contract. Strict canonical exact đạt `1/16 = 6,25%`; family duy
nhất đạt là ambiguity control C13. Kết quả này không phủ nhận tiến bộ QP2 priority lane: year intent
đạt `13/13`, column role rời rạc đạt `12/13`, answer mode và requested answer unit đều đạt `16/16`.
Khoảng cách chính là `QuestionPlan` chưa phát ra typed operand slots và program DAG/reference base như
contract đã khóa.

Legacy catalog compatibility trên 15 family có biến thể đạt `53/135 = 39,26%`. IBDS01 không được
trộn vào metric này vì precommit chỉ có một canonical question và không có biến thể được tạo từ trước.
Không sinh thêm câu IBDS01 sau khi nhìn kết quả.

Release gate strict core yêu cầu `16/16 = 100%`; baseline `1/16` nên fail. Báo cáo này không cấp quyền
thay runtime acceptance, verifier hoặc threshold.

## Strict canonical contract

| Dimension | Pass | Total | Tỷ lệ |
|---|---:|---:|---:|
| Requested years | 13 | 13 | 100% |
| Requested answer unit | 16 | 16 | 100% |
| Answer mode | 16 | 16 | 100% |
| Column roles rời rạc | 12 | 13 | 92,31% |
| Root operation | 9 | 13 | 69,23% |
| Direction intent | 8 | 13 | 61,54% |
| Program DAG | 8 | 13 | 61,54% |
| Reference base | 6 | 13 | 46,15% |
| Typed operand schema | 0 | 13 | 0% |
| Financial identity IBDS01 | 0 | 1 | 0% |

`Program DAG = 8/13` cho phép single-node DAG được chứng minh bằng root operation chính xác; PA08,
PA09 và IBDS01 vẫn bắt buộc DAG đa bước được biểu diễn rõ. Audit không dựng giả operand slots hoặc
multi-step DAG từ các tín hiệu rời rạc để cộng điểm.

### Gap theo family

| Family | Tín hiệu đã đúng | Gap ngoài typed operand schema |
|---|---|---|
| PA01 | SELECT, year, ACTUAL | Không có typed operand |
| PA02 | SELECT, year, PLAN | Không có typed operand |
| PA03 | RATIO_PERCENT, ACTUAL/PLAN | Thiếu `reference_base=plan` |
| PA04 | role và direction | Chưa nhận `trừ` thành SUBTRACT |
| PA05 | SUBTRACT, PLAN−ACTUAL, direction | Không có typed operand |
| PA06 | ACTUAL/PLAN | Chưa nhận `vượt` thành SUBTRACT/direction |
| PA07 | PERCENT_CHANGE, ACTUAL/PLAN | Bị gắn AMBIGUOUS sai; thiếu reference base |
| PA08 | PLAN/ACTUAL | Thiếu operation, direction, reference base và DAG SUBTRACT→RATIO_PERCENT |
| PA09 | ACTUAL/PLAN | Thiếu operation, direction, reference base và DAG SUBTRACT→RATIO_PERCENT |
| TY01 | SUBTRACT, NEW−OLD, hai năm | Không có typed operand |
| TY02 | PERCENT_CHANGE, NEW−OLD | Thiếu `reference_base=actual_2022` |
| TY04 | PERCENT_CHANGE, PLAN/ACTUAL | Direction bị đồng nhất sai thành NEW−OLD; thiếu reference base |
| IBDS01 | RATIO_PERCENT | Chưa nhận identity, ba operand role, reference base và ADD→RATIO_PERCENT DAG |
| C13 | Abstain đúng | Đạt strict canonical |
| C14 | Percent cue | Chưa abstain khi thiếu đối tượng/reference base |
| C15 | Nhìn thấy PLAN/ACTUAL | Chưa abstain khi câu hỏi không chọn column role |

## Legacy catalog robustness

Legacy scorer được giữ nguyên để so sánh với QP0/QP2, không được dùng thay strict metric.

| Family | Exact/Total |
|---|---:|
| C13 | 1/1 |
| C14, C15 | 0/1 mỗi family |
| PA01, PA02, TY01, TY02 | 9/11 mỗi family |
| PA03, PA05 | 8/11 mỗi family |
| PA04, PA06, PA07, PA08, PA09, TY04 | 0/11 mỗi family |

Theo variant, canonical đạt `7/15`; mỗi nhóm paraphrase/no-diacritics/long/clause-reordered/year-shift
đạt `6/12`; role-swap đạt `4/12`. Toàn bộ `12/12` hard-negative missing-role và `12/12` ambiguity
control chưa đạt hành vi mong đợi. Có `82/135` example chưa exact.

## Gap taxonomy và thứ tự sửa

1. **QP2A — typed operand schema:** 13/13 answer family bị ảnh hưởng. Thêm cấu trúc immutable cho
   từng operand: `id`, `metric`, `year`, `column_role`, `entity_scope`, `unit_family` và provenance của
   intent. Chỉ dùng bằng chứng từ câu hỏi/ontology/identity catalog; không dùng giá trị số, công ty,
   trang hoặc vị trí cột để suy role.
2. **QP2B — operation và direction:** sửa compositional cues chung cho `trừ`, `vượt`, `còn thiếu`,
   `thay đổi so với`; phân biệt semantic role direction với thứ tự năm. Sáu canonical family bị ảnh
   hưởng.
3. **QP2C — reference base và program DAG:** biểu diễn mẫu một bước và ba DAG đa bước PA08, PA09,
   IBDS01; reference base phải là operand ID, không phải candidate index. Chín family bị ảnh hưởng.
4. **QP2D — safe abstention:** compiler phải phát reason rõ cho missing reference base/missing requested
   column role, đồng thời không abstain nhầm PA07. Ba canonical example bị ảnh hưởng.
5. **QP2E — surface robustness:** replay nguyên 135 frozen catalog rows; không sửa riêng theo chuỗi câu
   canonical. Gate yêu cầu explicit-intent/counterfactual exact 100% trước QP3.
6. **QP3:** chỉ nối typed intent/proof vào acceptance sau counterfactual development-only pass. Không hạ
   threshold để bù compiler gap.

## Invariant

- Không đọc PDF hoặc nội dung tài liệu.
- Không load model/checkpoint để inference.
- Không đọc global/sealed validation, blind hoặc test data.
- Không fine-tune; không đổi checkpoint, threshold hay runtime.
- Phase D vẫn khóa tại baseline accepted precision `88,4739% < 90%`.

Machine-readable source of truth:
`reports/finhrm_thesis_core_compiler_baseline_v1.json`.
