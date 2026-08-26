# FinHRM QP3A — evidence-conditioned column-role proof precommit

Ngày hoàn tất: 13/08/2026  
Phân loại: development-only synthetic evidence precommit  
Trạng thái: khóa trước mọi thay đổi QP3 runtime

## Kết luận

QP3A precommit gate đạt. Đã khóa proof semantics, cohort, quota, gold state/action và runtime baseline cho bước tích
hợp column-role proof dựa trên evidence. Chưa sửa synthesis, verifier, executor hoặc acceptance; chưa chạy inference.

Cohort gồm 128 case:

- 96 case từ đúng 12 `hard_negative_missing_role` deferred sau QP2E;
- 32 counterfactual đổi metric/năm;
- 16 group, mỗi group có đúng tám scenario.

## Quy tắc đã precommit

Với câu hỏi không nêu column role:

1. gom evidence theo logical-role key gồm document, table, row, metric, period, entity và unit;
2. không dùng column role, value, candidate score hoặc candidate order trong group key;
3. deduplicate rendition theo physical-equivalence key;
4. nếu một logical group có từ hai explicit role khác nhau, proof là `INVALID` và hành động là `ABSTAIN`;
5. cùng numeric/value token vẫn là conflict nếu logical cells có role khác nhau;
6. nếu role inventory chưa chứng minh đầy đủ, proof là `UNKNOWN` và phải `ABSTAIN`;
7. chỉ khi inventory `EXPLICIT_COMPLETE` và mỗi requested logical group có đúng một compatible role thì obligation
   column-role uniqueness mới `VALID`;
8. `VALID` ở đây chỉ cho obligation role uniqueness; kết quả cuối vẫn cần mọi proof obligation khác đều VALID.

Với câu hỏi nêu role rõ:

- selected evidence cùng explicit role, không conflict → `VALID`;
- explicit role mismatch → `INVALID / REJECT`;
- metadata thiếu → `UNKNOWN / ABSTAIN`.

## Scenario matrix

Mỗi trong 16 group có đủ:

- explicit-role positive với inventory đầy đủ;
- explicit-role swap negative;
- no-role unique với inventory đầy đủ;
- no-role ACTUAL/PLAN conflict;
- no-role same-value ACTUAL/PLAN conflict;
- candidate-order permutation của conflict;
- no-role unique nhưng inventory UNKNOWN;
- cross-extractor conflict trên cùng physical evidence.

## Kết quả validation

| Kiểm tra | Kết quả |
|---|---:|
| Tổng examples | 128/128 |
| Oracle proof state | 128/128 |
| Oracle action | 128/128 |
| Deferred family groups | 12/12 |
| Generalization groups | 4/4 |
| Mỗi scenario | 16/16 |
| Same-value conflicts | 16/16 |
| Permutation invariance | 16/16 |
| Unknown-inventory controls | 16/16 |
| Distinct metrics | 6 |
| Distinct years | 7 |
| Malformed/leakage examples | 0 |
| Regression tests | 222/222 |

Sáu metrics gồm doanh thu, doanh thu hoạt động, lợi nhuận, chi phí, tài sản và hàng tồn kho. Bảy năm gồm 2022,
2023, 2024, 2027, 2028, 2029 và 2030. Vì vậy cohort không khóa vào một câu hỏi hoặc hai năm cụ thể.

## Provenance và leakage

- Không có numeric value hoặc numeric answer.
- Không có PDF/document content.
- Không có candidate score và candidate order không được dùng làm signal.
- Value chỉ là symbolic token để kiểm tra same-value invariance.
- Tất cả evidence provenance là synthetic và được gắn explicit source/table/row/column/span identifiers.
- Catalog, thesis-core contract và QP2E lock được kiểm tra SHA-256.

## Runtime boundary

Hash của `question_planner.py`, `program_synthesis.py`, `symbolic_verifier.py`, `document_hrm.py`,
`document_reasoning_engine.py` và `program_data.py` đều khớp QP2E baseline. Vì vậy:

- chưa tích hợp proof vào runtime;
- chưa đổi acceptance policy;
- chưa đổi threshold/checkpoint;
- không fine-tune;
- Phase D vẫn khóa tại accepted precision baseline `88,4739% < 90%`.

## Bước tiếp theo được phép

QP3B: implement đúng contract đã khóa trên frozen 128-case cohort. QP3B phải đạt 128/128 state/action,
permutation/same-value/unknown controls, đồng thời giữ regression QP2A–QP2E. Không thay threshold.

Machine-readable artifacts:

- `config/finhrm_qp3a_column_role_proof_precommit_v1.json`;
- `data/finhrm_qp3a_column_role_proof_development_v1.jsonl`;
- `reports/finhrm_qp3a_column_role_proof_precommit_v1.json`.
