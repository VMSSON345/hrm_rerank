# FinHRM QP3D — table-level role-inventory producer

Ngày: 13/08/2026  
Phân loại: development-only implementation and replay  
Trạng thái: hoàn tất QP3D; Phase D vẫn khóa

## Kết quả triển khai

QP3D đã triển khai contract QP3C vào ba tầng:

1. `program_data.py` tạo full-table structural manifest trong lúc còn giữ toàn bộ table block, trước khi candidate Top-K được chọn;
2. `evidence_schema.py` kiểm tra boundary/header/cell/inheritance/physical/extractor provenance, tạo state theo từng `(source, table)` và truyền vào `EvidenceGraph`;
3. `symbolic_verifier.py` chỉ dùng inventory của đúng table chứa selected logical evidence, không để state của table khác lan sang chương trình hiện tại.

Producer không dùng question text, numeric value/answer, candidate score/rank/order. Plaintext không có manifest và table chỉ có ordinary period columns như `2022/2023` giữ trạng thái không áp dụng; chúng không bị ép vào proof plan/actual.

## Frozen QP3C replay

| Hạng mục | Kết quả |
|---|---:|
| Inventory state exact | 96/96 |
| Reason-set exact | 96/96 |
| Development end-to-end controls | 8/8 |
| Pre-Top-K manifest control | đạt, manifest 80 cell trước khi prune còn 64 candidates |
| Runtime regression liên quan | 483/483 |

Tám end-to-end controls gồm complete table, no-role actual/plan ambiguity, partial boundary, role-bearing table có bare-year column, missing cell coverage, candidate permutation, plaintext legacy và ordinary-period non-applicability.

Hai runtime source không liên quan (`question_planner.py`, `program_synthesis.py`) cùng executor, reasoning engine và checkpoint giữ nguyên. QP3D chỉ thay `program_data.py`, `evidence_schema.py` và `symbolic_verifier.py`, đúng phạm vi producer + graph + table-scoped verifier.

## Real development-only replay: ABW

Query: `Doanh thu hoạt động thực hiện năm 2023 là bao nhiêu?`

- replay A tạo `SELECT(1,)`, chọn đúng `304.912.639.828 VND`, role `actual`, năm 2023, trang 10, nhưng `ABSTAIN` vì `0,517 < 0,846`;
- replay B tạo sai `ADD(3, 10)`, nhưng grounding gate bác vì `missing_explicit_addition_intent` và confidence `0,129 < 0,522`;
- cả hai replay đều `ABSTAIN`, false accept `0/2`; MinerU/OCR không được dùng.

Layout ABW là flattened native plaintext nên chưa có full-table manifest. QP3D không suy diễn completeness từ 64 candidates và giữ inventory undeclared. Hai neural proposal khác nhau cho thấy replay này chỉ chứng minh safe abstention, chưa chứng minh answer coverage ổn định. Đây là giới hạn end-to-end cần giữ tách biệt với gate producer 96/96.

## Invariants và giới hạn kết luận

- Không fine-tune, không đổi threshold/checkpoint.
- Không đọc global/sealed validation hoặc blind/test.
- QP3C precommit/cohort/lock giữ nguyên hash.
- Các test QP3C lịch sử yêu cầu `runtime_source_changed=false` đương nhiên không còn là active gate sau khi QP3D được triển khai; artifact lịch sử không bị sửa.
- Frozen/synthetic exact không phải validation accuracy và chưa chứng minh producer bao phủ flattened native/OCR layouts.
- Accepted precision chính thức vẫn là baseline `88,4739% < 90%`; Phase D vẫn khóa.

## Bước tiếp theo

QP3E precommit một development-only flattened-table provenance contract. Mục tiêu là tạo full-table manifest an toàn cho native plaintext/OCR layouts như ABW bằng explicit header-role cardinality, row/cell alignment và source-line boundaries; không được dùng câu hỏi, numeric answer hoặc Top-K. Sau khi khóa cohort/counterfactual mới được mở rộng producer và replay; threshold tiếp tục giữ nguyên.
