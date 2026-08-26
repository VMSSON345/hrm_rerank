# QP3-CORE-GENERALIZATION-COVERAGE-IMPLEMENTATION-PRECOMMIT

Status: **PRECOMMITTED — RUNTIME UNCHANGED**.

Đã đọc `CLAUDE.md` trước khi khóa scope. Thiết kế chọn thay đổi tối thiểu trong compiler, typed synthesis, symbolic verifier và final grounding; không thêm model hoặc subsystem.

## Admission

| Track | Baseline UNKNOWN | Admission |
|---|---:|---|
| period + provenance | 74 | khóa; không có cell-linked year/period-role evidence |
| operation + ADD grounding | 67 | mở exact generic scope |
| query metric | 33 | mở exact generic scope |

Cả 74 period failure đều thiếu `PERIOD`, explicit year và relative period role trên selected cell; header path cũng không chứa requested year/role. Chín context có nhắc requested year nhưng không có lineage nối nó với cell. Vì vậy chép năm từ question/context sẽ tạo coverage giả và bị cấm.

ADD được mở cho structural two-span hypotheses, giữ nhiều parse như ambiguity và chỉ materialize khi hai typed slots độc lập chứng minh hai logical cells khác nhau. Query metric được mở cho row + section + fully-provenanced header composition; giữ nguyên alias cutoffs và yêu cầu unique compatible logical cell.

## Frozen replay gates

- giữ `187/187` answer cũ, execution `363/363`, permutation `363/363`;
- counterfactual false accept `0`, hard-negative/counterfactual ANSWER `0`;
- ADD UNKNOWN phải giảm từ `67` trên ít nhất 3 company/3 layout;
- query-metric UNKNOWN phải giảm từ `33` trên ít nhất 3 company/3 layout;
- period không được chuyển VALID nếu thiếu explicit source lineage;
- không tính abstention hay synthetic layout là confirmation.

Không dùng ABW, protected evaluation, fine-tuning hoặc thay threshold/checkpoint. QP4 và Phase D vẫn khóa.

Bước tiếp theo chỉ là `QP3-CORE-GENERALIZATION-COVERAGE-IMPLEMENTATION` trong exact symbol whitelist đã khóa.
