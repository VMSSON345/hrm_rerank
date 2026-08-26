# FinHRM — chỉ mục tài liệu quan trọng

Các tài liệu định hướng được đặt cùng cấp với `strategy.md`:

1. `strategy.md` — lịch sử quyết định, invariant, phase gate và bước triển khai hiện tại.
2. `FINHRM_THESIS_PUBLICATION_ROADMAP.md` — phạm vi khóa luận, research gap, benchmark, baseline,
   ablation và lộ trình nộp bài.
3. `FINHRM_THESIS_CORE_SPEC.md` — đặc tả vertical slice, evidence graph, proof contract, benchmark,
   metric và gate đã precommit trước QP2/QP3.
4. `README.md` — cách cài đặt, chạy CLI và cấu hình hệ thống.
5. `DEMO.md` — hướng dẫn chạy giao diện local.
6. `mau.md` — tài liệu tham khảo VietFinDocAgent; không phải strategy của FinHRM.

Các báo cáo kỹ thuật bên dưới vẫn nằm trong `reports/` vì là artifact development/audit đã được
gắn hash hoặc lock; không di chuyển để tránh phá provenance:

- `reports/question_coverage_qp0_development_v1.md` — baseline question coverage QP0.
- `reports/question_pipeline_qp1_development_v1.md` — evidence/header-role pipeline QP1.
- `reports/column_role_runtime_priority_qp2_development_v1.md` — priority lane QP2/QP3.
- `reports/select_add_false_accepts_development_v1.md` — phân tích false accept SELECT/ADD.
- `reports/column_role_runtime_priority_qp2_lock_v1.json` — lock gần nhất của priority lane.
- `reports/finhrm_thesis_core_precommit_v1.json` — validation machine-readable của thesis core.
- `reports/finhrm_thesis_core_precommit_lock_v1.json` — lock manifest của thesis-core precommit.
- `reports/finhrm_thesis_core_compiler_baseline_v1.md` — baseline strict/legacy và gap QP2 theo family.
- `reports/finhrm_thesis_core_compiler_baseline_v1.json` — chi tiết machine-readable của 151 compiler checks.
- `reports/finhrm_thesis_core_compiler_baseline_lock_v1.json` — lock manifest của compiler baseline.
- `reports/finhrm_qp2a_operand_slots_development_v1.md` — kết quả typed operand schema QP2A.
- `reports/finhrm_qp2a_operand_slots_development_v1.json` — audit 13 canonical, 108 variants và 12 controls.
- `reports/finhrm_qp2a_operand_slots_lock_v1.json` — lock manifest của QP2A.
- `reports/finhrm_qp2b_operation_direction_development_v1.md` — kết quả operation/direction QP2B.
- `reports/finhrm_qp2b_operation_direction_development_v1.json` — audit 108 answer variants và counterfactual.
- `reports/finhrm_qp2b_operation_direction_lock_v1.json` — lock manifest của QP2B.

Quy tắc sử dụng:

- Đọc `FINHRM_THESIS_PUBLICATION_ROADMAP.md` để quyết định phạm vi nghiên cứu.
- Đọc `FINHRM_THESIS_CORE_SPEC.md` để biết chính xác câu hỏi, IR/DAG, proof và gate trong thesis core.
- Đọc cuối `strategy.md` để biết bước kỹ thuật tiếp theo.
- Dùng artifact trong `reports/` để kiểm tra số liệu; không sửa chúng như tài liệu kế hoạch.
