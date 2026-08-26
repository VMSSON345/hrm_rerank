# QP4 positive-target Candidate/Binding instrumentation cohort precommit

Precommit: **PASS**. Chưa chạy replay và chưa sửa runtime.

- Cohort khóa cứng: `5` positive targets; không được thêm, thay hoặc fallback case.
- Candidate: `3` case canonical / `3` company / `3` document layouts / `3` table layouts / `2` extraction families.
- Binding/Provenance: `2` case canonical / `2` company / `2` document layouts / `2` table layouts.
- Mỗi cluster chọn đúng một case trên mỗi document-layout family hiện hữu trong residual pool của cluster; đây là cohort nhỏ nhất không bỏ layout đang có.
- Replay kế tiếp phải dùng bản sao tạm `tmp/finhrm_qp4_positive_target_instrumentation_page_index_v1.sqlite3`. Page index gốc phải giữ hash `df8be071…84fd7` trước/sau; file tạm và sidecar phải xóa.
- Chỉ ghi candidate sequence/ranking, DAG/slot bindings và provenance/verification reasons. Output semantic envelope và decision phải bất biến `5/5`.

Cùng failing layer, candidate-support false, binding-exact false, empty slot binding hoặc cùng verification reason đều chưa đủ. Repair chỉ được mở nếu cùng runtime module/symbol, cùng branch/predicate và cùng causal signature đạt diversity đã khóa; nếu không kết luận `HOLD`.
