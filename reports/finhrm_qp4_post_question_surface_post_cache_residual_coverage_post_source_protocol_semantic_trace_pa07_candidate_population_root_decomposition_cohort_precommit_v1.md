# QP4 PA07 root-decomposition cohort precommit

## Kết quả

`PASS` cho **precommit contract**, chưa phải PASS cohort và không phải PASS runtime.

Bước này chỉ khóa một quy trình có thể replay để tìm evidence cho ba structural subroot PA07 trên development-only sources. Runtime, model, checkpoint, threshold, QP5 và Phase D vẫn khóa. Không có kết luận rằng ba subroot đã đủ coverage.

## Baseline và khả năng thực thi

- Parent lock khớp đúng bước được phép tiếp theo.
- Development catalog có 169 PDF/150 company group. Loại đúng hai ABW còn 167 PDF; `167/167` có locator duy nhất và SHA-256 khớp.
- Loại tiếp toàn bộ sáu company đã dùng trong diagnosis còn 160 PDF thuộc 143 company để xây cohort.
- Probe read-only chọn ba nguồn đầu theo `source_sha256`, không dùng filename, company, câu hỏi, giá trị hoặc kết quả model: `3/3` khớp hash; native page text, page method và word-geometry reader khớp số trang. Probe không tạo OCR/cache và không lưu raw text.
- Catalog chưa chứa sẵn layout/header oracle. Vì vậy bước build phải materialize và review geometry từ PDF; precommit không giả định cohort đã tồn tại.

## Ba structural signature đã khóa

1. `PA07_ROOT_HIERARCHICAL_MULTIROW_ROLE_PERIOD_HEADER_V1`: role và period nằm trên các tầng header khác nhau; quan hệ parent/child phải được chứng minh bằng span và bounding box.
2. `PA07_ROOT_FLATTENED_COMPACT_ROLE_YEAR_HEADER_V1`: native table bị flatten nhưng từng cột vẫn có role và period tường minh; cell phải thẳng hàng theo geometry.
3. `PA07_ROOT_FRAGMENTED_VISUAL_ROW_LABEL_V1`: một hàng thị giác bị tách thành nhiều native fragments; chỉ được ghép các fragment cùng row band/table grid.

Cả ba signature đều yêu cầu metric, ACTUAL, PLAN, cùng period, currency unit, entity scope và provenance tường minh. Không được suy từ thứ tự cột, nearby prose, câu hỏi, filename, value, rank, score hoặc confidence.

## Generalization gate

Mỗi signature được xét độc lập và chỉ đủ điều kiện khi có tối thiểu:

- 3 company khác nhau;
- 3 source PDF khác nhau;
- 3 **real table-layout family** khác nhau;
- 3 layout instance khác nhau;
- tối đa 1 anchor/company và 1 anchor/source;
- không overlap company/source/layout instance với diagnosis;
- mọi anchor có explicit same-metric, same-period ACTUAL/PLAN pair và geometry khớp visual review.

Layout family được tạo từ topology cấu trúc (header levels, role/period axis, column grouping, row shape, page span, orientation, native representation). Company, filename, source hash, page, year, metric literal, numeric value, example ID và question literal không được dùng để tạo layout family. Synthetic question variants không được tính là layout.

Không được pool evidence giữa ba signature để đủ gate. Nếu thiếu bất kỳ minimum nào sau khi quét đủ 160 PDF, cohort build phải `FAIL_CLOSED`; không được hạ gate hay chia layout giả để làm metric đẹp.

## Quy trình build tiếp theo

Bước duy nhất được mở là:

`QP4-POST-QUESTION-SURFACE-POST-CACHE-RESIDUAL-COVERAGE-POST-SOURCE-PROTOCOL-SEMANTIC-TRACE-PA07-CANDIDATE-POPULATION-ROOT-DECOMPOSITION-COHORT-BUILD`

Build phải quét đủ development universe trước khi chọn, xác minh source hash, dùng native text + word geometry + visual structural review, rồi mới chọn diversity-first. Question planner/HRM/ranker/evaluator không được tham gia eligibility. Sau khi source đã chọn mới được instantiate 11 PA07 surface từ catalog đã khóa.

## Trạng thái khóa

- Runtime edit: **không được phép**.
- OCR mới: **không được phép**.
- Fine-tune/model/checkpoint/threshold: **không thay đổi**.
- Protected splits/ABW: **không đọc/dùng**.
- QP4 hoàn tất: **chưa**.
- QP5/Phase D: **vẫn khóa**.

Hệ thống không tuyên bố hoạt động trên mọi PDF bất kể cấu trúc. Một phương pháp report-agnostic chỉ được thừa nhận trong structural support đã chứng minh; ngoài support hoặc thiếu proof phải `UNKNOWN → ABSTAIN`, conflict phải `REJECT`.
