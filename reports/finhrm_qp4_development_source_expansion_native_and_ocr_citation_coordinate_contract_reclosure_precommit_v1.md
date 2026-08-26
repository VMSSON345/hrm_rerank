# QP4 native/OCR citation-coordinate contract reclosure precommit

Trạng thái precommit: **PASS**. Trạng thái QP4: **chưa hoàn tất**.

Precommit này sửa đúng lỗi schema đã được execution trước chứng minh: citation native PDF không còn bị bắt phải có `content_list`, HTML table, logical row/column và source-cell ordinal chỉ tồn tại ở OCR. Nó không sửa runtime, không OCR hoặc tìm nguồn mới và không hạ proof requirement.

## Contract đã khóa

Mọi operand vẫn phải có chung: source path/hash, physical page, table bbox, raw/decimal value, header path, role, period, unit, metric và entity. Sau đó `carrier` bắt buộc chọn đúng một locator:

- `native_pdf_cell_bbox`: dùng bbox operand trong coordinate space native đã materialize. Không diễn giải lại đơn vị/toạ độ và không được tổng hợp bbox.
- `mineru_ocr_table_cell_ordinal`: dùng hash content-list/table HTML, table ID, logical row/column và half-open source-cell ordinal.

`table_ref_id`, `cell_ref_id` và `citation_id` đều được tính lại bằng canonical JSON UTF-8 và SHA-256. Trường bị thiếu trong một nhánh hợp lệ dẫn đến `UNKNOWN → ABSTAIN`; carrier hoặc coordinate-space mâu thuẫn dẫn đến `REJECT`. Rank, neural score và confidence không được dùng làm semantic proof.

## Feasibility đã chạy

- Parent/source hash: đúng; ba source `3/3`.
- OCR content-list và table HTML hash: `2/2`; OCR table ID tính lại đúng `2/2`.
- Carrier table references: `3/3`.
- Operand citation envelopes materialize và admit: `6/6`.
- Hai native cell references: `2/2`; bốn OCR logical cell references: `4/4`.
- Sáu missing/discriminator/coordinate-space negative controls: `6/6` fail closed đúng.
- Coverage dự kiến cho execution kế tiếp: `3/3/3/3` company/source/layout/instance. Đây chỉ là feasibility projection, chưa phải QP4 gate result.
- Frozen replay: execution `363/363`, permutation `363/363`, counterfactual false accept `0/141`, regression `360/360`.

Coverage geometry cũ không bị viết lại: hierarchical `1/1/1/1`, fragmented `0/0/0/0`. Contract mới là content-correctness/provenance contract có locator trung thực theo carrier, không phải tuyên bố pixel-geometry đã pass.

## Bước duy nhất được mở tiếp

`QP4-DEVELOPMENT-SOURCE-EXPANSION-NATIVE-AND-OCR-CITATION-COORDINATE-COHORT-RECLOSURE`

Bước đó chỉ replay đúng ba anchor và sáu operand đã khóa, chạy program/negative/frozen gates rồi báo kết quả. Không được tìm thêm source/page, OCR thêm, sửa runtime/model/renderer hoặc tự mở QP5.
