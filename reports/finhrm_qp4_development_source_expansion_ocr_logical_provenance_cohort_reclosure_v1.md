# QP4 OCR logical-provenance cohort reclosure

Trạng thái: **FAIL_CLOSED**. Bước này đã chạy đúng ba anchor được khóa, không tìm thêm nguồn, không OCR thêm và không sửa runtime/model/threshold/checkpoint/renderer.

## Kết quả thực chất

- Hash nguồn: `3/3`; hash content-list và table HTML của hai nguồn OCR: `2/2`.
- Binding/citation đầy đủ theo hợp đồng đã khóa: `2/3`.
- Phép tính `PERCENT_CHANGE` replay đúng từ các operand: `3/3`.
- Negative controls: `9/9` (`MISSING_PERIOD → ABSTAIN`, `DUPLICATE_ROLE → REJECT`, `MISSING_UNIT → ABSTAIN`); false accept `0`.
- Coverage đủ điều kiện: `2/2/2/2` company/source/layout/instance, chưa đạt mức khóa `3/3/3/3`.
- Frozen safe replay: execution `363/363`, candidate permutation `363/363`, `141` counterfactual với `0` false accept.
- Affected regression: `360/360` pass.

Hai anchor OCR THW trang 9 và TVA trang 21 có đầy đủ provenance logic: source/content/table hash, table ID, row/column, source-cell ordinal, role, period, unit, metric, entity và từng operand. Cả hai được `ADMIT`.

Anchor native ABB trang 34 có nội dung và phép tính đúng: thực hiện `779`, kế hoạch `1.000`, kết quả `-22,1%`. Nó cũng có source hash, trang, table bbox và bbox riêng cho từng operand. Tuy nhiên hợp đồng precommit lại bắt buộc toàn cục các trường chỉ có ở MinerU OCR như `content_list_path`, `table_html_sha256`, `table_id`, row/column và `cell_ordinal_span`. Những trường đó không tồn tại trong carrier native PDF. Vì không được bịa provenance, ABB phải `ABSTAIN` theo hợp đồng hiện tại.

Đây là lỗi thiết kế schema citation giữa hai carrier, không phải lỗi nội dung báo cáo, lỗi phép tính hay bằng chứng HRM yếu. Việc precommit trước đó ghi feasibility `PASS` cho cả ba anchor là không chính xác; execution lần này đã phát hiện và báo cáo trung thực thay vì ép coverage.

Coverage geometry cũ vẫn giữ nguyên: hierarchical `1/1/1/1`, fragmented `0/0/0/0`. Không dùng synthetic layout hoặc gộp signature để làm đủ chỉ tiêu.

## Bước duy nhất được mở tiếp

`QP4-DEVELOPMENT-SOURCE-EXPANSION-NATIVE-AND-OCR-CITATION-COORDINATE-CONTRACT-RECLOSURE-PRECOMMIT`

Bước đó chỉ được khóa một semantic citation envelope chung với locator phân nhánh rõ theo carrier:

- Native PDF: source hash, physical page, table bbox và operand cell bbox.
- OCR structured HTML: content-list/table hash, table ID, logical row/column và source-cell ordinal.

Các trường semantic chung như role, period, unit, metric, entity, raw value và decimal value vẫn bắt buộc cho cả hai. Bước tiếp theo phải tái sử dụng đúng ba anchor hiện tại; không mở source search, OCR, runtime, model hoặc renderer.
