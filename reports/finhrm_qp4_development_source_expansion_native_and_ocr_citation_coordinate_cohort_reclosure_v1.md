# QP4 native/OCR citation-coordinate cohort reclosure

Trạng thái cohort: **PASS**. Nhánh source-expansion/citation hiện tại đã đóng; toàn bộ QP4 chưa được tuyên bố hoàn tất cho đến final gate reclosure.

## Kết quả

- Precommit/parent artifact hash: `6/6`; source hash `3/3`.
- OCR content-list/table HTML: `2/2`; bốn row/column/source-ordinal operand khớp `4/4`.
- Hai native operand bbox nằm đúng trong table bbox: `2/2`.
- Carrier-tagged citation envelope đầy đủ và có citation ID riêng: `6/6`.
- Logical ACTUAL/PLAN binding: `3/3`; operand permutation: `3/3`.
- Program replay: `3/3`; accepted-wrong `0`.
- Semantic negatives: `9/9`; carrier/discriminator negatives: `6/6`; tổng `15/15`, false accept `0`.
- Coverage thực tế: `3/3/3/3` company/source/layout/instance, đạt minimum đã khóa.
- Frozen replay: execution `363/363`, candidate permutation `363/363`, counterfactual false accept `0/141`.
- Affected regression: `360/360` pass.

ABB native dùng source/page/table/cell bbox và không bị yêu cầu các trường OCR. THW và TVA OCR dùng hash content-list/table HTML, table ID, logical row/column và half-open source-cell ordinal. Không trường provenance nào được suy từ câu hỏi, nearby context hoặc sự vắng mặt.

Ba kết quả replay lần lượt là `-22,1%`, `3,103799642807903865566460074%` và `0,05928808251980106833671025972%`. So sánh sử dụng exact Decimal value; chỉ bỏ zero thập phân cuối cho biểu diễn ABB, không làm tròn.

Đây là development-only confirmation trên đúng ba real source/layout instance đã khóa. Nó chứng minh contract citation thực thi được trên hai carrier trong cohort, không phải claim unseen hoặc hoạt động hoàn hảo trên mọi báo cáo.

Coverage geometry cũ vẫn là hierarchical `1/1/1/1`, fragmented `0/0/0/0`; không đổi nhãn pixel-geometry gate và không dùng synthetic layout.

## Bước duy nhất được mở tiếp

`QP4-FINAL-GATE-RECLOSURE`

Bước này chỉ tổng hợp read-only toàn bộ gate QP4 đã khóa. Không sửa lỗi, bỏ gate hoặc diễn giải lại metric. Chỉ khi inventory đầy đủ cùng mọi hash đều PASS mới được đóng QP4 và mở QP5; Phase D vẫn khóa.
