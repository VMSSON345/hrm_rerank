# FinHRM QP2D — safe abstention cho conflict và underspecification

Ngày hoàn tất: 13/08/2026  
Phân loại: development-only synthetic compiler repair  
Phạm vi: question compiler; `abstention_reason` chưa được runtime consumer hoặc acceptance tiêu thụ

## Kết luận

QP2D đạt gate. Question compiler hiện phân biệt ba loại câu không đủ đặc tả trong thesis core:

- `ambiguous_signed_direction`: có hai toán hạng nhưng không nêu hướng chênh lệch;
- `missing_reference_base`: hỏi phần trăm so sánh/tỷ lệ nhưng thiếu đại lượng gốc hoặc mẫu số;
- `missing_column_role`: câu lookup nói rõ bảng có nhiều role loại trừ nhau nhưng không chọn role cần trả lời.

QP2D không dùng family ID, company, page, numeric value hoặc candidate position. Reason mới là trạng thái bất biến,
question-side và vẫn non-consuming. Nó chưa trực tiếp làm UI/runtime từ chối; việc nối reason vào proof/acceptance chỉ
được phép ở QP3 sau khi QP2E đạt gate.

## Kết quả

| Kiểm tra | Kết quả |
|---|---:|
| Canonical ambiguity controls C13–C15 | 3/3 |
| Missing-reference surface forms | 8/8 |
| Missing-column-role surface forms | 6/6 |
| Explicit-reference repair không false-abstain | 4/4 |
| Explicit-role repair không false-abstain | 3/3 |
| PA07 frozen answer variants không false-abstain | 9/9 |
| QP2A–QP2C frozen answer retention | 108/108 |
| Legacy collateral regression | 0 |
| Regression tests | 205/205 |

Strict thesis-core exact tăng từ `14/16 = 87,50%` lên `16/16 = 100%`. `behavior` và
`abstention_reason` cùng tăng `14/16 → 16/16`. Legacy catalog exact tăng `90/135 → 92/135`; đúng hai
improvement là `C14-canonical` và `C15-canonical`, không có regression.

Strict compiler gate đạt 100% chỉ chứng minh contract của 16 canonical family. Nó không đồng nghĩa runtime answer
quality đạt 100%, không mở Phase D và không cho phép thay threshold.

## Conflict contract cho column role

Compiler tách hai loại role evidence:

1. role mô tả bối cảnh bảng, ví dụ `bảng có cả kế hoạch và thực hiện`;
2. role thực sự được yêu cầu, ví dụ `doanh thu thực hiện năm 2023`.

Nếu bối cảnh nêu ít nhất hai role nhưng phần request không chọn role, compiler phát
`missing_column_role`, không biến hai role bối cảnh thành hai operand. Nếu request chọn rõ `ACTUAL` hoặc `PLAN`,
compiler vẫn tạo SELECT một slot tương ứng.

Counterfactual đã phát hiện và loại một rule quá rộng: chỉ có từ `bảng` không đủ tạo conflict. Các tiền tố hợp lệ
như `căn cứ bảng số liệu` phải giữ nguyên answerability. Context conflict chỉ được kích hoạt khi câu nói rõ
bảng/báo cáo `có`, `gồm`, `chứa` hoặc `thể hiện` đồng thời nhiều role.

## Underspecification contract cho reference base

Các câu có relational-percent cue như `lớn hơn`, `tăng`, `giảm`, `chiếm`, `đạt` hoặc `hoàn thành bao nhiêu phần
trăm` phải có reference base đã được compiler chứng minh. Nếu không có base/DAG hợp lệ, reason là
`missing_reference_base`. Khi câu bổ sung `so với kế hoạch`, kỳ gốc hoặc denominator rõ, reason biến mất và DAG
QP2C được giữ nguyên.

## Ranh giới an toàn

- Không đọc PDF hoặc nội dung tài liệu.
- Không load model/checkpoint để inference.
- Không đọc global/sealed validation, blind hoặc test data.
- Không fine-tune; không đổi checkpoint, threshold hoặc acceptance policy.
- Không sửa `program_synthesis.py`, `symbolic_verifier.py`, executor hoặc `program_data.py`.
- Scan năm runtime consumer xác nhận chưa file nào đọc `abstention_reason`.
- Compiler DAG và `program_identity_ids` tiếp tục non-consuming.
- Phase D vẫn khóa tại accepted precision baseline `88,4739% < 90%`.

## Gap còn lại và bước tiếp theo

Strict canonical compiler contract đã đạt `16/16`, nhưng frozen legacy surface robustness mới đạt `92/135`; còn
43 variants chưa exact trên 12 answer family. Bước tiếp theo là QP2E: phân loại và sửa surface robustness trên đúng
frozen 135-row catalog, giữ retention QP2A–QP2D và không nối các field mới vào runtime acceptance.

Machine-readable source of truth:
`reports/finhrm_qp2d_safe_abstention_development_v1.json`.
