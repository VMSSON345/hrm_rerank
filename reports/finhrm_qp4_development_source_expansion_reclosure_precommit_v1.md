# QP4 development-source expansion reclosure precommit

## Kết quả

Precommit **PASS** và chỉ mở `QP4-DEVELOPMENT-SOURCE-EXPANSION-COHORT-BUILD`. Runtime, model, checkpoint, threshold, OCR generation, source replay, QP5, Phase D và publication pilot vẫn khóa.

Lý do reclosure là phạm vi đã được xác minh lại: thư mục `pdf` có 2.127 PDF năm 2023–2024, trong khi cohort trước chỉ dùng 160 source thuộc build universe khóa từ catalog 169 tài liệu. Báo cáo trước không khẳng định đã QA toàn bộ 2.127 PDF, nhưng cụm từ “full catalog” không được tiếp tục diễn giải thành toàn bộ kho PDF.

## Source pool development-safe

Expansion chỉ nhận PDF chưa có trong catalog nhưng ticker ánh xạ một-một đến company đã được phân loại `development_only`. Không đọc validation, blind, test, sealed, holdout hoặc final evaluation. ABW và 15 company đã xuất hiện trong diagnosis, prior admitted anchors hoặc material-rejection review bị loại trước selection.

Sau exclusion có 112 PDF năm 2024 trên 112 company. Pool được hash-lock trước khi đọc nội dung:

- pool SHA-256: `8711cfdf491bc14fe5dbed6a161dd754c6de89f4d918be4b410006d6acc2c5ab`;
- transform: `FINHRM_QP4_DEV_SOURCE_EXPANSION_RANDOM_V1`;
- sample: 50 source trên 50 company, không replacement;
- selected-manifest SHA-256: `4af6cebd661df6cebf9249343de16c7b6740a40f8e9cd600b6b423f2a48ab238`.

Question, page, value, layout, OCR result, model output, rank, score, confidence và action không tham gia selection. Source scan-only hoặc không chứa target structure vẫn phải giữ trong mẫu và báo đúng terminal reason; không được thay bằng source thuận lợi hơn.

## Feasibility

Ba source đầu theo selection order đã được probe chỉ-đọc bằng native extraction và PyMuPDF word geometry, hash khớp `3/3`. NHA có `152/152` native/geometry page; CCP chỉ có `1/9` page có native word geometry; MBB có `175/184`. CCP vẫn được giữ nguyên, chứng minh precommit không loại source khó để làm đẹp metric. Không sinh OCR, không lưu raw text và script `_tmp` đã xóa.

## Build được phép

Build kế tiếp chỉ đọc 50 source đã khóa và tìm hai signature còn thiếu:

1. `PA07_ROOT_HIERARCHICAL_MULTIROW_ROLE_PERIOD_HEADER_V1`;
2. `PA07_ROOT_FRAGMENTED_VISUAL_ROW_LABEL_V1`.

`flattened` đã PASS `3 company / 3 source / 3 real layout` nên chỉ được giữ làm later replay control. Mỗi anchor mới phải có metric, hai physical operand cell ACTUAL/PLAN, cùng explicit period, unit trong cùng bảng, entity và cell provenance đầy đủ; geometry phải đồng ý với visual review.

Hierarchical cần ít nhất hai company/source/layout mới để cộng với anchor cũ thành `3/3/3`. Fragmented cần `3/3/3` mới. Không pool hai signature. Nếu thiếu, build phải `FAIL_CLOSED`, không tăng sample hậu kiểm, không thay source, không nới proof/layout definition và không sửa runtime.

## Bước duy nhất được mở

`QP4-DEVELOPMENT-SOURCE-EXPANSION-COHORT-BUILD` — đúng 1 final JSON, 1 final Markdown và 1 lock; mọi file tạm phải xóa sau verify.
