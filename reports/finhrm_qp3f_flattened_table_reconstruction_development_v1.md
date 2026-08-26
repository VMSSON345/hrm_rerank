# FinHRM QP3F — flattened native/OCR table reconstruction

Ngày: 14/08/2026  
Phân loại: development-only implementation and replay  
Trạng thái: hoàn tất QP3F; Phase D vẫn khóa

## Kết quả triển khai

QP3F triển khai frozen QP3E contract thành hai tầng:

1. `flattened_table_reconstruction.py` đánh giá đầy đủ boundary, header-role cardinality, row×column coverage, physical source order, geometry, repeated header và extractor conflict;
2. `program_data.py` hợp nhất các chunk chồng lấn cùng source/page bằng global line provenance trước Top-K, tạo flattened observations và chỉ chuyển chúng thành QP3D manifest khi state là `RECONSTRUCTABLE_COMPLETE`.

Native page index content version được nâng lên `8` để không chủ động xóa horizontal layout whitespace còn tồn tại trong PDF text layer. Các overlapping chunks chỉ được hợp nhất khi cùng source/page/extractor, global lines liên tục và phần overlap không xung đột.

`RECONSTRUCTABLE_COMPLETE` vẫn chỉ cấp quyền tạo manifest. QP3D inventory audit và toàn bộ proof obligations hiện hành tiếp tục quyết định acceptance.

## Frozen QP3E replay

| Hạng mục | Kết quả |
|---|---:|
| Reconstruction state exact | 128/128 |
| Reason-set exact | 128/128 |
| Manifest emission policy exact | 128/128 |
| Development-only end-to-end controls | 9/9 |
| Pre-Top-K full structure | 100 coordinates trước khi còn 64 candidates |
| Runtime/related regression | 306/306 |

End-to-end controls bao phủ complete native flattened table, table-scoped EvidenceGraph state, partial boundary, header-role cardinality mismatch, geometry conflict, ordinary-period non-applicability, candidate permutation, pre-Top-K coverage và overlapping-chunk reconstruction.

Producer không dùng question text, numeric value/answer, candidate score/rank/order hoặc Top-K để quyết định completeness.

## Real development-only replay: ABW

Query: `Doanh thu hoạt động thực hiện năm 2023 là bao nhiêu?`

- page-10 overlapping chunks đã được hợp nhất thành global-line run `1000001–1000037`;
- năm explicit column label vẫn được recover;
- native text layer tách ratio headers qua nhiều dòng và không giữ đủ column-coordinate provenance;
- QP3E geometry proof vì vậy không đạt và QP3F không phát manifest;
- neural head đề xuất sai `ADD(3, 13)`, nhưng confidence `0,125 < 0,522` và grounding gate chặn bởi `missing_explicit_addition_intent`;
- kết quả `ABSTAIN`, false accept `0`; MinerU/OCR không được dùng.

QP3F cải thiện đường reconstruction khi provenance đầy đủ nhưng chưa giải quyết answer coverage của ABW. Không được diễn giải frozen exact score là real-PDF accuracy.

## Invariants

- Không fine-tune, không đổi threshold/checkpoint/acceptance rule.
- Không đọc global/sealed validation hoặc blind/test.
- QP3E precommit/cohort/lock giữ nguyên hash.
- Các historical QP3E precommit tests yêu cầu runtime chưa thay đổi không còn là active implementation gate sau QP3F; artifacts lịch sử không bị sửa.
- Baseline accepted precision vẫn là `88,4739% < 90%`; Phase D vẫn khóa.

## Bước tiếp theo

QP3G precommit development-only word/bbox provenance contract cho native PDF và OCR. Mục tiêu là giữ word-level coordinates qua page index để chứng minh wrapped multi-line headers như ABW mà không dựa vào ordinal column position hoặc numeric values. Tách riêng một stability cohort cho neural operation proposals; threshold tiếp tục giữ nguyên.
