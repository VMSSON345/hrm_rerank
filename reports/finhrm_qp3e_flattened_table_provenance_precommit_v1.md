# FinHRM QP3E — flattened native/OCR table provenance precommit

Ngày: 14/08/2026  
Phân loại: development-only synthetic precommit  
Trạng thái: hoàn tất precommit; chưa triển khai runtime; Phase D vẫn khóa

## Mục tiêu

QP3E khóa contract trước khi mở rộng table reconstruction cho các bảng bị flatten thành native plaintext hoặc OCR observations như ABW. Contract tách hai quyền:

1. `RECONSTRUCTABLE_COMPLETE` chỉ cho phép tạo structural manifest;
2. manifest đó vẫn phải qua QP3D inventory audit và mọi proof obligation hiện hành trước khi một câu trả lời có thể được accept.

QP3E không trực tiếp tạo `EXPLICIT_COMPLETE`, `VALID`, `ACCEPT` hoặc đáp án.

## Contract đã khóa

Một flattened table chỉ được gắn `RECONSTRUCTABLE_COMPLETE` khi đồng thời chứng minh:

- source boundary hoàn chỉnh và không còn continuation mở;
- data-column IDs, declared width và header-role cardinality khớp chính xác;
- mỗi header có đúng một canonical role explicit hoặc inherited có anchor;
- row IDs đầy đủ và cell observations phủ đúng tích `rows × columns`;
- mỗi physical cell thuộc đúng một row và một column;
- header, row descriptor và cell đều có document/page/span/sequence/extractor/bbox provenance;
- header đứng trước data cells theo physical source order;
- column band của cell khớp header tương ứng;
- repeated headers và native/OCR views không xung đột;
- kết quả bất biến với thứ tự serialization.

Thiếu hoặc xung đột bất kỳ proof nào đều cho `UNKNOWN`. Contract cấm suy diễn completeness từ question text, numeric value/answer, Top-K, candidate score/rank/order.

## Frozen cohort

| Hạng mục | Kết quả |
|---|---:|
| Tổng examples | 128 |
| Layout groups | 8 |
| Distinct roles | 7 |
| Hard-negative types | 13 |
| `RECONSTRUCTABLE_COMPLETE` positives | 24/24 |
| `UNKNOWN` controls | 104/104 |
| State exact | 128/128 |
| Reason-set exact | 128/128 |
| Permutation invariance | 8/8 |
| Repeated-header positives | 8/8 |
| Boundary hard-negatives | 16/16 |
| Conflict hard-negatives | 32/32 |
| Tests QP3E + related QP3D/QP3B/QP2/QP1 | 90/90 |

Tám layout gồm native single-page, native multi-page, OCR bbox, native/OCR agreeing views, merged role header, bilingual header, revised plan và value/ratio columns.

Mười ba hard-negative types bao phủ partial/open boundary, header cardinality, width, unknown role, missing cell, overlapping physical span, ambiguous row membership, source order, cross-extractor disagreement, repeated-header disagreement, missing provenance và column geometry.

## Invariants

- Không sửa `program_data.py`, `evidence_schema.py`, verifier, planner, synthesis, HRM hoặc reasoning engine; runtime hash mismatch `0`.
- Không load PDF, model, global/sealed validation hoặc blind/test.
- Không fine-tune, không đổi threshold/checkpoint/acceptance policy.
- QP3D lock, QP3C lock và checkpoint giữ nguyên hash.
- Baseline accepted precision vẫn là `88,4739% < 90%`; Phase D vẫn khóa.

## Bước tiếp theo được phép

QP3F triển khai đúng contract đã freeze vào flattened native/OCR reconstruction. Chỉ observations đạt `RECONSTRUCTABLE_COMPLETE` mới được chuyển thành QP3D structural manifest. Sau đó replay nguyên frozen QP3E cohort và các development-only end-to-end cases; threshold tiếp tục giữ nguyên.
