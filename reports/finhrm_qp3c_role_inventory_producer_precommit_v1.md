# FinHRM QP3C — role-inventory producer precommit

Ngày: 13/08/2026  
Phân loại: development-only synthetic table provenance  
Trạng thái: hoàn tất precommit; chưa triển khai runtime producer; Phase D vẫn khóa

## Mục tiêu

QP3C khóa trước điều kiện để table reconstruction được phép gắn `column_role_inventory_state=EXPLICIT_COMPLETE`. Mục tiêu là ngăn runtime xem Top-K candidate hoặc một vài cell đã truy xuất như inventory đầy đủ.

## Contract đã khóa

Một table chỉ được gắn `EXPLICIT_COMPLETE` khi tất cả proof sau cùng đạt:

- table boundary là `COMPLETE` và header alignment là `ALIGNED`;
- reconstructed header IDs khớp chính xác toàn bộ expected data-column IDs;
- expected cell coordinates đúng tích Descartes `row IDs × data-column IDs` và observed coordinates khớp toàn bộ tập này;
- mọi data column có đúng canonical role với confidence `EXPLICIT` hoặc `INHERITED`;
- từng header-path element có provenance hợp lệ;
- role `INHERITED` có explicit anchor cùng role;
- từng observed cell có document/page/source-span/extraction-method provenance;
- không có header conflict, physical-equivalence role conflict hoặc cross-extractor conflict.

Thiếu bất kỳ proof nào đều cho `UNKNOWN`. Producer không có quyền dùng question text, numeric value/answer, candidate score, candidate rank, candidate order hoặc Top-K coverage để suy đoán completeness.

## Frozen cohort và kết quả

Cohort gồm `96` case: 8 layout group × 12 scenario. Tám layout bao phủ flat header, hierarchical merged header, repeated header, parallel entity panels, revised plan, forecast, ratio columns và bilingual header.

| Hạng mục | Kết quả |
|---|---:|
| Inventory state exact | 96/96 |
| Reason-set exact | 96/96 |
| `EXPLICIT_COMPLETE` positives | 24/24 |
| `UNKNOWN` controls | 72/72 |
| Permutation invariance | 8/8 |
| Inherited-header anchors | 8/8 |
| Conflict negatives | 24/24 |
| Distinct canonical roles | 6 |
| Distinct layout types | 8 |
| Malformed/leakage examples | 0 |
| Regression retention | 485/485 |

Các negative bao gồm thiếu header column, role không xác định, conflicting header role, cross-extractor disagreement, thiếu cell coverage, partial boundary, unaligned header width, thiếu physical provenance và duplicate physical role conflict.

## Invariants

- Không đọc PDF/document content.
- Không dùng numeric value/answer, question text hoặc candidate score.
- Không load model, không fine-tune.
- Không đọc global/sealed validation hoặc blind/test.
- Bảy runtime source hash khớp QP3B baseline; checkpoint không đổi.
- Không sửa threshold hay acceptance policy.
- Accepted precision chính thức vẫn là baseline `88,4739% < 90%`; Phase D vẫn khóa.

## Bước tiếp theo được phép

QP3D: triển khai đúng producer contract đã khóa, nối output vào `EvidenceGraph`, rồi replay nguyên frozen QP3C cohort và development-only end-to-end cases. QP3D không được đổi threshold và không được mở validation/blind/test.

Machine-readable artifacts:

- `config/finhrm_qp3c_role_inventory_producer_precommit_v1.json`;
- `data/finhrm_qp3c_role_inventory_producer_development_v1.jsonl`;
- `reports/finhrm_qp3c_role_inventory_producer_precommit_v1.json`.

