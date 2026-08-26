# FinHRM QP3B — evidence-conditioned column-role proof

Ngày: 13/08/2026  
Phân loại: development-only synthetic evidence  
Trạng thái: hoàn tất QP3B, Phase D vẫn khóa

## Mục tiêu

QP3B triển khai đúng proof contract đã khóa ở QP3A vào `EvidenceGraph`, symbolic verifier và grounding gate. Vòng này chỉ đánh giá cấu trúc evidence; không dùng numeric answer, candidate score, PDF/document content, validation, blind hay test data.

## Thay đổi runtime

- `EvidenceGraph` có trạng thái inventory tường minh: `EXPLICIT_COMPLETE`, `UNKNOWN`, hoặc không khai báo để tương thích ngược.
- Query có role tiếp tục dùng obligation `column_role_match`.
- Query không nêu role chỉ thêm required obligation `column_role_unique` khi inventory có trạng thái tường minh hoặc evidence có conflict thực tế.
- Nhiều role trong cùng logical identity, same-value khác role, hoặc cross-extractor disagreement đều cho `INVALID / ABSTAIN`.
- Inventory `UNKNOWN` cho required obligation `UNKNOWN / ABSTAIN` và grounding reason `missing_column_role_inventory`.
- Inventory chưa khai báo không được suy diễn là đầy đủ từ Top-64 và không làm giảm coverage legacy.
- `VALID` của column-role proof chỉ chuyển quyền quyết định cho các proof obligation khác; nó không tự cho phép trả lời.

## Kết quả frozen cohort

| Hạng mục | Kết quả |
|---|---:|
| Proof state exact | 128/128 |
| Action exact | 128/128 |
| Same-value conflicts | 16/16 |
| Permutation invariance | 16/16 |
| Unknown-inventory controls | 16/16 |
| Deferred family groups | 12/12 |
| Generalization groups | 4/4 |
| Regression liên quan | 477/477 |

QP3A artifacts vẫn giữ nguyên hash:

- precommit: `75cf5124b5107fce112c0e394cb7077c7c7e209b27fe461a23eb86af779d3c9f`;
- frozen cohort: `cfb7c5b38a55a721f81cc06cc4f00602e9df5f145c57590faf3bd5cb6634d1bf`;
- QP3A lock: `fbb79353e30473198c42fdcce89a0f8bbfab9a791e4f62046cbb53a4a901ea98`;
- checkpoint: `3a33276ded0d2ff2dd3d200c4c774a59eec0f8cad7102bc273cb263d983944de`.

## Giới hạn kết luận

Kết quả 128/128 chứng minh implementation khớp synthetic proof contract đã precommit. Nó chưa chứng minh end-to-end accuracy trên PDF thực tế và không phải kết quả validation. Runtime hiện không tự gọi inventory đầy đủ từ tập candidate; chỉ upstream table reconstruction có provenance đầy đủ mới được phép gắn `EXPLICIT_COMPLETE`.

Không fine-tune, không đổi checkpoint, không đổi threshold và không đọc global/sealed validation hoặc blind/test. Accepted precision chính thức vẫn là baseline `88,4739% < 90%`; vì vậy Phase D vẫn khóa.

## Bước tiếp theo

QP3C precommit một development-only provenance audit cho producer của role inventory ở table reconstruction. Chỉ những table có coverage header/cell đầy đủ và không có extractor conflict mới được gắn `EXPLICIT_COMPLETE`; mọi trường hợp còn lại phải là `UNKNOWN`. Sau khi khóa rule/cohort mới được nối producer này vào real runtime và replay development-only end-to-end, vẫn giữ nguyên threshold.
