# FinHRM — báo cáo đính chính alignment QP3

Ngày audit: 14/08/2026  
Phạm vi: development-only, read-only audit đối với runtime hiện có

## Kết luận

Chuỗi công việc gần đây đã lệch ở **cách chọn bước tiếp theo**: giới hạn word/bbox của smoke case ABW bị nâng
thành đề xuất `QP3G`, trong khi thesis-core QP3 vẫn còn các proof obligation chưa được nối đầy đủ. Đề xuất QP3G
được rút khỏi roadmap thực hiện ngay.

Không rollback QP3E–QP3F. Hai bước này là guard cấu trúc tổng quát, không chứa rule theo ABW và đã được kiểm tra
trên frozen synthetic cohorts. Xóa chúng không sửa được lỗi roadmap mà còn làm mất fail-closed provenance policy.

## Source of truth

Thứ tự ưu tiên được xác định bởi:

1. `config/finhrm_thesis_core_v1.json`;
2. `FINHRM_THESIS_CORE_SPEC.md`;
3. `FINHRM_THESIS_PUBLICATION_ROADMAP.md`;
4. mục 15.12 trở đi của `strategy.md`.

Phạm vi core là PA01–PA09, TY01/TY02/TY04, IBDS01 và C13–C15. TY05–TY12, OCR/layout robustness mở rộng,
conversational/cross-document và enterprise KG là deferred scope.

## Audit report-specific signal

Đã quét runtime `src/hrm_finance_qa` theo:

- `ABW`, `ABW_23CN`, `522bbb`;
- `472723394169`, `304912639828`, `363638445420`;
- rule theo company/page/year/value.

Kết quả: không có literal ABW hoặc ba giá trị diagnosis trong runtime source. Các giá trị này chỉ xuất hiện trong
test/report diagnosis. QP3E–QP3F dùng boundary, row/column coverage, header-role cardinality, physical source order,
span/bbox provenance, repeated-header và extractor conflict; không dùng câu hỏi, đáp án, numeric value hoặc neural
score để suy completeness.

## Completion matrix QP3 thesis-core

| Obligation | Trạng thái | Bằng chứng hiện có / khoảng trống |
|---|---|---|
| Typed operand slots và compiler DAG | Một phần | Compiler exact đã pass; chưa phải consumer proof đầy đủ |
| `column_role_match` | Có | Required cho explicit-role lane |
| Evidence-conditioned `column_role_unique` | Có | QP3A–QP3D, fail closed khi inventory đủ/xung đột |
| Table/flattened provenance inventory | Có, guard-only | QP3C–QP3F; không tự cấp `VALID/ACCEPT` |
| `reference_base_match` | Thiếu | Chưa có trong runtime proof constraint contract |
| `logical_cell_unique` | Thiếu | Chỉ từng là audit policy, chưa là core runtime obligation |
| `source_provenance_complete` | Thiếu | Chưa là required proof cho mọi answer family |
| `answer_mode_complete` | Thiếu | Chưa có runtime obligation |
| Proof cho mọi leaf/intermediate DAG | Thiếu | Chưa có `all_leaf_nodes_valid`, `all_intermediate_nodes_valid` |
| `dag_execution_valid` | Thiếu | DAG có thể materialize/execute nhưng chưa có node-level proof contract đầy đủ |

Vì vậy không được tuyên bố QP3 hoàn tất và chưa được chuyển sang QP4.

## Quyết định sửa

- Giữ QP3E–QP3F như auxiliary provenance hardening.
- Đánh dấu mọi ABW replay là diagnosis-only, non-gating.
- Rút QP3G khỏi immediate roadmap; chuyển word/bbox robustness sang extension sau thesis core.
- Bước kế tiếp là `QP3-CORE-PRECOMMIT`, khóa cohort và oracle cho các obligation còn thiếu trước khi sửa runtime.
- Chỉ sau `QP3-CORE-IMPLEMENT` và full frozen replay mới sang QP4.

## Invariant tiếp tục giữ

- Không đọc global/sealed validation hoặc blind/test.
- Không fine-tune.
- Không đổi checkpoint hay threshold.
- Không mở proof-conditioned acceptance mới.
- Phase D vẫn khóa tại accepted precision baseline `88,4739% < 90%`.

## Xác minh sau đính chính

- Runtime literal scan: `0` match cho tên/hash/ba numeric value của ABW trong `src/hrm_finance_qa`.
- Final contract/implementation regression: `33/33` pass, gồm thesis-core precommit, QP2E, QP3B, QP3D và QP3F.
- Không có runtime Python file nào được sửa trong lần đính chính này.

Không dùng các assertion hash của QP3A/QP3C/QP3E precommit làm gate hiện tại: chúng so runtime live với snapshot
trước QP3B/QP3D/QP3F, nên sau các implementation được phép sẽ báo `runtime_source_changed=true`. Frozen report/lock
lịch sử vẫn giữ nguyên; final implementation tests và lock tương ứng mới là regression gate cho trạng thái hiện tại.
