# QP3-CORE-GENERALIZATION-COVERAGE-PRECOMMIT

Status: **PRECOMMITTED — RUNTIME CHANGE NOT YET AUTHORIZED**.

Nguồn duy nhất là safe replay development/train đã khóa hash: 363 SELECT/ADD example, 121 company group và 6 layout family. Không dùng ABW; không đọc validation/blind/test; không fine-tune, đổi checkpoint hay threshold.

Baseline fail-closed: 187 ANSWER, 162 ABSTAIN, 14 REJECT; execution 363/363; permutation 363/363; explicit counterfactual false accept 0/123.

| Generic proof track | Unknown examples | Company groups | Layout groups | Admission |
|---|---:|---:|---:|---|
| period + provenance preservation | 74 | 73 | 2 | blocked cho tới khi có diagnostic layout thứ ba |
| operation + ADD grounding | 67 | 47 | 3 | đủ diversity diagnosis; runtime vẫn khóa |
| query metric | 33 | 26 | 6 | đủ diversity diagnosis; runtime vẫn khóa |

`source_provenance_complete`, `logical_cell_unique` và `reference_base_match` không tạo unknown trong baseline này, nên chúng là no-regression obligations chứ không phải lý do để nới proof. Track period mới chỉ trải trên hai layout, thấp hơn invariant tối thiểu ba layout; không được lách bằng một repair theo report.

Mọi repair sau này phải được kiểm bằng positive, hard negative, semantic counterfactual và candidate permutation trên ít nhất ba company và ba layout. Một answer mới chỉ hợp lệ khi toàn bộ required proof là VALID. Phải giữ toàn bộ 187 answer cũ, execution 363/363, permutation 100% và false accept bằng 0.

Bước duy nhất tiếp theo: `QP3-CORE-GENERALIZATION-COVERAGE-DIAGNOSTIC-MATRIX`. Chỉ xây control tổng quát; chưa sửa runtime và QP4 vẫn khóa.
