# QP0 — Question coverage và counterfactual baseline

Ngày khóa: 13/08/2026  
Phân loại: development-only + synthetic compiler contract  
Trạng thái: QP0 hoàn tất; chưa sửa runtime

## 1. Phạm vi đã khóa

- 59 question family, gồm 10 lookup, 10 single-step, 12 plan/actual, 15 câu
  phức tạp và 12 câu kết hợp hai năm.
- 299 câu compiler contract; riêng PA01–PA12 và TY01–TY12 có 11 biến thể/family:
  canonical, paraphrase, câu dài, không dấu, đổi thứ tự mệnh đề, minimal pair đổi role/year,
  hard-negative thiếu role và ambiguity control.
- Catalog không dùng nội dung PDF hoặc numeric answer.
- Catalog SHA-256: `69c4017afec7731f63407339ae0c3d717ac38690807b17908b77fb1892d8e31b`.

## 2. Compiler-only baseline

Current `QuestionPlan` đạt exact 7/299 = 2,34%. Bảy family canonical đạt là A01, A03,
A04, A05, A06, A08 và A09. Lookup, plan/actual, two-year complex và complex multi-clause
đều 0 exact theo target IR đã precommit.

| Dimension | Pass | Rate |
|---|---:|---:|
| Requested years | 299/299 | 100,00% |
| Entity mode | 299/299 | 100,00% |
| Requested unit family | 272/299 | 90,97% |
| Answer mode | 268/299 | 89,63% |
| Direction intent | 261/299 | 87,29% |
| Program DAG | 205/299 | 68,56% |
| Operation | 161/299 | 53,85% |
| Column roles | 70/299 | 23,41% |

`column_roles` pass chủ yếu ở câu/control không yêu cầu role. Current schema không biểu diễn
`ACTUAL/PLAN/REVISED_PLAN` theo từng operand, nên plan/actual exact bằng 0 là baseline dự kiến,
không phải regression do QP0.

## 3. S4.3-A counterfactual baseline

Replay candidate reselection dùng đúng frozen Top-64 v5 và năm company/report-disjoint fold:

- examples: 976;
- accepted: 937;
- signed-execution correct: 829;
- false accept: 108;
- accepted precision: 88,4739%;
- coverage: 96,0041%;
- accepted SELECT/ADD: 227;
- minimum fold raw gold program recall theo frozen cache manifest: 98,9474%.

| Policy audit-only | Precision | Coverage | F | T | New FA | Quyết định |
|---|---:|---:|---:|---:|---:|---|
| Required proof complete all | 89,12% | 87,60% | 15 | 67 | 0 | Fail precision/coverage/retention |
| ADD strict required proof | 89,04% | 87,91% | 14 | 65 | 0 | Loại quá nhiều true ADD |
| SELECT requested year | 88,47% | 96,00% | 0 | 0 | 0 | Không có tác dụng bổ sung |
| SELECT relative period | 88,96% | 95,59% | 5 | 0 | 0 | An toàn cục bộ nhưng chưa đủ gate |
| SELECT requested unit | 88,42% | 95,59% | 1 | 4 | 1 | Sinh false accept mới sau reselection |
| SELECT entity/scope | 89,44% | 84,43% | 23 | 92 | 2 | Fail coverage/retention |
| SELECT column role | 88,58% | 96,00% | 1 | 0 | 0 | Current corpus/header role chưa đủ |
| Logical cell unique | 89,65% | 93,03% | 14 | 16 | 0 | Cân bằng nhất nhưng chưa đạt 90% |
| SELECT qualifier bundle | 89,44% | 84,43% | 23 | 92 | 2 | Fail coverage/retention |
| Combined SELECT/ADD structural | 90,53% | 75,72% | 40 | 160 | 2 | Precision pass nhưng không an toàn |

`F` là baseline false accept được chuyển thành correct/abstain; `T` là baseline true accept bị
chuyển thành false/abstain. Metric transition được báo riêng để không che trường hợp policy
reselection sinh đáp án sai mới.

Không policy nào đồng thời đạt precision ≥90%, coverage ≥88,5%, `9F-T ≥143`, gold policy
retention ≥95% và new false accept bằng 0. Vì vậy không policy audit-only nào được phép bật runtime.

## 4. Kết luận QP0

Siết proof trên metadata hiện tại không giải quyết đúng bài toán. Policy ADD strict loại 14 false
accept nhưng mất 65 true accept; bundle SELECT bị chi phối bởi entity metadata thiếu; uniqueness
có tín hiệu tốt nhất nhưng vẫn loại 16 true accept và chỉ đạt 89,65% precision.

Bước hợp lệ tiếp theo là QP1:

1. thêm `column_role`, `column_role_source`, `header_path` và provenance vào `EvidenceCell`;
2. reconstruct merged/hierarchical header để tách actual, plan, revised plan và forecast;
3. đổi cache schema/fingerprint;
4. kiểm thử conflict/repeated header/parallel panel trước khi siết verifier.

Checkpoint, runtime, neural score và threshold không thay đổi; không fine-tune; Phase D vẫn khóa.

## 5. Hygiene

Các script QP0 có path guard và chỉ nạp synthetic catalog cùng frozen development fold/cache.
Tuy nhiên, trước khi chạy script, một lệnh tìm threshold tương tác quá rộng đã in một record từ
report protected cũ. Record đó không được dùng cho expected IR, policy hoặc metric. Vì vậy phiên
này không tuyên bố `zero protected read`; các metric QP0 vẫn là development-only theo input thực tế
của script.
