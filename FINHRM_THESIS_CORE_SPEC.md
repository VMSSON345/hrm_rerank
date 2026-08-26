# FinHRM — đặc tả lõi khóa luận v1

Ngày precommit: 13/08/2026  
Machine-readable contract: `config/finhrm_thesis_core_v1.json`

## 1. Mục đích

Tài liệu này khóa phạm vi thực nghiệm trước khi tiếp tục sửa QP2/QP3. Nó không tuyên bố các năng
lực bên dưới đã hoàn thành. Mọi expected program, proof, split và gate được định trước để các vòng
sau không đổi mục tiêu theo output của model.

Trong bước precommit này:

- không đọc validation, blind hoặc test;
- không fine-tune;
- không thay checkpoint, threshold hoặc runtime acceptance;
- không thêm LLM, multi-agent hoặc Neo4j production;
- Phase D tiếp tục khóa.

## 2. Claim và research question

Claim cần kiểm chứng:

> Cell-level financial roles và provenance-aware proof obligations cải thiện selective numerical
> reasoning khi evidence chứa các ô phản thực tế gần giống nhau.

Ba câu hỏi nghiên cứu:

1. Cell-level role-aware Evidence Graph có cải thiện operand/provenance grounding so với chunk hoặc
   section-level graph không?
2. Typed synthesis + three-state verifier có giảm semantic false accept mà giữ đúng gold program không?
3. Proof-aware acceptance có cải thiện risk–coverage so với neural-confidence-only không?

## 3. Phạm vi 16 family

### Plan–actual

| Family | Ý định | Program/DAG |
|---|---|---|
| PA01 | tra thực hiện | `SELECT(actual)` |
| PA02 | tra kế hoạch | `SELECT(plan)` |
| PA03 | tỷ lệ hoàn thành | `RATIO_PERCENT(actual, plan)` |
| PA04 | actual trừ plan | `SUBTRACT(actual, plan)` |
| PA05 | số tiền còn thiếu | `SUBTRACT(plan, actual)` |
| PA06 | số tiền vượt | `SUBTRACT(actual, plan)` |
| PA07 | actual thay đổi % so với plan | `PERCENT_CHANGE(actual, plan)` |
| PA08 | phần trăm còn thiếu | `RATIO_PERCENT(SUBTRACT(plan, actual), plan)` |
| PA09 | phần trăm vượt | `RATIO_PERCENT(SUBTRACT(actual, plan), plan)` |

### Hai năm

| Family | Ý định | Program |
|---|---|---|
| TY01 | actual 2023 trừ actual 2022 | `SUBTRACT(actual_2023, actual_2022)` |
| TY02 | tăng trưởng actual 2023 so với 2022 | `PERCENT_CHANGE(actual_2023, actual_2022)` |
| TY04 | plan 2024 so với actual 2023 | `PERCENT_CHANGE(plan_2024, actual_2023)` |

### Multi-step identity

`IBDS01` được chọn vì đã có implementation và cohort development:

```text
interest_bearing_debt = ADD(short_term_borrowings, long_term_borrowings)
answer = RATIO_PERCENT(interest_bearing_debt, total_capital)
```

Câu canonical yêu cầu rõ vay ngắn hạn, vay dài hạn, tổng nguồn vốn, cùng năm và cùng scope.
`total liabilities` không được dùng thay `total capital`; liabilities cũng không được dùng thay
interest-bearing borrowings.

### Ambiguity controls

| Family | Thiếu gì | Kết quả bắt buộc |
|---|---|---|
| C13 | chiều signed difference | `ABSTAIN: ambiguous_signed_direction` |
| C14 | reference base/mẫu số | `ABSTAIN: missing_reference_base` |
| C15 | actual hay plan | `ABSTAIN: missing_column_role` |

## 4. Expected operand identity

Mỗi leaf operand phải được xác định bằng:

```text
(document_id, table_id, row_id, column_id,
 metric, period, column_role, entity_scope, unit_scale)
```

Đúng giá trị số nhưng sai logical/physical cell vẫn là sai. Hai cell cùng value chỉ là grounded
equivalent khi provenance/semantic identity chứng minh chúng tương đương.

## 5. Evidence Graph contract

Graph chỉ cần phục vụ grounding, không phải enterprise KG:

```text
Report → Page → Table → Row/Column → Cell
Cell → Metric / Period / ColumnRole / EntityScope / UnitScale / Value / SourceSpan
```

Các edge proof/ranking gồm `SAME_METRIC`, `SAME_ENTITY`, `SAME_TABLE`, `SAME_ROW`, `SAME_UNIT`,
`PREVIOUS_PERIOD`, `NEXT_PERIOD`, `PART_OF_TOTAL` và `HEADER_DERIVED_ROLE`.

Mỗi relation hoặc thuộc tính evidence mang một state:

```text
EXPLICIT | INFERRED | UNKNOWN | CONFLICT
```

Backend có thể là in-memory hoặc SQLite. Neo4j không phải điều kiện hoàn thành khóa luận.

## 6. Proof contract đích

Mọi family ANSWER yêu cầu:

- execution và arity hợp lệ;
- operation khớp intent;
- metric, period, entity/scope và unit algebra hợp lệ;
- logical cell duy nhất;
- source provenance đầy đủ.

Áp dụng thêm:

- có role rõ → `column_role_match` bắt buộc;
- phép có chiều → `direction_match` bắt buộc;
- ratio/percent change → `reference_base_match`, denominator non-zero và comparable unit;
- multi-step → mọi leaf và intermediate node đều `VALID`.

Required proof `UNKNOWN` phải ABSTAIN; `INVALID` phải reject. Proof không được thay thế provenance.

## 7. Hard-negative bắt buộc

Mỗi positive real-PDF phải có ít nhất một negative đã human-review từ các lớp:

1. đổi `actual ↔ plan`;
2. đổi một operand year;
3. consolidated/separate/parent scope mismatch;
4. unit hoặc scale không tương thích;
5. cùng value nhưng khác semantic identity;
6. ô percent đứng cạnh ô absolute;
7. subtotal/detail hoặc liabilities/interest-bearing borrowing;
8. khác table/subtable;
9. wrong operands nhưng execution trùng gold;
10. bỏ role, direction, denominator hoặc reference base.

Negative không chỉ làm score thấp: program cũ phải `INVALID`, hoặc câu phải ABSTAIN vì required proof
`UNKNOWN`.

## 8. Annotation schema

Mỗi example phải có:

- question, family, answerability và answer/abstention reason;
- program DAG, operand order và reference base;
- company/report/source hash;
- hard-negative type;
- trạng thái review.

Mỗi operand phải có:

- value, metric, period, column role, entity/scope, unit/scale;
- logical-cell ID;
- document/page/table/row/column;
- source span hoặc coordinate;
- evidence state.

Sealed test phải double-review và adjudicate disagreement. Wrong program hoặc wrong provenance không
được tính đúng chỉ vì numerical answer trùng.

## 9. Company-disjoint split

- Group key là canonical company identity.
- Mọi báo cáo của một công ty chỉ thuộc một partition.
- Partition: development, calibration, sealed test.
- Assignment dùng SHA-256 với versioned salt; audit lại phân bố family/industry/layout/year.
- Không overlap company hoặc source-document hash.
- Công ty đã dùng diagnosis/repair không được vào sealed test.
- Sealed test chỉ mở một lần sau khi code/config/checkpoint/threshold/calibration hash đã khóa.
- Không chọn threshold trên sealed test.

Mục tiêu thesis: khoảng 300 example, ít nhất 30 company. Paper extension: 800–1.200 example, ít nhất
50 company. Đây là quota thiết kế, không phải lý do hạ chuẩn annotation.

## 10. Metric

Metric chính:

- provenance-correct execution;
- accepted precision và coverage;
- hard-negative false-accept rate;
- operand/program exact;
- evidence-cell recall và citation-cell accuracy;
- risk–coverage/AURC;
- verifier gold retention;
- answerability F1 và abstain-reason accuracy.

RAGAS chỉ là metric phụ. Confidence interval phải cluster theo company; comparison dùng paired test
trên cùng sealed examples.

## 11. Gate

### Compiler và counterfactual

- core explicit-intent exact = 100%;
- câu ngắn/dài tương đương phải có cùng IR;
- swapped-role false accept = 0;
- swapped-year false accept = 0;
- không tạo false accept mới.

### Grounding và selective reliability

- verifier gold retention từng fold ≥95%;
- thesis accepted precision point estimate ≥90%;
- overall coverage ≥88,5%;
- báo precision lower bound dù pass hay fail;
- publication chỉ pass khi one-sided 95% lower bound của accepted precision ≥90%.

Replay frozen S4 vẫn giữ gate `9F−T ≥143`, coverage ≥88,5%, common-operation regression không quá
1 điểm phần trăm. Không dùng thesis-core subset để thay thế gate toàn cục.

## 12. Baseline và ablation đã precommit

Baseline: BM25, dense/hybrid retrieval, section-level GraphRAG, LLM + Program-of-Thought, V8 neural-only
và flat-cell retrieval + deterministic executor.

Ablation:

```text
Full
− column-role/header lineage
− graph relations
− typed synthesis
− three-state verifier
− proof-conditioned acceptance
− operation calibration
```

Tất cả phải dùng cùng corpus, split, questions và gold evidence.

## 13. Bước tiếp theo sau precommit

1. Chạy compiler-only baseline riêng cho 16 family đã khóa.
2. Phân loại fail theo operation, role, direction, reference base, answer mode và DAG.
3. Sửa QP2 theo failure group tổng quát, không sửa theo từng câu.
4. Chỉ sau compiler audit mới nối các DAG còn thiếu vào QP3/runtime.

## 14. Addendum bắt buộc về tổng quát hóa trên báo cáo

Machine-readable addendum: `config/finhrm_report_generalization_invariant_v1.json`.

Một implementation duy nhất phải được áp dụng **không đổi** cho mọi báo cáo thuộc miền hỗ trợ đã khai báo. Miền
được xét theo công ty, năm, ngành, native/OCR, layout bảng, metric, numeric value và bề mặt câu hỏi. “Tổng quát trên
mọi báo cáo” không được hiểu là cam kết coverage 100% cho mọi PDF bất kỳ; report/layout ngoài miền hoặc thiếu
provenance phải `ABSTAIN` an toàn với reason tường minh, không được kích hoạt patch riêng.

Runtime bị cấm dùng company identity, filename/hash tài liệu, số trang cố định, năm cố định, numeric value/answer,
câu hỏi literal hoặc example/family ID làm rule quyết định. Runtime chỉ được dùng semantic type và provenance tổng
quát như metric, period, column role, entity/scope, unit/scale, logical/physical cell, table geometry, extractor
conflict và proof state.

Mọi repair mới phải được precommit và chỉ được nhận khi đồng thời:

- diagnosis document không nằm trong confirmation cohort;
- confirmation có ít nhất 3 công ty và 3 layout family độc lập với diagnosis;
- có positive, semantic hard-negative, counterfactual và permutation control;
- repair nhắm layout phải có layout-disjoint confirmation;
- replay toàn bộ frozen development cohorts không tạo regression;
- báo macro theo company/layout, worst group, unseen-company và unseen-layout;
- safe abstention không được tính là answered.

Ràng buộc này bổ sung cho thesis-core v1, không sửa hash artifact đã khóa và không cấp quyền đọc sealed/blind/test,
fine-tune, thay threshold hoặc mở Phase D.
