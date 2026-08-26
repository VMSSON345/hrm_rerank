# V12: Constraint-Guided Neuro-Symbolic Program Synthesis

## 1. Quyết định kiến trúc

V12 đi đúng theo hướng neuro-symbolic trong đó symbolic tham gia từ lúc tạo
chương trình, không chỉ kiểm tra kết quả sau HRM.

```text
Câu hỏi
   ↓
Hybrid semantic compiler
   - quy tắc rõ ràng khi câu hỏi nêu trực tiếp operation/period/entity
   - operation head của HRM khi ý định còn mơ hồ
   ↓
Typed Query IR
   (operation, metric, period, entity, requested unit, temporal direction)
   ↓
                         Báo cáo
                            ↓
                    Typed Evidence Graph
                            ↓
Symbolic synthesizer tạo khung chương trình và operand slots hợp kiểu
   ↓
HRM chấm điểm từng assignment bằng operation/pointer probability
   ↓
Three-state verifier + constraint solver
   ↓
Proof-guided program selection
   ↓
Decimal executor + confidence + abstention
```

Khác biệt cốt lõi:

```text
V8/V11: HRM sinh candidate tự do → symbolic kiểm tra muộn
V12:     symbolic sinh khung hợp kiểu → HRM chấm các slot
         → symbolic kiểm chứng lần cuối
```

V12 không train lại và không ghi đè checkpoint
`models/document_hrm_V8_candidate.pt`. V11 safe-residual được giữ làm ablation,
không phải runtime chính, vì nó cải thiện acceptance nhưng không tăng execution,
pointer pair hay program exact trên validation.

## 2. Vì sao không chỉ gắn CAS hoặc Lean sau HRM?

CAS có thể xác nhận phép tính `40.494 - 49.276`, nhưng không thể tự biết hai số
đó có phải doanh thu đúng năm, đúng công ty và đúng phạm vi báo cáo hay không.
Lỗi chính của QA tài chính nằm ở grounding:

- sai năm hoặc sai vai trò đầu kỳ/cuối kỳ;
- đúng metric nhưng sai thực thể/phạm vi;
- ghép hai hàng hoặc hai bảng không tương thích;
- lấy tỷ lệ phần trăm thay cho số tiền;
- cùng kết quả do ngẫu nhiên nhưng evidence sai.

Vì vậy, typed evidence và constraint-guided synthesis phải đứng trước CAS. Lean
có thể là thí nghiệm về sau để chứng minh định nghĩa công thức; nó không nằm
trên critical path của retrieval và grounding.

Nền tảng nghiên cứu tương thích với hướng này là tách evidence rồi mới thực hiện
symbolic aggregation/program reasoning: [TAT-QA](https://aclanthology.org/2021.acl-long.254/),
[FinQA](https://aclanthology.org/2021.emnlp-main.300/),
[DocFinQA](https://aclanthology.org/2024.acl-short.42/).

## 3. Nguyên tắc tổng quát hóa bắt buộc

Mọi thay đổi phải hoạt động trên nhiều công ty, năm và layout báo cáo:

1. Không hard-code tên công ty, ticker, năm, số trang, table ID hoặc vị trí cột.
2. Constraint chỉ dùng metric semantics, period role, entity/scope, unit và quan
   hệ row/column/table.
3. Metadata thiếu là `UNKNOWN`, không phải `INVALID`.
4. Chỉ mâu thuẫn được chứng minh bằng metadata mới bị loại cứng.
5. Reporting entity là mặc định; embedded entity chỉ được ưu tiên khi câu hỏi
   gọi tên thực thể đó.
6. Train, calibration và validation phải company/report-disjoint.
7. Báo cáo kết quả theo operation, company, report, layout và year; trung bình
   tổng không được che regression của một nhóm.
8. Blind test 1.500 câu cũ đã được dùng để chốt V8 và không được dùng phát triển
   V12.
9. Phiên bản nghiên cứu cuối cần một company-disjoint test mới được niêm phong.

## 4. Typed Query IR

`QuestionPlan` là IR dùng chung cho synthesis và verifier. Nó biểu diễn:

- operation;
- metric text;
- requested periods;
- hướng `from → to`, hoặc mặc định kỳ mới → kỳ cũ cho temporal change;
- reporting entity hoặc named entity;
- requested unit family;
- percent intent và temporal-change intent.

Ví dụ:

```text
Doanh thu thuần thay đổi bao nhiêu phần trăm
từ năm 2022 đến năm 2023?
```

được biên dịch thành:

```text
operation = PERCENT_CHANGE
metric = doanh_thu_thuan
answer_unit = percent
slot[0] = Cell[metric=M, period=2023, entity=E, unit=U]
slot[1] = Cell[metric=M, period=2022, entity=E, unit=U]
constraints = SAME_METRIC, SAME_ENTITY, SAME_UNIT, temporal_direction
```

Dấu `%` ở đây là kiểu đầu ra, không có nghĩa hai operand phải là cell phần trăm.

Hiện trạng V12-A dùng hybrid compiler: các intent được nêu rõ được biên dịch
bằng parser xác định; operation không rõ vẫn lấy từ HRM. Một neural semantic
parser được huấn luyện riêng là thí nghiệm sau, không được tuyên bố là đã có.

## 5. Typed Evidence Graph

Mỗi numeric candidate được chuẩn hóa thành `EvidenceCell`:

```text
value, metric, period, period_role, entity, unit,
table_id, subtable_id, row_id, column_id, source, page
```

Graph chỉ tạo quan hệ khi extraction metadata hỗ trợ:

```text
SAME_METRIC, SAME_ENTITY, SAME_TABLE, SAME_ROW, SAME_UNIT,
PREVIOUS_PERIOD, NEXT_PERIOD, PART_OF_TOTAL
```

Không suy diễn một quan hệ chắc chắn từ vị trí trang hoặc tên công ty cụ thể.

## 6. Typed-slot program synthesis

Implementation giai đoạn A: `src/hrm_finance_qa/program_synthesis.py`.

### 6.1 Typed lookup

```text
SELECT(Cell[metric=M, period=Y, entity=E, requested_unit=U])
```

Synthesis giới hạn slot theo metric, period, entity và đơn vị được hỏi. Extraction
method không phải điều kiện, nên layout/parser chưa từng thấy vẫn có thể tạo
search trace nếu metadata kiểu còn đủ.

### 6.2 Temporal same-metric template

```text
SUBTRACT(current[M,E,U], prior[M,E,U])
PERCENT_CHANGE(current[M,E,U], prior[M,E,U])
AVERAGE(period_a[M,E,U], period_b[M,E,U])
```

Hai operand không được có metric, entity hoặc comparable unit mâu thuẫn. Quan hệ
same-row, same-table và temporal edge là proof/ranking feature; metadata thiếu
không tự động loại program.

### 6.3 Explicit operand-role template

Khi câu hỏi nêu rõ các đại lượng:

```text
ADD(slot[nợ_ngắn_hạn], slot[nợ_dài_hạn])
RATIO_PERCENT(slot[part], slot[total])
MULTIPLY(slot[count], slot[par_value])
```

Mỗi slot có beam riêng. Beam không cho tái sử dụng cùng pointer và loại assignment
có entity/unit mâu thuẫn trước bước chọn chương trình.

### 6.4 Hợp candidate neural và symbolic

V12 không xóa candidate của V8:

```text
P = P_neural_baseline ∪ P_typed_slot_synthesis

nếu legacy structural anchors tồn tại:

P = P ∪ P_neural_structural_beam
```

Mỗi program có `generation_sources`. HRM operation probability và pointer
probability vẫn chấm điểm toàn bộ assignment; verifier quyết định constraint
state trước proof-guided selection. Đây là program synthesis có neural guidance,
không phải rule engine thay thế HRM.

Typed synthesis không được phép chỉ vì kích hoạt search mà mở toàn bộ neural
beam tự do. Khi legacy không có structural anchor, neural beam ngoài baseline
phải đi qua typed slots; quy tắc này ngăn candidate không hợp khung thắng bằng
proof heuristic ngẫu nhiên.

## 7. Three-state verifier và proof certificate

- `VALID`: metadata hiện có ủng hộ constraint.
- `INVALID`: metadata hiện có chứng minh mâu thuẫn.
- `UNKNOWN`: parser không thu được đủ metadata.

Chỉ `INVALID` bị loại cứng. Trace của program phải lưu:

- generation source;
- operation, metric, period, entity và unit match;
- same row/table/scope nếu biết;
- temporal direction;
- denominator safety;
- evidence source của từng operand.

Giai đoạn hiện tại dùng verifier ràng buộc xác định. SMT/CSP tổng quát và proof
serialization đầy đủ chưa được coi là hoàn thành.

## 8. Executor, CAS và unit algebra

Giai đoạn A dùng typed DSL và deterministic `Decimal` executor. Các bất biến:

```text
money - money -> money
money / money -> ratio
ratio * 100 -> percent
PERCENT_CHANGE(new, old) yêu cầu old != 0
```

`SAME_UNIT` là bằng chứng dương cho cộng/trừ và ratio, nhưng không phải bằng
chứng dương cho `MULTIPLY`; hai operand cùng đơn vị chỉ được xem là `UNKNOWN`
cho đến khi có operand-role unit algebra đầy đủ.

CAS/SMT ở pha sau chỉ được kiểm tra algebraic equivalence, multi-step program và
unit algebra. CAS không được tự chọn evidence cell. Unit scaling như đồng/nghìn
đồng/triệu đồng/tỷ đồng phải có provenance trước khi cho phép convert.

## 9. Positive program taxonomy và loss

Candidate cache tách bốn nhãn:

1. `canonical_exact`: đúng operation, operand và vai trò sau canonicalization;
2. `grounded_equivalent`: khác pointer vật lý nhưng cùng logical-cell signature,
   verifier `VALID` và execution đúng;
3. `execution_only`: cùng kết quả nhưng evidence không chứng minh tương đương;
4. `negative`: execution sai.

Rank loss ưu tiên `canonical_exact`. Chỉ nhóm không có canonical mới dùng
`grounded_equivalent`; `execution_only` không bao giờ là positive.

Loss mục tiêu sau khi candidate recall đủ cao:

\[
L = L_{slot\text{-}retrieval} + L_{program\text{-}set}
  + \lambda L_{hard\text{-}negative}
  + \mu L_{constraint} + \beta L_{confidence}
\]

Không fine-tune HRM chỉ để sửa từng câu. Pha đầu giữ V8 frozen; chỉ cân nhắc
fine-tune pointer head sau khi chứng minh bottleneck còn nằm ở neural scoring.

## 10. Audit đúng bottleneck

`no_trace` và `single_candidate` phải được tách nguyên nhân:

- single candidate đúng: không cần reranker;
- single candidate sai nhưng gold operand có trong retrieval: synthesis failure;
- gold operand không có trong retrieval: retrieval failure;
- gold được sinh nhưng verifier loại: verifier false rejection;
- gold hợp lệ có trong Top-K nhưng không thắng: ranking failure.

Các metric bắt buộc:

- gold operand recall@5/@10;
- gold program generated recall;
- gold eligible program recall;
- symbolic recall@5/@10;
- typed-synthesis gold generation rate;
- search trace coverage và single-eligible rate;
- verifier false-rejection rate;
- execution, pointer pair và program exact;
- latency và candidate count.

## 11. Feature flag và tương thích runtime

V12 mặc định tắt cho demo cho đến khi đạt gate:

```text
typed_program_synthesis_enabled = false
```

Bật thử nghiệm CLI:

```bash
--typed-program-synthesis
```

Bật web local:

```bash
TYPED_PROGRAM_SYNTHESIS=1 bash scripts/start_demo.sh
```

V8 và V11 sidecar không bị ghi đè. Cache reranker phải được regenerate nếu muốn
train trên candidate space V12.

## 12. Đánh giá validation-only

```bash
bash scripts/evaluate_document_hrm_v12_synthesis.sh
```

Output:

```text
reports/document_hrm_V12_typed_synthesis_validation.json
```

### Candidate-generation gate

- Gold generated recall tăng ít nhất 5 điểm phần trăm.
- Gold eligible recall không giảm.
- `no_trace` giảm.
- Pipeline gold rejection rate không tăng.

### Runtime-safety gate

- Execution, program exact và pointer pair không giảm.
- Không operation phổ biến nào giảm quá 1 điểm phần trăm.
- SHA-256 của V8 không đổi.

Nếu candidate gate đạt nhưng runtime gate không đạt, chỉ dùng synthesizer để tạo
Top-K/cache huấn luyện; không cho nó thay runtime winner. Khi đó train reranker
trên candidate space mới thay vì sửa rule theo từng câu validation.

## 13. Lộ trình nghiên cứu

### V12-A — đang triển khai và kiểm chứng

- Typed Query IR và requested unit family.
- Typed Evidence Graph.
- Typed-slot program synthesis.
- Neural + symbolic candidate union.
- Generation provenance.
- Positive taxonomy.
- Validation-only ablation và safety gates.

Chỉ đánh dấu hoàn thành sau khi unit/regression tests và report validation chạy
thành công.

Kết quả validation-only hiện tại trên 200 mẫu:

| Chỉ số | V8 | V12-A | Thay đổi |
|---|---:|---:|---:|
| Program exact / pointer pair | 71,0% | 74,5% | +3,5 điểm % |
| Execution | 80,5% | 83,5% | +3,0 điểm % |
| Gold program generated recall | 72,0% | 81,5% | +9,5 điểm % |
| Gold eligible recall | 67,5% | 76,0% | +8,5 điểm % |
| Search trace coverage | 82,5% | 92,5% | +10,0 điểm % |
| Accepted coverage | 87,5% | 89,5% | +2,0 điểm % |
| Accepted execution precision | 88,0% | 88,83% | +0,83 điểm % |

`runtime_safety_gate = true`: không operation phổ biến nào giảm; SUBTRACT tăng
10 điểm %, PERCENT_CHANGE tăng 20 điểm % và AVERAGE tăng 5 điểm %.

Candidate gate định trước vẫn `false` vì gold rejection tăng từ 9/144 lên
11/163, tương đương +0,50 điểm % theo tỷ lệ có điều kiện, dù số gold eligible
tăng 17 câu. Không đổi gate sau khi nhìn validation; giữ kết quả này làm mốc cho
V12-B.

Quyết định: V12-A đủ tốt để tạo candidate/cache cho V12-B, nhưng vẫn mặc định
tắt ở demo. Release gate tiếp theo yêu cầu accepted execution precision ít nhất
90% và candidate false rejection được xử lý trên train/calibration, không sửa
theo từng câu validation.

### V12.1 — role-complete planner và financial identities

V12.1 không phải một bộ luật dành riêng cho báo cáo AAM. AAM chỉ là external
smoke test để phát hiện failure mode; mọi constraint mới đều được định nghĩa
theo semantic type và phải qua validation nhiều báo cáo.

Các bổ sung kiến trúc:

1. `QuestionPlan` có thể phát nhiều operation hypothesis. Câu hỏi hệ số được
   biểu diễn bằng `SELECT(direct_coefficient)` hoặc `DIVIDE(numerator,
   denominator)`, thay vì ép một phép chia khi báo cáo đã công bố hệ số.
2. Candidate quota giữ bằng chứng theo operand role trước khi xếp theo độ lớn.
   Vì vậy một ratio nhỏ, số lượng cổ phần, vốn điều lệ hoặc dòng mệnh giá không
   bị loại chỉ vì tài liệu có nhiều số tiền lớn hơn.
3. Financial identity chỉ được materialize khi các cell nguồn cùng tài liệu và
   công thức kiểm chứng được. Trường hợp mệnh giá dùng:

   ```text
   par_value = charter_capital / issued_shares
   nominal_value_outstanding = outstanding_shares * par_value
   ```

   Intermediate phải mang provenance `symbolic-derived`, đơn vị `VND/share`,
   và phép chia phải cho kết quả nguyên dương. Không hard-code công ty, năm,
   trang hoặc giá trị.
4. Chương trình có hai role được nêu rõ phải chứng minh đủ cả hai role. Nếu
   không có bằng chứng, neural baseline không được thay bằng hai cell bất kỳ và
   symbolic confidence không được nâng kết quả; hệ thống phải `ABSTAIN`.
5. `shares × shares` là unit algebra `INVALID`; `shares × VND/share` là
   `VALID`.
6. Câu hỏi có hướng rõ `từ năm A đến năm B` giữ hiệu có dấu `B − A`. Quy ước
   magnitude dương chỉ còn áp dụng cho wording so sánh/tăng/giảm không yêu cầu
   signed direction.
7. Khi `ABSTAIN`, UI/API không hiển thị candidate bị bác bỏ như câu trả lời
   chính. Candidate chỉ nằm trong technical audit.

External smoke AAM kiểm tra bốn failure mode, không dùng để chọn trọng số:

- lookup 2024 và temporal subtraction dùng đúng metric/period;
- mệnh giá được suy ra bằng identity hai bước có provenance;
- tổng nợ thiếu hai role thì `ABSTAIN`;
- hệ số nợ/tổng tài sản thiếu numerator và direct coefficient thì `ABSTAIN`.

Validation-only 200 mẫu nhiều báo cáo sau V12.1:

| Chỉ số | Typed-off cùng runtime | V12.1 | Thay đổi |
|---|---:|---:|---:|
| Pointer pair / program exact | 70,0% | 73,0% | +3,0 điểm % |
| Execution | 79,5% | 82,0% | +2,5 điểm % |
| Gold program generated recall | 71,0% | 77,5% | +6,5 điểm % |
| Gold eligible recall | 66,0% | 71,5% | +5,5 điểm % |
| Search trace coverage | 82,5% | 88,5% | +6,0 điểm % |
| Accepted coverage | 85,0% | 88,0% | +3,0 điểm % |
| Accepted execution precision | 88,24% | 89,20% | +0,97 điểm % |

`runtime_safety_gate = true` so với typed-off cùng runtime; V8 SHA-256 không
đổi và blind/test không được đọc. Candidate gate vẫn `false` vì conditional
gold-rejection tăng 0,70 điểm %, nên V12.1 chưa đạt release gate nghiên cứu
90% accepted precision. So với mốc V12-A trước đó, safety guard đổi một phần
coverage/accuracy lấy việc không xuất kết luận sai khi thiếu role; vì vậy phải
giữ riêng cả hai báo cáo và không tuyên bố V12.1 là cải thiện tuyệt đối trên mọi
metric.

Báo cáo: `reports/document_hrm_V12_1_generic_financial_roles_validation.json`.

### V12.2 — YAML financial identity catalog (thử nghiệm)

Danh sách công thức chắc chắn được tách khỏi mã nguồn và lưu tại
`config/financial_identities.yaml`. Catalog chỉ mô tả semantic identity tổng
quát, không chứa công ty, trang, năm hoặc giá trị cụ thể. Mỗi identity khai báo
target, alias, operation, operand role, unit family, trạng thái và chính sách ưu
tiên giá trị công bố trực tiếp.

Runtime áp dụng theo thứ tự an toàn:

1. Nếu báo cáo có cell target trực tiếp, dùng `SELECT` thay vì tự tái tạo tổng.
2. Chỉ tổng hợp công thức khi đủ operand role và verifier xác nhận period,
   entity/scope, unit algebra và operation.
3. `active` được phép tham gia synthesis; `conditional` chỉ là tri thức lưu trữ,
   chưa được thực thi cho đến khi có sign policy hoặc multi-step DAG phù hợp.
4. YAML được kiểm tra schema/arity/operation/ID khi nạp; catalog sai làm runtime
   fail-fast thay vì âm thầm sinh chương trình sai.

Catalog được tách bằng feature flag riêng và mặc định tắt:

```text
financial_identity_catalog_enabled = false
```

Bật thử nghiệm web local cùng typed synthesis:

```bash
TYPED_PROGRAM_SYNTHESIS=1 FINANCIAL_IDENTITY_CATALOG=1 \
  bash scripts/start_demo.sh
```

Validation-only 200 mẫu nhiều báo cáo:

| Chỉ số | V12.1 | V12.2 YAML | Thay đổi |
|---|---:|---:|---:|
| Pointer pair / program exact | 73,0% | 72,5% | -0,5 điểm % |
| Execution | 82,0% | 81,5% | -0,5 điểm % |
| Accepted coverage | 88,0% | 84,5% | -3,5 điểm % |
| Accepted execution precision | 89,20% | 88,17% | -1,04 điểm % |

Vì V12.2 chưa vượt V12.1 nên catalog không được bật mặc định và không thay bản
demo chính. Kết quả này được giữ làm ablation; vòng tiếp theo phải cải thiện
identity matching/UNKNOWN handling trên train/calibration, không sửa theo từng
câu validation và không đọc lại blind test.

Báo cáo:
`reports/document_hrm_V12_2_yaml_financial_identity_validation.json`.

### V12.3 — safe additive identity overlay

V12.3 sửa nguyên nhân regression của V12.2 theo nguyên tắc catalog chỉ được bổ
sung, không được xóa giả thuyết V12.1:

1. Target activation dùng exact/high-precision semantic match; một metric hẹp
   như `nợ phải trả dài hạn` không được kích hoạt identity `tổng nợ phải trả`.
2. Giá trị target được công bố trực tiếp bổ sung giả thuyết `SELECT`, nhưng
   không cắt neural baseline hoặc operation beam hiện có.
3. Hard `financial_identity` constraint chỉ áp dụng cho chương trình do catalog
   sinh và nhánh `SELECT` của chính target identity. Chương trình neural khác
   không bị bác chỉ vì alias catalog chưa đầy đủ.
4. Identity `nợ vay chịu lãi = vay ngắn hạn + vay dài hạn` được thêm `active`.
   Financial-debt/equity, interest coverage và hai hệ số CFO được lưu
   `conditional`; DIO/DSO/DPO/CCC/Altman vẫn chờ typed constants và multi-step
   DAG, không thêm operator giả vào DSL hiện tại.

Validation-only 200 mẫu nhiều báo cáo:

| Chỉ số | V12.1 | V12.2 | V12.3 |
|---|---:|---:|---:|
| Operation | 95,0% | 94,5% | **95,0%** |
| Pointer pair / program exact | 73,0% | 72,5% | **73,0%** |
| Execution | 82,0% | 81,5% | **82,0%** |
| Accepted coverage | 88,0% | 84,5% | **88,0%** |
| Accepted execution precision | 89,20% | 88,17% | **89,20%** |
| Gold generated recall | 77,5% | 75,5% | **77,5%** |
| Gold eligible recall | 71,5% | 69,0% | **71,5%** |
| Pipeline gold rejection | 7,74% | 8,61% | **7,74%** |

Số accepted đúng/sai trở lại đúng V12.1: 176 accepted, 157 đúng và 19 sai.
V12.3 loại bỏ regression của catalog nhưng chưa làm tăng aggregate score trên
tập validation hiện tại; lợi ích mới là mở rộng an toàn các câu identity không
có trong bộ 200 mẫu. Vì chưa đạt accepted precision 90%, feature flag vẫn mặc
định tắt và chưa thay demo chính.

V8 SHA-256 giữ nguyên; `test.jsonl` và blind không được đọc. Báo cáo:
`reports/document_hrm_V12_3_safe_identity_overlay_validation.json`.

### V12.4 — operation-conditioned candidate recall + safe residual

V12.4 phát triển trực tiếp từ V12.3 và vẫn giữ V8 hoàn toàn đóng băng. Mục tiêu
không phải thêm luật cho từng báo cáo, mà sửa bottleneck candidate generation bằng
các ràng buộc có thể dùng lại cho mọi công ty, năm và layout:

1. Với phép toán hai toán hạng, candidate beam là hợp của joint pair logits và
   hai pointer-slot Top-K. Trước đây, khi có pair logits, các pointer riêng lẻ mạnh
   bị bỏ qua hoàn toàn.
2. Khi intent không tường minh, chỉ giữ một operation Top-K nhỏ. Một operation phụ
   chỉ được materialize nếu có typed/structural anchor, vì vậy không sinh toàn bộ
   tích Descartes của operation và cell.
3. Trace-complete fallback ghi lại cả trường hợp không có chương trình đủ điều kiện
   để phân biệt `not_generated`, `generated_but_ineligible` và `ranking_failure`.
   Fallback này chỉ phục vụ audit; grounding gate cuối vẫn bắt buộc `ABSTAIN`.
4. Safe residual chạy lại đúng search V12.3 trên cùng neural logits. Khi V12.3 đã
   có winner hợp lệ và chưa có reranker được calibration, V12.4 không được thay
   winner đó. Candidate mới vẫn được lưu để train/rerank và chỉ là runtime fallback
   khi V12.3 không có kết luận.
5. Parser SUM3/SUM4 chỉ nhận danh sách dấu phẩy khi câu hỏi thực sự mở đầu bằng
   yêu cầu `tổng`. Dấu phẩy trong tên chỉ tiêu và từ `và` nối hai kỳ không còn làm
   câu hỏi `bình quân` bị đổi thành SUM3.
6. Không có rule chứa tên công ty, số trang, năm cố định hoặc giá trị cụ thể.

Validation-only 200 mẫu, so trực tiếp V12.3 với V12.4:

| Chỉ số | V12.3 | V12.4 | Thay đổi |
|---|---:|---:|---:|
| Operation | 95,0% | **98,5%** | +3,5 điểm % |
| Pointer pair / program exact | 73,0% | **74,5%** | +1,5 điểm % |
| Execution | 82,0% | **83,5%** | +1,5 điểm % |
| Accepted coverage | 88,0% | **88,5%** | +0,5 điểm % |
| Accepted execution precision | 89,20% | **89,83%** | +0,63 điểm % |
| Gold generated recall | 77,5% | **83,0%** | +5,5 điểm % |
| Gold eligible recall | 71,5% | **74,5%** | +3,0 điểm % |
| Trace coverage | 88,5% | **100,0%** | +11,5 điểm % |

Theo operation, SUBTRACT execution tăng 70% → 75% và AVERAGE tăng 75% → 85%; các
operation phổ biến khác không giảm. `runtime_safety_gate = true`.

Hai giới hạn chưa đạt release gate:

- Pipeline gold rejection tăng 7,74% → 10,24%. Nguyên nhân còn lại tập trung ở
  tri-state role verification (`UNKNOWN` đang bị xem như hard mismatch) và unit
  algebra cho MULTIPLY/DIVIDE, không phải thiếu gold trong tài liệu.
- Runtime validation chậm hơn V12.3 khoảng 1,64 lần do số candidate trung bình
  tăng 50,91 → 74,23. Vì vậy beam mở rộng phù hợp trước hết để tạo cache/reranker;
  chưa bật mặc định ở demo cho đến khi pruning và accepted precision đạt 90%.

Feature flag thử nghiệm:

```bash
TYPED_PROGRAM_SYNTHESIS=1 FINANCIAL_IDENTITY_CATALOG=1 \
  CANDIDATE_RECALL_V12_4=1 bash scripts/start_demo.sh
```

V8 SHA-256 không đổi; `test.jsonl` và blind không được đọc. Báo cáo so sánh chuẩn:
`reports/document_hrm_V12_4_vs_V12_3_validation.json`.

### V12-B — tri-state pruned safe-residual reranker

V12-B dùng V12.4 làm candidate generator và vẫn giữ V8 đóng băng. Không fine-tune
semantic encoder, operation head hoặc pointer head trong giai đoạn này.

Pipeline đã triển khai:

1. Gộp `train.jsonl` và danh sách nhóm của `validation.jsonl`, sau đó loại toàn bộ
   company/report group xuất hiện trong validation. Trên dữ liệu hiện tại, 200 ID
   validation thuộc 7 nhóm làm 400/1.600 mẫu bị niêm phong; development còn 1.200
   mẫu, chia thành 920 train và 280 calibration với `group_overlap = 0`. Cả hai
   phần đều có đủ 8 operation chính.
2. Verifier cache dùng ba trạng thái độc lập với hard runtime gate:
   - `INVALID`: mâu thuẫn operation/period/entity/unit/structure rõ ràng, loại bỏ;
   - `UNKNOWN`: thiếu metadata hoặc fuzzy role match chưa đủ chắc chắn, giữ lại;
   - `VALID`: có proof phù hợp, giữ lại.

   `UNKNOWN` không tự động là negative. Positive/negative chỉ lấy từ canonical
   gold, grounded-equivalent và execution supervision. Mỗi candidate vẫn lưu
   `runtime_eligible`; candidate UNKNOWN không được tính accepted coverage.
3. Diversity pruning phân tầng theo operation, tri-state, generation source và
   first pointer; luôn giữ legacy winner. Beam mặc định tối đa 64. Cache generation
   dừng ngay nếu `pruning_gold_loss_count > 0`, có candidate INVALID được giữ, hash
   không khớp hoặc record count sai.
4. Cache có manifest chứa SHA-256 của V8, train, validation và hai tệp candidate.
   Cache khác model/data/split không được âm thầm tái sử dụng. Quá trình sinh cache
   được chia thành shard 64 mẫu; mỗi shard có fingerprint dữ liệu, SHA-256 record và
   audit riêng. Nếu tiến trình bị ngắt, lần chạy sau chỉ tái sử dụng shard đã xác minh
   hợp lệ rồi tiếp tục từ shard chưa hoàn thành, không phải sinh lại từ đầu.
5. Kết quả verify của các chương trình trùng nhau giữa V12.3 và beam V12.4 được
   memoize theo `(program, identity_constraint_required)`. Việc này chỉ loại tính toán
   verifier lặp lại, không thay đổi candidate, nhãn tri-state hoặc quyết định runtime.
6. Reranker tối ưu listwise loss + hardest-negative pairwise margin; operation
   sampling được cân bằng. Confidence head và threshold được học riêng trên
   calibration company/report-disjoint.
7. Override margin được calibration theo từng operation. Một override chỉ hợp lệ
   khi không làm giảm program exact, execution, runtime-eligible coverage hoặc
   accepted precision so với legacy symbolic winner.
8. Sidecar ghi các runtime flag bắt buộc. Runtime từ chối attach nếu quên bật typed
   synthesis, financial identity catalog hoặc candidate recall V12.4.

Kiểm tra split nhanh, không load V8:

```bash
PYTHONPATH=src venv/bin/python \
  scripts/train_document_hrm_v12b_reranker.py --split-audit-only
```

Sinh cache và kiểm tra pruning trước khi train:

```bash
bash scripts/train_document_hrm_v12b_reranker.sh --cache-only
```

Train sidecar; nếu cache hợp lệ, lệnh tự tái sử dụng cache:

```bash
bash scripts/train_document_hrm_v12b_reranker.sh
```

Đánh giá validation-only so trực tiếp với V12.4:

```bash
bash scripts/evaluate_document_hrm_v12b.sh
```

Chỉ bật demo V12-B nếu `reranker_release_gate = true`, accepted execution precision
đạt ít nhất 90%, coverage không giảm và không operation phổ biến nào regression.
Blind/test tiếp tục bị cấm trong toàn bộ train, calibration và model selection.

### V12-B.1 — supported tri-state + safe confidence residual

V12-B.1 sửa hai lỗi thiết kế được phát hiện từ ablation V12-B mà không fine-tune
V8 và không sinh lại candidate cache:

1. `VerificationResult.state` vẫn là kết luận runtime bảo thủ, nhưng cache có thêm
   `training_support_state`. Một chương trình không còn bị ép thành `UNKNOWN` chỉ vì
   metadata phụ như `same_row`, `graph_*` hoặc constraint không áp dụng bị thiếu.
   `INVALID` và mọi hard rejection vẫn không được nới lỏng.
2. Cache feature schema v2 được migrate có kiểm chứng sang v3 trong RAM. Trên cache
   hiện tại, 2.849/19.611 candidate train và 658/5.467 candidate calibration nhận
   được bằng chứng `VALID`; không cần chạy lại frozen V8.
3. Ranking no-regression được đánh giá không threshold. Confidence được so với
   legacy tại cùng target precision thay vì so coverage có threshold với coverage
   legacy 100% không threshold.
4. V12.3 reference winner được giữ nếu reranker chỉ được gọi nhưng không thực sự
   vượt override margin. Confidence sidecar cũng chỉ có quyền thay confidence V8
   khi có một override thật.
5. A/B validation chạy V12.4 và V12-B.1 với cùng code/runtime flags. Kết quả cuối:
   operation 98,5%, pointer/program 74,5%, execution 83,5%, accepted coverage
   88,5% và accepted execution precision 89,83% ở cả hai cấu hình. Mọi delta bằng
   0; latency sidecar tăng khoảng 1,4%.

Kết luận: `runtime_safety_gate = true` nhưng `reranker_release_gate = false` vì
precision validation chưa đạt 90%. V12-B.1 là ablation an toàn, chưa phải cải tiến
reasoning; demo tiếp tục dùng V8 + runtime V12.4. Không chỉnh threshold theo 200 mẫu
validation và không mở blind test.

### Kế hoạch tiếp theo — V12-B.2 generalization-first (IN PROGRESS)

**Trạng thái:** đang triển khai. Phase A, gate sống sót candidate của Phase B và
supervision Phase C đã hoàn tất trên development company/report-disjoint; chưa train
V12-B.2, chưa chạy lại validation niêm phong và chưa mở blind/test. Phase D–E vẫn
là kế hoạch.

#### 1. Chẩn đoán đã đóng băng

- V12-B.1 bằng V12.4 trên toàn bộ metric validation: program/pointer 74,5%,
  execution 83,5%, coverage 88,5%, accepted precision 89,83%.
- Trace/cache mềm giữ được gold program khoảng 95% trên validation, nhưng
  `gold_program_generated_recall` chỉ 83% và `gold_eligible_program_recall` chỉ
  74,5%. Khoảng cách chính nằm giữa candidate có mặt, verifier/runtime eligibility
  và winner cuối, không nằm ở operation head (98,5%).
- Cache train/calibration có rerankable recall trên 97%, cao hơn đáng kể so với
  báo cáo company-disjoint. Đây là dấu hiệu distribution shift theo công ty,
  nguồn báo cáo hoặc layout; không được tiếp tục tối ưu bằng cách nhìn từng lỗi
  validation.
- Reranker chỉ học được finite override margin cho `ADD`, nhưng validation không có
  override làm tăng accuracy. Confidence head không được phép thay confidence V8
  khi chương trình không thực sự bị override.

#### 2. Mục tiêu V12-B.2

Xây một verifier/reranker có thể tạo **correct override trên công ty và layout chưa
thấy**, trong khi V8 vẫn đóng băng. Không fine-tune HRM, không tăng catalog công thức
và không mở rộng DSL trong giai đoạn này.

Luồng dự kiến:

```text
Frozen V8 + V12.4 candidate generator
                ↓
Question-conditioned applicability mask
                ↓
Tri-state verifier trên required constraints
                ↓
Canonical multi-positive supervision + typed hard negatives
                ↓
Operation-aware listwise reranker
                ↓
Conditional override confidence + V8 fallback
```

#### 3. Phase A — applicability-aware verifier

Thay heuristic `training_support_state` bằng proof contract tường minh:

1. Planner sinh ba tập constraint cho từng câu:
   - `required`: bắt buộc phải có bằng chứng;
   - `optional`: dùng làm feature nhưng thiếu không kéo kết luận về `UNKNOWN`;
   - `not_applicable`: không được đưa vào mẫu số proof completeness.
2. Kết luận tri-state chỉ dựa trên `required`:
   - `INVALID`: có ít nhất một required constraint bị mâu thuẫn tường minh;
   - `UNKNOWN`: không có mâu thuẫn nhưng còn required constraint thiếu metadata;
   - `VALID`: mọi required constraint đều được chứng minh.
3. Required constraints theo họ câu hỏi:
   - `SELECT`: operation, metric, period nếu được hỏi và entity/scope nếu được hỏi;
   - temporal `SUBTRACT/PERCENT_CHANGE/AVERAGE`: same metric, đủ hai period và đúng
     temporal direction;
   - `DIVIDE/RATIO_PERCENT`: đúng numerator/denominator role, unit compatibility và
     denominator khác 0;
   - `MULTIPLY`: đúng operand role và unit algebra, đặc biệt
     `share_count × currency_per_share`;
   - `ADD`: explicit operand roles hoặc một financial identity active có proof;
   - mọi operation: execution hợp lệ và không reuse pointer trái phép.
4. Lưu thêm `required_valid_fraction`, `required_unknown_fraction`,
   `optional_support_fraction` và danh sách proof obligations vào trace/audit.
5. Không thay đổi hard runtime rejection cho đến khi audit chứng minh gold false
   rejection giảm trên development company-disjoint.

Gate Phase A:

- Gold verifier retention trên development ≥95%.
- Explicit contradiction precision ≥95% qua manual audit phân tầng.
- Không operation chính nào mất quá 1 điểm % eligible-gold recall.
- `VALID/UNKNOWN/INVALID` đều có mẫu; không trạng thái nào suy biến do constraint
  không áp dụng.

Kết quả Phase A development-only:

- Proof contract `required/optional/not_applicable` đã thay heuristic
  `training_support_state`; hard runtime rejection của V12.4 không đổi.
- Cache Top-64 mới lưu `required_valid_fraction`, `required_unknown_fraction`,
  `optional_support_fraction`, proof obligations và code hash của candidate/verifier
  pipeline; cache cũ không được tái sử dụng khi code hash thay đổi.
- Split có 920 mẫu/138 group train và 280 mẫu/20 group calibration, overlap = 0;
  200 validation ID và toàn bộ group trùng validation đã bị loại.
- Gold verifier retention đạt 739/739 = 100% trên train và 194/195 = 99,49% trên
  calibration. Mẫu bị giữ `INVALID` còn lại có question/gold tự mâu thuẫn giữa
  `31/12/2025` và cột `Số đầu năm`; vẫn được tính là false contradiction khi chấm
  gate bảo thủ.
- Manual audit phân tầng kiểm toàn bộ 24 invalid obligation có execution đúng và
  thêm 2 mẫu ổn định cho mỗi trong 11 constraint: 45/46 contradiction đúng =
  97,83%.
- Gold recall@64, post-prune recall và từng operation bằng baseline cùng split;
  pruning gold loss = 0. Cả ba trạng thái đều có mẫu.
- `phase_a_gate = true`. Report:
  `reports/document_hrm_V12B2_phase_a_audit.json`. Chưa được chuyển sang train
  reranker cho đến khi Phase B chứng minh raw gold program recall@64 và verifier
  retention trên đủ 5 fold.

#### 4. Phase B — dữ liệu generalization và cache không rò rỉ nhóm

1. Tạo manifest group theo `company + report + source + layout_family + period`.
2. Từ development hiện có, tạo 5 fold company/report-disjoint cho reranker và
   calibration. V8 là frozen generator đã thấy train nên kết quả fold phải được ghi
   rõ là **reranker-disjoint**, không được tuyên bố là end-to-end unseen-generator.
3. Ưu tiên bổ sung một development set mới từ công ty/layout không nằm trong
   train, validation, test hoặc blind. Tập này dùng để phát triển, không dùng làm
   final blind.
4. Mỗi fold cache Top-64 và lưu riêng:
   - raw gold operand/program recall;
   - recall sau verifier;
   - số `no_trace`, `single_candidate`, `rerankable`;
   - false rejection theo constraint, operation, layout và entity mode.
5. Deduplicate theo semantic cell signature, không chỉ theo giá trị số; cùng kết quả
   do ngẫu nhiên không được biến thành positive.

**Kết quả Phase B (development only, 09/08/2026):**

- Manifest có 1.200 mẫu, 154 công ty và đủ 5 fold; overlap company, report và
  composite group đều bằng 0. Kết quả chỉ được gắn nhãn
  `reranker-disjoint-not-end-to-end-unseen-generator`.
- Lần audit đầu phát hiện fold 4 chỉ đạt raw gold recall 94,32%, nên không train
  reranker. Nguyên nhân hệ thống là dấu phẩy/liên từ nằm trong tên metric làm ADD
  bị hiểu thành SUM3, parser tách nhầm `tại` trong `tài sản`, và unit mệnh giá bị
  kế thừa nhầm thành share count.
- Sau sửa candidate generation, raw gold program recall trên 8 operation tăng từ
  934/960 = 97,29% lên 953/960 = 99,27%. Fold thấp nhất đạt 98,46%; verifier gold
  retention thấp nhất đạt 99,43%; cả hai gate 95% đều đạt và pruning gold loss = 0.
- ADD tăng 92,50% -> 100%; MULTIPLY tăng 90,83% -> 99,17%; sáu operation còn lại
  không giảm. Cache dùng frozen V8 và fingerprint mới bao gồm cả
  `program_synthesis.py`.
- Fold cache deduplicate theo operation + typed semantic cell signature: loại 333
  chương trình trùng nghĩa. Có 784 candidate chỉ trùng execution và không candidate
  nào trong số đó được nâng thành positive.
- Chưa có development set mới từ công ty/layout hoàn toàn ngoài train,
  validation, test và blind trong workspace. Vì vậy không được tuyên bố kết quả
  end-to-end unseen-generator. Report:
  `reports/document_hrm_V12B2_phase_b_audit.json`.

Nếu raw gold program recall@64 <95%, dừng kế hoạch reranker và quay lại candidate
generation. Nếu raw recall cao nhưng verifier retention thấp, chỉ sửa Phase A. Chỉ
train reranker khi gold program đã sống sót trong candidate pool.

#### 5. Phase C — canonical multi-positive supervision

Positive set của mỗi câu gồm:

- canonical gold program;
- commutative-equivalent sau canonicalization;
- grounded-equivalent có cùng metric, period, entity, unit và semantic cell
  signature.

`execution_only` hoặc accidental same-result không phải positive. Hard negatives
được phân loại và lấy cân bằng:

- sai năm/đầu kỳ-cuối kỳ;
- sai metric nhưng cùng bảng;
- sai entity/scope;
- đảo numerator-denominator hoặc temporal direction;
- sai unit/constant;
- duplicate pointer;
- HRM score cao nhưng proof contract không đầy đủ.

Mỗi operation chỉ được bật learned override nếu có tối thiểu 30 group train và 15
group calibration chứa cả positive lẫn hard negative. Operation thiếu dữ liệu tiếp
tục dùng V12.4 fallback; không dùng oversampling để giả tạo độ phủ nhóm.

**Kết quả Phase C (development only, 09/08/2026):**

- Supervision dùng đúng ba positive class được phép: canonical gold,
  commutative-equivalent và grounded-equivalent có cùng semantic cell signature.
  Semantic dedup vẫn lưu member program của mỗi equivalence class nên không làm mất
  nhãn multi-positive.
- Có 0 label contradiction và 0 `execution_only` bị nâng thành positive. Hard
  negative đã được gắn taxonomy theo period, metric cùng bảng, entity/scope,
  direction/role đảo, unit/constant, duplicate pointer và high-neural incomplete
  proof; không dùng cùng kết quả số làm bằng chứng positive.
- Dữ liệu multi-positive thật còn rất ít: chỉ 1 example duy nhất có hơn một positive
  program sau semantic grouping. Đây là giới hạn dữ liệu, không được bù bằng
  duplicate hoặc oversampling.
- Áp cứng gate 30 group train + 15 group calibration trên từng fold: chỉ
  `RATIO_PERCENT` đạt ở cả 5 fold. `ADD`, `AVERAGE`, `DIVIDE`, `MULTIPLY`,
  `PERCENT_CHANGE`, `SELECT` và `SUBTRACT` phải giữ V12.4 fallback. Không hạ gate
  để tối ưu benchmark.
- `phase_c_supervision_ready_limited_coverage = true`. Report:
  `reports/document_hrm_V12B2_phase_c_audit.json`; supervision sidecar:
  `cache/v12b2_phase_c/manifest.json`. Chưa train Phase D.

**Bổ sung gate provenance và metric canonical (09/08/2026):**

- Exact program được tách thành ba lớp: `raw` giữ nguyên thứ tự serialization,
  `canonical` bỏ khác biệt thứ tự ở phép giao hoán, và `grounded semantic` chỉ nhận
  execution-equivalent khi operation cùng typed evidence signature. Không dùng
  execution-only accidental same-result làm program đúng.
- Phase-C schema v2 kiểm tra gold bằng typed evidence trước khi tạo positive. Với
  phép nhân giá trị theo mệnh giá, hai operand bắt buộc chứng minh được đúng
  `share_count × currency_per_share`; nhãn dòng ghi mệnh giá nhưng tagged cell là
  số ở cột lân cận bị loại khỏi supervision.
- Audit hai nguồn development có 2.421 mẫu: 2.344 eligible, 77 bị loại bởi hard
  contradiction; 29 mẫu MULTIPLY không chứng minh được unit algebra. Mojibake được
  báo cáo riêng là lỗi encoding/dữ liệu, không bị coi là OCR error và không tự động
  biến thành negative.
- Dựng lại 5 fold vẫn chỉ cho phép `RATIO_PERCENT` học override. MULTIPLY có 8–16
  calibration group/fold và SUBTRACT có 2–13, chưa đạt gate 15 ở mọi fold; không
  oversample và không fine-tune hai operation này.
- Validation 200 mẫu sau khi khóa rule tổng quát: operation 99,0%, canonical program
  80,5%, raw program 73,5%, execution 83,0%, accepted coverage 87,0% và accepted
  execution precision 89,08%. AVERAGE canonical tăng nhờ metric đúng; ADD đạt
  program 95%/execution 100%. SUBTRACT còn 70% và accepted precision vẫn dưới 90%,
  nên `phase_d_release_gate = false`; không chạy blind/test và không hạ threshold.
- Artifacts: `reports/program_supervision_integrity_development_v2.json`,
  `reports/document_hrm_V12B2_phase_c_v2_audit.json`,
  `cache/v12b2_phase_c_v2/manifest.json` và
  `reports/current_best_validation_phase_c_fixes_v3.json`.

### Kế hoạch cải thiện SUBTRACT — Direction Contract

Phạm vi duy nhất của nhánh tiếp theo là xác định đúng metric/pointer và chiều phép
trừ. Không thay đổi CAS/executor để che lỗi grounding, không tối ưu theo một báo cáo
hoặc một câu validation.

#### Trạng thái đã hoàn thành

- [x] Tách metric `raw`, `canonical` và `grounded semantic`; SUBTRACT tiếp tục giữ
  thứ tự operand vì không giao hoán.
- [x] Phân rã 6 lỗi SUBTRACT hiện tại: 4 lỗi đảo đúng hai operand, 2 lỗi chọn sai
  pointer/metric và 0 lỗi có gold operand âm.
- [x] Chốt policy: không dùng `abs()` mặc định, không canonicalize `A-B` thành
  `|A-B|`, không xem `A-B` và `B-A` là cùng program.
- [x] Sửa directional contract cho PERCENT_CHANGE có text đơn vị xen giữa năm,
  ví dụ `từ Năm 2024 (VND) đến Năm 2025 (VND)`; regression đạt lại 80%.
- [x] Phase-C v2 đã loại gold supervision có hard contradiction và giữ nguyên gate
  30 train group + 15 calibration group trên từng company-disjoint fold.
- [x] Xác nhận SUBTRACT hiện chưa đủ dữ liệu để fine-tune: 2–13 calibration
  group/fold; Phase D cho SUBTRACT vẫn khóa.

#### S1 — Biên dịch ý định chiều phép trừ

- [x] Thêm `DirectionIntent` dùng chung trong `QuestionPlan` với năm trạng thái:
  `NEW_MINUS_OLD`, `LEFT_MINUS_RIGHT`, `DECREASE_AMOUNT`, `ABSOLUTE_GAP` và
  `AMBIGUOUS`.
- [x] Viết rule theo cấu trúc ngôn ngữ, không theo công ty/số liệu: `từ A đến B`,
  `cuối kỳ so với đầu kỳ`, `A thấp hơn B bao nhiêu`, `A trừ B`, và câu hỏi nói rõ
  `độ lệch tuyệt đối`.
- [x] Câu chỉ nói `chênh lệch giữa A và B` nhưng không xác định signed/magnitude phải
  nhận `AMBIGUOUS`; không suy chiều bằng độ lớn con số.
- [x] Thêm test cho Unicode chuẩn, mojibake có thể phục hồi, năm/đơn vị khác nhau và
  hoán vị thứ tự candidate.

Kết quả S1: 30 test planner/direction/audit đạt; regression planner, grounding,
synthesis và supervision đạt 166 test. Contract mới chỉ đánh giá ý định/metadata,
chưa thay đổi executor, CAS hoặc quyết định accept ở runtime.

#### S2 — Một contract cho toàn bộ pipeline

- [x] Dùng cùng `DirectionIntent` để định hướng neural baseline, pair beam, typed
  synthesis, financial-identity program và winner cuối; không để mỗi nhánh tự chọn
  chiều riêng.
- [x] Giới hạn `_orient_subtract_magnitude` chỉ cho `ABSOLUTE_GAP` hoặc
  `DECREASE_AMOUNT` được chứng minh rõ; signed change không được tự đổi thành số dương.
- [x] Verifier đánh `INVALID` cho cặp đảo chiều khi contract rõ, `UNKNOWN` khi thiếu
  metadata và không biến `AMBIGUOUS` thành một chương trình được accept.
- [x] Executor tiếp tục thực thi đúng `A-B`; xử lý số âm bằng metadata dấu/vai trò kế
  toán ở nhánh riêng, không lấy `abs()` tự động.

Kết quả S2: bỏ hoàn toàn fallback ép `high-low`; `DECREASE_AMOUNT` dùng thứ tự
old/prior trừ new/current khi metadata chứng minh được, còn `ABSOLUTE_GAP` bị từ chối vì
DSL chưa có toán tử tương ứng. Neural baseline và pair beam gọi cùng orientation API;
typed/identity synthesis dùng cùng order trong plan; verifier chặn candidate đảo chiều
trước winner/reranker. 165 test trực tiếp và 41 test identity/reranker/supervision đạt.

#### S3 — Audit và supervision development-only

- [x] Tạo audit Direction Contract trên các nguồn development, chủ động từ chối path
  chứa `test`, `final`, `blind` hoặc frozen holdout.
- [x] Báo cáo riêng: đúng chiều, đảo chiều, sai metric, thiếu period, ambiguous wording,
  negative presentation và signed negative balance.
- [x] Loại câu có header/narrative tràn vào endpoint hoặc direction mâu thuẫn khỏi
  positive supervision; không sửa gold để khớp prediction.
- [x] Chỉ sinh reversed-pair hard negative khi direction đã được chứng minh; không
  biến cặp đảo của câu `AMBIGUOUS` thành negative giả.

Kết quả S3: integrity audit trên 2.221 development record loại 127 positive có hard
contradiction; direction gồm 44 ambiguous + 19 reversed. Hai nguồn hiện tại không có
gold endpoint được đánh dấu header/narrative, nhưng gate và regression tương ứng đã được
khóa. Phase-C v3 được dựng từ đúng 1.200 ID development, không nạp sealed validation và
loại 200 record ngoài manifest; 115 record fold-expanded bị loại khỏi positive. Chỉ 10
reversed-pair negative fold-expanded được giữ vì direction đã chứng minh; không có reverse
không chứng minh nào bị biến thành generic negative. SUBTRACT vẫn khóa do calibration
group theo fold chỉ đạt 2–12, dưới gate 15. Regression tổng hợp đạt 222 test.

#### S4 — Gate đánh giá trước Phase D

- [x] Đánh giá 5 fold theo company/report với `direction_accuracy`, pointer exact,
  signed execution, coverage và accepted precision theo operation.
- [x] Kiểm tra verifier gold retention ≥95%, không operation phổ biến nào giảm quá
  1 điểm %, và SUBTRACT tối thiểu 30 train group + 15 calibration group ở mọi fold.
  Hai gate đầu đạt; group gate không đạt.
- [x] Vì thiếu group, tiếp tục frozen fallback và thu thập/annotation thêm công ty;
  cấm oversampling để vượt gate.
- [ ] Chỉ sau khi code/config/hash và development gate đã khóa mới được đánh giá một
  lần trên validation; không điều chỉnh rule/threshold sau lần đánh giá đó.
- [x] Giữ Phase D cho SUBTRACT ở trạng thái khóa vì chưa đạt toàn bộ gate. Blind/test
  vẫn đóng cho đến khi release gate chung đạt.

Kết quả S4 development-only trên 930 mẫu integrity-eligible thuộc 5 fold disjoint:
signed execution 71,94% (+0,22 điểm % so với frozen winner), coverage 83,23% và
accepted precision 86,43%. Gold retention thấp nhất theo fold là 98,84%; không operation
phổ biến nào giảm quá 1 điểm %. Trên 229 câu có direction, 111 câu được chấp nhận, đúng
109 câu: direction accuracy có điều kiện 98,20% nhưng direction coverage chỉ 48,47%.
Riêng SUBTRACT có 114 mẫu: signed execution 31,58%, coverage 42,11%, accepted precision
75,00%, direction accuracy có điều kiện 97,92%. Số group train/calibration của năm fold
lần lượt là 24/12, 28/8, 29/7, 29/7 và 34/2, nên không đạt gate 30/15. Gate tổng thể cũng
không đạt accepted precision 90% và coverage 88,5%; `phase_d_release_gate = false`.
Báo cáo tái lập từ Top-64 candidate cache cố định, tính lại rejection cho operation có
direction theo contract hiện tại và ghi hash code/artifact tại
`reports/subtract_direction_s4_development_folds_v1.json`. Không nạp global validation,
không mở blind/test, không fine-tune và không thay threshold. Regression trực tiếp liên
quan planner/direction/synthesis/supervision/pipeline đạt 250/250 test. Bộ test legacy
toàn phần chưa thể nạp `tests/test_generate_document_derived_programs.py` vì file nguồn
`scripts/generate_document_derived_programs.py` hiện không tồn tại; không tính lỗi hạ tầng
này thành lỗi mô hình và không tự phục hồi ngoài phạm vi S4.

Audit `reports/subtract_direction_contract_development_v2.json` dùng 479 gold directional
development: 323 đúng chiều, 19 đảo chiều, 44 ambiguous, 40 thiếu period và 53 chưa đủ
metadata/ý định. Coverage chứng minh được chiều là 71,40%; accuracy trong phần chứng minh
được là 94,44%. Audit phát hiện và loại 200 record `synthetic-validation-*` nằm trong
`data/train.jsonl`; không dùng chúng để lập rule hay tính kết quả development. Verifier
direction loại đúng 63 trường hợp gồm 44 ambiguous + 19 đảo chiều; 40 trường hợp thiếu
period giữ `UNKNOWN`.

#### S4.1 — Bổ sung cohort new-company để đóng group gate

- [x] Khóa cohort đầu 40 công ty/53 PDF development-only, overlap company/report/hash
  với development/protected bằng 0; không dùng câu hỏi, số liệu hoặc model output để chọn.
- [x] Sinh và render QA 19 group OCR + 11 group native; chỉ giữ record có cùng row/table,
  hai endpoint duy nhất, `NEW_MINUS_OLD` được chứng minh và reversed-pair negative có proof.
- [x] Sửa các lỗi parser theo cấu trúc tổng quát: hàng ghép ngang có value đứng trước row,
  note code bị lệch vào cột period, amount mang metadata cột tỷ trọng, ratio nhận amount rất
  lớn, `công ty cổ phần` bị hiểu nhầm thành unit `shares`, và unit tiền bị từ `hệ số` của bảng
  bên cạnh ghi đè. Không có rule theo ticker, page hay con số benchmark.
- [x] Vì 30 group mới chỉ nâng tổng từ 36 lên 66, khóa cohort thứ hai độc lập 40 công ty/70
  PDF; audit overlap = 0 và 70/70 source hash khớp. Nhánh native/hybrid sinh thêm 16/16 group
  được visual QA và integrity audit chấp nhận.
- [x] Gộp tổng cộng 46 group mới vào năm fold cũ mà không đổi assignment/candidate cache
  cũ. Company/report/group overlap ở cả năm fold đều bằng 0. SUBTRACT train/calibration
  sau mở rộng lần lượt là `65/17`, `65/17`, `66/16`, `66/16`, `66/16`; group gate
  `30/15` đã đạt ở cả năm fold, không oversample.
- [x] Dựng Top-64 candidate cache và Phase-C supervision cho 46 record mới, sau đó chạy
  lại đúng S4 development-only. Chưa được fine-tune hoặc mở validation chỉ vì group gate
  vừa đạt; precision/coverage/gold-retention gate vẫn phải được xác nhận trên artifacts mới.

S4.1 đã hoàn tất trên cache Top-64 v5 đồng nhất cho toàn bộ 1.246 mẫu development; không
ghép cache khác candidate-pipeline hash. Cả 46/46 record mới có candidate record và proven
reversed-pair supervision; reversal probe chỉ được sinh khi shared direction contract chứng
minh chiều trực tiếp đúng và chiều đảo sai, luôn `INVALID`, `runtime_eligible=false`. Phase-C
bật SUBTRACT ở cả năm fold với train/calibration group lần lượt `103/21`, `97/27`, `98/26`,
`99/25`, `99/25`; label contradiction = 0, execution-only promoted = 0 và reversed pair
withheld = 0.

Kết quả S4 development-only trên 976 mẫu candidate-eligible: signed execution 84,94%,
coverage 96,00%, accepted precision 88,47%, verifier gold retention 99,59%; direction
coverage 98,91% và direction accuracy có điều kiện 98,90%. Raw gold recall thấp nhất theo
fold 98,95%, verifier retention thấp nhất 97,06%; group gate 30/15 đạt ở cả năm fold và
không operation phổ biến nào giảm quá 1 điểm %. Tuy nhiên accepted precision chưa đạt 90%,
nên `phase_d_release_gate = false`: không fine-tune, không mở validation/blind và không hạ
threshold. Báo cáo khóa tại `reports/subtract_direction_s4_development_folds_v2.json`.

Pipeline cũng đã sửa ba lỗi tổng quát phát hiện trong bước này: manifest fold development-only
không còn giữ đường dẫn sealed validation; calibration shard được canonicalize theo
`example_id` để độc lập Python hash seed; reconstruction gold-invalid đọc per-shard audit
thay vì diagnostics aggregate bị cap. Không có rule theo ticker, page, số liệu hay câu hỏi đơn.

Artifacts đã khóa: `data/subtract_new_company_development_reviewed_v1.jsonl`,
`data/subtract_native_development_reviewed_v1.jsonl`,
`data/subtract_native_development_reviewed_batch2_v1.jsonl`, fold mở rộng
`data/document_hrm_v12b2_reranker_disjoint_folds_subtract_v4.json` và audit
`reports/subtract_reviewed_cohort_fold_extension_v3.json`. Tổng 46/46 record mới có
`direction_supported`, không có integrity issue và không có reversed pair thiếu proof.
Regression liên quan đạt 320/320 test. Test legacy
`tests/test_audit_pointer_supervision.py` vẫn không thể nạp vì
`scripts/audit_document_derived_programs.py` không tồn tại; không khôi phục file archive
ngoài phạm vi nhánh này.

**Bước thực hiện ngay tiếp theo:** giữ frozen fallback và phân tích development-only các
false accept còn lại theo operation/proof obligation để tìm lỗi cấu trúc dùng chung. Không
fine-tune, không thêm `SUBTRACT_ABS`, không đọc validation/blind và không thay threshold;
Phase D chỉ được mở sau khi accepted precision đạt 90% cùng toàn bộ gate S4 còn lại.

#### S4.2 — Audit false accept SELECT/ADD theo proof obligation

- [x] Replay đúng năm calibration fold development company/report-disjoint từ cache Top-64
  v5; chỉ lấy winner được accept nhưng signed execution sai của gold `SELECT`/`ADD`.
- [x] Khóa phạm vi: không nạp global/sealed validation, blind/test/holdout; không train,
  không fine-tune và không thay threshold.
- [x] Phân loại theo required proof state và lỗi cấu trúc metric/period/role/entity/unit/
  table/subtable/logical-cell identity; không lập rule theo công ty, trang hoặc giá trị.

Audit có 42 false accept: SELECT 24/116 accepted và ADD 18/111 accepted, chiếm
42/108 = 38,89% toàn bộ false accept của S4.1. Chúng trải trên 35 source-page. Phân hoạch
proof contract: 26 winner sai được đánh `VALID`, 16 winner sai có required proof `UNKNOWN`,
và 0 winner có required proof `INVALID`; do đó không có explicit-contradiction leak.

Với ADD, 14/18 lỗi cùng mất cả intent lẫn hai operand role: planner trả operation/roles rỗng,
`operation_match=UNKNOWN` và `add_grounding=UNKNOWN`, nhưng tri-state runtime vẫn cho phép
candidate thắng. Nguyên nhân tổng quát là suffix period tương đối chưa được tách khỏi role và
fixed financial-anchor vocabulary không phủ nhiều row label hợp lệ. Bốn ADD lỗi còn lại được
đánh `VALID`: ba collision cùng physical semantic cell và một sai period role vì ADD chưa yêu
cầu period proof.

Với SELECT, 22/24 lỗi được đánh proof-complete. Contract hiện chủ yếu chứng minh metric và
năm, nên không phân biệt `01/01` với `31/12`, `Kế hoạch` với `Thực hiện`, hoặc relative period
khi không có năm số; unit family và reporting entity/scope cũng không phải required trong các
trường hợp sai tương ứng. Có 10 SELECT và 10 ADD false accept mang cùng typed signature nhưng
khác giá trị; 14/42 vẫn va chạm ngay cả khi tính đủ source/page/table/subtable/row/column/entity/
period/unit. Đây là ambiguity ở evidence identity, không phải ranking signal an toàn.

Chưa sửa runtime trong S4.2. Hướng hợp lệ tiếp theo chỉ là sửa tổng quát planner/proof/evidence:
tách period suffix khi compile ADD roles, thêm logical-cell uniqueness/ambiguity obligation,
biểu diễn SELECT column intent (exact date, relative role, plan/actual), và yêu cầu unit/entity
khi được hỏi. Sau đó phải replay lại đúng S4 development-only; Phase D vẫn khóa và gate chung
giữ nguyên 88,47%. Báo cáo: `reports/select_add_false_accepts_development_v1.json` và
`reports/select_add_false_accepts_development_v1.md`; script tái lập:
`scripts/analyze_select_add_false_accepts.py`.

#### S4.3 — Kế hoạch structural repair và counterfactual safety gate

**Mục tiêu:** nâng accepted execution precision development-only từ 88,47% lên ít nhất
90% bằng proof/grounding tổng quát, không tối ưu threshold và không dùng model mới để che lỗi
evidence. Baseline khóa là 976 mẫu candidate-eligible, 937 accepted, 829 signed-execution
correct và 108 false accept.

Nếu giữ nguyên 829 câu đúng thì tối đa chỉ được accept 921 câu để đạt 90%, tức phải loại ít
nhất 16 false accept. Gọi `F` là số false accept bị loại và `T` là số true accept bị loại,
điều kiện precision là `9F - T >= 143`; nếu chỉ loại đúng 16 false accept thì được phép mất
tối đa một true accept. Coverage có dư địa, nhưng không được dùng dư địa đó để abstain rộng
không có proof.

##### S4.3-A — Counterfactual audit trên toàn bộ accepted set

- [ ] Replay cùng frozen Top-64 v5 trên cả true accept và false accept, không chỉ 42 lỗi
  SELECT/ADD. Lập ma trận `VALID/UNKNOWN/INVALID × correct/incorrect` theo operation, fold,
  company/report, layout và extraction method.
- [ ] Với từng policy giả định, tính độc lập `F`, `T`, precision, coverage, signed execution,
  pointer exact, verifier gold retention và số abstain mới. Không ghép nhiều policy trước khi
  có kết quả riêng và giao nhau của chúng.
- [ ] Audit các policy: proof-complete cho explicit ADD; exact period/date/column qualifier;
  SELECT requested unit; entity/scope; logical-cell uniqueness; conflict khác giá trị trong
  cùng semantic/physical signature.
- [ ] Báo cáo collision trên toàn bộ accepted/gold pool để biết rule ambiguity loại bao nhiêu
  câu đúng. Không suy safety từ false-accept-only audit.
- [ ] Giữ counterfactual là replay eligibility/proof, không thay confidence threshold, neural
  score, candidate order hoặc checkpoint.

Mỗi structural policy chỉ được đưa vào runtime khi đồng thời đạt trên năm fold development:

1. không hard-code example ID, ticker, company, page, year hoặc numeric value;
2. raw gold program recall thấp nhất theo fold ≥95% và pruning gold loss = 0;
3. verifier gold retention thấp nhất theo fold ≥95%; không accepted candidate nào có required
   proof `INVALID`;
4. không operation phổ biến nào giảm signed execution quá 1 điểm %;
5. overall coverage ≥88,5%; precision không được tăng chỉ nhờ loại hàng loạt true answer;
6. bản cấu hình hợp nhất cuối đạt accepted precision ≥90% và thỏa `9F - T >= 143`;
7. kết quả giữ nguyên khi hoán vị candidate và khi đổi Python hash seed.

##### S4.3-B — Repair ADD intent và operand-role proof

- [ ] Tách suffix period/date khỏi cả hai role trước khi phân tích `Tổng A và B`, gồm explicit
  year/date, `đầu/cuối năm`, `năm nay/trước`, và header OCR dính liền đã chuẩn hóa được.
- [ ] Không dùng fixed financial-anchor vocabulary làm điều kiện duy nhất để sinh ADD. Parser
  tạo role hypothesis theo cấu trúc; typed synthesis chỉ giữ hypothesis khi hai role có beam
  evidence độc lập và role match được chứng minh.
- [ ] Không tách liên từ nằm trong một metric chuẩn như `doanh thu bán hàng và cung cấp dịch
  vụ`. Trường hợp có nhiều cách tách phải giữ ambiguity, không tự chọn bằng neural score.
- [ ] Với explicit two-role ADD, bắt buộc `operation_match`, đủ hai `operand_roles`,
  `add_grounding`, pointer uniqueness, requested period nếu có, entity/scope và unit
  compatibility. Thiếu proof giữ `UNKNOWN` để audit nhưng không được materialize thành câu
  trả lời nếu counterfactual gate chứng minh policy strict an toàn.
- [ ] Thêm regression theo biến thể Unicode/mojibake phục hồi được, period suffix, metric chứa
  dấu phẩy/liên từ, hoán vị candidate, duplicated row và parallel subtable.

##### S4.3-C — SELECT column-intent proof

- [ ] Mở rộng `QuestionPlan` bằng qualifier có kiểu: exact date endpoint, relative period role,
  numeric year, `plan/actual`, requested unit family và requested entity/scope. Không gộp tất
  cả thành một tập year.
- [ ] Tách proof obligation `period_coverage` thành các obligation áp dụng theo câu hỏi:
  `requested_year`, `exact_date_or_endpoint`, `relative_period_role` và `column_qualifier`.
  `01/01/2025` không tương đương `31/12/2025`; `Kế hoạch 2025` không tương đương
  `Thực hiện 2025`.
- [ ] Với SELECT, metric đúng là cần nhưng chưa đủ. Requested unit và named/reporting scope trở
  thành required khi câu hỏi nêu hoặc evidence có đối chứng rõ; explicit mismatch là
  `INVALID`, metadata thiếu là `UNKNOWN`.
- [ ] Khi cùng metric/year xuất hiện ở nhiều panel/subtable, yêu cầu qualifier hoặc lineage đủ
  để chọn duy nhất. Không cho proof score/neural score thay thế bằng chứng column intent.
- [ ] Thêm đối nghịch tổng quát cho đầu/cuối kỳ, plan/actual, amount/percent/ratio,
  reporting/embedded entity và repeated metric across subtables.

##### S4.3-D — Logical-cell identity và table reconstruction

- [ ] Bổ sung lineage ổn định tới `EvidenceCell`: source hash, page, table/subtable, row/column
  ID, source span hoặc physical cell coordinate và extraction method. Cache/dedup fingerprint
  phải đổi khi schema này đổi.
- [ ] Hai evidence cùng logical identity và cùng normalized value được deduplicate có lưu member
  provenance. Nếu khác value, đánh `logical_cell_conflict`; không chọn một value bằng score.
- [ ] Chỉ abstain/reject vì conflict sau khi đã tính source span/physical coordinate để không
  gộp nhầm hai cell thật ở parallel panel. Conflict phải được audit phân tầng native/OCR,
  merged horizontal table và repeated header.
- [ ] Sửa table reconstruction ở tầng tổng quát cho value gắn nhầm row/column, panel song song
  và merged header. Không sửa gold hoặc thêm ngoại lệ theo PDF cụ thể.
- [ ] Kiểm tra uniqueness trên cả gold và winner; báo false-rejection theo operation/layout.

##### S4.3-E — Hardening hoạt động runtime

- [ ] Đặt từng repair sau feature flag độc lập và mặc định tắt cho đến khi pass gate:
  ADD strict-role proof, SELECT column intent và logical-cell conflict handling. Frozen
  V8 + V12.4 luôn là rollback path.
- [ ] Trace/API lưu question plan, required/optional obligation state, selected lineage,
  ambiguity/rejection reason, generation source, confidence source và code/cache hash.
- [ ] Khi abstain, UI chỉ hiển thị lý do an toàn ở mức người dùng và citation có thể kiểm tra;
  candidate bị reject chỉ nằm trong technical audit, không trở thành câu trả lời chính.
- [ ] Thêm metric vận hành theo operation/layout: proof-UNKNOWN rate, conflict rate, abstention
  reason, candidate count, cache hit/hash mismatch, latency p50/p95 và peak memory.
- [ ] Khóa latency baseline trước repair. Cấu hình release không được tăng p95 quá 10% nếu
  không có ablation chứng minh lợi ích và ghi rõ budget; proof/dedup phải được tính một lần và
  tái dùng, không recompute theo từng nhánh winner.
- [ ] CI/regression bắt buộc permutation invariance, deterministic fold replay, stale-cache
  rejection, protected-path refusal và API/UI không lộ rejected answer.

##### S4.3-F — Replay, lock và quyết định

- [ ] Chạy unit/regression theo từng repair, sau đó ablation theo thứ tự: ADD planner → SELECT
  qualifier → logical-cell identity → cấu hình hợp nhất. Không sửa nhiều tầng rồi chỉ báo một
  số overall.
- [ ] Replay đúng năm fold development với frozen checkpoint và threshold. Báo cáo overall,
  per-operation, per-fold, company/report/layout, collision cohort và latency.
- [ ] Nếu bất kỳ gate nào fail, giữ repair sau feature flag, quay lại frozen fallback và chỉ
  dùng development evidence để điều tra. Không mở validation để chọn giữa các repair.
- [ ] Nếu toàn bộ gate pass, khóa code/config/checkpoint/data/cache/report hash trước khi đánh
  giá đúng một lần trên sealed validation. Sau lần đó không chỉnh rule/threshold theo lỗi
  validation; nếu validation fail thì đóng cấu hình làm ablation và không chạy blind.

Artifacts dự kiến:

- `reports/select_add_counterfactual_s4_development_v1.json`;
- `reports/select_add_structural_repairs_development_ablation_v1.json`;
- `reports/select_add_runtime_operational_gate_v1.json`;
- `reports/select_add_s4_development_folds_v3.json`;
- một lock manifest chứa toàn bộ code/config/checkpoint/data/cache hash.

**Bước thực hiện ngay:** chỉ triển khai S4.3-A để đo counterfactual trên toàn bộ 227 accepted
SELECT/ADD và collateral impact trên đủ 976 mẫu. Chưa sửa planner/verifier/runtime trước khi
có ma trận true/false accept theo proof state và gate safety cho từng policy riêng.

#### S4.4 — Bổ sung supervision group-disjoint cho SELECT/ADD

Structural repair phải ổn định trước khi bổ sung dữ liệu; không thu thập cohort để khớp một
rule chưa khóa. Phase-C hiện chưa cho phép learned override SELECT/ADD ở cả năm fold:

- ADD train/calibration group: `59/16`, `62/13`, `58/17`, `61/14`, `60/15`;
- SELECT train/calibration group: `80/17`, `76/21`, `74/23`, `72/25`, `86/11`.

ADD thiếu calibration group ở fold 1 và 3; SELECT thiếu ở fold 4. Vì operation eligibility
đòi đạt `30/15` ở mọi fold nên cả hai tiếp tục V12.4 fallback.

- [ ] Khóa cohort development company/report/layout mới bằng tiêu chí xác định trước model
  output. Ưu tiên explicit two-role ADD có period và SELECT có repeated metric, exact date,
  plan/actual, unit/entity và parallel subtable.
- [ ] Human/visual QA evidence, program, unit, entity, period và physical lineage; conflict
  hoặc ambiguous annotation không được biến thành positive.
- [ ] Rebuild deterministic five-fold assignment, audit company/report/hash overlap = 0 và
  không oversample/duplicate để vượt group gate.
- [ ] Chỉ bật learned override cho operation đạt 30 train + 15 calibration group ở mọi fold,
  có positive và typed hard negative thật. Execution-only same-result không là positive.
- [ ] Nếu sau structural repair gold candidate vẫn sống sót nhưng winner sai mới xem ranking
  là bottleneck. Nếu gold chưa sinh/eligible, tiếp tục sửa synthesis/verifier, không train.

Phase D chỉ được mở sau khi cả S4.3 development release gate và S4.4 operation group gate của
operation định train đều đạt. V8 encoder, operation head và pointer head vẫn frozen trong
Phase D; fine-tune chỉ có thể là một kế hoạch V12-D riêng sau ablation chứng minh retrieval/
ranking bottleneck còn tồn tại trên unseen-company development.

#### 6. Phase D — operation-aware listwise reranker

Giữ shared feature trunk nhỏ và dùng operation-specific output head. Loss dự kiến:

\[
L = L_{multi\text{-}positive\ listwise}
  + \lambda L_{typed\ hard\ negative}
  + \mu L_{proof\ consistency}
\]

- Listwise loss xếp tổng xác suất của toàn bộ grounded-positive cao hơn negative.
- Pairwise/hard-negative loss chỉ so positive với negative cùng operation và cùng
  candidate set.
- Proof-consistency không học `UNKNOWN = sai`; chỉ phạt explicit contradiction.
- Model selection dùng mean company-disjoint fold execution rồi program exact;
  không chọn epoch theo validation 200 câu hiện tại.
- V8 semantic encoder, operation head và pointer head tiếp tục frozen.

#### 7. Phase E — confidence chỉ cho override

1. V8/V12.4 giữ nguyên confidence cho mọi legacy fallback.
2. Confidence head mới chỉ dự đoán xác suất execution đúng **của một override**.
3. Calibration dùng group-disjoint fold khác fold train và báo cáo:
   - precision–coverage curve;
   - ECE/Brier score;
   - số override đúng, sai và bị từ chối theo operation.
4. Operation thiếu ít nhất 15 override calibration hoặc không đạt target precision
   90% bị khóa margin vô hạn.
5. Không hạ threshold dựa trên validation chỉ để vượt 89,83% → 90%.

#### 8. Ablation và release gate

Thứ tự ablation bắt buộc:

| Cấu hình | Câu hỏi nghiên cứu |
|---|---|
| V12.4 frozen | Baseline |
| + applicability-aware verifier | Có giảm gold false rejection không? |
| + canonical multi-positive | Positive taxonomy có cải thiện ranking không? |
| + operation-aware reranker | Có tạo correct override ngoài ADD không? |
| + conditional confidence | Có tăng precision mà không mất coverage không? |

Gate để V12-B.2 được đánh giá một lần trên validation đã niêm phong:

- Company/report/layout group overlap = 0.
- Raw gold program recall@64 ≥95% và verifier gold retention ≥95%.
- Cross-fold execution tăng trung bình ≥2 điểm %, program exact tăng ≥1,5 điểm %.
- Correct override nhiều hơn wrong override ở mọi operation được bật.
- Không operation phổ biến nào giảm quá 1 điểm %.
- Accepted execution precision ≥90% và coverage ≥ baseline 88,5%.
- Latency ≤1,25× V12.4.
- Bootstrap 95% CI của execution delta không có suy giảm lớn; dùng McNemar khi đủ
  cặp discordant.

Nếu validation không đạt gate, đóng băng V12-B.2 làm ablation và không chạy blind.
Blind chỉ được mở sau khi code/config/hash đã khóa, release gate đạt và không còn
quyết định model nào dựa trên kết quả blind.

#### 9. Phạm vi bị hoãn

- V12-C multi-step DSL, CAGR, SHARE_OF_TOTAL và CAS/SMT.
- Mở rộng formula catalog ngoài identity chắc chắn.
- Fine-tune pointer head hoặc toàn bộ HRM.
- Tối ưu riêng SUM3/SUM4.

Chỉ xem xét pointer-head fine-tuning sau V12-B.2 nếu raw candidate recall vẫn là
bottleneck trên development unseen-company; khi đó phải tạo một kế hoạch và ablation
riêng, không trộn vào thí nghiệm reranker này.

### V12-C

- Typed constants và unit scaling có provenance.
- Multi-step DAG programs.
- `CHANGE`, `SHARE_OF_TOTAL`, `CAGR`, `MAX/MIN`.
- CAS/SMT equivalence checking và proof serialization.

### V12-D — chỉ khi còn ranking bottleneck

- Verifier-guided hard-negative mining.
- Fine-tune pointer head với learning rate thấp.
- Giữ semantic encoder và operation head frozen.
- Bắt buộc no-regression gate theo report/layout/operation.

## 14. Ablation cho bài nghiên cứu

| Cấu hình | Đóng góp cần đo |
|---|---|
| V8 HRM | Baseline neural |
| V8 + hard verifier | Symbolic filtering muộn |
| V8 + typed synthesis | Candidate generation có constraint |
| V8 + synthesis + reranker | Learned proof-guided selection |
| Full − graph relations | Vai trò của cấu trúc evidence |
| Full − unit constraints | Vai trò của typed unit |
| Full − confidence calibration | Safe abstention |

Không dùng blind cũ để chọn rule, threshold, feature hay model. Kết quả blind V8
đã công bố được giữ nguyên như mốc lịch sử.

### Bổ sung development unseen-company/layout cho V12-B.2 (09/08/2026)

- Đã khóa một intake development-only từ `pdf/`: 40 công ty mới, 53 PDF, chọn xác
  định theo hash company identity; không dùng câu hỏi, đáp án, số liệu hay model output.
- Company overlap với development catalog cũ, train, validation, test và blind đều
  bằng 0; protected hash overlap = 0; 53/53 source hash được xác minh.
- Audit toàn bộ cohort có 4 nhóm cấu trúc: 32 scan-dominant, 9 native portrait,
  9 native mixed-orientation và 3 hybrid. Render đại diện xác nhận có bảng dọc, bảng
  ngang hai trang, infographic và bảng scan.
- Đây mới là intake cho OCR/table extraction và annotation, chưa phải QA supervision:
  `unseen_company_intake_ready = true`, `ready_for_phase_d_training = false`.
- Vấn đề phải xử lý tiếp trong đúng nhánh dữ liệu: 32/53 PDF là scan-dominant, nên phải
  OCR và kiểm tra bảng trước khi sinh câu hỏi/program; không được hạ gate hoặc dùng
  validation/test/blind để chọn mẫu.
- Manifest: `data/pdf_v12b2_unseen_development_intake_v1.json`; audits:
  `reports/pdf_v12b2_unseen_development_overlap_audit_v1.json` và
  `reports/pdf_v12b2_unseen_development_layout_audit_v2.json`.

### OCR/table extraction cho unseen scan cohort (09/08/2026)

- Từ 32 PDF scan-dominant, queue cấu trúc ban đầu có 327 trang. Pilot AIC đạt 11/11
  trang OCR và chọn đúng trang tài chính 25.
- Một trang HJS phủ đều nhưng có 30-38 vùng đồ họa làm hybrid VLM nghẽn. Nhánh table
  extraction được khóa lại ở 4 trang có điểm cấu trúc bảng cao nhất/PDF; không hạ chất
  lượng OCR và không bỏ cả công ty vì một trang bệnh lý.
- MinerU hybrid hoàn tất 128/128 trang trên 32/32 PDF, pipeline failure = 0.
- Scorer cấu trúc giữ 20 core financial page thuộc 18 PDF và 28 review page tính cả
  neighbor. 14 PDF không có core page không được giả nhãn âm hoặc đưa vào supervision.
- Render QA loại DBM trang 12: đây là statement-name mention với hai số giao dịch cổ
  đông, không phải bảng tài chính. Gate tổng quát mới yêu cầu statement-name có ít nhất
  6 số hoặc ít nhất 2 metric tài chính thực; DBM trang 6 vẫn được giữ.
- `scan_table_ocr_complete = true`, `ready_for_human_annotation = true` cho 18 PDF,
  nhưng `ready_for_phase_d_training = false` cho đến khi QA/program được kiểm tra.
- Report: `reports/pdf_v12b2_unseen_ocr_table_review_v1.md`; OCR manifest:
  `reports/pdf_v12b2_unseen_ocr_financial_pages_v2.json`.

### Sửa financial-page scorer theo tín hiệu nội bảng (09/08/2026)

- Recheck trực quan cho thấy kết quả 18/32 PDF của scorer v2 quá bảo thủ: regex cũ bỏ
  số thập phân/phần trăm và gate điểm loại nhầm bảng tỷ số hoặc bảng tóm tắt không có
  tên báo cáo chính thức. Không xử lý bằng cách hạ threshold.
- Scorer mới nhận dạng số tiền, số thập phân, số âm và phần trăm; đồng thời yêu cầu
  metric/summary signal nằm trong chính bảng số. Nhờ vậy cụm “tình hình tài chính” ở
  văn xuôi không được phép gán nhãn cho bảng quản trị/cổ đông không liên quan.
- Page taxonomy được tách thành `financial_summary`, `financial_note` và
  `non_financial`; score cũ chỉ còn phục vụ audit, không còn là selection gate.
- Chấm lại 128 trang từ cache cho 46 summary page thuộc 27 PDF, 7 note page với 2 PDF
  note-only, và 3 PDF không có financial evidence phù hợp. DBM trang 12 vẫn bị loại.
- Đây là kết quả development unseen-company có company/protected-hash overlap bằng 0;
  validation/test/blind không được đọc hoặc dùng để chọn rule. 25 kiểm thử liên quan qua.
- `financial_page_triage_ready = true`, nhưng `ready_for_phase_d_training = false` cho
  đến khi câu hỏi/program được sinh và QA. Report:
  `reports/pdf_v12b2_unseen_ocr_table_review_v2.md`; manifest:
  `reports/pdf_v12b2_unseen_ocr_financial_pages_v4.json`.

### Sinh QA/program có provenance từ unseen financial summary (09/08/2026)

- Không khôi phục generator pre-V12 đã archive. Nhánh mới chỉ đọc các trang
  `financial_summary` đã review và chỉ sinh khi metric-period duy nhất, cùng trang,
  cùng đơn vị; dòng tỷ số dạng công thức không được giả làm raw operand.
- Program dùng DAG có evidence theo PDF, page, dòng bảng, period, raw value, unit,
  extraction method, source hash và company key. Parser hỗ trợ cả `Năm: giá trị` lặp
  trong ô và ánh xạ năm từ header cột; trường hợp metric lặp khác scope bị abstain.
- Từ 46 summary page, batch v3 có 11 QA candidate thuộc 4 company group: 7 câu cộng
  nợ ngắn/dài hạn, 2 câu kiểm tra `Tổng tài sản - Tổng nguồn vốn`, 2 câu Current Ratio.
  Automatic program audit đạt 11/11; visual source QA bốn trang đạt 11/11.
- Chưa sinh `interest_bearing_debt_share`: không trang summary đã review nào chứa đồng
  thời, duy nhất và cùng period ba operand vay ngắn hạn, vay dài hạn, Tổng nguồn vốn.
  Không ghép khác trang hoặc suy diễn số thiếu chỉ để tăng coverage.
- Blind/test không được đọc. `qa_candidate_batch_ready_for_annotation = true`, nhưng
  `ready_for_phase_d_training = false` cho đến khi annotation được freeze và audit split.
- Dataset: `data/pdf_v12b2_unseen_financial_qa_candidates_v3.jsonl`; audits:
  `reports/pdf_v12b2_unseen_financial_qa_candidates_v3_audit.json` và
  `reports/pdf_v12b2_unseen_financial_qa_candidates_v3_visual_qa.json`.

### Đánh giá end-to-end trên unseen-company development holdout (09/08/2026)

- Đã khóa 11 câu/4 company group trước inference; dataset hash
  `9d921862f93d6439c3d2076090732c1504dad43715ce63f5774975050e196f72`.
  Mỗi câu dùng toàn trang OCR và top-64 candidate, không cấp riêng gold evidence.
- Runtime đúng cấu hình đang phát hành: V8 hash
  `3a33276ded0d2ff2dd3d200c4c774a59eec0f8cad7102bc273cb263d983944de`
  + typed synthesis + financial identity catalog + V12.4 candidate recall; không
  sidecar, không train, không đổi threshold, không đọc test/blind.
- Raw full-page candidate recall đạt 11/11 = 100%, nhưng gold operand recall@10 chỉ
  9/11, gold program recall@10 2/11, program exact 0/11 và execution 2/11 = 18,18%.
  Hệ thống accept 1/11, abstain 10/11 = 90,91%; accepted execution precision 100%.
- `ADD` có 7 mẫu nhưng operation accuracy 0%. Hai mẫu `SUBTRACT` và hai mẫu `DIVIDE`
  nhận đúng operation nhưng sai pointer, nên program/execution đều 0. Gold được sinh
  không bị verifier false reject, nhưng 100% bị loại ở eligibility/search pipeline.
- Kết luận: OCR/raw extraction không còn là bottleneck. Bottleneck chính là
  question-conditioned typed program generation + eligibility; pointer selection là
  bottleneck phụ. `generalization_gate = false` và tạm khóa Phase D training vì
  reranker không thể phục hồi gold program chưa được sinh/eligible.
- Không được sửa rule/threshold theo 11 câu holdout này. Bước kế tiếp duy nhất là sửa
  program generation/eligibility tổng quát trên development company-disjoint folds đã
  có, sau đó mới đánh giá lại holdout đã khóa.
- Reports: `reports/pdf_v12b2_unseen_company_holdout_v1_v12_4_evaluation.json` và
  `reports/pdf_v12b2_unseen_company_holdout_v1_generalization_summary.json`.

### Sửa program generation/eligibility tổng quát và đánh giá lại holdout đã khóa (09/08/2026)

- Sửa ở tầng tổng quát, không thêm rule theo công ty/số liệu: planner tách câu hỏi chính
  khỏi phần “hãy liệt kê/bao gồm/và cho biết”, nhận diện so sánh cùng kỳ giữa hai vai trò
  tài chính, và typed synthesis ưu tiên hai vai trò được nêu thay vì tự ép `SUBTRACT`
  thành so sánh thời gian.
- Câu hỏi yêu cầu tính hoặc kiểm tra thành phần bắt buộc dùng identity formula; dòng tổng
  công bố trực tiếp chỉ còn được ưu tiên cho câu hỏi lookup. Bổ sung alias `Current Ratio`
  và giữ nguyên checkpoint, threshold, V12.4 cùng cấu hình không reranker.
- Trước khi mở lại holdout, 59 kiểm thử planner/synthesis/identity/pipeline và 163 kiểm thử
  runtime `document_hrm` đều đạt. Hash code/config/checkpoint/dataset được khóa tại
  `reports/pdf_v12b2_generalization_fix_v1_lock.json`; không đọc test/blind và không train.
- Đánh giá lại đúng 11 câu/4 company group đã đóng băng: operation 11/11, program exact
  9/11, execution 11/11; gold program được sinh 11/11 và pipeline rejection giảm từ
  11/11 xuống 0/11. Accepted coverage tăng từ 1/11 lên 7/11, accepted execution precision
  vẫn 100%.
- Chưa được tuyên bố end-to-end generalization pass. Hai câu cân đối chọn nhầm hai ô
  phần-trăm thay đổi có cùng giá trị thay vì hai ô tổng tuyệt đối; kết quả 0 chỉ là
  execution-equivalent. Hai Current Ratio chọn đúng pointer/công thức nhưng confidence
  0,871111 thấp hơn frozen DIVIDE threshold 0,991747. Không hạ threshold theo holdout.
- `program_generation_eligibility_gate = true`, nhưng `semantic_grounding_gate = false`
  và `safe_end_to_end_generalization_gate = false`. Phase D chưa train. Bước tiếp theo duy
  nhất là tạo cohort development ngoài holdout cho bảng đồng thời có cột giá trị và cột
  phần-trăm thay đổi, rồi sửa same-period role grounding trên cohort đó trước khi đánh giá
  holdout thêm lần nữa.
- Reports:
  `reports/pdf_v12b2_unseen_company_holdout_v1_post_generalization_fix_v1_evaluation.json`,
  `reports/pdf_v12b2_unseen_company_holdout_v1_post_generalization_fix_v1_error_summary.json`
  và `reports/pdf_v12b2_unseen_company_holdout_v1_post_generalization_fix_v1_summary.json`.

### Chốt lỗi chọn cột phần trăm thay cho giá trị đúng kỳ (09/08/2026)

- Nguyên nhân gốc nằm ở eligibility/ranking: ô phần trăm thay đổi không có period vẫn được giữ
  cạnh ô giá trị tuyệt đối đúng kỳ và có thể thắng nhờ proof/neural score. Bản sửa áp dụng
  `requested-period dominance` độc lập trong từng operand role và trong typed synthesis sau
  metric-slot narrowing; không thêm rule theo công ty, năm hay giá trị.
- Fallback OCR thưa được bảo toàn: evidence không rõ kỳ chỉ bị loại khi chính role/metric đó có
  evidence khớp kỳ được hỏi; nếu không có explicit match thì evidence không rõ kỳ vẫn hợp lệ.
- Bảy kiểm thử đối nghịch/fallback và toàn bộ 222 kiểm thử planner, synthesis, identity và
  document runtime đều đạt. Audit chỉ dùng ba nguồn development, từ chối tên nguồn chứa
  `test/holdout/final/blind`, đạt 2.345/2.345 structural group, lỗi 0.
- Sau khi khóa hash code/config/checkpoint/dataset, đánh giá lại 11 câu frozen holdout đạt
  operation, program exact, execution và pointer pair 11/11. Theo operation: ADD 7/7,
  SUBTRACT 2/2, DIVIDE 2/2. Accepted coverage 9/11 và accepted program/execution precision
  100%; hai DIVIDE đúng program nhưng tiếp tục abstain dưới frozen threshold, không hạ threshold.
- `requested_period_grounding_gate = true` và scoped Phase-C program/execution gate = true.
  Phần tỷ trọng tổng nợ vay tài chính chịu lãi vẫn thuộc V12-C multi-step DAG/SHARE_OF_TOTAL,
  không được trộn vào bản sửa một lỗi này.
- Artifacts: `reports/requested_period_dominance_development_audit_v1.json`,
  `reports/pdf_v12b2_period_dominance_fix_v2_lock.json`,
  `reports/pdf_v12b2_unseen_company_holdout_v1_post_period_dominance_fix_v2_evaluation.json`
  và `reports/pdf_v12b2_period_dominance_fix_v2_summary.json`.

### V12-C — SHARE_OF_TOTAL cho nợ vay chịu lãi (09/08/2026)

- Chỉ triển khai một công thức: `(vay ngắn hạn + vay dài hạn) / Tổng nguồn vốn × 100`.
  Checkpoint V8 vẫn phát ra final operation `RATIO_PERCENT`; runtime hạ chương trình thành DAG
  hai node thực thi được: `ADD` rồi `SHARE_OF_TOTAL`, nên không thêm operator giả vào neural head.
- Intermediate nợ vay chịu lãi mang lineage pointer tới hai ô gốc. Câu trả lời/citation dùng ba
  evidence leaf gốc, không dùng riêng ô trung gian. Runtime tái thực thi DAG và chặn nếu giá trị
  materialized khác kết quả `ADD`.
- Chỉ sinh DAG khi ba role là duy nhất, cùng explicit requested period, cùng source/table/entity/
  scope và cùng currency scale; khác kỳ, khác scale, nhiều dòng trùng role hoặc denominator bằng 0
  đều không được phép trở thành câu trả lời.
- Chín kiểm thử mới và toàn bộ 281 regression liên quan đạt. Kiểm thử có hoán vị thứ tự candidate,
  năm 2024/2027, giới hạn top-K, neural pair đối nghịch và intermediate bị sửa; không dùng
  test/blind, không train và không đổi threshold.
- End-to-end bằng checkpoint thật trên fixture development năm 2027 chọn đúng DAG và tính đúng
  `10%`, với đủ ba evidence gốc. Kết quả vẫn `ABSTAIN` vì confidence thấp hơn sát ngưỡng
  RATIO_PERCENT đã đóng băng; không được hạ ngưỡng theo một fixture.
- Audit 1.342 mẫu thuộc ba nguồn development hiện có tìm được 0 trường hợp chứa đồng thời ba
  raw role cùng explicit period; cache OCR development chỉ có NWT với hai dòng vay nhưng thiếu
  Tổng nguồn vốn cùng bảng, còn CFM có hai dòng vay để trống. Vì vậy chưa có cohort
  company-disjoint hợp lệ để hiệu chỉnh acceptance; không ghép khác trang hoặc suy diễn denominator.
- `multi_step_dag_execution_gate = true`, `structural_grounding_gate = true`, nhưng
  `accepted_answer_calibration_gate = false`. Bước kế tiếp duy nhất là tạo cohort development
  company-disjoint có đủ ba raw role để hiệu chỉnh/đánh giá acceptance mà không dùng blind.
- Reports: `reports/interest_bearing_debt_share_v12c_development_v1.json` và
  `reports/interest_bearing_debt_share_v12c_cohort_discovery_v1.json`.

#### V12-C cohort development-only đã khóa (12/08/2026)

- Đã thay discovery cũ bằng discovery theo cấu trúc trên hai manifest native-page development
  đã khóa hash. Mọi path/ID chứa `test`, `holdout`, `final`, `blind` hoặc `validation` đều bị
  từ chối trước khi đọc. Báo cáo discovery v1 cũ từng liệt kê `validation.jsonl` nên không còn
  được dùng làm bằng chứng development-only; nội dung protected đó không được mở trong vòng này.
- Hai lỗi tổng quát được phát hiện và sửa tại tầng dựng cohort: ô tham chiếu thuyết minh như
  `19`/`21.2` không còn bị lấy làm amount; role label được bind với mã tài khoản gần nhất để
  bảng song song bên phải không dịch cửa sổ giá trị. Kỳ từ dòng Thông tư năm 2014 không còn bị
  nhận nhầm là kỳ báo cáo; `Số cuối năm` và `Số đầu năm` được giữ thành `31/12/YYYY` và
  `01/01/YYYY`, không đồng nhất sai hai endpoint.
- Cohort frozen v1 có 12 example, 6 source-page và 5 company-group. Mỗi example có đúng ba
  evidence leaf 320/338/440, cùng source, table, entity/scope, kỳ và currency scale; DAG vẫn là
  `ADD -> SHARE_OF_TOTAL`. Visual QA đã đối chiếu trực tiếp cả sáu trang PDF và khóa hash render,
  giá trị quan sát, quyết định review cùng hash candidate.
- Freeze audit tái thực thi toàn bộ phép tính, kiểm tra uniqueness, provenance/lineage, period,
  unit, denominator và sự đầy đủ của visual decision. Kết quả: role contract, lineage, arithmetic
  và visual QA đều pass; suite 277 kiểm thử liên quan đều đạt, trong đó có 8 kiểm thử
  builder/freezer mới.
- Không đọc validation/blind/test, không train/fine-tune, không sửa model/reranker và không đổi
  threshold. `ready_for_frozen_evaluation = true` nhưng `ready_for_calibration = false`; Phase D
  vẫn khóa. Bước kế tiếp là chạy frozen V8 inference trên chính cohort đã khóa, báo cáo riêng
  program/execution/accept-abstain theo company-group trước khi đề xuất bất kỳ calibration nào.
- Artifacts chính: `data/v12c_interest_bearing_debt_share_development_frozen_v1.jsonl`,
  `reports/v12c_interest_bearing_debt_share_cohort_freeze_v1.json`,
  `reports/v12c_interest_bearing_debt_share_visual_decisions_v1.json`,
  `reports/v12c_interest_bearing_debt_share_cohort_audit_v3.json` và
  `reports/v12c_interest_bearing_debt_share_render_queue_v3.json`.

#### V12-C frozen inference trên cohort đã khóa (12/08/2026)

- Đã chạy checkpoint V8 thật trên 12 example frozen với đúng runtime flags hiện hành:
  typed synthesis, financial identity catalog và candidate recall V12.4 bật; không gắn sidecar.
  Audit là evidence-conditioned trên ba raw leaf đã visual-QA, không tuyên bố đo retrieval.
- Overall program/execution đạt 8/12 = `66,67%`. Phân tầng endpoint cho thấy hai outcome tách
  biệt hoàn toàn: 8 closing endpoint đạt program 8/8 và execution 8/8; 4 opening endpoint đạt
  0/4 vì runtime không materialize intermediate `interest_bearing_debt`.
- Nguyên nhân opening endpoint là lỗi period-role tổng quát: evidence có semantic chính xác
  `01/01/YYYY`, nhưng `_candidate_period_years` coi marker `01/01` là role `prior` rồi lùi
  explicit year thêm một lần. Kết quả là `missing_requested_period`,
  `operand_outside_requested_period` và `missing_required_multi_step_derivation`. Hệ thống
  fail-closed; không có false accept trong cohort này.
- Cả 12/12 đều abstain. Tám chương trình đúng có confidence `0,901176`, thấp hơn frozen
  RATIO_PERCENT threshold `0,990989`; bốn opening case có confidence khoảng `0,589–0,656`
  và thêm structural rejection. Vì accepted count bằng 0, accepted precision là không quan sát
  được (`null`), không được diễn giải thành 0% hoặc dùng để hạ threshold.
- Theo company-group: hai group đạt program/execution 2/2, một group đạt 2/4, hai group đạt
  1/2; tất cả có accepted coverage 0. V8 SHA-256 trước/sau giữ nguyên
  `3a33276ded0d2ff2dd3d200c4c774a59eec0f8cad7102bc273cb263d983944de` và suite 281
  kiểm thử liên quan đạt.
- Gate: `closing_endpoint_program_execution_gate = true`,
  `overall_program_execution_gate = false`, `acceptance_observable = false`,
  `ready_for_calibration = false`, `phase_d_unlocked = false`. Phase D vẫn khóa ở baseline
  overall accepted precision `88,47%`; không train, không sửa reranker và không đổi threshold.
- Bước tiếp theo duy nhất: sửa contract period development-only để exact date `01/01/YYYY`
  không bị lùi năm lần hai, thêm counterfactual cho `Số đầu năm`/`Số cuối năm` và explicit
  `01/01`/`31/12`, rồi replay chính frozen cohort. Chỉ sau khi program/execution đạt trên cả
  hai endpoint mới được thiết kế calibration group-disjoint; tuyệt đối chưa thay threshold.
- Artifact chính: `reports/v12c_interest_bearing_debt_share_frozen_inference_v2.json` và
  `scripts/evaluate_v12c_interest_bearing_debt_share_frozen.py`.

#### V12-C exact opening-period repair và frozen replay (12/08/2026)

- Đã sửa period contract tại tầng dùng chung, không thêm rule theo công ty/năm/giá trị. Khi
  descriptor chứa ngày tuyệt đối đầy đủ `dd/mm/YYYY`, calendar year được giữ nguyên trước khi
  áp dụng role prior/current; do đó `01/01/2025` vẫn là opening/prior endpoint nhưng không còn
  bị biến thành năm 2024. Identity endpoint không bị làm phẳng: `01/01/2025` và `31/12/2025`
  vẫn có role khác nhau.
- `_augment_interest_bearing_debt_share` nay truyền question contract vào period resolution.
  Counterfactual khóa cả hai phía: explicit `01/01`/`31/12` giữ đúng calendar year; relative
  header `Số đầu năm 2025`/`Số cuối năm 2025` không có exact date vẫn giữ legacy mapping
  prior/current, nên bản sửa không mở rộng sang semantics tương đối chưa được chứng minh.
- Replay nguyên frozen cohort, cùng V8 và cùng runtime flags, đạt materialized derivation 12/12,
  program exact 12/12 và execution 12/12. Opening endpoint tăng từ 0/4 lên 4/4; closing endpoint
  giữ 8/8. Cả 5 company-group đều đạt program/execution 100%.
- Acceptance không đổi: 12/12 tiếp tục abstain vì confidence `0,901176` thấp hơn frozen
  RATIO_PERCENT threshold `0,990989`; không còn structural rejection. Accepted precision vẫn
  không quan sát được (`null`), không phải 0%. Đây là bằng chứng repair chỉ sửa correctness,
  không âm thầm nới acceptance.
- Suite regression liên quan đạt 283/283; bốn evaluator tests cuối đạt sau khi khóa hash report.
  SHA-256 checkpoint V8 trước/sau vẫn là
  `3a33276ded0d2ff2dd3d200c4c774a59eec0f8cad7102bc273cb263d983944de`.
- Gate sau sửa: `overall_program_execution_gate = true`,
  `closing_endpoint_program_execution_gate = true`, `acceptance_observable = false`,
  `ready_for_calibration = false`, `phase_d_unlocked = false`. Phase D vẫn khóa ở baseline
  accepted precision `88,47%`; không train, không gắn reranker và không thay threshold.
- Bước kế tiếp không được tự động hạ threshold hoặc nâng confidence floor. Trước calibration
  cần một protocol development-only group-disjoint đủ sức đo precision/coverage và phải tách
  calibration cohort khỏi cohort 12 mẫu vừa dùng để phát triển/sửa period contract. Cho đến khi
  có cohort độc lập đó, giữ acceptance gate đóng.
- Artifact: `reports/v12c_interest_bearing_debt_share_frozen_inference_post_period_fix_v2.json`,
  `src/hrm_finance_qa/document_hrm.py`, `src/hrm_finance_qa/document_reasoning_engine.py` và
  `tests/test_evaluate_v12c_interest_bearing_debt_share_frozen.py`.

#### V12-C independent development calibration cohort (12/08/2026)

- Đã khóa tập loại trừ từ cohort 12 mẫu dùng để phát triển/sửa exact opening-period trước khi
  discovery. Discovery chỉ đọc catalog `development_only`, từ chối path/ID chứa
  `test`, `holdout`, `final`, `blind` hoặc `validation`, kiểm tra hash 169/169 PDF rồi quét
  8.649 trang. Kết quả shortlist 13 trang thuộc 10 company-group; không có company overlap
  với năm group repair.
- Visual QA trực tiếp 13 render chấp thuận 11 trang thuộc 9 company-group và loại hai trang:
  một bảng chỉ có debt summary nhưng thiếu raw denominator cùng bảng, một bảng là tổng nợ phải
  trả chứ không phải vay chịu lãi. Hai endpoint có ô vay dài hạn là dấu gạch cũng bị loại, không
  suy diễn thành zero. Cohort frozen cuối cùng có 20 positive answerable example, 11 source-page,
  9 company-group và overlap với repair cohort bằng 0.
- Frozen V8 inference evidence-conditioned đạt materialized derivation, program exact và execution
  20/20. Checkpoint trước/sau giữ nguyên SHA-256
  `3a33276ded0d2ff2dd3d200c4c774a59eec0f8cad7102bc273cb263d983944de`.
  Cả 20 mẫu vẫn abstain vì confidence `0,901176` thấp hơn frozen RATIO_PERCENT threshold
  `0,990989`; accepted coverage bằng 0 và accepted precision tiếp tục là `null`.
- Cohort này chứng minh repair tổng quát hóa program/execution ngoài 12 mẫu sửa lỗi, nhưng chưa
  chứng minh một operating point acceptance an toàn. Đây là cohort positive-only, không có
  negative/ambiguity control và tại threshold đóng băng không có accepted sample; vì vậy không thể
  đo false-accept risk hoặc dùng kết quả 20/20 để hạ threshold.
- Gate: `independent_positive_cohort_gate = true`, `acceptance_observable = false`,
  `ready_for_threshold_calibration = false`, `threshold_change_authorized = false` và
  `phase_d_unlocked = false`. Phase D vẫn khóa ở baseline overall accepted precision `88,47%`;
  không đọc validation/blind/test, không train/fine-tune, không sửa reranker và không thay threshold.
- Bước tiếp theo duy nhất là đóng băng một negative/ambiguity control cohort development-only,
  company-disjoint dưới cùng evidence-conditioned protocol. Chỉ sau khi đo false-accept risk chung
  với positive cohort độc lập này mới được đề xuất calibration; threshold hiện tại tiếp tục bất biến.
- Artifacts chính: `data/v12c_interest_bearing_debt_share_independent_calibration_frozen_v1.jsonl`,
  `reports/v12c_interest_bearing_debt_share_independent_calibration_freeze_v1.json`,
  `reports/v12c_interest_bearing_debt_share_independent_calibration_frozen_inference_v1.json` và
  `reports/v12c_interest_bearing_debt_share_independent_calibration_adequacy_v1.json`.

#### V12-C negative/ambiguity control cohort — hoàn tất (12/08/2026)

- Trước discovery đã loại theo cả cohort repair 12 mẫu và positive calibration 20 mẫu bằng
  ticker/company key; mọi path/ID chứa `test`, `holdout`, `final`, `blind` hoặc `validation`
  tiếp tục bị từ chối trước khi đọc. Sau exclusion, discovery kiểm tra hash và quét 159 PDF
  development-only với 7.798 trang, shortlist 27 near-miss page thuộc 14 company-group.
- Đã render 27 trang và visual-QA theo proof obligation, tối đa một trang đại diện mỗi company.
  Cohort frozen giữ 13 trang/13 company-group, tạo 26 period control và có overlap bằng 0 với
  cả repair lẫn positive cohort. Một trang NHA mà visual QA cho thấy đủ ba raw role bị loại khỏi
  negative label; kết quả regex discovery không được dùng thay cho nhãn trực quan.
- Trong 26 control, 14 trường hợp là `INCOMPLETE_REQUIRED_ROLE_ABSENT`: thiếu vay ngắn hạn,
  vay dài hạn hoặc raw Tổng nguồn vốn; ô dấu gạch không bị suy diễn thành zero, và không ghép
  denominator từ trang khác. 12 trường hợp là `INVALID_ROLE_SEMANTICS`: hàng Nợ ngắn hạn/Nợ
  dài hạn chỉ là tổng liabilities, không phải raw interest-bearing borrowings.
- Frozen V8 inference tại đúng RATIO_PERCENT threshold `0,990989` cho false accept `0/26`,
  safe abstention `26/26`; cả 26 đều dưới threshold. Structural grounding chặn 20/26, nhưng sáu
  semantic-confound của SBB, VID và VLW không bị structural gate chặn và chỉ an toàn nhờ threshold.
  Confidence negative nằm trong `0,505601–0,950636`; checkpoint trước/sau giữ nguyên SHA-256
  `3a33276ded0d2ff2dd3d200c4c774a59eec0f8cad7102bc273cb263d983944de`.
- Joint audit với positive cohort cho thấy score không tách được: 20 positive đúng đều có confidence
  `0,901176`, trong khi 6 negative không bị structural rejection có confidence lớn hơn hoặc bằng
  mức đó và negative ceiling là `0,950636`. Separation margin bằng `-0,049460`;
  `confidence_only_separable = false`. Vì vậy kết quả 0 false accept tại threshold hiện tại không
  phải bằng chứng cho phép hạ threshold.
- Gate đã đánh dấu: `negative_control_cohort_complete = true`,
  `current_threshold_zero_false_accept_gate = true`,
  `current_threshold_positive_coverage_gate = false`, `structural_negative_gate = false`,
  `confidence_separation_gate = false`, `ready_for_threshold_calibration = false`,
  `threshold_change_authorized = false` và `phase_d_unlocked = false`. Không chạy alternative
  threshold, không đọc validation/blind/test, không train/fine-tune và không sửa model/reranker.
- Bước tiếp theo duy nhất là sửa phân biệt semantic tổng quát giữa liabilities và interest-bearing
  borrowings trên development-only, để sáu control nói trên bị structural fail-closed; sau đó replay
  nguyên positive và negative frozen cohorts. Threshold phải giữ nguyên trong vòng sửa này.
- Regression liên quan đạt 291/291. Artifacts chính:
  `data/v12c_interest_bearing_debt_share_negative_ambiguity_frozen_v1.jsonl`,
  `reports/v12c_interest_bearing_debt_share_negative_ambiguity_freeze_v1.json`,
  `reports/v12c_interest_bearing_debt_share_negative_ambiguity_frozen_inference_v1.json` và
  `reports/v12c_interest_bearing_debt_share_joint_calibration_adequacy_v1.json`.

#### V12-C liability/borrowing semantic gate repair — hoàn tất (12/08/2026)

- Nguyên nhân gốc là fuzzy alias score: hàng broad liability `Nợ ngắn hạn` đạt khoảng `0,8375`
  so với alias `Nợ vay ngắn hạn`, vượt admission gate `0,82` dù thiếu semantic anchor quyết định
  `vay`. Vì vậy ba bộ SBB/VID/VLW từng materialize sai intermediate `interest_bearing_debt`.
- Đã thêm semantic admission anchor trước fuzzy score, không dùng tên công ty/năm/layout/giá trị:
  role short/long-term borrowing bắt buộc có token `vay`, đúng endpoint ngắn/dài và loại asset
  context `phải thu về cho vay`/`cho vay`; role denominator bắt buộc có đủ family
  `Tổng + nguồn + vốn`. Fuzzy score chỉ còn dùng sau khi accounting concept đã được chứng minh.
- Counterfactual khóa hai phía: `Nợ ngắn hạn/Nợ dài hạn` và `Phải thu về cho vay` không còn
  materialize debt identity; `Vay ngắn hạn`, `Nợ vay ngắn hạn` và
  `Vay và nợ thuê tài chính ngắn hạn` vẫn được nhận. Không chỉnh confidence, threshold hay
  neural score.
- Replay nguyên positive frozen cohort hash
  `58b1304afabbfd87f2f12cdf9b9694d996210a40768646368bff8e9274ca7133` giữ materialized,
  program exact và execution `20/20`; accepted vẫn `0/20` tại threshold đóng băng `0,990989`.
- Replay nguyên negative frozen cohort hash
  `17d8bc460346409200fa167ad485a7f3b056b497f1b3c4bec19b38618f5e1882` giữ false accept
  `0/26` và tăng structural rejection từ `20/26` lên `26/26`. Sáu semantic-confound trước đây
  không còn materialize derived debt và đều có `missing_required_multi_step_derivation`.
  Nhóm `INVALID_ROLE_SEMANTICS` tăng structural rejection từ `6/12` lên `12/12`; nhóm thiếu
  raw role giữ `14/14`.
- Joint audit sau sửa có `structural_negative_gate = true` và số negative không bị structural
  rejection ở hoặc trên positive confidence floor giảm từ 6 xuống 0. Confidence thô vẫn không
  separable (`positive floor 0,901176`, `negative ceiling 0,938633`), nhưng các negative score cao
  nay đã bị fail-closed bằng proof contract thay vì phụ thuộc threshold.
- Gate đã đánh dấu: `semantic_liability_borrowing_gate_repaired = true`,
  `positive_program_execution_regression_free = true`, `structural_negative_gate = true`, nhưng
  `current_threshold_positive_coverage_gate = false`, `ready_for_threshold_calibration = false`,
  `threshold_change_authorized = false` và `phase_d_unlocked = false`. Checkpoint SHA-256 trước/sau
  vẫn là `3a33276ded0d2ff2dd3d200c4c774a59eec0f8cad7102bc273cb263d983944de`;
  regression đạt 293/293.
- Bước tiếp theo không phải hạ threshold trên hai cohort vừa replay. Cần thiết kế một calibration
  protocol development-only mới, company-disjoint với cả repair/positive/negative cohorts và có
  đồng thời answerable positives cùng semantic hard negatives; precommit operating-point rule và
  precision lower-bound trước khi chạy. Cho đến lúc đó threshold giữ nguyên và Phase D tiếp tục khóa.
- Artifacts: `reports/v12c_interest_bearing_debt_share_semantic_gate_repair_lock_v1.json`,
  `reports/v12c_interest_bearing_debt_share_independent_calibration_post_semantic_gate_v1.json`,
  `reports/v12c_interest_bearing_debt_share_negative_ambiguity_post_semantic_gate_v1.json` và
  `reports/v12c_interest_bearing_debt_share_joint_calibration_post_semantic_gate_v1.json`.

#### V12-C fresh joint calibration protocol — feasibility stop (12/08/2026)

- Trước khi discovery nguồn mới, đã precommit protocol và khóa SHA-256 catalog development,
  checkpoint, runtime source cùng ba exclusion cohort. Exclusion hợp nhất có 27 ticker và 27
  company key; bị áp dụng trước khi mở PDF. Không được dùng validation/blind/test, không train,
  không đổi model/reranker, không chạy alternative threshold và không thay runtime threshold.
- Operating-point gate được định trước: accepted precision point estimate và Clopper-Pearson
  one-sided 95% lower bound đều phải ít nhất 90%; ít nhất 30 accepted answer từ 15 company;
  positive accepted coverage ít nhất 50%; negative false-accept phải bằng 0 ở overall và từng
  failure stratum. Cohort trước visual QA phải có tối thiểu 30 positive/15 company và
  30 negative/15 company/3 failure strata, đồng thời positive-negative company-disjoint.
- Feasibility discovery sau exclusion kiểm tra hash rồi quét 145 PDF/6.590 trang development-only;
  24 catalog document bị loại trước khi mở, missing/ambiguous PDF bằng 0 và hash mismatch bằng 0.
  Kết quả có `0` positive candidate page/`0` positive company, cùng 29 negative candidate page
  thuộc 23 company nhưng chỉ có hai failure strata: 27 invalid borrowing-role semantics và hai
  missing-required-role page.
- Stop rule đã kích hoạt trước render/visual QA/inference. Positive thiếu 30 page và 15 company;
  negative thiếu một page và một failure stratum. Vì visual QA chỉ có thể loại candidate chứ không
  thể tạo thêm candidate hợp lệ, catalog hiện tại không thể đạt quota đã precommit. Không hạ quota,
  không tái dùng company đã exclusion và không đọc protected source để lấp mẫu.
- Gate đã đánh dấu: `precommit_valid = true`, `joint_protocol_feasible_before_visual_qa = false`,
  `stop_rule_triggered = true`, `ready_for_visual_qa = false`, `ready_for_inference = false`,
  `ready_for_threshold_calibration = false`, `threshold_change_authorized = false` và
  `phase_d_unlocked = false`. Threshold giữ nguyên `0,990989`; đây là infeasibility của nguồn dữ
  liệu hiện tại, không phải kết quả precision mới.
- Bước tiếp theo duy nhất là bổ sung một catalog PDF development-only có provenance/hash được audit,
  gồm company mới và đủ complete raw-role triplet cùng ít nhất ba hard-negative failure strata;
  sau đó chạy lại nguyên protocol precommit này. Không được sửa quota hoặc operating-point rule.
- Artifacts: `config/v12c_joint_calibration_protocol_v1.json`,
  `reports/v12c_joint_calibration_protocol_v1_precommit_audit.json`,
  `reports/v12c_joint_calibration_feasibility_discovery_v1.json` và
  `reports/v12c_joint_calibration_protocol_v1_feasibility_stop.json`.
#### V12-C fresh joint calibration source extension và frozen replay — hoàn tất (12/08/2026)

- Đã bổ sung và audit hết phần PDF development-only local còn đủ điều kiện thành hai source block
  company-disjoint. Block 1 có 399 PDF/240 company, SHA-256 manifest
  `e4c49887fddc2dd3d7f813a3cb60d13cc0726042bd2cf8c7387506181970548c`; block 2 có
  391 PDF/239 company, SHA-256
  `9c20c84ccf323c46b8f733cfae0aad2352c59d7fa3bd7823b13d7beb1b8c0015`.
  Tổng cộng 790 PDF/479 company mới và 46.558 trang; 100% source có provenance/hash, không overlap
  ba cohort exclusion. Protocol gốc giữ nguyên SHA-256
  `e797925a706f3a2f61ac4f7e58515919ccc569770fb5217b6543d0c91a53f3bc`.
- Discovery không dùng model tìm được 27 positive candidate page/20 company và 326 hard-negative
  page/137 company thuộc đủ ba strata. Queue visual QA khóa tối đa một page/company/stratum và tối đa
  hai period control/page. Positive visual QA chấp thuận 18 page/18 company = 36 control; DXG và POW
  bị loại vì chỉ là management overview, không chứng minh unique same-table triplet. Negative visual QA
  khóa 16 page/16 company = 31 control: 8 missing-required-role, 22 invalid borrowing semantics và
  1 ambiguity control. Positive-negative company overlap bằng 0.
- Hai frozen cohort đều vượt quota precommit mà không hạ quota: positive tối thiểu 30/15 và negative
  tối thiểu 30/15/3 strata. SHA-256 positive cohort là
  `41c873a0cdb5ceb5fa5a22448693ebdf72d961fbd5b6987854f5d0c4a9a5a28f`; SHA-256 negative cohort là
  `2b4676a8216eaeb4e711216b74ffa3f2754c9fa1c5439437a54ec3b05525a4fe`.
- Replay checkpoint frozen cho positive: derivation 36/36, program exact 36/36 và execution exact
  36/36. Tuy nhiên confidence đồng nhất `0,901176470588235`, dưới RATIO_PERCENT threshold frozen
  `0,9909889376955401`, nên accepted `0/36` và positive accepted coverage bằng 0.
- Replay negative: false accept `0/31`, safe abstention `31/31`, structural rejection `31/31` và
  confidence nằm trong `0,551189–0,954401`. Cả ba failure strata đều có 0 false accept. Checkpoint hash
  trước/sau giữ nguyên; không chạy alternative threshold.
- Operating-point rule đã precommit được áp dụng nguyên trạng. Vì accepted denominator bằng 0,
  accepted precision point estimate và Clopper-Pearson one-sided 95% lower bound đều là `null`, không
  được diễn giải thành 100%. Các gate precision, minimum 30 accepted, minimum 15 accepted company và
  minimum 50% positive coverage đều fail; zero-negative-FA, checkpoint-hash và frozen-threshold gate pass.
  `all_conditions_pass = false`, `threshold_change_authorized = false`, `phase_d_unlocked = false`.
- Phase D tiếp tục khóa ở baseline overall accepted precision `88,47%`. Vòng này không đọc
  validation/blind/test, không train/fine-tune, không sửa model/reranker và không thay threshold. Bước
  “bổ sung nguồn mới” được đánh dấu hoàn tất và không nên lặp lại trên cùng catalog local: cohort hiện đã
  đủ quota thống kê; blocker thực tế là acceptance calibration không quan sát được tại threshold frozen.
- Bước kế tiếp hợp lệ là phân tích development-only cơ chế confidence/abstention cho các positive có
  proof-complete và program/execution đúng, đồng thời giữ semantic hard-negative fail-closed. Mọi đề xuất
  operating point mới phải là một quyết định riêng được precommit và kiểm chứng trên cohort khác; kết quả
  hiện tại không tự cấp quyền hạ threshold.
- Artifacts chính: `config/v12c_joint_calibration_source_extension_v2.json`,
  `reports/v12c_joint_calibration_visual_decisions_v1.json`,
  `data/v12c_interest_bearing_debt_share_joint_calibration_fresh_positive_v1.jsonl`,
  `data/v12c_interest_bearing_debt_share_joint_calibration_fresh_negative_v1.jsonl`,
  `reports/v12c_interest_bearing_debt_share_joint_calibration_fresh_positive_inference_v1.json`,
  `reports/v12c_interest_bearing_debt_share_joint_calibration_fresh_negative_inference_v1.json` và
  `reports/v12c_interest_bearing_debt_share_joint_calibration_fresh_operating_point_v1.json`.
- Hygiene note: toàn bộ manifest/discovery/freeze/inference ở trên chỉ nạp development-only source và
  protected-path guard đều pass; không artifact hay quyết định cohort nào dùng protected evaluation data.
  Tuy nhiên một lệnh tìm kiếm tương tác quá rộng trong lúc kiểm tra schema đã vô tình in ra một record từ
  `data/document_hrm_holdout.jsonl`. Record đó không được dùng trong phân tích hay phép đo, nhưng vì vậy
  phiên làm việc này không được tuyên bố là `zero protected read` tuyệt đối. Phase D vẫn khóa; nếu cần
  chứng thư clean-room nghiêm ngặt, phải tái chạy các script hash-locked trong một phiên mới chỉ allowlist
  các path development-only.
- Regression liên quan đạt 16/16 (13 cohort/evaluator/source-extension tests và 3 operating-point tests).
  Full-suite collection không hoàn tất vì workspace thiếu 17 script cũ được các test ngoài phạm vi import;
  đây là lỗi `FileNotFoundError/ModuleNotFoundError` ở collection, không phải assertion regression của vòng này.

#### V12-C proof-conditioned abstention repair — hoàn tất (13/08/2026)

- Audit trên nguyên hai frozen cohort development-only xác định `36/36` positive đều có
  `proof_contract_state=VALID`, `required_valid_fraction=1`, `required_unknown_fraction=0`, derivation
  `interest_bearing_debt` được materialize và grounding rejection bằng 0. Ngược lại `31/31` negative
  có proof contract `INVALID/UNKNOWN`, không materialize derivation và đều bị structural rejection.
- Nguyên nhân abstention là schema mismatch: confidence floor `grounded-symbolic-fusion-v12.1` dùng
  `min(0,95; 0,76 + 0,20 * valid/all constraints)`. Các optional layout constraint `UNKNOWN` làm tất cả
  positive proof-complete nhận cùng confidence `0,901176`, rồi bị so với threshold neural-calibrated
  `0,9909889376955401`. Không có bằng chứng hợp lệ để chỉ nâng hằng số confidence.
- Đã thêm acceptance basis tổng quát `complete_required_proof`, chỉ cho winner multi-step có executable
  derived DAG, nguồn `typed_slot_synthesis`, không search/grounding rejection và toàn bộ required obligation
  explicit `VALID`. Optional obligation có thể `UNKNOWN`. `SELECT`, `ADD`, flat neural program và bất kỳ
  required `UNKNOWN/INVALID` đều bị loại rõ ràng; do đó lane này không mở lại false-accept surface đã biết
  của S4.1/S4.2. Không thêm rule theo company/page/năm/giá trị.
- Replay positive frozen cohort hash
  `41c873a0cdb5ceb5fa5a22448693ebdf72d961fbd5b6987854f5d0c4a9a5a28f`: program/execution
  đúng `36/36`, accepted tăng từ `0/36` lên `36/36`; cả 36 có acceptance basis
  `complete_required_proof`. Confidence vẫn `0,901176` và threshold vẫn `0,990989`.
- Replay negative frozen cohort hash
  `2b4676a8216eaeb4e711216b74ffa3f2754c9fa1c5439437a54ec3b05525a4fe`: false accept `0/31`,
  structural rejection `31/31`, safe abstention `31/31`; cả ba failure strata tiếp tục 0 false accept.
- Operating-point measurement sau repair: accepted `36` từ `18` company, precision point `100%`,
  Clopper-Pearson one-sided 95% lower bound `92,0153%`, positive coverage `100%`, negative false accept
  `0/31`; toàn bộ measurement condition precommit pass. Checkpoint vẫn có SHA-256
  `3a33276ded0d2ff2dd3d200c4c774a59eec0f8cad7102bc273cb263d983944de`, protocol hash
  `e797925a706f3a2f61ac4f7e58515919ccc569770fb5217b6543d0c91a53f3bc`; không fine-tune,
  không thay threshold và không chạy alternative threshold.
- Gate đánh dấu `proof_conditioned_abstention_repaired=true`, `positive_acceptance_observable=true`,
  `hard_negative_fail_closed=true`; nhưng `threshold_change_authorized=false` và `phase_d_unlocked=false`.
  Lý do: hai frozen cohort này đã được dùng để chẩn đoán/thiết kế repair, nên replay là repair evidence
  chứ không phải independent confirmation. Baseline overall accepted precision vẫn là `88,47%`.
- Regression cuối cùng đạt 246 tests + 22 subtests, 0 failure. Artifacts chính:
  `reports/v12c_proof_conditioned_abstention_audit_v1.json`,
  `reports/v12c_proof_conditioned_abstention_repair_lock_v1.json`,
  `reports/v12c_interest_bearing_debt_share_joint_calibration_fresh_positive_post_proof_abstention_v3.json`,
  `reports/v12c_interest_bearing_debt_share_joint_calibration_fresh_negative_post_proof_abstention_v3.json` và
  `reports/v12c_interest_bearing_debt_share_joint_calibration_fresh_post_proof_abstention_operating_point_v3.json`.
- Bước kế tiếp hợp lệ là precommit một confirmation cohort development-only mới, company-disjoint
  với tất cả cohort đã dùng cho diagnosis/repair. Giữ nguyên checkpoint, proof-acceptance contract,
  threshold, quota và precision lower-bound rule; chỉ cohort chưa tham gia thiết kế mới có thể cung cấp
  bằng chứng xác nhận độc lập cho Phase D.

## 15. Kế hoạch hiện hành — Question Coverage và structural repair plan/actual (13/08/2026)

### 15.1 Quyết định phạm vi

Lỗi smoke-test development-only trên `ABW_23CN_BCTN.pdf` xác nhận một khoảng trống cấu trúc
không thuộc giao diện, MinerU hoặc threshold. Câu hỏi yêu cầu `Doanh thu hoạt động thực hiện
năm 2023`, retrieval tìm đúng dòng có cả `Kế hoạch 2023 = 472.723.394.169` và
`Thực hiện 2023 = 304.912.639.828`, nhưng candidate chỉ giữ `COLUMN/PERIOD = 2023` và không giữ
vai trò cột. Winner vì vậy chọn nhầm kế hoạch; frozen SELECT threshold `0,846` đã fail-closed
đúng và không được hạ.

Audit code hiện tại cũng cho thấy `QuestionPlan` mới biểu diễn operation, metric, year,
period direction, entity, unit và operand role tổng quát; chưa có qualifier có kiểu cho
`plan/actual/revised plan`. Proof contract SELECT hiện yêu cầu `query_metric` và
`period_coverage`, chưa có `column_role_match`. Các cụm `đạt bao nhiêu phần trăm kế hoạch`,
`hoàn thành kế hoạch` và `vượt kế hoạch` chưa có compiler contract tường minh, nên có thể rơi
về neural operation/pointer và chọn `SELECT` thay cho phép tính.

Track này là phần mở rộng cụ thể của S4.3-C/D, không xóa các kết quả V12-C trước đó. Tuy nhiên,
không tạo confirmation cohort V12-C mới trước khi khóa repair này: mọi runtime repair sau đó
sẽ làm confirmation cũ không còn xác nhận đúng cấu hình cuối. Tài liệu ABW chỉ là diagnosis
example và phải bị exclusion khỏi independent confirmation sau này.

Bất biến của toàn bộ vòng:

- chỉ dùng development-only; không đọc validation, blind hoặc test;
- checkpoint, neural score, per-operation threshold và operating-point rule giữ nguyên;
- không fine-tune, không thêm rule theo company/page/year/value và không sửa gold để khớp output;
- câu hỏi dài phải được biên dịch về cùng typed intent như câu ngắn tương đương; độ dài không
  được là feature để nới proof;
- metadata thiếu vai trò bắt buộc là `UNKNOWN` và phải fail-closed; vai trò trái yêu cầu là
  `INVALID`;
- chỉ được tuyên bố hỗ trợ một họ câu hỏi khi có positive, swapped-role negative, ambiguity
  control và replay collateral trên toàn bộ development set.

### 15.2 Typed intent đích dùng chung

`QuestionPlan` cần được mở rộng mà không phá schema cũ:

```text
answer_mode: SINGLE_SCALAR | SCALAR_WITH_DERIVED_LABEL | MULTI_ANSWER | AMBIGUOUS
operation / program_dag
metric / component_metrics
entity / consolidation_scope / segment_scope
requested_year / exact_date / relative_period_role
column_role: ACTUAL | PLAN | REVISED_PLAN | FORECAST | UNKNOWN
comparison_role_left / comparison_role_right
reference_base: PLAN | PRIOR_ACTUAL | CURRENT_ACTUAL | EXPLICIT_OTHER
direction_intent
requested_unit_family / source_unit / scale_conversion
required_outputs
```

`period_role` và `column_role` là hai trục khác nhau. Ví dụ `Thực hiện 2023` và
`Kế hoạch 2023` cùng year nhưng khác column role; `Thực hiện đầu năm 2023` còn có thêm
relative period role. Không được nhét cả hai vào một chuỗi year/column mơ hồ.

Mỗi operand cuối phải có proof tối thiểu:

```text
metric_match
requested_year_match
exact_date_or_relative_period_match (khi được hỏi)
column_role_match (khi được hỏi hoặc bảng có đối chứng plan/actual)
entity_and_scope_match
unit_and_scale_match
logical_cell_unique
source_lineage_complete
```

Program nhiều operand còn phải chứng minh operation, ordered operand role, denominator/reference
base, direction, same-scope compatibility và completeness của mọi intermediate DAG node.

### 15.3 Taxonomy câu hỏi phải bao phủ

#### A. Lookup một ô

| ID | Họ câu hỏi | Ví dụ | Typed target |
|---|---|---|---|
| L01 | Metric + năm | `Doanh thu thuần năm 2023 là bao nhiêu?` | `SELECT(metric, year)` nếu logical cell duy nhất |
| L02 | Actual | `Doanh thu thực hiện năm 2023 là bao nhiêu?` | `SELECT(column_role=ACTUAL, year=2023)` |
| L03 | Plan | `Doanh thu kế hoạch năm 2023 là bao nhiêu?` | `SELECT(column_role=PLAN, year=2023)` |
| L04 | Revised plan/forecast | `Kế hoạch điều chỉnh năm 2024 là bao nhiêu?` | `SELECT(REVISED_PLAN)`; không đồng nhất với PLAN gốc |
| L05 | Relative endpoint | `Tiền cuối kỳ/đầu kỳ là bao nhiêu?` | `SELECT(relative_period_role)` |
| L06 | Exact date | `Tổng tài sản tại 31/12/2023 là bao nhiêu?` | `SELECT(exact_date)`; không chỉ match year |
| L07 | Unit-qualified | `Doanh thu thực hiện năm 2023 theo triệu đồng?` | `SELECT` + unit/scale proof |
| L08 | Entity/scope-qualified | `Doanh thu công ty mẹ năm 2023?` | `SELECT(named entity/scope)` |
| L09 | Consolidated/standalone | `Lợi nhuận hợp nhất năm 2023?` | `SELECT(consolidation_scope)` |
| L10 | Repeated metric across panel | cùng metric ở nhiều segment/subtable | chỉ answer khi lineage/qualifier chọn duy nhất |

`L01` không mặc nhiên an toàn khi cùng metric/year có hai giá trị plan và actual. Nếu câu hỏi
không nêu role và evidence có conflict khác giá trị, target là `AMBIGUOUS`, không được chọn theo
confidence.

#### B. Phép tính một bước

| ID | Họ câu hỏi | Ví dụ | Typed target |
|---|---|---|---|
| A01 | Tổng hai khoản | `Tổng nợ ngắn hạn và nợ dài hạn?` | `ADD(A,B)` |
| A02 | Tổng 3–4 khoản | `Tổng A, B và C?` | `SUM3/SUM4` với đủ role độc lập |
| A03 | Chênh lệch theo thời gian | `2023 thay đổi bao nhiêu so với 2022?` | `SUBTRACT(current, prior)` |
| A04 | Mức giảm dương | `2023 thấp hơn 2022 bao nhiêu?` | `SUBTRACT(prior, current)` |
| A05 | Tăng trưởng phần trăm | `Tăng bao nhiêu phần trăm từ 2022 đến 2023?` | `PERCENT_CHANGE(current, prior)` |
| A06 | Tỷ trọng | `Lợi nhuận chiếm bao nhiêu % doanh thu?` | `RATIO_PERCENT(part,total)` |
| A07 | Hệ số | `Nợ trên tổng tài sản là bao nhiêu lần?` | direct `SELECT` hoặc `DIVIDE`, có proof tương đương |
| A08 | Bình quân | `Bình quân đầu kỳ và cuối kỳ?` | `AVERAGE(begin,end)` |
| A09 | Nhân | `Số cổ phiếu × mệnh giá?` | `MULTIPLY(count,par_value)` |
| A10 | Kiểm tra cân đối | `Tài sản có bằng nguồn vốn; nếu lệch là bao nhiêu?` | ordered `SUBTRACT`, không suy từ equality text đơn thuần |

#### C. Họ plan/actual bắt buộc xử lý cùng một thể

| ID | Ý định | Ví dụ rút gọn | Program đích |
|---|---|---|---|
| PA01 | Tra actual | `Thực hiện 2023 là bao nhiêu?` | `SELECT(actual_2023)` |
| PA02 | Tra plan | `Kế hoạch 2023 là bao nhiêu?` | `SELECT(plan_2023)` |
| PA03 | Tỷ lệ hoàn thành | `Thực hiện đạt bao nhiêu % kế hoạch?` | `RATIO_PERCENT(actual, plan)` |
| PA04 | Chênh lệch có dấu | `Thực hiện trừ kế hoạch?` | `SUBTRACT(actual, plan)` |
| PA05 | Thiếu kế hoạch theo tiền | `Thực hiện thấp hơn kế hoạch bao nhiêu?` | `SUBTRACT(plan, actual)` |
| PA06 | Vượt kế hoạch theo tiền | `Thực hiện vượt kế hoạch bao nhiêu?` | `SUBTRACT(actual, plan)` |
| PA07 | Sai lệch có dấu theo % | `Thực hiện thay đổi bao nhiêu % so với kế hoạch?` | `PERCENT_CHANGE(actual, plan)` |
| PA08 | Thiếu kế hoạch theo % | `Còn thiếu bao nhiêu % để đạt kế hoạch?` | `RATIO_PERCENT(SUBTRACT(plan,actual), plan)` |
| PA09 | Vượt kế hoạch theo % | `Vượt kế hoạch bao nhiêu %?` | `RATIO_PERCENT(SUBTRACT(actual,plan), plan)` |
| PA10 | Kế hoạch mới so actual cũ | `Kế hoạch 2024 tăng bao nhiêu % so với thực hiện 2023?` | `PERCENT_CHANGE(plan_2024, actual_2023)` |
| PA11 | Plan/actual của tổng cấu thành | `Tổng A+B thực hiện đạt bao nhiêu % tổng A+B kế hoạch?` | hai intermediate `ADD`, sau đó `RATIO_PERCENT` |
| PA12 | Plan/actual nhiều entity | `Công ty mẹ đạt bao nhiêu % kế hoạch hợp nhất?` | scope mismatch; mặc định `INVALID` trừ khi câu hỏi yêu cầu rõ phép so sánh chéo scope |

Các từ đồng nghĩa cần canonicalize theo nghĩa, không theo chuỗi cố định:

- actual: `thực hiện`, `thực tế`, `đã đạt`, `kết quả đạt được`;
- plan: `kế hoạch`, `mục tiêu`, `dự kiến`; nhưng `dự báo/forecast` và `kế hoạch điều chỉnh`
  phải giữ role riêng nếu bảng phân biệt;
- completion: `hoàn thành`, `đạt ... kế hoạch`, `thực hiện trên kế hoạch`;
- shortfall/exceed: `chưa đạt`, `còn thiếu`, `thấp hơn`, `vượt`, `cao hơn`.

Không dùng danh sách từ đồng nghĩa để tự chứng minh evidence. Từ ngữ chỉ tạo query role; header
lineage trong bảng phải cung cấp evidence role tương ứng.

#### D. Câu dài và phức tạp

| ID | Dạng khó | Ví dụ/hành vi bắt buộc |
|---|---|---|
| C01 | Nhiều qualifier | Câu có metric + company + scope + year + actual/plan + unit; mọi qualifier đều required |
| C02 | Mệnh đề dẫn dài | `Theo bảng..., căn cứ nghị quyết..., cho biết...`; bỏ discourse nhưng không bỏ semantic qualifier |
| C03 | Parenthetical/bilingual | `(Current Ratio)`, `(đơn vị: triệu đồng)`; tách alias và unit, không nhập vào metric |
| C04 | Hai năm, hai role | plan 2024 so actual 2023; không gán cả hai chỉ bằng temporal order |
| C05 | Hai metric, hai role | lợi nhuận actual / doanh thu plan; role gắn theo từng slot, không gắn toàn câu |
| C06 | Multi-step 3–4 leaf | cộng component actual, cộng component plan, rồi tính ratio; mọi intermediate phải có proof |
| C07 | Câu hỏi kèm giải thích | `... và cho biết vì sao`; hệ thống chỉ kết luận phần có evidence định lượng, không sinh nguyên nhân ngoài báo cáo |
| C08 | Một câu hỏi, nhiều output | actual + plan + completion; phải decompose thành ba subquery/structured outputs hoặc abstain `multi_answer_not_supported` |
| C09 | Số + nhãn định tính | `đã hoàn thành kế hoạch chưa và vượt/thiếu bao nhiêu`; numeric DAG trước, label suy ra xác định sau |
| C10 | Điều kiện/loại trừ | `không bao gồm X`, `sau khi loại Y`; chỉ hỗ trợ khi subtraction set và evidence scope được chứng minh |
| C11 | Đại từ/hội thoại | `chỉ tiêu này`, `năm đó`, `công ty trên`; deferred cho đến khi có conversation-state proof, không đoán từ chat cũ |
| C12 | Nhiều tài liệu | so sánh hai báo cáo/công ty; ngoài single-document contract hiện tại, phải từ chối hoặc chạy protocol riêng |
| C13 | Câu hỏi mơ hồ về hướng | `chênh lệch giữa actual và plan`; yêu cầu làm rõ hoặc trả absolute gap chỉ khi người dùng nói `độ lệch tuyệt đối` |
| C14 | Mơ hồ về mẫu số | `lớn hơn bao nhiêu %`; phải xác định rõ `so với` đại lượng nào, nếu không thì abstain |
| C15 | Role không nêu | metric/year trùng cả plan và actual; `logical_cell_conflict` và abstain thay vì chọn top score |

#### E. Câu hỏi kết hợp hai năm và nhiều phép so sánh

Hai năm không chỉ là hai `period` của cùng một phép trừ. Mỗi ô vẫn phải được định danh đầy đủ
bằng `(metric, year, column_role, entity, scope, unit)`. Đặc biệt, không được dùng quy tắc
`năm mới = actual` hoặc `năm cũ = plan`; role phải đến từ câu hỏi và header evidence.

| ID | Ý định hai năm | Ví dụ rút gọn | Program/DAG đích |
|---|---|---|---|
| TY01 | Actual hai năm, chênh lệch tiền | `Thực hiện 2023 tăng bao nhiêu so với thực hiện 2022?` | `SUBTRACT(actual_2023, actual_2022)` |
| TY02 | Actual hai năm, tăng trưởng % | `Thực hiện 2023 tăng bao nhiêu % so với 2022?` | `PERCENT_CHANGE(actual_2023, actual_2022)` |
| TY03 | Plan hai năm | `Kế hoạch 2024 tăng bao nhiêu so với kế hoạch 2023?` | `SUBTRACT(plan_2024, plan_2023)` hoặc `PERCENT_CHANGE` theo output unit |
| TY04 | Mixed role/mixed year | `Kế hoạch 2024 cao hơn thực hiện 2023 bao nhiêu %?` | `PERCENT_CHANGE(plan_2024, actual_2023)` |
| TY05 | Tỷ lệ hoàn thành của mỗi năm | `Tỷ lệ hoàn thành kế hoạch năm 2022 và 2023?` | hai DAG `actual_y / plan_y × 100` |
| TY06 | Thay đổi tỷ lệ hoàn thành theo điểm % | `Tỷ lệ hoàn thành 2023 cao hơn 2022 bao nhiêu điểm phần trăm?` | `SUBTRACT(completion_2023, completion_2022)` |
| TY07 | Thay đổi tương đối của tỷ lệ hoàn thành | `Tỷ lệ hoàn thành 2023 tăng bao nhiêu % so với 2022?` | `PERCENT_CHANGE(completion_2023, completion_2022)` |
| TY08 | So sánh và gắn nhãn | `Năm nào hoàn thành kế hoạch cao hơn và cao hơn bao nhiêu?` | tính hai completion, deterministic comparison, rồi magnitude |
| TY09 | Bốn raw value + hai derived value | `Cho actual, plan và tỷ lệ hoàn thành của cả 2022 và 2023` | structured multi-answer gồm 4 `SELECT` + 2 ratio DAG |
| TY10 | Tổng cấu thành qua hai năm | `Tổng A+B actual 2023 tăng bao nhiêu % so với tổng A+B actual 2022?` | hai `ADD` intermediate rồi `PERCENT_CHANGE` |
| TY11 | Hai metric qua hai năm | `Biên lợi nhuận 2023 so với 2022 thay đổi thế nào?` | ratio mỗi năm rồi so sánh, không lấy nhầm cell % lân cận |
| TY12 | Khác entity/scope qua hai năm | `Công ty mẹ 2023 so với hợp nhất 2022?` | mặc định scope mismatch; chỉ chạy khi cross-scope được yêu cầu tường minh và được gắn nhãn |

Phải phân biệt hai phép đo dễ nhầm:

```text
completion_2022 = actual_2022 / plan_2022 * 100
completion_2023 = actual_2023 / plan_2023 * 100

điểm phần trăm = completion_2023 - completion_2022
thay đổi tương đối (%) = (completion_2023 - completion_2022) / completion_2022 * 100
```

Nếu câu hỏi chỉ nói `thay đổi thế nào` mà không yêu cầu đơn vị, hệ thống có thể trả hai tỷ lệ và
nhãn tăng/giảm xác định; không được tự chọn giữa `điểm phần trăm` và `% thay đổi` để phát ra một
con số duy nhất. Nếu API chưa hỗ trợ structured multi-answer, phải abstain với reason riêng thay
vì trả một phần.

Hai câu dài sau phải biên dịch cùng intent và cùng program với câu PA03 ngắn:

```text
Doanh thu hoạt động thực hiện năm 2023 đạt bao nhiêu phần trăm kế hoạch năm 2023?

Theo bảng kết quả hoạt động sản xuất kinh doanh, doanh thu hoạt động thực hiện
năm 2023 của Công ty đạt bao nhiêu phần trăm so với mức kế hoạch được phê duyệt
cho cùng năm, sử dụng các số liệu cùng đơn vị trong chính bảng đó?
```

Nếu thêm một qualifier làm thay đổi nghĩa, plan phải thay đổi đúng slot thay vì coi đó là filler:

```text
... kế hoạch năm 2024 so với thực hiện năm 2023
... của công ty mẹ so với kế hoạch hợp nhất
... theo kế hoạch điều chỉnh thay vì kế hoạch ban đầu
```

### 15.4 Cấp độ kiểm thử

| Level | Đặc trưng | Kỳ vọng |
|---|---|---|
| Q0 | lookup một metric, một year, một cell duy nhất | SELECT đúng và citation đúng |
| Q1 | lookup có date/role/entity/unit/scope | qualifier bắt buộc được chứng minh |
| Q2 | một phép tính, hai operand | operation, order, role và denominator đúng |
| Q3 | multi-step DAG, 3–4 leaf hoặc mixed year/role | mọi intermediate proof-complete |
| Q4 | multi-clause hoặc multi-output | decompose xác định hoặc fail-closed có lý do |
| Q5 | ambiguity, missing role, conflict, cross-scope | không false accept |

Mỗi intent PA01–PA12 và TY01–TY12 phải có ít nhất các biến thể ngôn ngữ sau trên cùng evidence để tách
language robustness khỏi retrieval:

1. câu canonical ngắn;
2. ba paraphrase giữ nguyên nghĩa;
3. một câu dài tối thiểu 30 token có mệnh đề dẫn;
4. một bản không dấu hoặc punctuation khác;
5. một bản đổi thứ tự mệnh đề nhưng giữ operand role;
6. hai minimal-pair hoán đổi `plan ↔ actual` hoặc year;
7. một hard-negative thiếu role/header;
8. một ambiguity control có cùng metric/year nhưng khác value.

Compiler-only suite không cần mở PDF và phải khóa expected IR trước sửa code. Evidence integration
suite dùng development-only source đã có trước; chỉ kích hoạt Data Creation Gate nếu inventory audit
chứng minh thiếu layout/company, không mặc định đi tìm thêm PDF.

### 15.5 Thứ tự triển khai đã lập lại

#### QP0 — Freeze taxonomy và baseline

- [x] Materialize catalog câu hỏi ở 15.3 thành JSONL có `family_id`, question, expected IR,
  expected program/DAG hoặc expected abstention reason.
- [x] Ghi hash checkpoint, runtime, threshold config, existing Top-64 cache và toàn bộ development
  cohort; ABW mang nhãn diagnosis-only và exclusion cho confirmation.
- [x] Chạy compiler-only audit để đo exact operation, role, year, entity, unit, answer mode;
  không sửa runtime trong bước này.
- [x] Hoàn tất S4.3-A trên toàn bộ 976 development example để đo collateral của từng proposed
  obligation trước khi bật strict policy.

#### QP1 — Evidence schema và table-header reconstruction

- [x] Thêm `column_role`, `column_role_source`, `header_path` và confidence/provenance xác định
  vào `EvidenceCell`; đổi cache fingerprint/schema version.
- [x] Reconstruct hierarchical/merged headers để phân biệt `2022 actual`, `2023 plan`,
  `2023 actual`, `% actual/plan`, `% actual/prior`; không suy role chỉ từ thứ tự trái-phải.
- [x] Chuẩn hóa `actual/plan/revised_plan/forecast` nhưng giữ raw header và source span.
- [x] Khi hai parser/native-OCR cho role khác nhau, đánh conflict; không để neural score tự chọn.
- [x] Unit-test repeated header, merged cell, parallel panel, no-diacritic/OCR noise và candidate
  permutation.

#### QP2 — Question compiler cho role và answer mode

- [ ] Mở rộng `QuestionPlan` theo 15.2; parser phải gắn role theo từng operand, không chỉ toàn câu.
- [ ] Compile đầy đủ PA01–PA12; `vượt/chưa đạt/còn thiếu` phải khóa chiều và reference base;
  PA12 mặc định sinh cross-scope rejection trừ khi comparison scope được yêu cầu tường minh.
- [ ] Compile TY01–TY12 với role gắn riêng theo từng year; phân biệt `điểm phần trăm` với
  `% thay đổi`, và không dùng temporal order để thay thế column-role proof.
- [ ] Tách primary quantitative request khỏi discourse/follow-up dài mà không xóa entity, date,
  unit, role hoặc scope qualifier.
- [ ] Nhận diện multi-output C08 và context-dependent C11/C12 để decompose hoặc abstain rõ ràng;
  không âm thầm trả một phần câu hỏi.
- [ ] Giữ operation neural fallback chỉ cho intent thực sự không explicit; explicit plan/actual
  không được rơi về operation không ràng buộc.

#### QP3 — Synthesis, DAG và proof obligation

- [ ] Typed synthesis tạo beam riêng cho từng `(metric, year, column_role, entity, unit)` slot.
- [ ] Thêm `column_role_match`, `reference_base_match`, `answer_mode_complete` và
  `logical_cell_unique` vào proof contract với applicability tường minh.
- [ ] Candidate trái role/year/scope là `INVALID`; thiếu metadata là `UNKNOWN`; cả hai không được
  acceptance ở strict plan/actual lane.
- [ ] PA08/PA09/PA11 chỉ materialize khi mọi leaf và intermediate DAG node `VALID`; denominator
  non-zero và cùng unit/scope.
- [ ] TY05–TY11 chỉ materialize khi từng completion/ratio intermediate có đủ actual-plan role,
  year, denominator và unit proof; comparison label phải được suy ra từ giá trị executor đã xác minh.
- [ ] Không mở proof-conditioned acceptance hiện dành cho multi-step complete proof sang flat
  SELECT/ADD nếu chưa pass S4.3 counterfactual gate.

#### QP4 — Counterfactual và long-question stress

- [ ] Với mỗi positive, sinh minimal pair đổi actual/plan, year, entity, unit và direction; program
  cũ phải trở thành `INVALID`, không chỉ giảm score.
- [ ] So sánh câu ngắn/câu dài tương đương: expected IR, eligible program set và winner phải bất biến.
- [ ] Thử clause permutation, punctuation, không dấu, bilingual parenthetical, distractor clause,
  repeated metric và conflicting cell.
- [ ] Audit riêng câu multi-output, qualitative follow-up, anaphora và cross-document; báo coverage
  thật, không tính safe abstention là answered.

#### QP5 — Frozen replay và release gate

- [ ] Chạy ablation riêng: header role → question compiler → strict proof → multi-step DAG → hợp nhất.
- [ ] Replay development-only plan/actual cohort, toàn bộ 227 accepted SELECT/ADD và đủ 976 S4
  example bằng checkpoint/threshold frozen.
- [ ] Báo per-family, per-operation, per-level Q0–Q5, company/report/layout/extraction method;
  đo true accept loss lẫn false accept removal.
- [ ] Chỉ lock repair khi: explicit-intent compiler exact 100%; swapped-role false accept bằng 0;
  verifier gold retention thấp nhất mỗi fold ≥95%; overall coverage ≥88,5%; accepted precision
  development ≥90%; `9F - T >= 143`; operation phổ biến không giảm quá 1 điểm %; deterministic
  permutation/hash-seed; latency p95 tăng không quá 10% hoặc có ablation/budget được duyệt.
- [ ] Nếu fail, rollback feature flag và tiếp tục development diagnosis. Không mở validation để
  chọn rule, không hạ threshold và không fine-tune.
- [ ] Chỉ sau khi code/config/data/cache/report hash đã khóa mới precommit confirmation cohort
  development-only company-disjoint mới. Confirmation đó phải giữ nguyên operating-point rule;
  sau independent confirmation mới xem xét sealed validation/Phase D.

### 15.6 Artifacts dự kiến

- `config/question_family_catalog_plan_actual_v1.jsonl`;
- `reports/question_plan_compiler_audit_development_v1.json`;
- `reports/plan_actual_header_role_audit_development_v1.json`;
- `reports/plan_actual_counterfactual_development_v1.json`;
- `reports/plan_actual_long_question_stress_development_v1.json`;
- `reports/plan_actual_structural_repair_ablation_development_v1.json`;
- `reports/s4_question_coverage_release_gate_v1.json`;
- lock manifest chứa code/config/checkpoint/threshold/data/cache/report hash.

### 15.7 Ranh giới tích hợp LLM cho phân tích chỉ số (future extension)

LLM không phải điều kiện để hỗ trợ các câu so sánh định lượng TY01–TY12. Phần lõi phải tiếp tục
dùng question compiler, typed evidence, deterministic executor và verifier cho truy xuất số,
phép tính, chiều so sánh, nhãn tăng/giảm và citation. Thêm LLM sớm không được dùng để che lỗi
header reconstruction, column-role binding hoặc proof `UNKNOWN`.

Sau khi QP5 pass và cấu hình định lượng đã khóa, có thể mở một track riêng `LLM-A` với vai trò
giới hạn:

1. phân rã câu dài thành các subquery/typed IR hypothesis;
2. nhận diện paraphrase và discourse nhưng không tự tạo evidence role;
3. diễn giải bằng ngôn ngữ tự nhiên từ structured result đã được executor/verifier xác nhận;
4. tóm tắt nhiều chỉ số và nêu citation tương ứng;
5. phát hiện yêu cầu định tính/nguyên nhân để chuyển sang evidence retrieval bổ sung hoặc abstain.

Mọi output của LLM là untrusted proposal. Numeric value, operation, operand, unit, year, role,
scope và citation phải được kiểm tra lại bằng lõi neuro-symbolic. LLM không được tự tính số cuối,
không được thay threshold và không được biến candidate bị reject thành câu trả lời.

Phân biệt phạm vi kết luận:

- `Doanh thu 2023 tăng 12% so với 2022`: kết luận định lượng, không cần LLM;
- `Xu hướng tăng/giảm`: nhãn xác định từ kết quả đã kiểm chứng, không cần LLM;
- `Tăng là tốt hay xấu`: cần policy/ngữ cảnh nhiều chỉ số, LLM chỉ diễn giải trên evidence;
- `Vì sao doanh thu giảm`: chỉ kết luận nguyên nhân khi báo cáo nêu bằng chứng có citation;
  nếu tài liệu chỉ có số, trả `đã xác minh mức giảm nhưng chưa đủ bằng chứng về nguyên nhân`.

Đánh giá LLM phải tách khỏi accepted numeric precision: semantic-decomposition exactness,
unsupported-claim rate, citation entailment, numeric consistency và safe-abstention rate. Track
LLM không mở Phase D, không thay operating-point rule của lõi và không được dùng validation/blind
để prompt-tune trong giai đoạn development.

### 15.8 Kết quả QP0 — hoàn tất (13/08/2026)

- Đã khóa 59 question family thành 299 compiler contract synthetic không dùng PDF/numeric answer.
  PA01–PA12 và TY01–TY12 mỗi family có 11 biến thể gồm câu dài, không dấu, clause reorder,
  minimal pair role/year, missing-role hard-negative và ambiguity control. Catalog SHA-256 là
  `69c4017afec7731f63407339ae0c3d717ac38690807b17908b77fb1892d8e31b`.
- Compiler-only baseline đạt exact `7/299 = 2,34%`. Requested year và entity mode đạt 100%,
  operation `161/299 = 53,85%`, program DAG `205/299 = 68,56%`, column role chỉ
  `70/299 = 23,41%`; plan/actual và two-year category đều `0` exact. Bảy canonical family đang
  đạt là A01/A03/A04/A05/A06/A08/A09. Đây là baseline trước repair, không sửa parser/runtime.
- S4.3-A replay đúng frozen Top-64 trên 976 development example tái lập baseline `937` accepted,
  `829` correct, `108` false accept, precision `88,4739%`, coverage `96,0041%`; accepted
  SELECT/ADD đúng `227`. Frozen cache manifest giữ minimum fold raw gold program recall `98,9474%`.
- Không policy audit-only nào pass toàn bộ counterfactual gate. `logical_cell_unique` cân bằng nhất
  nhưng chỉ đạt precision `89,6476%`, coverage `93,0328%`, loại `F=14` false accept và mất
  `T=16` true accept. Policy hợp nhất đạt precision `90,5277%` nhưng coverage tụt còn `75,7172%`,
  mất `160` true accept, gold retention `78,4836%` và sinh hai false accept mới sau reselection;
  do đó tuyệt đối không được bật runtime.
- Kết luận: siết verifier trên metadata hiện tại gây collateral quá lớn. Phải thực hiện QP1 để
  tạo evidence role/provenance trước, rồi mới audit lại strict proof. Checkpoint hash giữ
  `3a33276ded0d2ff2dd3d200c4c774a59eec0f8cad7102bc273cb263d983944de`; threshold, runtime và
  neural score không đổi; không fine-tune; Phase D vẫn khóa.
- Regression liên quan đạt 67/67. Artifacts:
  `config/question_family_catalog_plan_actual_v1.jsonl`,
  `reports/question_family_catalog_plan_actual_v1_lock.json`,
  `reports/question_plan_compiler_audit_development_v1.json`,
  `reports/select_add_counterfactual_s4_development_v1.json`,
  `reports/question_coverage_qp0_development_v1.md` và
  `reports/question_coverage_qp0_lock_v1.json`.
- Hygiene: các script QP0 chỉ nạp allowlisted development/synthetic artifacts. Một broad search
  tương tác trước khi chạy script đã in một protected-report record cũ nhưng record không được dùng
  cho expected IR, policy hoặc metric; vì vậy phiên này không tuyên bố `zero protected read`.

**Trạng thái sau QP0:** QP1 — mở rộng EvidenceCell/cache schema và table-header reconstruction
để truyền `column_role`, `column_role_source`, `header_path` cùng provenance xác định. Chưa sửa
strict verifier/acceptance cho đến khi QP1 có positive, conflict và counterfactual extraction tests;
giữ nguyên checkpoint, threshold, runtime acceptance và Phase D lock.

### 15.9 Kết quả QP1 — hoàn tất (13/08/2026)

- Đã mở rộng `EvidenceCell` bằng `column_role`, nguồn role, trạng thái xác định
  `EXPLICIT/INHERITED/CONFLICT/UNKNOWN`, evidence, raw `header_path`, provenance từng tầng,
  raw header, extraction method và source span. Role không được suy từ giá trị số hoặc thứ tự
  trái-phải.
- Đã reconstruct Markdown header nhiều tầng/merged/repeated/parallel-panel và native plaintext
  header bị dàn nhiều dòng. Các role chuẩn gồm `actual`, `plan`, `revised_plan`, `forecast`,
  `actual_to_plan_ratio`, `actual_to_prior_ratio`; bất đồng native/OCR trên cùng logical cell trở
  thành `CONFLICT` thay vì để neural score chọn.
- Cache identity đã nâng từ `v12b2-applicability-proof-candidate-cache-v4` lên `v5`, gắn
  evidence schema `evidence-cell-header-role-v2` có fingerprint
  `de4c47be90bced3f4700a39aeebb53efdad081585a8169cfd547c77c6b3c89b0`. Cache cũ chưa rebuild
  và không hợp lệ cho schema mới.
- Audit development-only trên 1.246 example/37.781 frozen candidate ghi nhận 1.766 role rõ,
  36.015 `UNKNOWN`, 9 conflict candidate thuộc 6 example và 1.458/1.458 counterfactual
  `actual ↔ plan` flip đúng. Chỉ 115 example có role rõ vì cohort cũ chủ yếu là bảng theo kỳ;
  không dùng tỷ lệ `UNKNOWN` này làm accuracy/release metric.
- ABW diagnosis-only trang 10 đã tái dựng đúng năm cột: `363.638.445.420 = actual 2022`,
  `472.723.394.169 = plan 2023`, `304.912.639.828 = actual 2023`, `-35,50% = actual/plan`,
  `-16,15% = actual/prior`. ABW vẫn bị loại khỏi confirmation và không dùng chọn threshold.
- Regression liên quan đạt 109/109. Báo cáo:
  `reports/question_pipeline_qp1_development_v1.md`,
  `reports/plan_actual_header_role_audit_development_v1.json`,
  `reports/abw_qp1_diagnosis_only_v1.json`. Không fine-tune, không thay checkpoint/threshold,
  không bật strict verifier/acceptance; Phase D vẫn khóa.

**Trạng thái sau QP1:** QP2 — mở rộng `QuestionPlan` và compiler để gắn column role, year,
entity, unit và answer mode theo từng operand; compile PA01–PA12/TY01–TY12, gồm câu dài,
hai năm và multi-output. Chưa thêm `column_role_match` vào acceptance cho đến QP3.

### 15.10 Kết quả QP2/QP3 priority lane — hoàn tất (13/08/2026)

- Đã ưu tiên nối QP1 vào runtime cho explicit `actual/plan`: `QuestionPlan.column_roles` giữ role
  theo operand; lookup role rõ khóa `SELECT`; completion ratio khóa `ACTUAL ÷ PLAN`; typed synthesis
  dùng slot `(metric, year, column_role, entity, unit)`.
- Proof contract đã có required `column_role_match`; runtime loại role mismatch, missing hoặc
  conflict. Đây là strict grounding, không phải threshold mới và không cho neural score tự chọn role.
- Đã sửa lỗi lexical `hoạt động` bị hiểu nhầm là đơn vị `đồng`; metric ABW hiện giữ đúng
  `doanh thu hoạt động`.
- Compiler-only catalog giữ nguyên đạt `93/299 = 31,10%`, tăng từ `7/299`; column-role pass
  `212/299 = 70,90%`, operation pass `213/299 = 71,24%`, plan/actual exact `42/132`, two-year
  exact `36/132`. Đây là tiến bộ của priority lane, chưa phải QP2 complete.
- Full runtime ABW diagnosis-only đã chọn đúng `304.912.639.828 = actual 2023` thay vì
  `472.723.394.169 = plan 2023`. Required proof đạt 100% và `column_role_match=VALID`, nhưng vẫn
  ABSTAIN vì confidence `0,516992 < 0,845693`; threshold giữ nguyên.
- Audit-only frozen S4 chỉ có 2 accepted explicit-role SELECT: strict filter loại `F=1`, mất `T=0`,
  gold retention `2/2`; conservative precision chỉ `88,568%`, coverage `95,902%`, `9F−T=9`.
  Vì fail gate 90%/143 và cỡ mẫu quá nhỏ, proof-conditioned SELECT acceptance không được bật.
- Regression liên quan đạt 264/264. Không fine-tune, không đọc validation/blind/test, không đổi
  checkpoint/threshold; Phase D vẫn khóa. Artifacts:
  `reports/column_role_runtime_priority_qp2_development_v1.md`,
  `reports/question_plan_compiler_audit_qp2_priority_development_v1.json`,
  `reports/abw_column_role_runtime_qp2_diagnosis_only_v1.json`,
  `reports/column_role_select_priority_qp2_development_v1.json`.

**Bước thực hiện ngay:** tiếp tục QP2 cho PA04–PA12/TY, answer mode, câu dài/multi-output và
document-conditioned ambiguity; đồng thời tạo hoặc reuse cohort development-only company-disjoint
explicit-role positive + hard-negative đủ lớn. Chỉ sau counterfactual gate mới xem xét
proof-conditioned acceptance cho flat SELECT; không hạ threshold.

### 15.11 Chốt hướng khóa luận và publication track (13/08/2026)

Đã chốt định vị FinHRM là **selective neuro-symbolic financial document reasoning over a
provenance-aware, role-aware evidence graph**. Không chuyển hướng sang multi-agent RAG tổng quát,
không lấy Neo4j/Knowledge Graph deployment hoặc LLM sinh Python làm novelty trung tâm. Claim nghiên
cứu tập trung vào cell-level semantic identity `(metric, period, column_role, entity, unit)`, typed
program/DAG, three-state proof và selective acceptance trên counterfactual hard-negative.

Ba research question được khóa cho thesis/paper:

1. cell-level role-aware evidence graph có cải thiện operand/provenance grounding so với chunk hoặc
   section-level graph không;
2. typed synthesis + verifier có giảm semantic false accept mà giữ gold/true accept không;
3. proof-aware acceptance có cải thiện risk–coverage so với neural confidence threshold thuần không.

Phạm vi thesis core ưu tiên PA01–PA09, TY01/TY02/TY04, một multi-step financial identity và
ambiguity controls C13–C15. PA10–PA12, TY05–TY12, conversational/cross-document, causal analysis,
chart reasoning, multi-agent memory và enterprise KG được giữ làm extension, không được làm chậm
vertical slice. Graph nghiên cứu có thể là in-memory/SQLite typed graph; Neo4j không phải điều kiện.

Publication track yêu cầu benchmark real-PDF company-disjoint có gold logical/physical cell,
program/DAG, role/year/entity/unit/provenance, answerability và counterfactual hard-negative. Metric
trung tâm là provenance-correct execution, operand/program exact, accepted precision/coverage,
risk–coverage, hard-negative rejection và citation accuracy; RAGAS chỉ là metric phụ. Mọi baseline
phải chạy trên cùng corpus/split; không so trực tiếp RAGAS của `mau.md` với accepted precision.

Gate trước bài mạnh: sealed company-disjoint test mới; benchmark được human-audit; baseline
BM25/dense/section-GraphRAG/LLM-code/neural-only; ablation column-role/graph/synthesis/verifier/
calibration; confidence interval và statistical test; artifact tái lập. Trạng thái hiện tại chưa đủ
để hứa conference hoặc journal Q1/Q2 vì QP2/QP3 chưa hoàn chỉnh, precision còn dưới gate và chưa có
independent sealed test cho cấu hình cuối.

Roadmap đầy đủ: `FINHRM_THESIS_PUBLICATION_ROADMAP.md`.

**Bước thực hiện ngay được thay rõ hơn:** precommit thesis-core artifact gồm family list, expected
IR/DAG, annotation schema, company split rule, hard-negative taxonomy, metric và gates; review artifact
xong mới tiếp tục QP2/QP3. Không thêm LLM, không xây Neo4j production, không fine-tune và không thay
threshold trong bước precommit.

### 15.12 Thesis-core precommit — hoàn tất (13/08/2026)

Đã khóa machine-readable contract `config/finhrm_thesis_core_v1.json` trước mọi thay đổi QP2/QP3
tiếp theo. Contract định nghĩa 16 question family thuộc vertical slice: PA01–PA09, TY01/TY02/TY04,
IBDS01 và ambiguity controls C13–C15. Trong đó có 13 family phải trả lời, 3 family phải `ABSTAIN`,
3 family có DAG đa bước. Mỗi family đã có expected IR/DAG, typed operand slot, root operation,
answer mode, proof obligation và hành vi khi thiếu/xung đột bằng chứng.

Contract đồng thời khóa:

- evidence graph ở mức logical/physical cell với identity
  `(metric, period, column_role, entity, unit)` và provenance edge;
- three-state proof `VALID/INVALID/UNKNOWN`, trong đó mọi required obligation phải `VALID` mới được
  phép trả lời;
- 10 loại semantic hard-negative, gồm swap actual/plan, swap năm, sai entity/scope/unit/metric,
  thiếu role, role conflict và thiếu operand;
- annotation schema, company-disjoint split rule, metric, baseline/ablation và release gate;
- checkpoint, question-family catalog và financial-identity catalog bằng SHA-256.

Validation contract đạt 16/16 family, không lỗi và không warning; catalog alignment đạt 15/15 family
có trong catalog, còn IBDS01 được neo vào financial-identity catalog. Sáu unit test đều đạt. Báo cáo
validation ghi nhận `model_loaded=false`, `pdf_loaded=false`, `validation_loaded=false`,
`blind_test_loaded=false`, `fine_tuning_performed=false`, `threshold_changed=false` và
`runtime_changed=false`.

Artifacts:

- `FINHRM_THESIS_CORE_SPEC.md` — đặc tả đọc bởi người;
- `config/finhrm_thesis_core_v1.json` — source of truth machine-readable;
- `scripts/validate_finhrm_thesis_core.py` — validator và report generator;
- `tests/test_finhrm_thesis_core_precommit.py` — regression contract;
- `reports/finhrm_thesis_core_precommit_v1.json` — kết quả validation;
- `reports/finhrm_thesis_core_precommit_lock_v1.json` — manifest khóa artifact.

Không sửa parser, program synthesis, verifier hay acceptance runtime trong bước này; không đổi
checkpoint/threshold, không fine-tune và không đọc validation/blind/test. Phase D vẫn khóa ở baseline
accepted precision `88,4739% < 90%`.

**Bước thực hiện ngay:** chạy compiler-only baseline trên đúng 16 family thesis core và phân loại gap
theo từng trường IR/DAG/proof. Sau đó sửa QP2 theo thứ tự: operand role/year/entity/unit → DAG đa bước
→ answer mode/multi-output → ambiguity/abstention. Chỉ nối proof vào acceptance ở QP3 sau khi
counterfactual development-only pass; giữ nguyên threshold.

### 15.13 Thesis-core compiler-only baseline — hoàn tất (13/08/2026)

Đã chạy audit compiler-only trên đúng scope đã precommit, không đọc PDF/tài liệu, không load model hoặc
checkpoint để inference và không execution. Audit tách hai thước đo để không che gap kiến trúc:

1. `strict_canonical_contract`: 16 canonical family, bắt buộc compiler phải phát typed operand slots,
   reference base và DAG như thesis contract;
2. `legacy_catalog_robustness`: giữ nguyên scorer QP0/QP2 trên 135 frozen variant của 15 family có
   trong catalog. IBDS01 không được trộn vào metric này vì chỉ có canonical precommit; không sinh thêm
   variant sau khi nhìn kết quả.

Strict exact đạt `1/16 = 6,25%`, chỉ C13 đạt. Đây là baseline trước QP2A, không phải regression của
accepted answer. Các tín hiệu hiện có cho thấy hướng sửa có thể tái sử dụng phần lớn parser:

- requested year `13/13`, requested answer unit `16/16`, answer mode `16/16`;
- column role rời rạc `12/13`;
- root operation `9/13`, direction `8/13`;
- typed operand schema `0/13`, reference base `6/13`, financial identity IBDS01 `0/1`.

Legacy exact đạt `53/135 = 39,2593%`. PA01/PA02/TY01/TY02 đạt `9/11`; PA03/PA05 đạt `8/11`;
PA04/PA06/PA07/PA08/PA09/TY04 đều `0/11`. Có `82/135` variant chưa exact; toàn bộ 12 missing-role
hard-negative và 12 ambiguity-control variant đều chưa đạt hành vi mong đợi. Metric này chỉ dùng so
với lịch sử, không thay gate strict.

Gap đã phân hoạch và khóa thứ tự sửa:

1. `QP2A_TYPED_OPERAND_SCHEMA`: 13/13 answer family — thêm operand ID/metric/year/column role/entity/
   unit dưới dạng compiler-emitted immutable slots; audit tuyệt đối không tự dựng slot để cộng điểm;
2. `QP2B_OPERATION_DIRECTION`: PA04, PA06, PA07, PA08, PA09, TY04;
3. `QP2C_REFERENCE_BASE_PROGRAM_DAG`: chín family, gồm DAG đa bước PA08/PA09/IBDS01;
4. `QP2D_SAFE_ABSTENTION`: C14/C15 và false abstain PA07;
5. `QP2E_SURFACE_ROBUSTNESS`: replay nguyên 135 frozen rows;
6. chỉ mở QP3 proof integration sau khi explicit-intent và counterfactual compiler gate pass.

Artifacts:

- `scripts/audit_finhrm_thesis_core_compiler.py`;
- `tests/test_finhrm_thesis_core_compiler_audit.py`;
- `reports/finhrm_thesis_core_compiler_baseline_v1.json`;
- `reports/finhrm_thesis_core_compiler_baseline_v1.md`;
- `reports/finhrm_thesis_core_compiler_baseline_lock_v1.json`.

Regression contract/audit liên quan đạt 16/16. Precommit lock đã loại `strategy.md`, document index và
roadmap khỏi artifact hash vì đây là tài liệu mutable; immutable spec/contract/validator/test/report
vẫn được giữ hash. Không fine-tune, không đổi runtime/checkpoint/threshold, không đọc
global/sealed validation hoặc blind/test; Phase D vẫn khóa.

**Bước thực hiện ngay:** QP2A — mở rộng `QuestionPlan` bằng typed operand schema tổng quát và compiler
provenance cho intent. Trước khi nối vào synthesis, phải đạt unit test cho PA01/PA02/PA03/PA05/TY01/
TY02 cùng role/year swap; sau đó replay strict 16 + frozen 135 để đo collateral. Không sửa acceptance
trong QP2A.

### 15.14 QP2A typed operand schema — hoàn tất (13/08/2026)

Đã mở rộng `QuestionPlan` bằng immutable `OperandSlot` schema `question-operand-slot-v1` gồm
`id`, `metric`, `year`, `column_role`, `entity_scope`, `unit_family` và `intent_sources`. Slot chỉ
biên dịch question-side intent; không chứa numeric value, candidate index, page/table coordinate hoặc
evidence position. Field mới chưa được synthesis/verifier tiêu thụ và chưa nối vào acceptance.

Quy tắc QP2A:

- chỉ materialize plan/actual slot khi câu hỏi có role lexical rõ; thiếu role giữ slot rỗng;
- gắn từng role vào explicit year, tạo stable ID theo role/year cho câu hai năm;
- giữ role/year counterfactual; `A thấp hơn B` dùng thứ tự `B−A` thay vì hard-code PLAN−ACTUAL;
- loại discourse prefix/arithmetic suffix khỏi metric identity;
- phân biệt answer unit với leaf unit (`percent` answer nhưng currency operands);
- IBDS01 tạo ba leaf `short_term_borrowings`, `long_term_borrowings`, `total_capital`; role actual mặc
  định của financial-statement leaf có provenance riêng. Chưa kích hoạt identity/DAG trong QP2A.

QP2A gate đạt:

- canonical answer slot exact `13/13`;
- frozen answer variant slot exact `108/108`;
- missing-role non-materialization `12/12`;
- duplicate slot ID `0`, provenance violation `0`, collateral regression `0`;
- strict thesis-core exact tăng `1/16 → 5/16`; typed operand schema tăng `0/13 → 13/13`;
- legacy exact tăng `53/135 → 54/135`, improvement duy nhất là PA05 role-swap;
- regression liên quan đạt `116/116`.

`question_planner.py` là runtime source nên source hash đã thay đổi, nhưng slot mới chưa được runtime
synthesis/verifier đọc; acceptance policy, threshold và checkpoint không đổi. Audit không đọc PDF,
không load model, không fine-tune, không đọc global/sealed validation hoặc blind/test. Phase D vẫn khóa.

Artifacts:

- `tests/test_question_operand_slots_qp2a.py`;
- `scripts/audit_qp2a_operand_slots.py`;
- `tests/test_audit_qp2a_operand_slots.py`;
- `reports/finhrm_thesis_core_compiler_qp2a_v1.json`;
- `reports/finhrm_qp2a_operand_slots_development_v1.json`;
- `reports/finhrm_qp2a_operand_slots_development_v1.md`;
- `reports/finhrm_qp2a_operand_slots_lock_v1.json`.

**Bước thực hiện ngay:** QP2B — sửa operation/direction tổng quát cho PA04, PA06, PA07, PA08,
PA09 và TY04. Replay canonical 16, frozen answer variants 108 và role/year counterfactual; giữ slot
non-consuming. Chưa thêm reference base/DAG (QP2C) và chưa sửa acceptance (QP3).

### 15.15 QP2B operation/direction — hoàn tất (13/08/2026)

Đã sửa question compiler theo semantic cue tổng quát cho PA04, PA06, PA07, PA08, PA09 và TY04;
không dùng family ID, company, page, numeric value hoặc candidate position. Hai direction đa bước mới
được thêm là `PLAN_MINUS_ACTUAL` và `ACTUAL_MINUS_PLAN`; chúng chỉ mô tả intent của root
`RATIO_PERCENT`, chưa tạo executable DAG.

Quy tắc chính:

- `A trừ B` và `A vượt B bao nhiêu đồng` → `SUBTRACT + LEFT_MINUS_RIGHT`;
- `A thay đổi ... phần trăm so với B` → `PERCENT_CHANGE + LEFT_MINUS_RIGHT` khi A/B có role khác;
- so sánh PLAN/ACTUAL qua hai năm ưu tiên semantic language order, không để recency ép thành NEW−OLD;
- temporal cùng role như ACTUAL 2023/ACTUAL 2022 vẫn giữ `NEW_MINUS_OLD`;
- `còn thiếu ... để đạt` → root `RATIO_PERCENT` cùng target−observed direction;
- `vượt ... bao nhiêu phần trăm` → root `RATIO_PERCENT` cùng observed−base direction;
- mọi role-swap đổi direction theo lexical roles, không hard-code ACTUAL/PLAN.

QP2B gate đạt:

- operation+direction exact trên tất cả frozen answer variants `108/108`;
- sáu target family `54/54`;
- role-swap `12/12`, year-shift `12/12`;
- long/no-diacritic/clause-reordered `36/36`;
- QP2A retention giữ canonical slot `13/13` và frozen slot `108/108`;
- strict exact tăng `5/16 → 7/16`; operation `9/13 → 13/13`, direction `8/13 → 13/13`;
- legacy exact tăng `54/135 → 90/135`, 36 improvement và 0 regression;
- regression liên quan đạt `131/131`.

QP2B làm thay đổi planning behavior vì operation/direction là runtime field cũ. Tuy nhiên
`program_synthesis.py`, `symbolic_verifier.py`, acceptance policy, threshold và checkpoint không đổi;
typed operand slots vẫn non-consuming. Audit không đọc PDF, không load model, không fine-tune và không
đọc global/sealed validation hoặc blind/test. Phase D vẫn khóa.

Artifacts:

- `tests/test_question_operation_direction_qp2b.py`;
- `scripts/audit_qp2b_operation_direction.py`;
- `tests/test_audit_qp2b_operation_direction.py`;
- `reports/finhrm_thesis_core_compiler_qp2b_v1.json`;
- `reports/finhrm_qp2b_operation_direction_development_v1.json`;
- `reports/finhrm_qp2b_operation_direction_development_v1.md`;
- `reports/finhrm_qp2b_operation_direction_lock_v1.json`.

**Bước thực hiện ngay:** QP2C — thêm immutable `reference_base` và compiler-side program DAG cho
PA03, PA07, PA08, PA09, TY02, TY04 và IBDS01. DAG phải tham chiếu operand slot ID, pass topological/
unit/reference-base validation và replay 108 frozen answer variants. Chưa nối DAG mới vào runtime
execution/acceptance trong QP2C.

### 15.16 QP2C reference base và program DAG — hoàn tất (13/08/2026)

Đã thêm schema compiler-side bất biến cho reference base và program DAG. Mỗi DAG chỉ tham chiếu typed
operand slot ID hoặc output node đứng trước; không chứa numeric value, candidate index, page, table hoặc
vị trí vật lý của evidence. Validator fail-closed kiểm tra node uniqueness, arity, topological references,
unit compatibility, root operation và denominator/reference-base consistency.

Toàn bộ 13 answer family hiện có canonical DAG:

- SELECT/SUBTRACT đơn bước dùng một node `answer`;
- PA03 và PA07 dùng `plan` làm reference base;
- PA08 dùng `SUBTRACT(plan, actual) → RATIO_PERCENT(shortfall, plan)`;
- PA09 dùng `SUBTRACT(actual, plan) → RATIO_PERCENT(excess, plan)`;
- TY02 dùng `actual_2022` làm base cho thay đổi ACTUAL 2023/2022;
- TY04 dùng `actual_2023` làm base cho PLAN 2024 so với ACTUAL 2023;
- IBDS01 dùng `ADD(short_term_debt, long_term_debt) → RATIO_PERCENT(debt, total_capital)` và
  `program_identity_ids=(balance.financial_debt)`.

`program_identity_ids` được tách khỏi `financial_identity_ids` cũ. Việc tách này là bắt buộc để mô tả
identity ở compiler contract mà không vô tình kích hoạt consumer runtime hiện hữu. `program_dag`,
`reference_base` và `program_identity_ids` mới chưa được nối vào synthesis, verifier, executor hoặc acceptance.

QP2C gate đạt:

- canonical answer contract exact `13/13`;
- frozen answer variants exact `108/108`;
- ABSTAIN/ambiguity controls không sinh DAG `27/27`;
- role-swap `12/12`, year-shift `12/12`;
- PA08/PA09 multi-step variants `18/18`, IBDS01 canonical đạt;
- QP2A operand-slot retention `108/108` và QP2B operation/direction retention `108/108`;
- DAG validation error `0`, provenance violation `0`, legacy collateral regression `0`;
- strict thesis-core exact tăng `7/16 → 14/16`; reference base và program DAG đều `13/13`;
- regression đạt `193/193`.

QP2C có sửa runtime planning source, nhưng compiler DAG mới vẫn non-consuming. Không sửa source của
`program_synthesis.py`, `symbolic_verifier.py`, executor hoặc `program_data.py`; không đọc PDF, không load
model, không đọc global/sealed validation hoặc blind/test; không fine-tune và không đổi checkpoint,
threshold hay acceptance policy. Phase D vẫn khóa tại `88,4739% < 90%`.

Artifacts:

- `tests/test_question_program_dag_qp2c.py`;
- `scripts/audit_qp2c_reference_dag.py`;
- `tests/test_audit_qp2c_reference_dag.py`;
- `reports/finhrm_thesis_core_compiler_qp2c_v1.json`;
- `reports/finhrm_qp2c_reference_dag_development_v1.json`;
- `reports/finhrm_qp2c_reference_dag_development_v1.md`;
- `reports/finhrm_qp2c_reference_dag_lock_v1.json`.

**Bước thực hiện ngay:** QP2D — sửa safe abstention cho C14/C15 bằng conflict/underspecification contract
tổng quát. Replay toàn bộ frozen compiler cohorts và giữ QP2A–QP2C retention. Chưa nối compiler DAG hoặc
`program_identity_ids` vào runtime execution/acceptance; không thay threshold.

### 15.17 QP2D safe abstention — hoàn tất (13/08/2026)

Đã thêm immutable `QuestionPlan.abstention_reason` và compiler contract question-side cho ba trường hợp:

- `ambiguous_signed_direction`: có hai toán hạng nhưng thiếu hướng chênh lệch;
- `missing_reference_base`: relational-percent request thiếu đại lượng gốc/mẫu số;
- `missing_column_role`: bảng được mô tả có nhiều role loại trừ nhau nhưng request không chọn role.

Column-role conflict parser tách role thuộc context bảng khỏi role thực sự được hỏi. `Bảng có cả kế hoạch và
thực hiện` không còn tự tạo hai operand. Nếu phần request chọn rõ `thực hiện` hoặc `kế hoạch`, compiler vẫn tạo
SELECT một slot đúng role. Counterfactual gate đã bắt một phiên bản matcher quá rộng vì hiểu mọi tiền tố có từ
`bảng` là conflict; rule cuối chỉ kích hoạt khi bảng/báo cáo được nói rõ là có/gồm/chứa/thể hiện đồng thời nhiều
role.

Reference-base contract yêu cầu relational-percent cue như `lớn hơn`, `tăng`, `giảm`, `chiếm`, `đạt` hoặc
`hoàn thành bao nhiêu phần trăm` phải có base đã compile. Nếu câu bổ sung `so với`, target hoặc denominator rõ,
reason không được phát và DAG QP2C phải được giữ nguyên.

QP2D gate đạt:

- canonical ambiguity controls C13–C15 `3/3`;
- missing-reference surfaces `8/8`, missing-column-role surfaces `6/6`;
- explicit-reference repair `4/4`, explicit-role repair `3/3`;
- PA07 false-abstain control `9/9`;
- QP2A–QP2C frozen answer retention `108/108`;
- strict thesis-core exact tăng `14/16 → 16/16`; behavior và abstention reason đều `16/16`;
- legacy exact tăng `90/135 → 92/135`, đúng hai improvement C14/C15 và 0 regression;
- regression đạt `205/205`.

Strict compiler `16/16` không phải runtime accuracy và không mở Phase D. `abstention_reason` mới chưa được
`program_synthesis.py`, `symbolic_verifier.py`, `document_hrm.py`, `document_reasoning_engine.py` hoặc
`program_data.py` tiêu thụ. Compiler DAG và `program_identity_ids` vẫn non-consuming; acceptance policy,
threshold và checkpoint không đổi. Không đọc PDF, model, global/sealed validation hoặc blind/test; không
fine-tune. Phase D vẫn khóa tại `88,4739% < 90%`.

Artifacts:

- `tests/test_question_safe_abstention_qp2d.py`;
- `scripts/audit_qp2d_safe_abstention.py`;
- `tests/test_audit_qp2d_safe_abstention.py`;
- `reports/finhrm_thesis_core_compiler_qp2d_v1.json`;
- `reports/finhrm_qp2d_safe_abstention_development_v1.json`;
- `reports/finhrm_qp2d_safe_abstention_development_v1.md`;
- `reports/finhrm_qp2d_safe_abstention_lock_v1.json`.

**Bước thực hiện ngay:** QP2E — audit và sửa 43 surface variants còn chưa exact trên đúng frozen 135-row
compiler catalog. Phân loại theo missing-role, ambiguity và surface form trước khi sửa; giữ QP2A–QP2D retention,
giữ các field compiler mới non-consuming và không thay threshold/acceptance.

### 15.18 QP2E frozen surface robustness — hoàn tất (13/08/2026)

Đã phân hoạch 43 frozen surface failures sau QP2D thành `behavior=24` và `column_roles=19`. Không sửa theo
family ID, câu cụ thể, metric `doanh thu` hoặc năm 2022–2024.

Hai sửa đổi tổng quát:

- thêm immutable `program_column_roles`, chiếu leaf role theo use-site references trong program DAG; operand slot
  vẫn là unique identity, denominator reuse không tạo duplicate slot và legacy runtime `column_roles` không đổi;
- thêm `ambiguous_metric_or_column_role` khi câu tự mô tả bảng/báo cáo có nhiều cột nhưng request không
  xác định role. Một keyword đơn lẻ như `bảng` không đủ kích hoạt reason.

QP2E safe gate đạt:

- raw frozen exact tăng `92/135 → 123/135 = 91,11%`, 31 improvement và 0 regression;
- compiler-identifiable subset đạt `123/123`;
- answer variants retention `108/108`, ambiguity controls `12/12`;
- column-role use-site projection `135/135`, strict canonical thesis core `16/16`;
- counterfactual đổi metric/năm đạt `7/7`: lợi nhuận 2028, chi phí 2027/2026, tài sản 2029/2027;
- regression đạt `216/216`.

Không tuyên bố `135/135`. Mười hai residual đều là `hard_negative_missing_role` và không khả định bằng question
alone. Frozen catalog có collision trực tiếp trên cùng normalized question:

- `A05-canonical`: `Doanh thu năm 2023 tăng bao nhiêu phần trăm so với năm 2022?` → ANSWER;
- `TY02-hard_negative_missing_role`: đúng cùng câu → ABSTAIN.

Vì vậy không được ép mọi câu thiếu keyword `thực hiện/kế hoạch` thành ABSTAIN. Mười hai residual được phân loại
`evidence_conditioned_deferred`: QP3 chỉ được từ chối nếu evidence chứng minh tồn tại nhiều logical cell cùng
metric/year/entity/unit nhưng khác column role và câu hỏi không chỉ rõ role. Nếu evidence có một cell duy nhất phù
hợp, thiếu role keyword không đủ để từ chối.

`program_column_roles`, `abstention_reason`, compiler DAG và `program_identity_ids` vẫn non-consuming. Không sửa
synthesis/verifier/executor, không đọc PDF/model/global/sealed validation hoặc blind/test, không fine-tune và không
đổi threshold/acceptance. Phase D vẫn khóa tại `88,4739% < 90%`.

Artifacts:

- `tests/test_question_surface_robustness_qp2e.py`;
- `scripts/audit_qp2e_surface_robustness.py`;
- `tests/test_audit_qp2e_surface_robustness.py`;
- `reports/finhrm_thesis_core_compiler_qp2e_v1.json`;
- `reports/finhrm_qp2e_surface_robustness_development_v1.json`;
- `reports/finhrm_qp2e_surface_robustness_development_v1.md`;
- `reports/finhrm_qp2e_surface_robustness_lock_v1.json`.

**Bước thực hiện ngay:** QP3A precommit — định nghĩa evidence-conditioned column-role proof cho 12 deferred hard
negatives, gồm unique-role positive, multiple-role conflict, role-swap và no-role counterfactual. Chưa sửa runtime
trước khi khóa cohort/gate; giữ nguyên threshold.

### 15.19 QP3A evidence-conditioned column-role proof precommit — hoàn tất (13/08/2026)

Đã khóa proof semantics, cohort, quota, gold state/action và runtime hashes trước mọi thay đổi QP3 implementation.
QP3A chỉ tạo development-only synthetic evidence; không đọc PDF/content, không load model và không sửa runtime.

Evidence-conditioned rule đã khóa cho query không nêu role:

- logical-role group key gồm document/table/row/metric/period/entity/unit, loại column role, numeric value,
  candidate score và candidate order khỏi key;
- deduplicate theo physical-equivalence key trước khi đếm role;
- từ hai explicit role khác nhau trong cùng requested logical group → `INVALID / ABSTAIN`;
- same-value khác role vẫn conflict;
- cross-extractor role disagreement trên cùng physical evidence → `INVALID / ABSTAIN`;
- role inventory chưa chứng minh đầy đủ → `UNKNOWN / ABSTAIN`;
- chỉ `EXPLICIT_COMPLETE` cùng đúng một compatible role trên mỗi logical group mới cho obligation này `VALID`;
- `VALID` không tự cho phép answer: mọi applicable proof obligation khác vẫn phải VALID.

Query có explicit role giữ policy hiện hành: match → VALID, mismatch → INVALID/REJECT, thiếu metadata →
UNKNOWN/ABSTAIN.

Cohort đã freeze gồm `128` examples:

- 12 deferred family groups × 8 scenario = 96;
- 4 metric/year generalization groups × 8 scenario = 32;
- tám scenario gồm explicit positive, role-swap negative, no-role unique, multi-role conflict, same-value conflict,
  permutation, unknown inventory và cross-extractor conflict.

QP3A precommit gate đạt:

- oracle state/action `128/128`;
- same-value conflict `16/16`, permutation invariance `16/16`, unknown inventory `16/16`;
- 12 deferred groups và 4 generalization groups đủ quota;
- 6 metric và 7 year (`2022`, `2023`, `2024`, `2027`, `2028`, `2029`, `2030`);
- malformed/leakage `0`; không numeric value/answer, document content hoặc candidate score;
- runtime baseline sáu file không có hash mismatch;
- regression đạt `222/222`.

Artifacts:

- `config/finhrm_qp3a_column_role_proof_precommit_v1.json`;
- `scripts/build_qp3a_column_role_proof_cohort.py`;
- `data/finhrm_qp3a_column_role_proof_development_v1.jsonl`;
- `scripts/validate_qp3a_column_role_proof_precommit.py`;
- `tests/test_qp3a_column_role_proof_precommit.py`;
- `reports/finhrm_qp3a_column_role_proof_precommit_v1.json`;
- `reports/finhrm_qp3a_column_role_proof_precommit_v1.md`;
- `reports/finhrm_qp3a_column_role_proof_precommit_lock_v1.json`.

Không sửa synthesis/verifier/executor/acceptance, không thay threshold/checkpoint, không fine-tune và không mở
Phase D.

**Bước thực hiện ngay:** QP3B — implement đúng evidence-conditioned role proof đã precommit trên frozen cohort
128 case; bắt buộc state/action 128/128, same-value/permutation/unknown controls và QP2A–QP2E retention. Không
thay threshold.

### 15.20 QP3B evidence-conditioned column-role proof — hoàn tất (13/08/2026)

Đã triển khai proof contract QP3A vào evidence schema, symbolic verifier và grounding gate. `EvidenceGraph` có
inventory state tường minh `EXPLICIT_COMPLETE`/`UNKNOWN`; trạng thái rỗng được giữ riêng để tương thích ngược và
không được suy diễn là inventory đầy đủ từ Top-64.

Quy tắc runtime:

- query có explicit role tiếp tục dùng required obligation `column_role_match`;
- query không nêu role chỉ kích hoạt required `column_role_unique` khi inventory được khai báo hoặc evidence có
  conflict thực tế;
- nhiều role trong cùng logical identity, same-value khác role và cross-extractor disagreement đều
  `INVALID / ABSTAIN`;
- inventory `UNKNOWN` tạo required proof `UNKNOWN / ABSTAIN` cùng grounding reason
  `missing_column_role_inventory`;
- inventory không khai báo và không có conflict giữ obligation `not_applicable`, nên không gây coverage regression;
- `VALID` chỉ defer sang các obligation khác, không tự cấp quyền answer.

Frozen development-only cohort đạt state/action `128/128`; same-value, permutation và unknown-inventory đều
`16/16`. Regression liên quan QP0–QP2E, evidence/verifier, identity, direction, safety và `document_hrm` đạt
`477/477`. Hash QP3A
precommit/cohort/lock và checkpoint giữ nguyên. Không dùng numeric answer, candidate score, PDF/document content,
global/sealed validation hoặc blind/test; không fine-tune và không đổi threshold/checkpoint. Phase D vẫn khóa tại
accepted precision baseline `88,4739% < 90%`.

Artifacts:

- `src/hrm_finance_qa/evidence_schema.py`;
- `src/hrm_finance_qa/question_planner.py`;
- `src/hrm_finance_qa/symbolic_verifier.py`;
- `src/hrm_finance_qa/document_hrm.py`;
- `scripts/evaluate_qp3b_column_role_proof.py`;
- `tests/test_qp3b_column_role_proof.py`;
- `tests/test_column_role_runtime_qp2.py`;
- `reports/finhrm_qp3b_column_role_proof_development_v1.json`;
- `reports/finhrm_qp3b_column_role_proof_development_v1.md`;
- `reports/finhrm_qp3b_column_role_proof_lock_v1.json`.

**Bước thực hiện ngay:** QP3C precommit development-only provenance audit cho producer của role inventory trong
table reconstruction. Chỉ table có header/cell coverage đầy đủ và không có extractor conflict mới được gắn
`EXPLICIT_COMPLETE`; trường hợp còn lại phải `UNKNOWN`. Khóa rule/cohort trước khi nối producer vào real runtime và
replay end-to-end; tiếp tục giữ nguyên threshold.

### 15.21 QP3C role-inventory producer precommit — hoàn tất (13/08/2026)

Đã khóa contract, cohort, oracle state/reason, quota và runtime baseline trước khi triển khai producer vào table
reconstruction. QP3C chỉ dùng synthetic table provenance development-only; không đọc PDF/content, không load model
và không sửa runtime.

Một table chỉ được gắn `EXPLICIT_COMPLETE` khi đồng thời chứng minh được:

- table boundary `COMPLETE`, header alignment `ALIGNED`;
- reconstructed header IDs khớp chính xác toàn bộ expected data-column IDs;
- expected cell coordinates đúng tích `row IDs × data-column IDs`, observed coordinates phủ đúng toàn bộ tập;
- mọi data column có canonical role với confidence `EXPLICIT` hoặc `INHERITED` và provenance hợp lệ;
- role inherited có explicit anchor cùng role;
- mọi observed cell có document/page/source-span/extraction-method provenance;
- không có header, physical-equivalence hoặc cross-extractor conflict.

Thiếu bất kỳ proof nào đều cho `UNKNOWN`. Contract cấm dùng Top-K candidates, question text, numeric value/answer,
candidate score/rank/order để suy đoán completeness.

Frozen cohort có `96` case = 8 layout × 12 scenario. Kết quả:

- inventory state và exact reason-set `96/96`;
- `EXPLICIT_COMPLETE` positive `24/24`, `UNKNOWN` control `72/72`;
- permutation invariance `8/8`, inherited-anchor positive `8/8`, conflict negative `24/24`;
- 6 canonical role và 8 layout type; malformed/leakage `0`;
- regression retention `485/485`;
- bảy runtime hash và checkpoint giữ nguyên QP3B baseline.

Không fine-tune, không đổi threshold/acceptance/checkpoint, không đọc global/sealed validation hoặc blind/test.
Phase D vẫn khóa tại accepted precision baseline `88,4739% < 90%`.

Artifacts:

- `config/finhrm_qp3c_role_inventory_producer_precommit_v1.json`;
- `scripts/build_qp3c_role_inventory_producer_cohort.py`;
- `data/finhrm_qp3c_role_inventory_producer_development_v1.jsonl`;
- `scripts/validate_qp3c_role_inventory_producer_precommit.py`;
- `tests/test_qp3c_role_inventory_producer_precommit.py`;
- `reports/finhrm_qp3c_role_inventory_producer_precommit_v1.json`;
- `reports/finhrm_qp3c_role_inventory_producer_precommit_v1.md`;
- `reports/finhrm_qp3c_role_inventory_producer_precommit_lock_v1.json`.

**Bước thực hiện ngay:** QP3D — implement đúng table-level role-inventory producer đã khóa, nối state vào
`EvidenceGraph`, replay nguyên frozen QP3C cohort và development-only end-to-end cases. Không đổi threshold và chưa
mở validation/blind/test.

### 15.22 QP3D table-level role-inventory producer — hoàn tất (13/08/2026)

Đã triển khai producer theo đúng QP3C lock tại ba tầng:

- `program_data.py` tạo full-table structural manifest trước candidate Top-K;
- `evidence_schema.py` audit boundary/header/cell/inheritance/physical/extractor provenance và truyền inventory theo
  từng `(source, table)` vào `EvidenceGraph`;
- `symbolic_verifier.py` chỉ dùng inventory của đúng table chứa selected logical evidence, không để state từ table
  khác lan sang chương trình hiện tại.

Producer không dùng question text, numeric value/answer hoặc candidate score/rank/order. Table chỉ có ordinary
period columns (`2022/2023`, đầu/cuối kỳ) và plaintext không có manifest giữ proof role-inventory ở trạng thái không
áp dụng; chúng không bị ép thành plan/actual ambiguity.

Kết quả:

- frozen QP3C inventory state/reason-set `96/96`;
- development-only end-to-end controls `8/8`;
- pre-Top-K control chứng minh manifest giữ đủ `80` table cells dù candidates bị prune còn `64`;
- regression liên quan `483/483`;
- runtime change set đúng ba file `program_data.py`, `evidence_schema.py`, `symbolic_verifier.py`; checkpoint,
  question planner, synthesis, executor và reasoning engine không đổi.

Hai real development-only ABW replay cho query `Doanh thu hoạt động thực hiện năm 2023 là bao nhiêu?` đều
`ABSTAIN`, false accept `0/2`. Replay A tạo `SELECT(1,)`, chọn đúng `304.912.639.828 VND`, role actual, năm 2023,
trang 10 nhưng confidence `0,517 < 0,846`. Replay B tạo sai `ADD(3, 10)` nhưng bị grounding gate chặn bởi
`missing_explicit_addition_intent` và confidence `0,129 < 0,522`. Neural proposal chưa ổn định, vì vậy kết quả này
chỉ chứng minh safe abstention chứ chưa chứng minh answer coverage. ABW là flattened native plaintext nên chưa có
full-table manifest; QP3D không suy diễn completeness từ Top-64 và giữ inventory undeclared.

Không fine-tune, không đọc global/sealed validation hoặc blind/test, không đổi threshold/checkpoint. Frozen QP3C
artifacts giữ nguyên. Phase D vẫn khóa tại accepted precision baseline `88,4739% < 90%`.

Artifacts:

- `src/hrm_finance_qa/program_data.py`;
- `src/hrm_finance_qa/evidence_schema.py`;
- `src/hrm_finance_qa/symbolic_verifier.py`;
- `scripts/evaluate_qp3d_role_inventory_producer.py`;
- `tests/test_qp3d_role_inventory_producer.py`;
- `reports/finhrm_qp3d_role_inventory_producer_development_v1.json`;
- `reports/finhrm_qp3d_role_inventory_producer_development_v1.md`;
- `reports/finhrm_qp3d_abw_development_replay_v1.json`;
- `reports/finhrm_qp3d_role_inventory_producer_lock_v1.json`.

**Bước thực hiện ngay:** QP3E precommit development-only flattened-table provenance contract cho native
plaintext/OCR layouts như ABW. Phải khóa explicit header-role cardinality, row/cell alignment, source-line boundary,
positive và counterfactual hard-negative trước khi mở rộng producer. Không dùng question/answer/Top-K và không đổi
threshold.

### 15.23 QP3E flattened native/OCR provenance precommit — hoàn tất (14/08/2026)

Đã khóa contract, synthetic development-only cohort, oracle state/reason, quota và runtime baseline trước khi sửa
flattened table reconstruction. QP3E không đọc PDF/content, không load model và không sửa runtime.

Contract tách rõ hai quyền: `RECONSTRUCTABLE_COMPLETE` chỉ cho phép tạo structural manifest; manifest vẫn phải qua
QP3D inventory audit và mọi proof obligation hiện hành. Trạng thái này không trực tiếp tạo `EXPLICIT_COMPLETE`,
`VALID`, `ACCEPT` hoặc đáp án.

Một flattened native/OCR table chỉ đạt `RECONSTRUCTABLE_COMPLETE` khi đồng thời chứng minh:

- source boundary hoàn chỉnh và table continuation đã đóng;
- declared column IDs, data width và header-role cardinality khớp chính xác;
- mỗi header có một canonical role explicit hoặc inherited có anchor;
- row IDs đầy đủ và cell observations phủ đúng tích `rows × columns`;
- mỗi physical cell thuộc đúng một row/column, không có span overlap conflict;
- header, row descriptor và cell có document/page/span/sequence/extractor/bbox provenance;
- header đứng trước cells theo physical source order và column bands thẳng hàng;
- repeated headers cùng native/OCR views không xung đột;
- kết quả bất biến với serialization order.

Thiếu hoặc xung đột bất kỳ proof nào đều cho `UNKNOWN`. Question text, numeric value/answer, Top-K, candidate
score/rank/order bị cấm làm completeness signal.

Frozen cohort có `128` case = 8 layout × 16 scenario:

- state và exact reason-set `128/128`;
- positive `RECONSTRUCTABLE_COMPLETE` `24/24`, `UNKNOWN` controls `104/104`;
- permutation invariance `8/8`, repeated-header positive `8/8`;
- boundary hard-negative `16/16`, conflict hard-negative `32/32`;
- 13 loại hard-negative, 7 canonical role và 8 layout type;
- malformed/leakage `0`, runtime hash mismatch `0`;
- QP3E cùng related QP3D/QP3B/QP2/QP1 tests đạt `90/90`.

Không fine-tune, không đổi threshold/checkpoint/acceptance, không đọc global/sealed validation hoặc blind/test.
QP3D lock, QP3C lock và checkpoint giữ nguyên. Phase D vẫn khóa tại accepted precision baseline
`88,4739% < 90%`.

Artifacts:

- `config/finhrm_qp3e_flattened_table_provenance_precommit_v1.json`;
- `scripts/build_qp3e_flattened_table_provenance_cohort.py`;
- `data/finhrm_qp3e_flattened_table_provenance_development_v1.jsonl`;
- `scripts/validate_qp3e_flattened_table_provenance_precommit.py`;
- `tests/test_qp3e_flattened_table_provenance_precommit.py`;
- `reports/finhrm_qp3e_flattened_table_provenance_precommit_v1.json`;
- `reports/finhrm_qp3e_flattened_table_provenance_precommit_v1.md`;
- `reports/finhrm_qp3e_flattened_table_provenance_precommit_lock_v1.json`.

**Bước thực hiện ngay:** QP3F — implement đúng frozen flattened native/OCR reconstruction contract. Chỉ
observations đạt `RECONSTRUCTABLE_COMPLETE` mới được chuyển thành QP3D structural manifest; replay nguyên frozen
QP3E cohort và development-only end-to-end cases. Không đổi threshold.

### 15.24 QP3F flattened native/OCR reconstruction — hoàn tất (14/08/2026)

Đã triển khai frozen QP3E contract vào runtime:

- `flattened_table_reconstruction.py` đánh giá boundary, continuation, header-role cardinality, declared width,
  row×column coverage, physical source order, geometry, repeated-header và extractor conflict;
- `program_data.py` tạo flattened observations trước candidate Top-K và chỉ chuyển thành QP3D manifest khi state
  là `RECONSTRUCTABLE_COMPLETE`;
- các chunk chồng lấn chỉ được hợp nhất khi cùng source/page/extractor, global lines liên tục và phần overlap không
  xung đột;
- native page-index content version tăng lên `8` để không chủ động xóa horizontal layout whitespace còn giữ được
  từ PDF text layer.

`RECONSTRUCTABLE_COMPLETE` chỉ cấp quyền tạo structural manifest. QP3D inventory audit và toàn bộ proof obligation
hiện hành vẫn quyết định acceptance; QP3F không trực tiếp cấp `VALID` hoặc `ACCEPT`.

Kết quả development-only:

- frozen QP3E state `128/128`, exact reason-set `128/128`;
- manifest emission policy `128/128` — chỉ 24 complete positives phát manifest, 104 unknown controls không phát;
- end-to-end controls `9/9`, gồm overlapping-chunk reconstruction;
- pre-Top-K control giữ `100` physical coordinates trước khi candidate set bị prune còn `64`;
- runtime/related regression `306/306`.

Real ABW replay đã hợp nhất page-10 chunks thành run `1000001–1000037` và recover năm explicit column labels. Tuy
nhiên native text layer tách ratio header qua nhiều dòng và không giữ đủ column-coordinate provenance, nên QP3E
geometry proof không đạt; QP3F không suy diễn theo ordinal position và không phát manifest. Neural head đề xuất sai
`ADD(3, 13)` nhưng bị chặn bởi confidence `0,125 < 0,522` cùng `missing_explicit_addition_intent`; kết quả
`ABSTAIN`, false accept `0`. Vì vậy QP3F chứng minh safe reconstruction policy nhưng chưa chứng minh ABW answer
coverage.

Không fine-tune, không đổi threshold/checkpoint/acceptance, không đọc global/sealed validation hoặc blind/test.
Frozen QP3E artifacts giữ nguyên. Phase D vẫn khóa tại accepted precision baseline `88,4739% < 90%`.

Artifacts:

- `src/hrm_finance_qa/flattened_table_reconstruction.py`;
- `src/hrm_finance_qa/program_data.py`;
- `src/hrm_finance_qa/document_sources.py`;
- `src/hrm_finance_qa/document_page_index.py`;
- `scripts/evaluate_qp3f_flattened_table_reconstruction.py`;
- `tests/test_qp3f_flattened_table_reconstruction.py`;
- `reports/finhrm_qp3f_flattened_table_reconstruction_development_v1.json`;
- `reports/finhrm_qp3f_flattened_table_reconstruction_development_v1.md`;
- `reports/finhrm_qp3f_abw_development_replay_v1.json`;
- `reports/finhrm_qp3f_flattened_table_reconstruction_lock_v1.json`.

**Đính chính roadmap (14/08/2026):** đề xuất `QP3G` word/bbox provenance ở trên **bị rút khỏi chuỗi
thực hiện ngay**. Đây là một extension OCR/layout có ích, nhưng được nảy sinh từ giới hạn của smoke case ABW và
không phải obligation kế tiếp của thesis-core precommit. Không dùng ABW để đổi thứ tự roadmap.

### 15.25 Strategy alignment correction — hoàn tất (14/08/2026)

Đã audit lại QP2–QP3F theo source of truth `config/finhrm_thesis_core_v1.json`,
`FINHRM_THESIS_CORE_SPEC.md` và `FINHRM_THESIS_PUBLICATION_ROADMAP.md`. Kết luận:

- ABW chỉ là **diagnosis-only smoke case**, bị loại khỏi confirmation, calibration và mọi release gate;
- runtime `src/hrm_finance_qa` không chứa tên/hash file ABW, ba giá trị ABW hoặc rule theo company/page/year/value;
- QP3E–QP3F được giữ lại như provenance safeguard tổng quát vì rule chỉ dùng boundary, header/cell coverage,
  source span, geometry và extractor conflict; frozen synthetic cohorts và regression không phụ thuộc ABW;
- kết quả ABW trong các report QP1/QP2/QP3D/QP3F chỉ là lịch sử chẩn đoán, không phải bằng chứng
  generalization và không được tính vào metric/gate;
- `QP3G` cũ không còn là bước được phép thực hiện ngay. Word/bbox/OCR robustness chuyển về extension sau thesis
  core, chỉ mở lại khi có cohort real-PDF nhiều công ty/layout được precommit độc lập.

Đối chiếu thesis-core cho thấy QP3 **chưa hoàn tất**. Các phần đã có gồm typed operand slots/compiler DAG,
strict `column_role_match`, evidence-conditioned `column_role_unique` và fail-closed inventory provenance.
Các phần còn thiếu trong runtime proof contract là:

1. `reference_base_match` cho ratio/percent-change;
2. `logical_cell_unique` và `source_provenance_complete` cho mọi answer family;
3. `answer_mode_complete` khi request có nhiều output hoặc không thể trả một phần;
4. `all_leaf_nodes_valid`, `all_intermediate_nodes_valid` và `dag_execution_valid` cho PA08, PA09 và IBDS01;
5. nối typed operand slots/program DAG vào synthesis–verifier–executor theo proof, thay vì chỉ giữ compiler IR
   hoặc materialize DAG sau khi neural flat program đã thắng.

Không rollback QP3E–QP3F vì audit không tìm thấy report-specific runtime signal và rollback sẽ loại bỏ guard tổng
quát đã pass frozen controls. Không tuyên bố QP3 hoàn tất, không chuyển sang QP4, không mở acceptance mới và không
dùng ABW để tối ưu coverage.

**Bước thực hiện ngay đã sửa:** `QP3-CORE-PRECOMMIT` — khóa một cohort development-only,
company/layout-disjoint với mọi diagnosis document, bao phủ đủ 13 answer family của thesis core và C13–C15. Cohort
phải có positive, swapped role/year, sai reference base, duplicate logical cell, thiếu provenance, leaf/intermediate
`UNKNOWN/INVALID`, cùng các DAG PA08/PA09/IBDS01. Khóa oracle proof state/action, gold retention, false-accept gate
và runtime hash **trước khi sửa consumer**. Sau khi precommit pass mới triển khai `QP3-CORE-IMPLEMENT`; sau đó mới
sang QP4 counterfactual/long-question stress. Giữ nguyên checkpoint, threshold và Phase D lock
`88,4739% < 90%`; không đọc global/sealed validation hoặc blind/test, không fine-tune.

### 15.26 Ràng buộc report-agnostic generalization — đã khóa (14/08/2026)

Từ bước này, mọi thay đổi parser/retrieval/reconstruction/compiler/synthesis/verifier/executor/acceptance phải tuân
theo addendum `config/finhrm_report_generalization_invariant_v1.json`:

- một implementation duy nhất áp dụng không đổi cho mọi báo cáo trong declared supported domain;
- cấm rule theo company, filename/hash, page, fixed year, numeric value/answer, câu literal hoặc example/family ID;
- diagnosis document chỉ dùng tìm lỗi, không được dùng làm confirmation, calibration hoặc release evidence;
- mỗi repair phải precommit cohort có ít nhất 3 công ty và 3 layout family company-disjoint với diagnosis, gồm
  positive, semantic hard-negative, counterfactual và permutation controls;
- sửa layout phải có layout-disjoint confirmation; mọi frozen development cohort phải được replay nguyên vẹn;
- kết quả phải báo macro/worst-group theo company và layout, cùng unseen-company/unseen-layout;
- report ngoài miền hoặc thiếu proof/provenance phải `ABSTAIN`; safe abstention không được tính là answered.

Đây là ràng buộc về **phương pháp tổng quát và fail-closed behavior**, không phải tuyên bố coverage 100% cho mọi PDF
trên thế giới. Chỉ sealed company/layout-disjoint evaluation sau khi khóa method mới có quyền chứng minh mức độ tổng
quát hóa; development diagnosis không có quyền đó.

Ràng buộc này có hiệu lực ngay cho `QP3-CORE-PRECOMMIT` và mọi bước sau. Không sửa thesis-core v1 đã khóa, không
thay threshold/checkpoint, không fine-tune và không mở Phase D.

Artifacts khóa: `config/finhrm_report_generalization_invariant_v1.json`,
`tests/test_finhrm_report_generalization_invariant.py` và
`reports/finhrm_report_generalization_invariant_lock_v1.json`. Contract validation đạt `11/11` khi chạy cùng
thesis-core precommit tests.

### 15.27 QP3-CORE-PRECOMMIT — hoàn tất (14/08/2026)

Đã khóa proof-oracle development-only **trước khi sửa consumer runtime**. Cohort có `122` fixture typed-proof trên
đủ `16` family PA01–PA09, TY01/TY02/TY04, IBDS01 và C13–C15. Trong đó có `113` answer-family case và `9`
ambiguity-control case; positive gold/permutation, swapped role/year, sai reference base, duplicate logical cell,
thiếu source provenance, incomplete multi-output và leaf/intermediate/DAG `UNKNOWN/INVALID` đều được khóa tường minh.

Để không biến một diagnosis report thành confirmation, cohort không đọc hay sao chép nội dung PDF thật. Sáu company
identity và sáu layout family nằm trong namespace synthetic confirmation riêng, disjoint-by-construction với mọi
diagnosis source; ABW không xuất hiện trong cohort và không tham gia oracle, metric hoặc gate. Đây là **contract
precommit**, không phải bằng chứng empirical rằng runtime đã generalize trên sáu công ty/layout thật. Bằng chứng đó
chỉ được báo sau `QP3-CORE-IMPLEMENT` khi replay nguyên các development cohort company/layout-disjoint đã khóa; sealed
evaluation vẫn chưa được mở.

Kết quả khóa:

- oracle proof state/action `122/122`, false accept `0`;
- positive gold retention `26/26`, candidate permutation `16/16`;
- macro/worst-group oracle exact theo company `1,000/1,000`, theo layout `1,000/1,000`; safe abstention không được tính
  là answered và không có universal coverage claim;
- runtime hash mismatch `0`; checkpoint, threshold, acceptance, training và Phase D giữ nguyên;
- regression liên quan QP2E/QP3B/QP3D/QP3F cùng thesis-core/generalization đạt `54/54`.

Artifacts khóa: `config/finhrm_qp3_core_precommit_v1.json`,
`data/finhrm_qp3_core_development_v1.jsonl`, `scripts/build_qp3_core_precommit_cohort.py`,
`scripts/validate_qp3_core_precommit.py`, `tests/test_qp3_core_precommit.py`,
`reports/finhrm_qp3_core_precommit_v1.json`, `reports/finhrm_qp3_core_precommit_v1.md` và
`reports/finhrm_qp3_core_precommit_lock_v1.json`.

**Bước được phép tiếp theo:** `QP3-CORE-IMPLEMENT` — nối typed operand slots/DAG vào
synthesis–verifier–executor theo oracle đã khóa; required `UNKNOWN` phải `ABSTAIN`, `INVALID` phải `REJECT`, không mở
acceptance mới và không đổi threshold. Sau implementation phải replay toàn bộ compiler, proof, QP3A–QP3F, frozen S4
và development company/layout-disjoint trước khi được xét sang QP4.

### 15.28 QP3-CORE-IMPLEMENT — implementation hoàn tất, replay gate còn khóa (14/08/2026)

Đã nối immutable typed operand slots và compiler DAG vào synthesis, symbolic verifier, executor và final acceptance.
Runtime hiện chứng minh và gate tường minh `reference_base_match`, `logical_cell_unique`,
`source_provenance_complete`, `answer_mode_complete`, `all_leaf_nodes_valid`,
`all_intermediate_nodes_valid` và `dag_execution_valid`. Required `UNKNOWN` bắt buộc `ABSTAIN`, `INVALID` bắt buộc
`REJECT`; neural confidence không thể bypass proof contract. Typed column roles cũng chứng minh hướng phép toán mà
không nhìn dấu hay độ lớn numeric value. Không có rule theo company/file/hash/page/fixed year/value/question literal
hoặc family ID trong runtime.

Kết quả implementation development-only synthetic contract:

- runtime oracle `122/122`, false accept `0`, gold retention `26/26`;
- macro/worst-group theo 6 synthetic company namespace là `1,000/1,000` và theo 6 synthetic layout namespace là
  `1,000/1,000`;
- đây là proof-contract result, không phải bằng chứng empirical hoặc tuyên bố coverage cho mọi PDF.

Replay hợp lệ đã chạy:

- QP3A–QP3F cùng QP3-CORE tests `66/66`;
- compiler/proof/generalization/company-disjoint tests `145/145`;
- document runtime tests `181/181`;
- reviewed new-company development direction cohort `19/19` direction-supported trên 19 company/source key. Cả 19
  vẫn có final verifier state `UNKNOWN` vì cohort cũ thiếu proof QP3 đầy đủ, nên không được tính là answered hoặc
  confirmation coverage.

Frozen S4 **chưa replay**: audit hiện mở các file tên `validation_candidates.jsonl`, xung đột trực tiếp với ràng
buộc không đọc validation/blind/test. Không chạy lách ràng buộc và không tính S4 là pass. Vì vậy implementation đã
hoàn tất nhưng post-implementation replay gate chưa đóng; QP4 và Phase D tiếp tục khóa.

Không dùng ABW làm confirmation/calibration/gate, không đọc protected evaluation, không fine-tune, không đổi
threshold/checkpoint. Checkpoint vẫn có SHA-256
`3a33276ded0d2ff2dd3d200c4c774a59eec0f8cad7102bc273cb263d983944de`.

Artifacts: `scripts/evaluate_qp3_core_implementation.py`, `tests/test_qp3_core_implementation.py`,
`reports/finhrm_qp3_core_implementation_development_v1.json`,
`reports/finhrm_qp3_core_implementation_lock_v1.json`,
`config/finhrm_qp3_core_replay_authorization_v1.json`,
`reports/finhrm_qp3_core_postimplementation_replay_v1.json` và
`reports/finhrm_qp3_core_postimplementation_replay_v1.md`.

**Bước được phép tiếp theo:** `QP3-CORE-S4-SAFE-REPLAY` — precommit hoặc tái tạo nguồn replay tương đương S4 chỉ
từ development data, không có đường dẫn hay record validation/blind/test. Chỉ sau khi replay này pass mới được xét
sang QP4.

### 15.29 QP3-CORE-S4-SAFE-REPLAY — hoàn tất stress gate, coverage generalization còn khóa (14/08/2026)

Đã thay đường replay S4 không hợp lệ bằng một precommit development/train riêng, không đọc bất kỳ đường dẫn hoặc
record validation/blind/test nào. Nguồn `document_hrm_development_v1_train.jsonl` được khóa SHA-256 trước replay,
không chứa ABW hoặc protected marker. Replay chạy trên **toàn bộ** example SELECT/ADD đủ điều kiện, không chọn một
báo cáo và không dùng candidate score/rank làm semantic proof.

Mỗi example có ba control: gold program, đảo toàn bộ candidate serialization với remap pointer, và semantic
counterfactual chỉ khi metadata chứng minh rõ sai năm, entity scope hoặc metric. Một row label khác không tự động bị
coi là negative; alias/equivalent disclosure được giữ lại bằng semantic alias contract chung.

Kết quả:

- `363` example trên `121` company group và `6` layout family; ADD `123`, SELECT `240`;
- gold execution `363/363`; candidate permutation `363/363`;
- `123` explicit counterfactual gồm wrong-year `72`, wrong-metric `48`, wrong-entity `3`; false accept `0`;
- gold proof-complete/answered `187/363`; company macro/worst answer rate `0,347/0,000`, layout macro/worst
  `0,501/0,000`.

Vì nguồn là development/train, kết quả chỉ đóng **validation-free regression stress gate**, không thay thế unseen
company/layout hoặc sealed confirmation. Worst-group coverage bằng 0 cho thấy hệ thống chưa có quyền tuyên bố hoạt
động đầy đủ trên mọi báo cáo. Các abstention lớn nhất đến từ `period_coverage`, `operation_match`, `add_grounding` và
`query_metric` còn `UNKNOWN`; không nới proof hoặc threshold để tăng coverage giả.

Không dùng ABW, không đọc validation/blind/test, không fine-tune, không đổi threshold/checkpoint/runtime, không mở
Phase D. QP4 tiếp tục khóa.

Artifacts: `config/finhrm_qp3_core_s4_safe_replay_precommit_v1.json`,
`scripts/evaluate_qp3_core_s4_safe_replay.py`, `tests/test_qp3_core_s4_safe_replay.py`,
`reports/finhrm_qp3_core_s4_safe_replay_v1.json`,
`reports/finhrm_qp3_core_s4_safe_replay_v1.md` và
`reports/finhrm_qp3_core_s4_safe_replay_lock_v1.json`.

**Bước được phép tiếp theo:** `QP3-CORE-GENERALIZATION-COVERAGE-PRECOMMIT` — khóa trước các category repair tổng
quát cho missing period/provenance, operation/add grounding và query metric trên nhiều company/layout group. Cấm
repair theo report/example; QP4 chỉ được xét sau confirmation company/layout-disjoint thật.

### 15.30 QP3-CORE-GENERALIZATION-COVERAGE-PRECOMMIT — khóa proof-coverage theo nhóm (14/08/2026)

Đã khóa precommit cho ba track proof coverage từ **toàn bộ** safe replay development/train, không chọn report hoặc
example riêng. Baseline giữ nguyên: `363` example, `121` company group, `6` layout family; `187` ANSWER, `162`
ABSTAIN, `14` REJECT; execution `363/363`, permutation `363/363`, counterfactual false accept `0/123`.

Failure taxonomy chỉ dùng tên typed proof obligation và thống kê nhóm:

- period + provenance preservation: `period_coverage=UNKNOWN` trên `74` example, `73` company nhưng mới `2` layout;
- operation + ADD grounding: `operation_match/add_grounding=UNKNOWN` trên cùng `67` example, `47` company, `3`
  layout;
- query metric: `query_metric=UNKNOWN` trên `33` example, `26` company, đủ `6` layout.

`source_provenance_complete`, `logical_cell_unique` và `reference_base_match` không tạo UNKNOWN trong baseline nên là
no-regression obligations, không phải lý do nới proof. Track period chưa đạt invariant tối thiểu ba layout và bị khóa
fail-closed cho tới khi diagnostic matrix development-only có layout thứ ba bằng biến đổi cấu trúc bảo toàn semantics.
Không được bù thiếu diversity bằng rule theo company, report, page, year, numeric value hoặc question literal.

Diagnostic matrix trước mọi runtime edit phải có positive, hard negative, explicit semantic counterfactual và candidate
permutation trên ít nhất ba company và ba layout cho từng repair. Sau repair phải giữ toàn bộ `187` answer cũ, execution
không regression, permutation `1,0`, false accept `0`; answer mới chỉ được materialize khi mọi required proof là VALID.
Macro và worst-group phải báo cáo riêng; abstention không được tính là answered.

Không dùng ABW làm confirmation, không đọc validation/blind/test, không fine-tune, không đổi threshold/checkpoint,
không mở Phase D hoặc QP4. Artifacts:
`config/finhrm_qp3_core_generalization_coverage_precommit_v1.json`,
`tests/test_qp3_core_generalization_coverage_precommit.py`,
`reports/finhrm_qp3_core_generalization_coverage_precommit_v1.md` và
`reports/finhrm_qp3_core_generalization_coverage_precommit_lock_v1.json`.

**Bước được phép tiếp theo:** `QP3-CORE-GENERALIZATION-COVERAGE-DIAGNOSTIC-MATRIX` — chỉ dựng control tổng quát và
đóng diversity gate cho cả ba track; chưa sửa runtime. QP4 tiếp tục khóa.

### 15.31 QP3-CORE-GENERALIZATION-COVERAGE-DIAGNOSTIC-MATRIX — hoàn tất control matrix (14/08/2026)

Đã dựng deterministic diagnostic matrix từ safe replay development/train đã khóa hash. Matrix chỉ giữ aggregate
source-derived bindings và generic transformation recipe; không persist company key, example id, question hoặc numeric
answer. Mỗi source binding được phủ bởi bốn control: semantic positive, hard negative, explicit counterfactual và đảo
candidate serialization kèm remap pointer.

Kết quả diversity gate:

- period + provenance: `74` source example, `73` company, `2` source layout; thêm đúng `3` projection từ ba company
  sang `DEVELOPMENT_SYNTHETIC_OCR_CHANNEL_SEMANTICS_PRESERVING`, thành `3` diagnostic layout và `308` control
  instance;
- operation + ADD grounding: `67` source example, `47` company, `3` layout, `268` control instance;
- query metric: `33` source example, `26` company, `6` layout, `132` control instance.

Tổng cộng `708` track-control instance trong `48` aggregate matrix cell. Projection period chỉ relabel extraction
channel, giữ cell text/value/structure/provenance; nó đóng diagnostic diversity gate nhưng **không** phải unseen layout,
confirmation hoặc bằng chứng publication. Hard negative và counterfactual oracle đều cấm ANSWER; permutation oracle
yêu cầu giữ nguyên proof decision và execution.

Runtime hashes vẫn trùng precommit. Không dùng ABW làm confirmation, không đọc validation/blind/test, không fine-tune,
không đổi threshold/checkpoint và không thêm rule theo company/report/page/year/value/question. QP4 và Phase D tiếp tục
khóa.

Artifacts: `config/finhrm_qp3_core_generalization_coverage_diagnostic_matrix_v1.json`,
`scripts/build_qp3_core_generalization_coverage_diagnostic_matrix.py`,
`tests/test_qp3_core_generalization_coverage_diagnostic_matrix.py`,
`reports/finhrm_qp3_core_generalization_coverage_diagnostic_matrix_v1.json`,
`reports/finhrm_qp3_core_generalization_coverage_diagnostic_matrix_v1.md` và
`reports/finhrm_qp3_core_generalization_coverage_diagnostic_matrix_lock_v1.json`.

**Bước được phép tiếp theo:** `QP3-CORE-GENERALIZATION-COVERAGE-IMPLEMENTATION-PRECOMMIT` — khóa chính xác các
generic typed-semantics edits và replay gates trước khi chạm runtime. Chưa được sửa runtime trong bước hiện tại.

### 15.32 QP3-CORE-GENERALIZATION-COVERAGE-IMPLEMENTATION-PRECOMMIT — exact edit scope (14/08/2026)

Đã đọc lại toàn bộ `CLAUDE.md` trước khi khóa scope. Precommit chọn thay đổi tối thiểu, không thêm model/subsystem và
không mở một repair chỉ vì category có nhiều abstention.

Aggregate diagnosis trên safe development source cho thấy:

- cả `74/74` period failure có selected `PERIOD` rỗng, không explicit year, không relative period role; header path
  không chứa requested year/role. Chín context có nhắc requested year nhưng không có lineage nối năm đó với cell và chỉ
  hai example có physical provenance đầy đủ;
- `67/67` ADD failure có hai selected metric label nhưng compiler tạo `0` ADD operation, `0` pair operand role và `0`
  typed operand slot: đây là compiler intent gap, không phải lý do nới verifier;
- query metric có `33` UNKNOWN: `25` row-only alias ở trạng thái giữa, `8` thiếu row metric. Row + section + fully
  provenanced header composition đưa đúng `8` candidate qua **cutoff hiện hữu**, trải trên `3` company/`3` layout;
  symmetric evidence riêng chỉ hỗ trợ `4` và không được dùng làm threshold bypass.

Vì vậy period/provenance runtime repair **không được admission**: thiếu cell-linked evidence phải giữ UNKNOWN, không chép
năm/role từ question, nearby context, candidate order hoặc synthetic layout. Exact implementation chỉ mở cho:

1. structural ADD role hypotheses trong `question_planner.py`, giữ mọi separator parse hợp lệ như ambiguity;
2. independent two-slot binding và provenanced metric-context proposal trong `program_synthesis.py`;
3. complete ADD-hypothesis proof và unique compositional query-metric proof trong `symbolic_verifier.py`;
4. final ADD hypothesis grounding/fail-closed ambiguity trong `document_hrm.py`.

`evidence_schema.py`, `financial_identities.py`, `direction_contract.py` và `document_reasoning_engine.py` bị freeze. Alias,
proof, acceptance, model và retrieval thresholds giữ nguyên.

Postimplementation gate phải giữ `187/187` answer cũ, gold execution `363/363`, permutation `363/363`, false accept
`0`, hard-negative/counterfactual ANSWER `0`. ADD UNKNOWN phải giảm dưới `67` và query-metric UNKNOWN dưới `33`, mỗi
track trên ít nhất `3` company/`3` layout. Period không được giảm nếu không có explicit source lineage. Macro/worst-group
phải báo cáo; synthetic projection và abstention không phải confirmation.

Không dùng ABW, không đọc validation/blind/test, không fine-tune, không đổi threshold/checkpoint, không mở QP4/Phase D.
Artifacts: `config/finhrm_qp3_core_generalization_coverage_implementation_precommit_v1.json`,
`tests/test_qp3_core_generalization_coverage_implementation_precommit.py`,
`reports/finhrm_qp3_core_generalization_coverage_implementation_precommit_v1.md` và
`reports/finhrm_qp3_core_generalization_coverage_implementation_precommit_lock_v1.json`.

**Bước được phép tiếp theo:** `QP3-CORE-GENERALIZATION-COVERAGE-IMPLEMENTATION` — chỉ sửa exact admitted symbols cho
ADD và query metric. Period/provenance relaxation, mọi file ngoài whitelist và QP4 tiếp tục khóa.

### 15.33 QP3-CORE-GENERALIZATION-COVERAGE-IMPLEMENTATION — runtime hoàn tất, diversity gate ADD chưa đóng (20/08/2026)

Đã triển khai đúng whitelist typed-semantics cho structural ADD và compositional query metric. Không sửa các runtime file bị
freeze; không thay alias/proof/acceptance/retrieval threshold, checkpoint hoặc model; không fine-tune; không dùng ABW làm
confirmation và không đọc validation/blind/test. Period/provenance tiếp tục fail-closed: `74 -> 74` UNKNOWN, không sao chép
year/relative role từ question hoặc nearby context khi cell không có source lineage.

Safe development replay cho kết quả:

- giữ `187/187` ANSWER cũ, thêm `72` safe ANSWER; gold execution `363/363`, candidate permutation `363/363`;
- `141` explicit counterfactual, false accept `0`;
- operation/add grounding UNKNOWN `67 -> 3`, cải thiện `64` example trên `44` company nhưng mới chỉ `1` real layout;
- query metric UNKNOWN `33 -> 25`, cải thiện `8` example trên `3` company và `3` real layout;
- company/layout macro và worst-group được ghi trong implementation report; abstention và synthetic projection không được
  tính là confirmation.

Implementation controls gồm positive, compound/ambiguity/same-cell hard negatives, wrong role/period/entity/unit và wrong
metric counterfactual, provenance omission, competing logical cell, relative-period corroboration và candidate permutation;
`18/18` control tests và `269/269` affected regression/report tests pass. Hai historical diagnostic-matrix tests kiểm tra
runtime phải còn đúng preimplementation hash không còn applicable sau authorized runtime implementation; frozen matrix artifact
và recipes vẫn là parent exact, không bị dùng như postimplementation confirmation.

Overall implementation gate **chưa đóng** vì duy nhất `add_improvement_minimum_3_layouts` chưa đạt. Ba ADD-labelled example còn
lại không cho phép đóng gate an toàn: một câu không liệt kê đủ component, một hypothetical removal có operation/period không
phù hợp ADD, và một câu có hai header role rõ nhưng chỉ bổ sung layout thứ hai. Không được hard-code, ép UNKNOWN thành ANSWER,
hay dùng synthetic layout để đạt đủ ba layout.

Artifacts: `scripts/evaluate_qp3_core_generalization_coverage_implementation.py`,
`tests/test_qp3_core_generalization_coverage_implementation.py`,
`tests/test_qp3_core_generalization_coverage_implementation_report.py`,
`reports/finhrm_qp3_core_generalization_coverage_implementation_replay_v1.json`,
`reports/finhrm_qp3_core_generalization_coverage_implementation_v1.json`, markdown và implementation lock tương ứng.

**Bước duy nhất còn mở:** tiếp tục `QP3-CORE-GENERALIZATION-COVERAGE-IMPLEMENTATION` với development-safe, explicit two-role
ADD evidence trên ít nhất hai real layout family bổ sung rồi replay lại frozen gates. QP4 và Phase D vẫn khóa.

### 15.34 Rà soát tài liệu neuro-symbolic và khóa hướng answer presentation (20/08/2026)

Đã rà soát bốn tài liệu trong `tham_khao/`, gồm survey neuro-symbolic reasoning over KG, integration cho KG completion,
enterprise KG và VeNRA/UFL cho financial reasoning. Kết luận không đổi thesis claim và không chuyển sang enterprise KG, GNN/KGE
hay KG completion. FinHRM tiếp tục là selective document reasoning trên typed, role-aware, provenance-aware evidence graph.

Các chiến lược được tiếp thu là typed variable ledger, physical + semantic double-lock grounding, schema-driven separation của
actual/plan/limit/formula, lexical/typed gate trước acceptance, deterministic execution và ecological counterfactuals. Chính sách
open-world tiếp tục bắt buộc: thiếu evidence là `UNKNOWN`, không phải false/zero. Không tiếp thu threshold của paper khác, 3B
Sentinel/fine-tuning, online rule learning hoặc fallback bypass graph bằng raw-text extraction.

Phân tích runtime cho thấy `DocumentReasoningAnswer` đã có value, program, operand evidence, source và page nhưng renderer hiện là
technical audit, chưa phải product answer. Vì vậy khóa một `Verified Answer Envelope v1` gồm claim typed, canonical/display value,
program/DAG, citation từng operand, proof state, warnings và summary coverage. Ba mode trình bày tương lai:

1. `FACT`: câu trả lời tự nhiên tối thiểu gồm metric, period, value/unit và document/statement/page/row/column citation;
2. `COMPUTATION`: kết quả, formula, từng operand với citation riêng, phép tính replay được và kết luận;
3. `VERIFIED_SUMMARY`: compile yêu cầu “nội dung chính” thành inventory subquery, verify từng row rồi mới render bảng
   Nội dung — Số liệu/so sánh — Điểm nổi bật — Nguồn.

Renderer không được tạo số/citation, không được làm tròn trước execution, không được đổi formula, và không được biến partial
summary thành complete. Nếu chỉ chứng minh `N/M` mục thì phải ABSTAIN theo `answer_mode_complete` hoặc ghi rõ verified partial
coverage; qualitative cause chỉ được nêu khi có cited text evidence. Metric tương lai phải gồm numeric/unit-scale exact, citation
exact, derivation replay, unsupported-claim rate, summary completeness và safe abstention.

Review đầy đủ: `reports/finhrm_reference_strategy_review_answer_presentation_v1.md`.

**Thứ tự không đổi ở hiện tại:** bước duy nhất còn mở vẫn là
`QP3-CORE-GENERALIZATION-COVERAGE-IMPLEMENTATION` để đóng ADD real-layout diversity. Sau khi QP3 core pass, mở riêng
`QP3-ANSWER-PRESENTATION-PRECOMMIT` trước QP4; deterministic renderer đi trước, LLM synthesis/explanation vẫn để sau QP5 và
phải được kiểm lại từ verified envelope. QP4/Phase D chưa được mở bởi design note này.

### 15.35 QP3-CORE-GENERALIZATION-COVERAGE-IMPLEMENTATION — đóng real-layout diversity gate (20/08/2026)

Đã khóa thêm một confirmation development-only từ toàn bộ ADD record đủ điều kiện trong
`document_pointer_common_development_semantic_canonical_v5.jsonl`. Selection chỉ dùng operation, typed question intent,
physical/logical provenance và extraction/layout metadata; không dùng company, filename, page, year, numeric value, answer,
example ID, question literal, candidate rank/score hoặc confidence làm runtime rule/proof. Runtime không thay đổi.

Cohort được tách fail-closed thành:

- `10` positive source-backed ADD có explicit two-role syntax, hai physical cell riêng, cùng requested period và provenance
  đầy đủ; `10/10` ANSWER proof-complete trên `5` company và `5` real PDF source;
- `2` physical-source-span collision control: row/table label khác nhưng cùng `(source, page, source_span)`, giữ nguyên
  `INVALID -> REJECT`, không được dùng làm confirmation positive;
- hai real layout family bổ sung là `DEVELOPMENT_NATIVE_MULTI_TABLE` (`6` positive) và
  `DEVELOPMENT_OCR_OR_HYBRID_TABLE` (`4` positive). Hợp với layout cải thiện cũ
  `DEVELOPMENT_NATIVE_STRUCTURED_SPARSE`, ADD đạt đủ `3` real layout family;
- gold execution `12/12`, candidate permutation `12/12`, `12` semantic counterfactual và false accept `0`;
- positive company/layout macro và worst-group answer rate đều `1,000`; hai collision control vẫn được báo riêng, không
  che bằng macro positive.

Các câu confirmation được mechanically derived nhưng selected cells/layout đến từ PDF thật; vì vậy chúng chứng minh
development source-backed real-layout coverage cho contract đã khóa, không phải synthetic layout projection, sealed/unseen
confirmation hay publication claim. ABW và protected evaluation không tham gia selection, metric hoặc gate.

Visual QA trên bảy relevant PDF page của sáu source xác nhận các header, row, period column và physical page layout tồn tại
thật: OCR/hybrid nhiều cột ở ADS/AGR và native multi-table/section ở DMC/VNT/TNH. Trang SCG cũng xác nhận hai vùng bảng khác
nhau nhưng extraction provenance hiện gán trùng source span cho hai selected cell; vì vậy hai case này tiếp tục là collision
hard negative `REJECT`, không được sửa provenance từ quan sát bằng mắt hoặc dùng filename/page làm runtime signal.

Toàn bộ frozen safe replay được chạy lại vào output tạm và khớp byte-for-byte với implementation replay hash
`a1254f19ffedb5682d7aae509d8d68893a33d27819c9cfb6fd150bb37ac45291`: giữ `187/187` ANSWER cũ, thêm `72` safe ANSWER,
gold execution `363/363`, permutation `363/363`, `141` counterfactual và false accept `0`; period UNKNOWN giữ `74 -> 74`.
Affected functional/report regression đạt `275/275`. Năm historical preimplementation-hash assertion trong một batch rộng
không applicable đúng như §15.33 vì chúng yêu cầu runtime/`strategy.md` quay về hash trước authorized implementation; không có
functional/proof regression.

Overall implementation gate hiện **PASS**. Không fine-tune, không đổi threshold/checkpoint/model, không sửa answer renderer,
không mở QP4 hoặc Phase D.

Artifacts mới: `config/finhrm_qp3_core_add_real_layout_confirmation_precommit_v1.json`,
`scripts/evaluate_qp3_core_add_real_layout_confirmation.py`,
`tests/test_qp3_core_add_real_layout_confirmation.py`,
`reports/finhrm_qp3_core_add_real_layout_confirmation_v1.json`, markdown và confirmation lock; implementation report/lock
được replay và cập nhật để tính đúng hợp ba real layout family.

**Bước được phép tiếp theo:** `QP3-ANSWER-PRESENTATION-PRECOMMIT` — chỉ khóa Verified Answer Envelope v1, ba mode
`FACT/COMPUTATION/VERIFIED_SUMMARY`, citation/derivation replay, unsupported-claim và `answer_mode_complete` controls trước khi
sửa renderer runtime. QP4 và Phase D tiếp tục khóa.

### 15.36 Hiệu chỉnh claim diversity: candidate-level confirmation không đóng QP3 core (20/08/2026)

§15.35 đã diễn giải quá mạnh evidence hiện có. Runtime ADD không được sửa theo ADS/AGR/DMC/VNT/SCG/TNH và các kiểm tra safety
vẫn hợp lệ, nhưng `10` positive bổ sung chỉ replay proof contract từ candidates đã materialize. Chúng không chạy lại đầy đủ chuỗi
PDF thật → extraction/reconstruction → retrieval/intent proposal → symbolic binding → verification/execution. Vì vậy chúng là
`SUPPORTING_ONLY_NOT_GATE_CLOSING`, không được cộng vào số layout cải thiện của frozen replay.

Audit diversity fail-closed cho thấy:

- `DEVELOPMENT_NATIVE_MULTI_TABLE`: `6` positive, `3` company, `3` PDF;
- `DEVELOPMENT_OCR_OR_HYBRID_TABLE`: `4` positive, chỉ `2` company, `2` PDF;
- toàn bộ `10/10` positive vẫn có `review_status=pending_independent_review`, nên số positive được review độc lập là `0/10`;
- extraction/reconstruction, retrieval/intent proposal và symbolic binding end-to-end chưa được evidence này kiểm tra;
- candidate permutation và counterfactual safety vẫn pass, false accept vẫn bằng `0`; hai physical collision vẫn `REJECT`.

Do đó ADD improvement chính thức vẫn là `64` example trên `44` company nhưng chỉ `1` real layout; gate
`add_improvement_minimum_3_layouts` tiếp tục fail. Khóa bổ sung yêu cầu mỗi additional real layout có tối thiểu `3` company và
`3` source PDF, mỗi company/layout có positive được review độc lập, toàn bộ pipeline stage được replay, và có company/layout-disjoint
confirmation. Các ngưỡng này áp dụng theo layout, không dùng tổng gộp để che một layout hẹp.

Không có runtime rollback vì không có runtime change trong confirmation vừa rồi. Không fine-tune, không đổi model/checkpoint/threshold,
không đọc protected split, không dùng ABW/synthetic projection và không mở renderer.

Artifacts hiệu chỉnh: `config/finhrm_qp3_core_add_diversity_confirmation_precommit_v1.json`,
`scripts/evaluate_qp3_core_add_diversity_confirmation.py`,
`tests/test_qp3_core_add_diversity_confirmation.py`, report/markdown/lock tương ứng, cùng implementation report/lock đã hạ về
`implementation_executed_gate_not_closed`.

**Bước duy nhất còn mở:** `QP3-CORE-GENERALIZATION-COVERAGE-IMPLEMENTATION`. `QP3-ANSWER-PRESENTATION-PRECOMMIT`, QP4 và
Phase D đều tiếp tục khóa cho đến khi diversity gate mới pass.

### 15.37 QP3-CORE-GENERALIZATION-COVERAGE-IMPLEMENTATION — đóng bằng reviewed end-to-end cohort (20/08/2026)

§15.37 supersede trạng thái mở ở §15.36 bằng evidence mới đúng contract: toàn bộ reviewed cohort được chạy từ PDF thật và câu hỏi,
không truyền gold candidate/program vào engine. Chuỗi được kiểm tra gồm document discovery, native/OCR extraction, chunk retrieval,
numeric candidate materialization, frozen neural operation/operand proposal, symbolic binding, deterministic verification/execution.

Kết quả end-to-end development-only:

- `6/8` positive pass trên `6` company và `6` PDF khác nhau; `2` residual không đủ proof đều `ABSTAIN`, accepted-wrong `0`;
- `NATIVE_TEXT_TABLE`: `3/4` positive trên `3` company/`3` PDF;
- `OCR_OR_HYBRID_TABLE`: `3/4` positive trên `3` company/`3` PDF;
- positive trải trên `4` real layout subfamily: native portrait, native mixed-orientation, scanned-image-dominant và hybrid scan/text;
- mỗi additional layout stratum đạt ngưỡng khóa `3` company/`3` PDF. Cộng với frozen improvement layout cũ,
  `add_improvement_minimum_3_layouts` đạt `3` real layout development groups.

Các sửa runtime chỉ dùng đặc trưng cấu trúc chung đã được cohort gap chứng minh: bảo toàn ADD role hypotheses qua candidate quota,
coi numeric outline marker là phi-semantic nhưng vẫn bắt buộc đủ compound-role tokens, map header một dòng theo explicit
year/share column roles, không cho ADD dùng percent cells nếu query không yêu cầu ratio/percent, tái sử dụng OCR cache đúng provenance,
và chia sparse MinerU selection thành contiguous page runs. Không có branch theo company, filename, page, year, value, example ID hoặc
question literal; candidate rank/score/confidence không trở thành semantic proof.

Frozen safe replay cuối giữ `187/187` ANSWER cũ và `72` safe ANSWER mới: gold execution `363/363`, permutation `363/363`,
`141` counterfactual với false accept `0`; actions `ANSWER=259`, `ABSTAIN=90`, `REJECT=14`. Affected regression đạt `319/319`.
Checkpoint, threshold và model không đổi; không fine-tune, không dùng ABW, không đọc protected evaluation, không tính synthetic layout
là confirmation và không claim unseen/universal generalization.

Overall `QP3-CORE-GENERALIZATION-COVERAGE-IMPLEMENTATION` hiện **PASS**. Candidate-level confirmation ở §15.35 vẫn chỉ là
supporting evidence; gate được đóng bởi reviewed PDF+question end-to-end replay nêu trên.

**Bước duy nhất được phép tiếp theo:** `QP3-ANSWER-PRESENTATION-PRECOMMIT` — chỉ khóa Verified Answer Envelope v1, ba mode
`FACT/COMPUTATION/VERIFIED_SUMMARY`, citation/derivation replay, unsupported-claim và `answer_mode_complete` controls trước mọi sửa
renderer runtime. QP4 và Phase D tiếp tục khóa; thứ tự roadmap không đổi.

### 15.38 QP3-ANSWER-PRESENTATION-PRECOMMIT — khóa contract và mở implementation có whitelist (20/08/2026)

Đã đóng precommit cho `Verified Answer Envelope v1` mà không sửa runtime. Contract khóa ba trạng thái
`ANSWER/ABSTAIN/REJECT`, ba mode `FACT/COMPUTATION/VERIFIED_SUMMARY`, typed claim, exact canonical/display value, executed program DAG,
citation theo từng operand, proof obligations, warnings và summary coverage. Presentation chỉ được tiêu thụ action hiện hữu cùng exact
verifier/proof snapshot trên full candidate evidence graph; candidate rank, neural score, confidence và warning text không phải semantic
proof và không được presentation dùng để nâng `ABSTAIN/REJECT` thành `ANSWER`.

Các rule fail-closed đã được khóa:

- `FACT` chỉ nhận `SELECT` proof-complete với logical-cell citation; field provenance không được chứng minh thì bỏ khỏi citation, không
  sinh report title, statement/note hoặc `page X`;
- `COMPUTATION` render formula trực tiếp từ executed typed DAG, mọi leaf có citation riêng, mọi intermediate và kết quả cuối replay exact;
  display value không được quay lại execution và v1 không cho lossy rounding;
- `VERIFIED_SUMMARY` là inventory các sub-answer độc lập. Chỉ `ANSWER` khi `verified_count=requested_count`, mọi row verified và
  `answer_mode_complete=VALID`; partial phải giữ overall `ABSTAIN` và ghi rõ `N/M`, conflict phải `REJECT`; qualitative claim chỉ được
  render khi có direct cited text span;
- unsupported claim, false complete summary, parent-action promotion và hard-negative false accept đều có gate bằng `0`.

Development generalization cohort cho implementation phải có tối thiểu `3` verified example, `3` company, `3` real layout family và
`3` source document cho **mỗi mode**, kèm company/layout-disjoint confirmation. Synthetic fixture chỉ được test schema, không được tính
coverage. Hard negatives đã khóa gồm wrong page/statement/year, swapped operand, scale `x1000`, missing summary row, uncited qualitative
cause và parent-action promotion.

Exact runtime scope của bước tiếp theo chỉ gồm file mới `src/hrm_finance_qa/answer_presentation.py` và hai symbol
`DocumentReasoningAnswer`, `DocumentHRMReasoningEngine.answer` trong `document_reasoning_engine.py` để ghi exact proof snapshot và nối
adapter presentation. Retrieval, proposal, binding, verifier, executor, acceptance, threshold, checkpoint/model cùng technical audit
renderer đều bị freeze.

Precommit validator PASS, `12/12` contract tests PASS. Frozen safe replay chạy lại PASS với actions
`ANSWER=259, ABSTAIN=90, REJECT=14`, execution `363/363`, candidate permutation `363/363`, `141` counterfactual và false accept `0`.
Affected regression batch đạt `315/315`. Runtime hash và checkpoint
`3a33276ded0d2ff2dd3d200c4c774a59eec0f8cad7102bc273cb263d983944de` không đổi; không dùng ABW/protected split, không fine-tune,
không dùng synthetic layout làm confirmation và không claim unseen/universal generalization.

Artifacts: `config/finhrm_qp3_answer_presentation_precommit_v1.json`,
`scripts/validate_qp3_answer_presentation_precommit.py`, `tests/test_qp3_answer_presentation_precommit.py`, report/markdown và
`reports/finhrm_qp3_answer_presentation_precommit_lock_v1.json`.

**Bước duy nhất được phép tiếp theo:** `QP3-ANSWER-PRESENTATION-IMPLEMENTATION` trong exact two-file scope và toàn bộ presentation,
regression, permutation, counterfactual, frozen safe replay gates nêu trên. QP4 và Phase D tiếp tục khóa; thứ tự roadmap không đổi.

### 15.39 QP3-ANSWER-PRESENTATION-IMPLEMENTATION — deterministic renderer hoàn tất (21/08/2026)

Đã triển khai `Verified Answer Envelope v1` trong đúng exact runtime whitelist: file mới
`src/hrm_finance_qa/answer_presentation.py` và additive proof snapshot/adapter trong `DocumentReasoningAnswer`,
`DocumentHRMReasoningEngine.answer`. Retrieval, neural proposal, symbolic binding, verifier, executor, acceptance, threshold,
checkpoint và model không thay đổi. Technical audit `DocumentReasoningAnswer.render()` được giữ nguyên; output người dùng đi qua
`render_verified()` riêng.

Implementation cung cấp:

- ba trạng thái fail-closed `ANSWER/ABSTAIN/REJECT` từ existing runtime action cộng exact verifier snapshot trên full candidate
  evidence graph; presentation không đọc rank, neural score, confidence hoặc warning text làm proof và không thể promotion action;
- `FACT` với exact Decimal canonical/display value và provenance fingerprint cho source/page/span/table/row/column/period/entity/unit;
  field không được chứng minh bị bỏ, page thiếu không sinh placeholder;
- `COMPUTATION` với công thức render trực tiếp từ executed typed DAG, citation riêng từng operand, exact replay mọi intermediate/final,
  non-commutative operand swap và scale mutation đều fail validation;
- `VERIFIED_SUMMARY` theo requested inventory order, mỗi row là verified sub-answer; complete mới `ANSWER`, partial giữ `ABSTAIN` và
  ghi rõ `N/M`, conflict `REJECT`, uncited qualitative claim bị loại và không được render;
- deterministic candidate/citation/summary permutation cùng same-envelope same-bytes.

Development confirmation chọn **toàn bộ** record đủ điều kiện bằng review/proof/provenance/layout metadata từ reviewed end-to-end
real-PDF cohort, không chọn theo company, filename, page, year, value, example ID, question literal, rank/score/confidence. Có `5` record
trên `5` company, `5` PDF và `3` real layout family: native portrait, native mixed-orientation và scanned-image-dominant. Kết quả theo
mode:

- `FACT`: `10/10` ANSWER;
- `COMPUTATION`: `5/5` ANSWER;
- `VERIFIED_SUMMARY`: `5/5` ANSWER.

Mỗi mode đạt tối thiểu `3` company/`3` real layout/`3` source document; company/layout macro và worst-group answer rate đều `1,000`.
Candidate permutation `5/5`, citation permutation `15/15`, summary-input permutation `5/5`; tám presentation hard negative có false
accept `0`, gồm wrong page/statement/year, scale `x1000`, swapped non-commutative operand, missing summary row, uncited qualitative cause
và parent-action promotion. Synthetic fixture không được tính coverage; derived FACT/SUMMARY chỉ materialize các leaf/computation đã được
review và end-to-end verify trên PDF thật, không tạo claim unseen/universal.

Frozen core replay cuối giữ actions `ANSWER=259, ABSTAIN=90, REJECT=14`, execution `363/363`, candidate permutation `363/363`,
`141` counterfactual và false accept `0`. Không dùng ABW/protected evaluation, không fine-tune, không đổi threshold/checkpoint/model và
không mở Phase D. Presentation/report tests đạt `22/22`; affected functional/report regression đạt `293/293`.

Artifacts: `config/finhrm_qp3_answer_presentation_development_cohort_v1.json`,
`scripts/evaluate_qp3_answer_presentation_implementation.py`, hai implementation/report test module,
`reports/finhrm_qp3_answer_presentation_implementation_v1.json`, markdown và implementation lock.

Overall QP3 hiện **COMPLETE**. **Bước duy nhất được phép tiếp theo:** `QP4-PRECOMMIT` để khóa minimal-pair counterfactual,
short/long-question equivalence, clause/permutation stress, multi-output/qualitative/anaphora/cross-document audit và exact scope trước
mọi sửa runtime QP4. QP4 runtime change chưa được phép; Phase D tiếp tục khóa và thứ tự roadmap không đổi.

### 15.40 QP4-PRECOMMIT — khóa counterfactual và long-question stress contract (21/08/2026)

`QP4-PRECOMMIT` đã khóa contract development-only mà không sửa runtime. Sáu dimension minimal-pair bắt buộc là
`COLUMN_ROLE`, `OPERAND_YEAR`, `ENTITY_SCOPE`, `UNIT_SCALE`, `DIRECTION` và `REFERENCE_BASE`. Mỗi pair chỉ được đổi đúng một
dimension semantic; program trước mutation không được tiếp tục là ANSWER. Mismatch tường minh phải làm proof áp dụng `INVALID` và
`REJECT`; thiếu qualifier/proof bắt buộc phải là `UNKNOWN` và `ABSTAIN`. Việc score giảm hoặc kết quả số tình cờ trùng không phải
counterfactual pass.

Meaning-preserving stress khóa bảy category: short/long tương đương, discourse prefix/suffix, clause permutation còn giữ rõ attachment,
punctuation, không dấu, bilingual parenthetical đã xác minh và distractor clause không mang qualifier cạnh tranh. Semantic projection,
eligible-program set, winner/action, selected provenance và exact execution đều phải bất biến `100%`; độ dài câu, rank, neural score và
confidence không được dùng làm proof. Repeated metric, conflicting/competing cell, percent cạnh absolute, table/subtable distractor và
candidate permutation là evidence stress riêng: thiếu proof giữ `ABSTAIN`, contradiction giữ `REJECT`.

Compiler-only synthetic variants phải phủ đủ `13` answer family và `3` ambiguity control nhưng không được tính company/layout coverage.
End-to-end confirmation phải dùng PDF development thật, independently reviewed; với mỗi counterfactual dimension áp dụng phải có ít nhất
`3` positive anchor trên `3` company, `3` source document và `3` real layout family, company-disjoint khỏi diagnosis. Long/surface stress
cũng phải đạt tối thiểu `3/3/3`; cohort QP3 cũ chỉ có thể seed eligibility và không tự động đóng QP4 gate. Safe abstention không được tính
là answered. Multi-output, qualitative follow-up, anaphora và cross-document được audit riêng; cross-document chưa có coverage claim.

Exact conditional runtime whitelist cho bước implementation sau chỉ gồm `question_planner.py`, `program_synthesis.py`,
`symbolic_verifier.py` và `direction_contract.py`. Trước mọi edit phải có frozen baseline chỉ ra layer fail; chỉ symbol group nhỏ nhất trực
tiếp gây fail mới được sửa, các layer pass giữ byte-frozen. `document_hrm.py`, reasoning engine, answer presentation, evidence schema,
program data, retrieval/model/checkpoint/threshold/acceptance đều freeze. Sau mỗi runtime edit phải replay functional regression,
short/long/surface, minimal-pair counterfactual, permutation/hash-seed, source-backed company/layout cohort và full frozen safe replay.

Artifacts: `config/finhrm_qp4_precommit_v1.json`, validator/test riêng, report JSON/Markdown và
`reports/finhrm_qp4_precommit_lock_v1.json`. Precommit này không cấp quyền sửa runtime trong cùng bước. Bước sau chỉ có thể là build/freeze
QP4 development cohort và baseline diagnostic rồi mới triển khai trong whitelist khi có yêu cầu riêng. QP5 và Phase D tiếp tục khóa;
thứ tự roadmap không đổi.

### 15.41 QP4 development cohort và baseline diagnostic — hoàn tất, implementation gate còn khóa (21/08/2026)

Development cohort hai tầng đã được materialize và chạy mà không sửa runtime. Compiler tier phủ `13/13` core answer family và `3/3`
ambiguity control; `91` meaning-preserving surface record và `57` semantic minimal pair được chấm bằng frozen semantic projection, không
dùng surface text, rank, score hoặc confidence. Source-backed tier chạy `50/50` lượt từ public boundary `question + PDF path` trên toàn bộ
`5` eligible independently reviewed anchor thuộc `5` company, `5` PDF và `3` real layout family. Chỉ dùng OCR cache development hiện hữu;
không sinh OCR artifact mới trong stress run. Query quá `15` giây là baseline failure, không phải answered hay safe-abstention pass.

Baseline hiện không pass QP4 gates:

- compiler canonical `13/13`, ambiguity `3/3`;
- compiler surface invariance `73/91`: short/long, clause permutation, punctuation và no-diacritics pass `13/13`; discourse prefix/suffix
  `1/13`, bilingual parenthetical `10/13`, irrelevant distractor `10/13`;
- compiler counterfactual detection `44/57`: column role `12/12`, operand year `13/13`, entity/scope `13/13`, direction `3/3` và
  reference base `3/3`; unit/scale `0/13` vì `QuestionPlan` chưa biểu diễn requested scale;
- source surface winner invariance `20/35`: punctuation, no-diacritics, bilingual parenthetical và irrelevant distractor đều `5/5`;
  short/long, discourse prefix/suffix và clause permutation đều `0/5`;
- source semantic counterfactual có operand-year false accept `0/5`, entity/scope false accept `0/5`, nhưng unit/scale old-winner false
  accept `4/5`, trải trên `4` company và đủ `3` real layout;
- source ADD anchors không applicable để xác nhận `COLUMN_ROLE`, `DIRECTION` hoặc `REFERENCE_BASE`; ba dimension này giữ
  `NOT_CONFIRMED`, không dùng compiler fixture để thay source-backed evidence.

Evidence generic hiện đủ để quy lỗi typed-intent `UNIT_SCALE` về `question_planner`: fail `13` family và tái hiện end-to-end trên `4`
company/`3` layout. Discourse/bilingual/distractor compiler failures cũng là question-only generic groups, nhưng source residual chỉ được
attribution lại sau replay. Ngược lại, short/long và clause-permutation compiler projection đã đúng trong khi source winner fail `5/5`;
phải isolate retrieval/binding và hiện nằm ngoài automatic four-file whitelist, nên không được sửa runtime từ report này.

Artifacts: `config/finhrm_qp4_development_cohort_v1.json`, evaluator, hai test module,
`reports/finhrm_qp4_source_backed_baseline_v1.json`, baseline report/Markdown và lock. `cohort_and_baseline_complete=true` nhưng
`baseline_gates_passed=false`; runtime, checkpoint, threshold và model giữ nguyên. Bước tiếp theo là xây independently reviewed
ACTUAL/PLAN ratio/direction source anchors cho ba dimension còn thiếu và review exact pre-authorization cho các generic failure groups.
QP4 runtime, QP5 và Phase D tiếp tục khóa.
