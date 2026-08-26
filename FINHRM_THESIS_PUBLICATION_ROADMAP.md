# FinHRM: lộ trình từ khóa luận tốt nghiệp đến bài báo

Ngày chốt hướng: 13/08/2026  
Trạng thái: roadmap nghiên cứu; không thay đổi model, threshold, dữ liệu niêm phong hoặc Phase D

## 1. Quyết định cuối cùng

Hướng nên theo là:

> **FinHRM — selective neuro-symbolic financial document reasoning over a provenance-aware, role-aware evidence graph.**

Pipeline nghiên cứu:

```text
PDF báo cáo tài chính
  → OCR/layout/table reconstruction
  → cell-level Financial Evidence Graph
  → typed question compiler
  → neural-guided program synthesis
  → deterministic executor + three-state proof verifier
  → answer + operands + derivation + source citation
    hoặc ABSTAIN + proof failure
```

Không chọn ba hướng sau làm đóng góp trung tâm:

1. `PDF → Knowledge Graph → LLM trả lời`: graph QA và GraphRAG đã là một hướng đông người làm.
2. `multi-agent RAG + reviewer`: khó tạo khác biệt so với `mau.md` và các agentic financial QA mới.
3. `LLM sinh Python để tính`: FinQA, TAT-QA và nhiều hệ thống Program-of-Thought đã dùng chương trình thực thi.

Đóng góp trung tâm phải là:

> **Grounding đúng semantic identity của từng ô tài chính, đặc biệt các ô cùng metric/year nhưng khác actual/plan/entity/unit; sau đó chỉ cho phép chương trình được trả lời khi từng toán hạng mang proof và provenance đầy đủ.**

Đây là hướng phù hợp nhất vì nó đồng thời:

- xuất phát từ lỗi thật của FinHRM, không phải novelty được nghĩ thêm sau khi làm hệ thống;
- kế thừa phần code hiện có thay vì viết lại toàn bộ;
- khác rõ với section-level GraphRAG và multi-agent RAG;
- tạo được hypothesis, hard-negative, ablation và metric kiểm chứng;
- có một phạm vi nhỏ đủ hoàn thành khóa luận và một phạm vi lớn hơn để viết bài.

## 2. Đánh giá trung thực trạng thái hiện tại

### 2.1 Những gì đã đủ nền tảng

- Typed Query IR và question-family catalog đã tồn tại.
- Evidence đã có metric, period, entity, unit, table/row/column/page provenance và mới bổ sung `column_role`.
- Typed synthesis, deterministic Decimal executor và three-state verifier đã tồn tại.
- Hệ thống có search trace, proof obligation, confidence và ABSTAIN.
- Có development company-disjoint, audit hash/provenance và các cohort hard-negative.
- Lỗi ABW chứng minh giá trị của hướng role-aware: cùng metric và cùng year nhưng `plan` và `actual` dẫn tới hai đáp án khác nhau.

### 2.2 Những gì chưa đủ để nộp bài mạnh

- QP2/QP3 mới hoàn thành priority lane, chưa hoàn chỉnh PA04–PA12, TY và multi-output.
- Overall accepted precision development hiện còn dưới release gate 90%.
- Chưa có test mới, độc lập và company-disjoint được niêm phong cho cấu hình cuối.
- Chưa có benchmark real-PDF được annotation đầy đủ ở mức cell, role, program và proof.
- Chưa có baseline chạy trên cùng corpus/split với `mau.md`, GraphRAG, LLM Program-of-Thought và neural-only.
- Chưa có ablation cuối chứng minh riêng tác dụng của evidence graph, column role, typed synthesis, verifier và calibration.
- Demo hiện làm nổi bật lookup nhiều hơn multi-step reasoning nên chưa truyền đạt đúng đóng góp.

Kết luận: **hướng nghiên cứu có tiềm năng, nhưng kết quả hiện tại chưa đủ để tuyên bố sẵn sàng cho conference hoặc journal Q1/Q2.**

## 3. Research gap sau khi đối chiếu công trình gần đây

Các hướng liên quan đã bao phủ khá rộng:

- [FinQA](https://aclanthology.org/2021.emnlp-main.300/) cung cấp financial QA với gold executable programs.
- [TAT-QA](https://aclanthology.org/2021.acl-long.254/) kết hợp evidence extraction từ table/text với symbolic aggregation.
- [MultiHiertt](https://aclanthology.org/2022.acl-long.454/) mở rộng sang nhiều bảng phân cấp và annotation supporting facts/reasoning.
- [DocFinQA](https://aclanthology.org/2024.acl-short.42/) đưa FinQA vào full-document context; 7.437 câu có context trung bình khoảng 123 nghìn từ, cho thấy retrieval trên tài liệu dài vẫn khó.
- [Hybrid Graphs for Table-and-Text QA](https://aclanthology.org/2025.naacl-long.39/) đã dùng unified graph và question-conditioned pruning với LLM, không fine-tune.
- [FinLFQA](https://aclanthology.org/2025.findings-emnlp.908/) đánh giá riêng supporting evidence, intermediate numerical reasoning và financial knowledge attribution.
- [FinRank](https://arxiv.org/abs/2608.07400) — hiện là preprint — nhấn mạnh rằng đáp án số đúng vẫn có thể dựa trên evidence sai, và đưa hard-negative theo entity, period và disclosure context.
- Selective prediction đã có nền tảng risk–coverage rõ, ví dụ [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html).

Vì vậy không được claim rằng FinHRM là hệ thống đầu tiên dùng graph, chương trình tính, provenance hoặc abstention. Research gap khả thi và hẹp hơn là:

> **Provenance-sensitive, role-aware selective program synthesis for Vietnamese financial tables, evaluated with counterfactual cell-level hard negatives and proof completeness.**

Khoảng trống cụ thể:

1. Nhiều hệ thống đánh giá answer correctness nhưng không bắt buộc đúng physical/logical evidence cell.
2. Section-level graph không biểu diễn đủ `metric × period × column role × entity × unit` của từng ô.
3. Program execution xác nhận phép toán nhưng không tự xác nhận semantic identity của operand.
4. Confidence threshold thuần neural có thể under-confident với program đúng hoặc over-confident với evidence sai.
5. Actual/plan/revised-plan, consolidated/separate và cùng-year repeated cells tạo hard-negative đặc thù cho báo cáo tài chính Việt Nam.

## 4. Câu hỏi nghiên cứu nên chốt

Không nên dùng quá ba research question trong khóa luận.

### RQ1 — Cell-role grounding

**Financial Evidence Graph ở mức cell có cải thiện việc chọn đúng operand semantic identity so với chunk retrieval hoặc graph ở mức section không?**

Đo bằng:

- evidence-cell Recall@K;
- operand exact match;
- role/year/entity/unit exact match;
- provenance-correct answer accuracy.

### RQ2 — Proof-guided synthesis

**Typed program synthesis kết hợp three-state verifier có giảm false accept trên semantic hard-negative mà vẫn giữ true accept không?**

Đo bằng:

- program exact và execution accuracy;
- hard-negative false-accept rate;
- verifier gold retention;
- `VALID/UNKNOWN/INVALID` confusion theo obligation;
- counterfactual invariance và contradiction rejection.

### RQ3 — Selective reliability

**Kết hợp neural confidence với proof state có cải thiện risk–coverage so với threshold confidence thuần neural không?**

Đo bằng:

- accepted precision và coverage;
- risk–coverage curve/AURC;
- precision lower bound;
- calibration error/Brier score theo operation;
- tỷ lệ đúng của ABSTAIN trên câu unanswerable/ambiguous.

## 5. Phạm vi đủ để hoàn thành khóa luận

Khóa luận không cần hoàn thành toàn bộ 59 family. Cần một vertical slice có chiều sâu và có negative control.

### 5.1 Các family bắt buộc

| Nhóm | Family tối thiểu | Năng lực |
|---|---|---|
| Lookup có role | PA01, PA02 | SELECT actual/plan đúng cell |
| Plan–actual | PA03–PA06 | completion, signed gap, shortfall, exceed |
| Plan–actual phần trăm | PA07–PA09 | percent change, shortfall %, exceed % |
| Hai năm | TY01, TY02, TY04 | money change, growth %, mixed year/role |
| Multi-step | một identity đã có, ưu tiên interest-bearing debt share | ADD/subtotal → ratio |
| Unanswerable | C13–C15 tương ứng | ambiguous direction/base/column role phải ABSTAIN |

Deferred khỏi thesis core:

- conversational anaphora;
- cross-document/cross-company reasoning;
- qualitative causal analysis;
- chart reasoning;
- multi-agent memory;
- full PA10–PA12/TY05–TY12 nếu không ảnh hưởng core claim;
- Neo4j deployment quy mô lớn.

### 5.2 Sản phẩm khóa luận phải trình diễn

Demo phải có ít nhất bốn case khác nhau:

1. `SELECT actual` và `SELECT plan` trên cùng hàng/cùng năm.
2. `actual ÷ plan × 100` với hai operand và citation riêng.
3. So sánh hai năm hoặc mixed role/year.
4. Một câu mơ hồ có top candidate hợp lexical nhưng hệ thống từ chối với proof reason.

Mỗi câu trả lời phải hiển thị:

```text
Answer
Typed program / DAG
Operand values
Metric, year, column role, entity, unit của từng operand
Page/table/row/column citation
Required proof obligations
Acceptance basis hoặc abstention reason
```

Nếu demo chỉ hiện “doanh thu = X”, phần đóng góp học thuật sẽ bị che khuất dù backend có proof.

## 6. Financial Evidence Graph nên mở rộng đến đâu

### 6.1 Schema cần thiết

Không cần xây enterprise KG chứa mọi quan hệ doanh nghiệp. Graph nghiên cứu chỉ cần phục vụ grounding:

```text
Report → Page → Table → Row/Column → Cell
Cell → Metric
Cell → Period
Cell → ColumnRole
Cell → Entity/Scope
Cell → Unit/Scale
Cell → Value
Cell → SourceSpan/Coordinate
```

Các edge suy luận:

```text
SAME_METRIC
SAME_ENTITY
SAME_TABLE
SAME_ROW
SAME_UNIT
PREVIOUS_PERIOD / NEXT_PERIOD
PART_OF_TOTAL
HEADER_DERIVED_ROLE
```

Mỗi thuộc tính/edge phải lưu:

```text
state = EXPLICIT | INFERRED | UNKNOWN | CONFLICT
source = native_header | OCR_header | text_span | ontology
provenance = document/page/table/coordinate
```

### 6.2 Không bắt buộc Neo4j

Novelty không nằm ở database. In-memory typed graph hoặc SQLite adjacency vẫn đủ cho paper nếu schema, construction và query/proof reproducible. Chỉ dùng Neo4j nếu cần baseline GraphRAG hoặc truy vấn nhiều báo cáo; không để hạ tầng graph làm trễ QP2/QP3.

### 6.3 Quan hệ giữa graph và neuro-symbolic

Graph là nguồn evidence có kiểu; neuro-symbolic là cơ chế reasoning:

```text
neural component:
  question intent, candidate recall, assignment scoring

symbolic component:
  slot type, role/year/entity/unit compatibility,
  program construction, execution, proof, abstention
```

Không gọi toàn bộ graph là “neuro-symbolic KG”. Tên chính xác hơn là **role-aware Financial Evidence Graph được sử dụng bởi neuro-symbolic reasoner**.

## 7. Benchmark để biến khóa luận thành bài báo

### 7.1 Hai cấp dữ liệu

**Thesis set tối thiểu:**

- khoảng 300 câu real-PDF;
- ít nhất 30 company;
- company-disjoint train/development/sealed test;
- cân bằng lookup, single-step, multi-step và unanswerable;
- mỗi positive có ít nhất một counterfactual hard-negative.

**Paper-grade extension:**

- mục tiêu 800–1.200 câu;
- ít nhất 50 company, nhiều ngành, năm và layout;
- native, scan và hybrid PDF;
- hai annotator cho test và hard-negative quan trọng;
- adjudication và inter-annotator agreement;
- công bố annotation/schema/code, cùng URL/hash/provenance của báo cáo nếu không thể phát hành lại PDF.

Con số trên là mục tiêu thiết kế, không phải điều kiện bảo đảm được nhận bài. Chất lượng annotation quan trọng hơn số lượng.

### 7.2 Annotation bắt buộc cho mỗi câu

```text
question
answerability
answer / abstention reason
operation hoặc program DAG
operand order và reference base
logical-cell identity
physical source cell/span
metric, period, column role, entity/scope, unit/scale
page/table/row/column provenance
hard-negative type
```

### 7.3 Hard-negative taxonomy

- same metric + same year + different role (`actual ↔ plan`);
- same metric + different year;
- same label + consolidated/separate entity;
- same value + different semantic identity;
- percent cell cạnh absolute cell;
- subtotal/detail hoặc liabilities/interest-bearing borrowings;
- same table vs confusable table khác;
- correct execution by coincidence but wrong operands;
- missing role/base/direction, tức câu thực sự unanswerable.

Đây phải là phần dữ liệu đặc trưng của paper, không chỉ là synthetic paraphrase.

## 8. Baseline và ablation bắt buộc

Mọi baseline phải chạy trên cùng PDF, split, question và gold evidence. Không so trực tiếp metric hiện tại của FinHRM với bảng RAGAS trong `mau.md`.

### 8.1 Baseline

| ID | Hệ thống | Mục đích |
|---|---|---|
| B1 | BM25 / keyword | retrieval lexical |
| B2 | dense hoặc hybrid retrieval | semantic retrieval |
| B3 | section-level GraphRAG kiểu `mau.md` | lợi ích hierarchy graph |
| B4 | LLM + retrieved context + Program-of-Thought/code | baseline LLM reasoning |
| B5 | V8/HRM neural-only | baseline nội bộ |
| B6 | flat cell retrieval + deterministic executor | tách graph/proof khỏi arithmetic |
| Full | evidence graph + typed synthesis + verifier + calibrated abstention | FinHRM |

Nếu ngân sách hạn chế, ưu tiên một open-weight LLM có thể tái lập và một API mạnh làm reference; không cần chạy nhiều agent framework gần giống nhau.

### 8.2 Ablation

```text
Full
− column role/header lineage
− graph relations
− typed synthesis
− three-state verifier
− proof-conditioned acceptance
− per-operation calibration
− counterfactual hard-negative training/audit
```

Phải báo cả lợi ích và collateral:

- false accept loại được `F`;
- true accept mất `T`;
- gold retention;
- coverage;
- latency;
- kết quả theo operation/company/layout, không chỉ overall.

## 9. Metric chính và kiểm định

### 9.1 Không dùng RAGAS làm metric trung tâm

RAGAS có thể là metric phụ cho context và answer presentation. Metric trung tâm phải phản ánh numerical correctness và provenance:

1. Evidence-cell Recall@K/MRR.
2. Operand grounding exact match.
3. Program/DAG exact match.
4. Execution accuracy.
5. **Provenance-correct execution:** answer đúng và tất cả operand đúng logical/physical evidence.
6. Accepted precision, coverage và false-accept rate.
7. Risk–coverage curve/AURC.
8. Hard-negative rejection và answerability F1.
9. Citation page/cell accuracy.
10. Latency p50/p95 và candidate count.

### 9.2 Thống kê

- bootstrap confidence interval theo company, không bootstrap từng câu độc lập nếu cùng report;
- McNemar hoặc paired bootstrap cho hai hệ thống trên cùng test;
- Clopper–Pearson lower bound cho accepted precision;
- inter-annotator agreement cho role/program/answerability;
- kết quả theo company/layout/operation và macro average để công ty lớn không chi phối.

## 10. Gate hoàn thành theo hai tầng

### 10.1 Gate khóa luận

- core family compiler exact đạt 100% trên contract suite đã freeze;
- swapped-role/year counterfactual false accept bằng 0 trong core family;
- provenance-correct execution được báo trên real-PDF test;
- accepted precision point estimate ít nhất 90%; lower bound phải được báo trung thực;
- verifier gold retention ít nhất 95% trên từng fold;
- demo có lookup, two-operand, multi-step và ambiguity rejection;
- code/data/report hash và hướng dẫn tái lập đầy đủ.

Nếu accepted precision chưa đạt 90%, vẫn có thể hoàn thành khóa luận như một kết quả nghiên cứu có phân tích lỗi, nhưng không được tuyên bố production-ready.

### 10.2 Gate trước khi nộp conference/journal

- sealed company-disjoint test chỉ mở một lần sau khi method/config đã freeze;
- ít nhất một external/public benchmark hoặc một Vietnamese benchmark được phát hành đủ để tái lập;
- Full vượt neural-only, retrieval và GraphRAG baseline trên provenance-correct execution;
- ablation cho thấy mỗi claim chính có đóng góp riêng;
- có confidence interval/statistical test;
- có error taxonomy và limitation rõ;
- artifact/code/data statement và license hợp lệ;
- không dùng câu hỏi do model sinh chưa kiểm duyệt làm gold test.

## 11. Lộ trình triển khai tối ưu

### M0 — Freeze paper claim

- Chốt ba RQ ở mục 4.
- Chốt core family và deferred scope.
- Không thêm multi-agent, causal analysis hoặc enterprise KG.

### M1 — Hoàn tất vertical slice QP2/QP3

- Compiler contract PA01–PA09, TY01/TY02/TY04, IBDS01 và C13–C15 đã được khóa và compiler-only suite đã pass.
- Role inventory/provenance guard đã có, nhưng không được dùng ABW làm confirmation hoặc đổi thứ tự roadmap.
- Việc còn lại của M1 là nối `reference_base_match`, `logical_cell_unique`, `source_provenance_complete`,
  `answer_mode_complete` và proof từng leaf/intermediate DAG vào consumer runtime trên frozen core cohort.
- Mọi repair M1 trở đi phải tuân `config/finhrm_report_generalization_invariant_v1.json`: không rule theo report,
  confirmation company/layout-disjoint với diagnosis và fail closed trên report ngoài miền hỗ trợ.
- Sau khi QP3 core pass, khóa `QP3-ANSWER-PRESENTATION-PRECOMMIT` trước QP4: `Verified Answer Envelope`, deterministic
  `FACT/COMPUTATION/VERIFIED_SUMMARY` renderers, citation từng claim/operand và `answer_mode_complete`. Đây là presentation
  contract trên verified result, không phải quyền thêm LLM hoặc nới proof gate.

### M2 — Materialize cell-level evidence graph

- Biến EvidenceCell/header lineage/provenance thành schema graph versioned.
- Synthesis lấy operand qua typed graph query.
- Chưa cần Neo4j production.

### M3 — Thesis benchmark

- Annotation real-PDF company-disjoint.
- Tạo counterfactual hard-negative từ gold cell, sau đó human review.
- Freeze train/dev/test manifest trước khi chạy final experiments.

### M4 — Reliability repair

- Replay frozen development.
- Phân tích retrieval/synthesis/verifier/ranking/calibration theo đúng bottleneck.
- Chỉ fine-tune khi ablation chứng minh ranking còn là bottleneck; không fine-tune để sửa từng mẫu.

### M5 — Final experiment

- Chạy B1–B6 và Full.
- Chạy ablation, risk–coverage, statistical test và efficiency.
- Mở sealed test đúng protocol.

### M6 — Thesis/demo/paper package

- UI hiển thị câu trả lời tự nhiên cùng program, operand, proof và citation; câu tính phải có formula/operand lineage, còn
  “nội dung chính” phải là bảng các verified sub-answer với coverage đầy đủ hoặc partial status tường minh.
- Viết thesis theo RQ thay vì kể lịch sử V8–V12.
- Chuẩn hóa repository, model card, data statement và reproducibility script.

### M7 — Post-thesis paper extension

- Mở benchmark lên 800–1.200 câu/50+ company.
- Thêm external benchmark/cross-lingual experiment nếu mapping schema khả thi.
- Thêm OCR/layout robustness và annotation analysis.
- Chỉ thêm LLM explanation khi numeric/citation consistency được verifier kiểm tra.

## 12. Chiến lược nộp bài

### 12.1 Phân biệt conference và Q1/Q2

Conference NLP thường không được gọi Q1/Q2 theo cùng cách với journal. Với NLP, ACL/EMNLP/NAACL/Findings và các workshop chuyên ngành được đánh giá theo venue/review track; ACL Rolling Review hiện có các area phù hợp trực tiếp như `NLP and Symbolic Reasoning`, `Question Answering`, `Resources and Evaluation`, `Information Retrieval` và `Multilingualism` ([ARR scope](https://github.com/acl-org/aclrollingreview/blob/main/cfp.md)).

Q1/Q2 phụ thuộc năm và category của journal; phải kiểm tra lại JCR/SJR tại thời điểm nộp. Không được ghi “journal Q1” như một thuộc tính vĩnh viễn.

### 12.2 Venue theo độ trưởng thành

**Conference/Findings phù hợp nếu trọng tâm là method + benchmark NLP:**

- ACL/EMNLP/NAACL Findings qua ARR;
- COLING/EACL/AACL tùy lịch;
- FinNLP hoặc workshop table reasoning làm bước đầu nếu benchmark/method còn hẹp.

**Journal phù hợp nếu có full-scale benchmark và system evaluation:**

- [Knowledge-Based Systems](https://www.sciencedirect.com/journal/knowledge-based-systems): hợp với knowledge representation, reasoning và intelligent systems;
- [Expert Systems with Applications](https://www.sciencedirect.com/journal/expert-systems-with-applications): hợp với hệ thống AI ứng dụng trong finance/accounting và thực nghiệm rộng;
- [Information Processing & Management](https://www.sciencedirect.com/journal/information-processing-and-management/publish/guide-for-authors): hợp nếu paper nhấn retrieval, provenance và information access;
- Decision Support Systems chỉ hợp khi có thêm đánh giá tác động đến quyết định hoặc user study, không chỉ benchmark QA.

Khuyến nghị thực tế:

1. Hoàn thành thesis và viết một conference/Findings paper tập trung vào method + counterfactual benchmark.
2. Sau phản biện, mở rộng dữ liệu, OCR/layout, external benchmark và user-facing audit để thành journal paper.
3. Không nộp đồng thời hai nơi và không chia nhỏ claim; journal extension phải có đóng góp thực chất mới.

## 13. FinHRM khác `mau.md` ở đâu

`mau.md` mô tả VietFinDocAgent như multi-agent RAG có outline/section graph, hybrid retrieval, formula expansion, code calculation và LLM Reviewer. Graph của nó được thừa nhận chưa phải full `Company–Metric–Year–Value` graph; benchmark hiện dùng RAGAS-generated QA như bước trung gian.

FinHRM không nên cạnh tranh bằng cách thêm nhiều agent hơn. Khác biệt cần được chứng minh như sau:

| Trục | `mau.md` | FinHRM paper claim |
|---|---|---|
| Graph granularity | report/section/subsection + table retrieval | logical/physical financial cell |
| Query representation | expansion prompt/heuristic formula | typed operand-level IR |
| Reasoning | code calculation + reviewer | typed program/DAG + deterministic verifier |
| Evidence validity | retrieved/reviewed context | proof obligation theo metric/role/year/entity/unit |
| Uncertainty | reviewer/missing operand | selective prediction + risk–coverage |
| Negative control | general retrieval noise | counterfactual semantic hard-negative |
| Correctness | answer/RAGAS | answer + program + provenance + acceptance |

Chỉ khi chạy hai hệ thống trên cùng split mới được kết luận hệ thống nào tốt hơn. Các số `RAGAS context precision = 0,89` trong `mau.md` và `accepted execution precision = 88,47%` của FinHRM đo hai khái niệm khác nhau, không được so trực tiếp.

## 14. Những claim không nên dùng

- “Hệ thống đầu tiên dùng knowledge graph cho financial QA.”
- “Hệ thống đầu tiên dùng neuro-symbolic reasoning cho financial QA.”
- “Proof VALID đồng nghĩa đáp án chắc chắn đúng.”
- “Precision 90% nghĩa hệ thống production-ready.”
- “Vietnamese dataset tự nó đủ tạo novelty.”
- “RAGAS cao chứng minh numerical reasoning đúng.”
- “Một câu ABW đúng chứng minh hệ thống generalize.”

Claim nên dùng:

> FinHRM studies whether cell-level financial roles and provenance-aware proof obligations improve selective numerical reasoning under counterfactual evidence ambiguity in Vietnamese financial reports.

## 15. Tên đề tài và tên bài đề xuất

### Tên khóa luận tiếng Việt

**FinHRM: Suy luận neuro-symbolic có kiểm chứng trên đồ thị bằng chứng báo cáo tài chính tiếng Việt**

### Tên khóa luận/bài tiếng Anh

**FinHRM: Proof-Guided Selective Program Synthesis over Role-Aware Financial Evidence Graphs**

Nếu dataset trở thành đóng góp chính:

**VietFinRole-QA: A Counterfactual Benchmark for Provenance-Sensitive Numerical Reasoning over Vietnamese Financial Reports**

Không cần đổi tên sản phẩm FinHRM; chỉ cần tách rõ tên hệ thống và tên benchmark.

## 16. Bước thực hiện ngay

Không xây Neo4j, thêm LLM hoặc mở OCR/layout extension theo một diagnosis report ngay. Thesis-core precommit và QP2
compiler suite đã hoàn tất; QP3 mới hoàn tất column-role/inventory priority lane, chưa hoàn tất toàn bộ proof contract.
Thứ tự tiếp theo:

1. `QP3-CORE-PRECOMMIT`: khóa development-only cohort cho reference base, logical-cell uniqueness, source
   provenance, answer completeness và node-level DAG proof trên toàn bộ core family; diagnosis document như ABW bị
   loại khỏi confirmation/gate. Cohort phải có ít nhất 3 công ty và 3 layout family độc lập với diagnosis theo
   report-generalization invariant.
2. `QP3-CORE-IMPLEMENT`: nối typed slots/DAG vào synthesis–verifier–executor, required `UNKNOWN` phải ABSTAIN và
   `INVALID` phải reject; không mở acceptance mới.
3. Replay nguyên compiler, proof, frozen S4 và company/layout-disjoint development cohorts; chỉ báo kết quả, không
   đổi threshold.
4. Sang QP4 counterfactual/long-question stress khi và chỉ khi QP3 core pass.
5. Sau QP4/QP5 mới materialize graph/no-graph ablation và pilot real-PDF 30–50 câu theo benchmark protocol.

Word/bbox OCR robustness được giữ ở M7 extension trừ khi một cohort đa công ty/layout độc lập chứng minh nó là
bottleneck chung của thesis core. Một smoke case không đủ quyền mở nhánh roadmap mới.

### Cập nhật trạng thái 20/08/2026 — không đổi thứ tự roadmap

QP3 core và generalization-coverage gate đã đóng bằng frozen safe replay cùng source-backed ADD confirmation trên ba real
layout family. Confirmation bổ sung có 10 positive proof-complete trên 5 company/5 PDF source ở native multi-table và
OCR/hybrid; hai physical-source-span collision vẫn `REJECT`. Không dùng ABW, synthetic layout projection hoặc protected split
làm confirmation; không đổi runtime, threshold, checkpoint hoặc model.

Bước kế tiếp theo đúng thứ tự M1 đã khóa là `QP3-ANSWER-PRESENTATION-PRECOMMIT`. Bước này chỉ được khóa schema/gate cho
Verified Answer Envelope và deterministic `FACT/COMPUTATION/VERIFIED_SUMMARY`; chưa được sửa renderer trước precommit.
QP4 và Phase D vẫn khóa. Các mốc M2–M7 giữ nguyên thứ tự.

### Hiệu chỉnh trạng thái 20/08/2026 — QP3 core vẫn đang mở

Cập nhật ngay phía trên bị supersede vì đã gộp candidate-level source-backed confirmation vào evidence đóng gate. Audit chặt hơn
cho thấy OCR/hybrid mới có `2` company/`2` PDF, toàn bộ positive còn `pending_independent_review`, và replay chưa đi qua
extraction/reconstruction, retrieval/intent proposal và symbolic binding end-to-end. Vì vậy không được tuyên bố hệ thống đã hoạt động
đa dạng chỉ từ `10` positive trên `5` PDF.

Trạng thái đúng là `QP3-CORE-GENERALIZATION-COVERAGE-IMPLEMENTATION` **chưa đóng**. Candidate-level confirmation chỉ là supporting
safety evidence; ADD frozen improvement vẫn mới có `1` real layout. Bước tiếp theo vẫn là xây development-safe,
company/layout-disjoint, independently reviewed, end-to-end ADD confirmation với ít nhất `3` company/`3` PDF trong từng additional
layout, rồi replay toàn bộ frozen gates. `QP3-ANSWER-PRESENTATION-PRECOMMIT`, QP4 và Phase D tiếp tục khóa; thứ tự M1–M7 không đổi.

### Cập nhật chốt 20/08/2026 — reviewed end-to-end cohort đóng QP3 core

Cập nhật này supersede trạng thái mở ngay phía trên bằng evidence mới không phải candidate-only. Reviewed cohort chạy engine với duy nhất
PDF thật và câu hỏi: `6/8` positive pass, hai residual `ABSTAIN`, accepted-wrong `0`. `NATIVE_TEXT_TABLE` và
`OCR_OR_HYBRID_TABLE` đều đạt `3` positive company trên `3` PDF độc lập; positive phủ `4` real layout subfamily. Frozen safe replay
giữ `259` ANSWER (`187` cũ + `72` mới), execution/permutation `363/363`, counterfactual false accept `0/141`.

Vì vậy `QP3-CORE-GENERALIZATION-COVERAGE-IMPLEMENTATION` đã đóng đúng gate development generalization; không có claim unseen/universal.
Không dùng ABW/protected split/synthetic projection, không fine-tune và không đổi model/checkpoint/threshold.

Bước kế tiếp theo đúng thứ tự đã khóa là `QP3-ANSWER-PRESENTATION-PRECOMMIT`. Chỉ được khóa schema/gate cho Verified Answer Envelope,
deterministic `FACT/COMPUTATION/VERIFIED_SUMMARY`, citation và derivation replay trước khi sửa renderer runtime. QP4 và Phase D vẫn khóa;
M1–M7 không đổi thứ tự.

### Cập nhật 20/08/2026 — answer presentation precommit đã đóng

`QP3-ANSWER-PRESENTATION-PRECOMMIT` đã PASS mà không sửa runtime. `Verified Answer Envelope v1` khóa typed claim, exact
canonical/display value, executed program DAG, citation từng operand, proof snapshot và summary coverage cho ba mode
`FACT/COMPUTATION/VERIFIED_SUMMARY`. Renderer deterministic không được tạo số, page, citation, formula hoặc claim; không được dùng
rank/score/confidence làm proof và không được nâng action hiện hữu. Thiếu proof hoặc partial summary giữ `ABSTAIN` với coverage `N/M`;
contradiction giữ `REJECT`; complete summary chỉ hợp lệ khi mọi requested row được verify độc lập và `answer_mode_complete=VALID`.

Gate implementation yêu cầu ít nhất `3` verified example, `3` company, `3` real layout family và `3` source document cho mỗi mode,
company/layout-disjoint; synthetic fixtures không tính coverage. Frozen replay vẫn giữ execution/permutation `363/363`, false accept
`0/141`, actions `259/90/14`; affected regression `315/315`. Không dùng ABW/protected split, không fine-tune và không đổi
model/checkpoint/threshold.

Bước kế tiếp theo đúng thứ tự là `QP3-ANSWER-PRESENTATION-IMPLEMENTATION`, chỉ trong whitelist file mới
`answer_presentation.py` và adapter proof snapshot ở `document_reasoning_engine.py`. Retrieval, proposal, binding, verifier, executor,
acceptance và technical audit output tiếp tục freeze. QP4 và Phase D vẫn khóa; M1–M7 không đổi thứ tự.

### Cập nhật 21/08/2026 — QP3 và answer presentation implementation hoàn tất

`QP3-ANSWER-PRESENTATION-IMPLEMENTATION` đã PASS trong exact two-file runtime scope. Hệ thống đã có deterministic Vietnamese renderer
cho `FACT`, `COMPUTATION` và `VERIFIED_SUMMARY` từ `Verified Answer Envelope v1`; technical audit output vẫn tách riêng. Renderer không
tạo số/page/citation/formula/claim, không dùng rank/score/confidence làm proof và không promotion `ABSTAIN/REJECT`.

Source-backed development confirmation gồm toàn bộ `5` eligible reviewed end-to-end PDF record trên `5` company, `5` PDF và `3` real
layout family. Kết quả là FACT `10/10`, COMPUTATION `5/5`, VERIFIED_SUMMARY `5/5`; mỗi mode vượt gate `3` company/`3` layout/`3` source.
Candidate/citation/summary permutation lần lượt `5/5`, `15/15`, `5/5`; presentation hard-negative false accept `0/8`. Partial summary
giữ `ABSTAIN` và hiện `N/M`, conflict `REJECT`, uncited qualitative cause không được render.

Frozen core replay vẫn giữ actions `259/90/14`, execution/permutation `363/363` và counterfactual false accept `0/141`. Không dùng
ABW/protected split/synthetic coverage, không fine-tune và không đổi model/checkpoint/threshold. Presentation/report tests đạt `22/22`;
affected regression đạt `293/293`.

M1/QP3 hiện hoàn tất. Bước tiếp theo theo đúng thứ tự là `QP4-PRECOMMIT` để khóa counterfactual/long-question stress contract và exact
scope. Việc sửa runtime QP4 chưa được phép; Phase D vẫn khóa và M2–M7 không đổi thứ tự.

### Cập nhật 21/08/2026 — QP4 precommit đã khóa

`QP4-PRECOMMIT` đã PASS mà không sửa runtime. Contract đóng sáu minimal-pair dimension: column role, một operand year, entity/scope,
unit/scale, signed direction và denominator/reference base. Program cũ phải mất tính hợp lệ semantic sau mutation; score giảm hoặc numeric
result trùng không được tính pass. Meaning-preserving short/long, discourse, clause permutation, punctuation, không dấu, bilingual
parenthetical và distractor clause phải giữ đúng semantic IR, eligible program set, winner/action, provenance và exact execution.

Compiler variants synthetic chỉ dùng cho semantic contract, không tính tổng quát hóa. Source-backed QP4 confirmation phải là
development-only PDF thật, independently reviewed, company-disjoint khỏi diagnosis; mỗi counterfactual dimension áp dụng và long/surface
track đều cần tối thiểu `3` company, `3` source document và `3` real layout family. Repeated metric/conflicting cell phải fail closed;
multi-output, qualitative follow-up, anaphora và cross-document báo coverage riêng, không tính safe abstention là answered.

Runtime whitelist tương lai được khóa có điều kiện ở bốn file compiler/synthesis/verifier/direction, nhưng chưa được phép sửa trong bước
precommit này. Trước implementation phải freeze development cohort và baseline diagnostic, sau đó chỉ sửa layer fail nhỏ nhất và replay
toàn bộ regression, equivalence, counterfactual, permutation cùng frozen safe gates. QP4 runtime, QP5 và Phase D vẫn khóa; M2–M7 không đổi
thứ tự.

### Cập nhật 21/08/2026 — QP4 development cohort và baseline đã hoàn tất

QP4 baseline đã chạy đủ hai tầng mà không sửa runtime. Compiler phủ `13` answer family, `3` ambiguity control, `91` surface variant và
`57` minimal pair. End-to-end chạy `50` stress query trên `5` independently reviewed company/PDF và `3` real layout; synthetic question
không được tính layout confirmation, safe abstention không tính answered và baseline không tạo OCR artifact mới.

Kết quả diagnostic: compiler surface `73/91`, counterfactual detection `44/57`, source winner invariance `20/35`, source old-winner
counterfactual false accept `4/15`. Lỗi unit/scale là generic: compiler fail `13/13` family và runtime giữ nguyên old winner ở `4` company
trên đủ `3` real layout. Operand-year và entity/scope giữ false accept `0/5`. Column-role, direction và reference-base chưa có source-backed
ACTUAL/PLAN ratio anchors nên tiếp tục `NOT_CONFIRMED`. Short/long và clause projection đúng ở compiler nhưng source winner fail `5/5`, vì
vậy cần isolate retrieval/binding và không được tự mở file ngoài whitelist.

Cohort/baseline lock hoàn tất nhưng QP4 implementation gate chưa pass. Bước tiếp theo là bổ sung independently reviewed source anchors cho
ba dimension còn thiếu và review pre-authorization chỉ cho failure group generic đã được chứng minh. Runtime, QP5, Phase D và thứ tự
M2–M7 tiếp tục khóa.
