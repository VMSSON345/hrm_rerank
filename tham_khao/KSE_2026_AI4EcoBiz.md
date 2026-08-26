# VietFinDocAgent: A Multi-Agent Retrieval-Augmented Framework for Financial Statement Information Retrieval in the Vietnamese Stock Market

## Page 1

VietFinDocAgent: A Multi-Agent
Retrieval-Augmented Framework for Financial
Statement Information Retrieval in the Vietnamese
Stock Market
Abstract—Answering questions about Vietnamese financial
statements requires more than locating a relevant paragraph. The
system must navigate long, table-heavy, hierarchically structured
PDF reports, identify the correct reporting period and statement
version, and perform reliable calculations over values drawn from
one or more tables. Existing approaches address only parts of this
workflow. Graph-based retrieval preserves document structure
but is mainly designed for fact lookup; agentic document readers
such as DocAgent support outline-aware navigation but do not
explicitly ground financial formulas; and multi-round systems
such as FinLongDocAgent support verified calculation but target
English-language SEC filings. We present VietFinDocAgent, a
multi-agent retrieval-augmented framework for Vietnamese financial reports. The implemented system uses a PDF extraction
and OCR stage to obtain page-level report content, builds
a Djago-compatible Neo4j graph, combines keyword, dense,
table-cell, and graph retrieval, applies formula-oriented query
expansion, performs code-based calculation when a formula
can be grounded, and evaluates outputs with official RAGAS
metrics. Experiments on a generated Vietnamese financial-report
benchmark show that VietFinDocAgent achieves the strongest
overall RAGAS performance among the evaluated systems, with
answer relevancy of 0.65, context precision of 0.89, context
recall of 0.98, and faithfulness of 0.82. The framework follows
the original design of FinVNLongDocQA, while the current
experiment pipeline uses RAGAS-generated question-answer sets
and benchmark files to support reproducible evaluation.
Key words—Large language models, financial statement question answering, multi-agent systems, retrieval-augmented generation, Vietnamese financial reports, numerical reasoning
I. INTRODUCTION
Financial analysts often need to retrieve information from
quarterly and annual financial statements, annual reports,
explanatory notes, and audit reports issued by Vietnam’s listed
companies. These documents are often long, table-heavy, and
available in several versions, such as consolidated or separate
statements and audited or unaudited reports. Unlike general
document question answering, financial analysis often requires
calculations over multiple line items rather than a direct text
lookup. A reliable system must therefore identify the correct
values, preserve their units and reporting periods, and provide
page- or table-level evidence for every operand used in the
final answer.
Recent work addresses different parts of this problem.
DocAgent [1] treats long, multimodal document understanding
as an active-reading process. It represents a document as a
hierarchical outline, allows an Actor agent to inspect selected
sections through interactive tools, and uses a Reviewer agent to
verify the answer against additional evidence. A memory component stores reusable correction patterns. Although DocAgent
performs well on report-style documents, it is not designed for
explicit, verifiable financial calculations.
FinLongDocQA and its accompanying FinLongDocAgent
framework [2] focus on document-level numerical reasoning
over complete SEC annual reports. FinLongDocAgent uses an
Expansion Agent to translate a question into a formula and
a list of required operands, a retriever to gather candidate
evidence, a Solving Agent to compute the answer, and an
Evaluation Agent to request additional retrieval when information is missing. The framework improves over single-round
retrieval and long-context prompting, but it is designed for
English SEC filings. Moreover, the reported error analysis
shows that retrieval remains the main bottleneck.
FR-Graph RAG [3] addresses the structure of Vietnamese
financial reports by representing each report as a graph
containing Company, Report, Sector, Section, Subsection, and
Time nodes. Questions are answered through Cypher queries
over this graph. The approach preserves document hierarchy
and reports higher faithfulness and context precision than
vector-based alternatives. However, its retrieval units remain
largely at the section and subsection level, and the framework
does not include an explicit formula-expansion or verifiedcalculation stage for multi-table numerical reasoning.
Taken together, these systems leave three requirements insufficiently addressed: (i) efficient navigation through long
and hierarchically structured Vietnamese reports; (ii) formuladriven, multi-round retrieval for questions whose evidence is
distributed across tables and notes; and (iii) robust handling
of Vietnamese reporting conventions, including scanned documents, company-specific accounting terminology, consolidated
and separate statements, and number formats that use dots
for thousands, commas for decimals, and parentheses for
negative values. VietFinDocAgent is designed to address these
requirements in a unified pipeline.
This study investigates the following research questions:
• RQ1: Does combining outline-aware navigation with
formula-driven, multi-round retrieval improve evidence

## Page 2

localization compared with single-round vector retrieval?
• RQ2: Does a code-based calculation engine, together
with a Reviewer that checks operands, periods, statement
versions, and units, reduce numerical errors in cross-table
questions?
• RQ3: Can a memory module that stores Vietnamese
accounting terminology and recurring correction patterns
improve query expansion across companies and industry
sectors?
The main contributions of this paper are:
• We propose VietFinDocAgent, a multi-agent retrievalaugmented framework that combines outline-aware navigation with formula-driven, multi-round retrieval and
verified calculation.
• We implement a retrieval strategy that combines keyword/BM25 retrieval, dense embeddings for baseline
modes, table-cell retrieval, and Djago-compatible graph
retrieval over Neo4j. Page-image retrieval remains part of
the broader design for scanned or layout-broken reports.
• We introduce a calculation engine that normalizes currency units, Vietnamese thousands and decimal separators, parenthesized negative values, and reporting periods
before executing formulas in code.
• We provide an experimental pipeline for RAGAS-based
question generation, answer generation, baseline comparison, and official RAGAS evaluation, while retaining
the FinVNLongDocQA benchmark design as the target
dataset schema.
The remainder of the paper is organized as follows. Section II reviews related work on financial question answering, agentic document understanding, and long-document numerical reasoning. Section III presents the research method
and implemented VietFinDocAgent architecture. Section IV
describes the proposed benchmark and RAGAS-based data
generation method. Section V reports the experimental setup
and results. Section VI discusses limitations and future work,
and Section VII concludes the paper.
II. RELATED WORKS
A. Financial document question answering with LLMs and
RAG
Modern large language models are based on the Transformer
architecture [4]. Recent studies suggest that, with suitable
prompting and task design, these models can perform competitively on financial statement analysis [5], [6]. Research
on financial numerical reasoning has progressed from table–
text benchmarks such as FinQA [7], TAT-QA [8], and ConvFinQA [9] to more structurally demanding settings represented by MultiHiertt [10] and the long-context DocFinQA
benchmark [11].
Financial reports are particularly difficult for standard
chunk-based RAG because related evidence may be separated
across table rows, pages, sections, or explanatory notes. Splitting a report into fixed-size passages can therefore remove
the structural context needed to interpret a value correctly.
Figure 1. Knowledge Graph Ontology for Financial Reports
FR-Graph RAG addresses this issue by representing a Vietnamese report as a six-node ontology containing Company,
Report, Sector, Section, Subsection, and Time nodes. Relations such as AT_TIME, BELONGS_TO, HAS_REPORT,
HAS_SECTION, and HAS_SUBSECTION preserve the organization of the source document, as illustrated in Fig. 1. Instead
of relying only on vector similarity, the system generates
Cypher queries over the graph. VietFinDocAgent follows the
same principle of structure preservation, but extends it with
table-cell retrieval, formula expansion, and calculation.
B. Retrieval-augmented and self-reflective generation
Retrieval-augmented generation is a natural fit for long
financial reports because the language model should answer
from external evidence rather than from parametric memory
alone. However, standard single-round RAG is sensitive to
retrieval noise and often fails when the required evidence is
distributed across several pages or table blocks. Self-reflective
retrieval methods such as Self-RAG [12] address part of
this limitation by letting the model judge whether additional
retrieval or verification is needed. RAGAS [13] provides
reference-free and reference-based metrics for evaluating RAG
pipelines, including answer relevancy, faithfulness, context
precision, and context recall. VietFinDocAgent adopts this
evaluation view but specializes the retrieval and verification
process for Vietnamese financial statement QA.
C. Agentic long-document understanding
DocAgent formulates long, multimodal document question
answering as an active-reading task rather than a single-pass
prompting task. Each document is converted into a hierarchical
XML outline in which sections form parent nodes and headings, paragraphs, images, and tables form child nodes. Page
ranges and unique identifiers are attached to relevant nodes so

## Page 3

that detailed content can be retrieved on demand. To keep the
initial context compact, long paragraphs and visual elements
are represented by short hints, such as a first sentence or a
caption.
An Actor agent receives the question, the outline, and five
tools: search, get_section_content, get_image,
get_page_images, and get_table_image. It calls
these tools iteratively until it has gathered enough evidence
or reaches the tool-call limit. A Reviewer then examines the
Actor’s reasoning path and may retrieve additional evidence
before producing the final answer. A reflection step updates
the memory when the Reviewer corrects the Actor. VietFinDocAgent adapts these ideas through lightweight document
tools, answer review, and evidence-focused prompting over the
extracted report content.
D. Numerical reasoning over long financial documents
FinLongDocQA evaluates numerical question answering
over complete annual reports rather than isolated tables. The
benchmark is built from SEC filings and focuses on questions
that require evidence from multiple pages or combinations of
table and narrative evidence. FinLongDocAgent mirrors the
workflow of a financial analyst: an Expansion Agent converts
the question into a formula-like expression, a retriever returns
candidate evidence, a Solving Agent computes the answer, and
an Evaluation Agent requests more evidence when information
is missing.
VietFinDocAgent adopts this multi-round reasoning pattern,
but adapts it to Vietnamese financial reports. The implementation provides a Vietnamese-oriented expansion module,
value extraction, unit normalization, calculation, and reviewer
checks. It also includes a legacy adapter to the original Fin-
LongDocAgent codebase for direct baseline comparison when
the required dependencies and model endpoint are available.
E. Vietnamese financial reporting challenges
Vietnamese financial reports introduce structural and typographic issues that are not fully addressed by the systems
above. Many reports are distributed as scanned documents
and therefore require reliable Vietnamese OCR. Accounting
terms may vary across companies and industries and are often
presented bilingually. The same reporting period may also
include both consolidated and separate statements, which must
not be mixed. In addition, Vietnamese number formatting
commonly uses dots as thousands separators, commas as
decimal separators, and parentheses for negative values.
FR-Graph RAG addresses document structure and Vietnamese terminology but does not provide verified multitable calculation. DocAgent supports outline-aware navigation
and answer review but does not explicitly ground financial
formulas. FinLongDocAgent supports formula-driven retrieval
and calculation but is tailored to English SEC filings. VietFinDocAgent combines these complementary strengths and
adds normalization and verification steps specifically designed
for Vietnamese financial statements.
III. RESEARCH METHOD
A. Overview
VietFinDocAgent is designed as a verifiable financialanalysis workflow rather than a general document chatbot.
Given a Vietnamese financial report and a user question, the
intended system performs six stages: (1) ingesting and parsing
the source document; (2) constructing a compact document
representation; (3) retrieving candidate evidence through complementary indexes; (4) expanding the question into a financial
formula and a set of operands when possible; (5) grounding
the operands and executing the calculation in code; and (6)
reviewing the result for missing evidence, inconsistent periods,
incorrect statement versions, and unit errors.
Fig. 2 summarizes the implemented end-to-end workflow. The current codebase exposes this workflow through
command-line stages for dataset generation, graph construction, answer generation, baseline execution, and RAGAS evaluation.
B. Problem statement
The problem studied in this paper is document-grounded
question answering over Vietnamese financial statements. Unlike open-domain QA, the answer must be grounded in the
selected report, must preserve the correct year, unit, and
statement version, and may require numerical computation
over several tables or explanatory notes. Following the activereading formulation of DocAgent [1] and the multi-round
numerical reasoning loop of FinLongDocAgent [2], we define
a Vietnamese financial report as a collection of page-level and
table-level evidence units
D = {c1, c2, . . . , cN},
(1)
where each unit ci is associated with metadata such as page
number, report identifier, fiscal period, unit, and statement
version. The document is also represented by an outline tree
OD and a Djago-compatible graph GD = (V, E) containing
report, section, subsection, company, sector, and time nodes.
Given a question q, the system aims to produce an answer ˆy
and a compact evidence set ˆE ⊆D:
(ˆy, ˆE) = arg max
y,E⊆D p(y, E | q, OD, GD).
(2)
At round t, the actor-style retrieval state is defined as
st = (q, OD, GD, E<t, ∆t−1, m) ,
(3)
where E<t is the accumulated evidence, ∆t−1 is the set
of missing or inconsistent operands found in the previous
round, and m denotes reusable memory such as terminology
mappings. The next retrieval or answer action is sampled from
at ∼ρA(a | st),
(4)
while the reviewer verifies the answer trace τt and produces
a missing-component set:
∆t = ρR(q, ˆyt, τt, Et).
(5)
If ∆t = ∅, the system terminates; otherwise, the missing
components are fed back into the next query-expansion round.

## Page 4

Figure 2. Implemented VietFinDocAgent workflow.
C. Solution strategy
VietFinDocAgent solves the problem by decomposing it
into three coupled subproblems: evidence localization, formula
grounding, and answer verification. Evidence localization retrieves a small evidence set Et from the full report using keyword, dense, table-cell, and graph signals. Formula grounding
maps the question to a formula Ft and operand set Ot when
the question is numerical. Answer verification checks whether
the retrieved evidence supports all operands and whether the
final answer is faithful to the report. This design follows
the selective reading behavior of DocAgent and the iterative
expansion–solving–evaluation loop of FinLongDocAgent, but
adds Vietnamese-specific normalization for accounting terms,
statement versions, and number formats.
D. Data ingestion for Vietnamese financial statements
The source collection may include company disclosure
portals, investor-relations websites, annual reports, and audited
financial statements. During ingestion, each document is assigned metadata such as company ticker, reporting period,
statement type, consolidation status, and audit status. Textnative PDFs are processed with a layout-aware extraction stage,
whereas scanned reports are processed with a Vietnamese
OCR engine. The implemented pipeline receives the pagelevel content produced by this PDF extraction and OCR stage,
reconstructs page documents, infers metadata, and extracts
table cells when table markup is available.
1) Vietnamese financial outline construction: Following the
outline-based navigation used by DocAgent and the section
hierarchy of FR-Graph RAG, each report is converted into
a compact representation. The current implementation stores
page-level documents, table-cell records, and graph nodes for
sections and subsections. Each evidence unit retains metadata
such as document identifier, page number, period, statement
version, audit status, and source path. This representation
allows the retriever and reviewer to reason over page, table, and
report-level provenance without repeatedly sending the entire
report to the language model.
Following the FS-Graph RAG construction in [3], the initial
pages of a report D are used to infer a constrained metadata
set
M = fθ(Dinit, C) = {m1, m2, m3, m4},
(6)
where C denotes schema constraints and permissible values,
while m1, m2, m3, and m4 correspond to company name,
sector, report type, and reporting period. For page labeling, a
page is represented as a set of text lines Lp = {l1, l2, . . . , ln}
and the candidate component labels are denoted by Z =
{z1, z2, . . . , zm}. A line-label score combines semantic similarity and exact phrase matching:
Score(li, zj) = sim(li, zj) + δ 1[zj ⊂li].
(7)
The best heading line and page label are then selected as
l∗= arg max
li∈Lp max
zj∈Z Score(li, zj),
(8)
z∗= arg max
zj∈Z Score(l∗, zj).
(9)
If the maximum score is below a threshold τ, the page is
assigned a null label:
label(p) =
(
None,
maxli,zj Score(li, zj) < τ,
z∗,
otherwise.
(10)
Uncertain pages are normalized by inheriting the nearest
previous valid section,
S(pi) = S(pi−1),
if label(pi) = None,
(11)
and section-to-subsection segmentation is performed over candidate titles
SubTitle = {s1, s2, . . . , sK},
(12)
with each detected subsection defined as
subi = {l | l ∈Sk, si ≤l < si+1}.
(13)
These equations formalize how the implemented loader preserves the report hierarchy before graph construction and
retrieval.
E. Hybrid retrieval
No single retrieval method works equally well for all financial questions. Exact line-item names are often best handled
by keyword search, paraphrased questions require semantic
retrieval, and numerical lookups benefit from table-aware
indexing. VietFinDocAgent therefore follows the five-layer
retrieval design in Table I. In the current code, keyword/BM25,
dense retrieval, table-cell retrieval, graph retrieval, and optional evidence reranking are implemented across the main
pipeline and applicable evaluation modes. Page-image retrieval
is retained as a design layer for scanned or parsing-broken
pages, but it is not the primary runtime path in the current
experiments.

## Page 5

For an expanded query qt and candidate evidence unit ci,
the hybrid retrieval score is modeled as
score(qt, ci) =λb BM25(qt, ci) + λd cos(ϕ(qt), ϕ(ci))
+ λg 1[ci ∈NGD(qt)] + λt TableMatch(qt, ci),
(14)
where ϕ(·) is the dense embedding function, NGD(qt) denotes
graph-neighbor evidence obtained from the Djago-compatible
graph, and TableMatch rewards row, column, period, and
unit matches in extracted tables. The coefficients λb, λd, λg, λt
control the contribution of keyword, dense, graph, and tablecell retrieval.
The graph backend follows the Djago-compatible FR-
Graph RAG schema. It builds Company, Sector, Time,
Report, Section, and Subsection nodes and links them
through relations such as HAS_REPORT, HAS_SECTION,
HAS_SUBSECTION, BELONGS_TO, and AT_TIME. To
avoid mixing consolidated and separate reports, each report
keeps a document identifier derived from the source report
rather than only company and year metadata.
The graph-only retrieval route follows the FS-Graph RAG
formulation. Let the financial-statement graph be
G = (V, E),
(15)
where V contains report, section, subsection, company, sector,
and time nodes, and E contains the structural relations among
them. Given a user question q, a graph-query generation
function maps the natural-language request into a Cypher
query:
Cq = fcypher(q).
(16)
The Cypher query is executed over G to obtain the relevant
subgraph
Sq = Execute(G, Cq),
Sq ⊆G.
(17)
The retrieved subgraph is serialized into a textual context
cq = fagg(Sq),
(18)
which is then combined with the answer prompt P:
yq = LLM(P, cq).
(19)
In VietFinDocAgent, this graph route is used both as an
independent baseline and as one evidence source inside the
full hybrid retrieval pipeline.
F. Query expansion into financial formulas
The Expansion Agent converts a natural-language question
into a formula-like expression and a list of required operands
when the question matches a supported financial pattern. For
example, for the question “What was the debt-to-equity ratio in
2024?”, the agent identifies short-term borrowings (Bs), longterm borrowings (Bl), and owners’ equity (E), and produces
D/E = Bs + Bl
E
.
(20)
It then generates Vietnamese and English variants for each
operand and extracts constraints such as fiscal year, quarter,
company, currency unit, and statement version. In the current
implementation, this expansion is heuristic and is complemented by LLM answer generation when no formula can be
reliably identified.
More generally, at each retrieval round the expansion module produces a formula candidate Ft, operand set Ot, and
search-query set Qt:
(Ft, Ot, Qt) = Expand(q, ∆t−1).
(21)
The retrieved evidence is then
Rt = TopKci∈D max
qt∈Qt score(qt, ci).
(22)
The accumulated context used for reasoning is compressed as
Et = RerankCompress(E<t ∪Rt).
(23)
After evidence extraction, the calculation module estimates
a numerical answer by applying the grounded formula to
normalized operand values:
ˆyt = Calc (Ft, {value(o, Et) | o ∈Ot}) .
(24)
G. Calculation engine
The calculation engine executes formulas in code instead of
relying on the language model to perform arithmetic. Before
computation, each retrieved value is normalized. Currency
units such as VND, thousand VND, million VND, and billion
VND are converted to a common base. Parenthesized values
are interpreted as negative numbers, and Vietnamese thousands
and decimal separators are parsed explicitly. If an operand
cannot be grounded with sufficient confidence, the system
records the missing operand and either performs another
retrieval round or falls back to evidence-grounded answer
generation.
H. Reviewer and memory
The Reviewer combines ideas from DocAgent, FinLong-
DocAgent, and self-reflective retrieval systems such as Self-
RAG [12]. It checks common failure types: missing operands,
incorrect reporting periods, confusion between consolidated
and separate statements, inconsistent units, and selection of
an incorrect accounting line item. The current implementation
records review issues in the result file and uses evidence
compression and reranking to improve answer grounding. A
terminology memory is part of the design and is represented
through reusable term mappings; a persistent company- and
sector-level memory bank remains a future extension.
When the reviewer detects a correction or a missing
operand, the memory state can be updated as
mt+1 = ρM(q, τt, ∆t, mt),
(25)
where ρM stores reusable guidance such as alternative Vietnamese account names, unit conventions, and recurring correction patterns. The resulting multi-round workflow is described
below.

## Page 6

Table I
HYBRID RETRIEVAL LAYERS IN VIETFINDOCAGENT
Layer
When used
Implementation status and example
Keyword / BM25
Exact accounting terminology
Implemented for exact terms such as
borrowings and finance lease payables.
Dense embeddings
Natural-language questions
Implemented in vector/keyword,
FinLongDocAgent-style, and full
VietFinDocAgent modes.
Table-cell
Locate a value by row, column, and period
Implemented when the PDF extraction stage
preserves table markup.
Page image
Scanned or parsing-broken tables
Design layer retained for scanned reports; not
the main runtime path in the current code.
Graph
Preserve report hierarchy and section/subsection links
Implemented through the Djago-compatible
Neo4j graph backend.
Algorithm 1 starts with an empty missing-component set
∆0 and an empty evidence set ˆE. At each round, the system
expands the question into a formula candidate, operands, and
search queries; retrieves evidence from text, tables, dense
indexes, and the graph; compresses the accumulated evidence;
and then either calculates the answer in code or generates
an evidence-grounded response. The Reviewer checks missing
operands, periods, units, and statement versions. If no issue
is found, the loop stops early; otherwise, the detected issues
guide the next retrieval round. Finally, the memory is updated
and the system returns the answer with its supporting evidence.
Algorithm 1 VietFinDocAgent multi-round QA workflow
Require: Report D, question q, graph GD, outline OD, memory m, top-k,
maximum rounds T
Ensure: Final answer ˆy and evidence set ˆE
1: ∆0 ←∅, ˆE ←∅
2: for t = 1 to T do
3:
(Ft, Ot, Qt) ←Expand(q, ∆t−1)
4:
Rt ←HybridRetrieve(Qt, D, GD, k)
5:
ˆE ←RerankCompress( ˆE ∪Rt)
6:
(ˆyt, τt) ←SolveOrCalculate(q, Ft, Ot, ˆE)
7:
∆t ←Review(q, ˆyt, τt, ˆE)
8:
if ∆t = ∅then
9:
break
10:
end if
11: end for
12: m ←UpdateMemory(q, τt, ∆t, m)
13: return ˆyt, ˆE
IV. PROPOSED METHOD
A. Benchmark design
This section describes the proposed benchmark and datageneration method used to evaluate VietFinDocAgent. No
public benchmark currently evaluates document-level question
answering over Vietnamese financial statements with explicit
evidence and executable calculations. We therefore retain the
FinVNLongDocQA design, which covers the five question categories summarized in Table II. The taxonomy is informed by
the evaluation settings of FR-Graph RAG and FinLongDocQA.
The target benchmark will be constructed from reports
of Vietnamese listed companies across multiple industries,
reporting periods, and statement versions. Candidate questions
should be filtered to remove incoherent questions, invalid
calculations, unsupported answers, and cross-table candidates
that can actually be solved from a single table. Each remaining
item should be manually reviewed for financial relevance, clear
specification of year, segment, unit, and statement version, and
sufficient supporting evidence.
B. RAGAS-based data generation
The current code implements a reproducible intermediate path using the official RAGAS test-set generator. The
generate-dataset command samples extracted report
content, calls the official RAGAS generator, and writes benchmark records containing question, reference answer, report
identifier, metadata, and reference contexts. This supports
controlled experiments before the manually verified FinVN-
LongDocQA release is completed.
V. EXPERIMENT
A. Experimental setup
The planned baselines in the current experimental design are: (i) vector/keyword RAG; (ii) FR-Graph-style graphonly RAG; (iii) a Vietnamese adaptation of FinLongDocAgent without outline-aware navigation; (iv) a DocAgent-style
adapter; and (v) the complete VietFinDocAgent framework.
These systems are implemented in the current CLI as vector_keyword, graph_only, finlongdocagent_vi,
docagent_txt, and full_vietfindocagent. Additional legacy adapters are provided for the original FR-Graph
RAG project, DocAgent, and FinLongDocAgent when their
dependencies are available.
The implemented graph_only baseline uses the same
Djago-compatible
Neo4j
graph
as
the
full
system
but
restricts
retrieval
to
graph
evidence.
The
implemented
full_vietfindocagent baseline combines BM25, dense
retrieval, table-cell retrieval, graph retrieval, optional crossencoder reranking, answer generation, calculation, and review.
The main run-experiment command runs the integrated
VietFinDocAgent pipeline over the selected reports and benchmark questions, appending each answer to the output file as
soon as it is generated.

## Page 7

Table II
PROPOSED FINVNLONGDOCQA QUESTION CATEGORIES
Category
Goal
Example
Single-table
Locate a value in one table
Net revenue for 2024
Cross-table
Combine values from multiple tables
Borrowings / owners’ equity
Text–table
Combine a table value with explanatory text
Interest expense from the debt note
Calculation-intensive
Perform a multi-step calculation
FCF, ROIC, or cash conversion ratio
Statement-version-aware
Compare report versions or periods
Separate versus consolidated statement
B. Metrics
The evaluation reports both retrieval and answer quality.
The original protocol includes page-level Precision, Recall, F1,
and the percentage of document pages retrieved. The current
implementation records retrieval recall when gold pages are
available and also reports completion count, skip count, Exact
Match, Tolerance Accuracy, mean numeric error, mean number
of rounds, and mean evidence count.
RAGAS metrics [13] are implemented through the official
RAGAS library. The current evaluation reports Faithfulness,
Answer Relevancy, Context Precision, and Context Recall.
Numerical performance is measured with Exact Match and
Tolerance Accuracy when the benchmark contains a numerical
gold answer. Token-level F1 and full page-level Precision/F1
remain part of the planned benchmark evaluation once manually verified gold annotations are available.
Following the metric definitions used in FS-Graph RAG [3],
Faithfulness is computed from the atomic statements in the
generated answer. If S is the set of generated statements and
V ⊆S is the subset supported by the retrieved context, then
Faithfulness = |V |
|S| .
(26)
Context Precision evaluates whether relevant contexts are
ranked near the top:
CP =
PK
k=1 Precision@k · vk
rK
,
(27)
where vk denotes the relevance value of the context at rank k
and rK is the number of relevant contexts among the top-K
results. The precision at rank k is
Precision@k =
TP@k
TP@k + FP@k.
(28)
Context Recall measures how many reference-answer statements are covered by the retrieved context C:
CR = num(si ∈C)
num(si)
.
(29)
C. Expected outcomes
For RQ1, we expect graph-augmented and multi-round
retrieval to improve evidence recall without retrieving a
large fraction of the report, particularly for cross-table and
statement-version-aware questions. For RQ2, we expect codebased calculation and Reviewer checks to reduce errors caused
Table III
RAGAS EVALUATION RESULTS ON THE GENERATED BENCHMARK
System
Relevancy Precision Recall Faithfulness
Vector/Keyword RAG
0.65
0.80
0.94
0.73
FR-Graph RAG
0.61
0.87
0.97
0.74
FinLongDocAgent
0.62
0.84
0.97
0.71
DocAgent
0.61
0.83
0.98
0.74
VietFinDocAgent
0.65
0.89
0.98
0.82
by missing operands, inconsistent units, and period mismatches. For RQ3, we expect terminology memory to improve
retrieval across companies and sectors whose line-item labels
differ from those seen during initial development. These gains
may increase latency because additional tool calls and retrieval
rounds are required; the experiments therefore report both
answer quality and runtime-facing summary statistics.
D. Results
Table III reports the RAGAS evaluation results of VietFinDocAgent and the baseline systems on the generated Vietnamese
financial report question-answering benchmark.
The results indicate that VietFinDocAgent achieves the
strongest overall RAGAS performance among the evaluated
systems. It obtains the highest precision and faithfulness scores
and ties for the highest relevancy and recall scores after
rounding. Vector/keyword RAG retrieves broad contexts and
therefore remains competitive in recall, while FR-Graph RAG
improves structure-aware retrieval. The full VietFinDocAgent
pipeline benefits from combining keyword, dense, table-cell,
and graph retrieval with formula-oriented reasoning and review.
VI. DISCUSSION AND FUTURE WORK
The current work has several limitations. The graph follows
the Djago report-section-subsection schema and is not yet a
full financial knowledge graph with explicit Company–Metric–
Year–Value nodes for every extracted value. Page-image retrieval is retained in the design, but current experiments
primarily evaluate extracted page content, table cells, dense
retrieval, and graph retrieval. Building the final benchmark
still requires careful manual verification of evidence and
calculations. Future work will complete the manually verified
FinVNLongDocQA dataset, strengthen page-image retrieval
for scanned reports, extend graph representation to financial

## Page 8

values and cross-period links, and evaluate component ablations at larger scale.
VII. CONCLUSION
This
paper
presents
VietFinDocAgent,
a
multi-agent
retrieval-augmented framework for question answering over
Vietnamese financial statements. The framework combines
outline-aware document navigation, hybrid retrieval, formulaoriented query expansion, code-based calculation, and iterative
review. It extends prior work with processing steps for scanned
Vietnamese reports, accounting-term variation, consolidated
and separate statements, and Vietnamese number formats. The
experimental results show that VietFinDocAgent achieves the
strongest overall RAGAS performance among the evaluated
systems, especially in precision and faithfulness.
ACKNOWLEDGMENT
This work was supported by Vietnam National University,
Hanoi, under research project QG.26.68.
REFERENCES
[1] L. Sun, L. He, S. Jia, Y. He, and C. You, “Docagent: An agentic
framework for multi-modal long-context document understanding,” in
Proceedings of the 2025 Conference on Empirical Methods in Natural
Language Processing. Association for Computational Linguistics, 2025,
pp. 17 588–17 606.
[2] Y.-C. Wang, W.-A. Wang, and C.-S. Chen, “Document-level numerical
reasoning across single and multiple tables in financial reports,” arXiv
preprint arXiv:2604.03664, 2026.
[3] D.-A. Nguyen, T. T. T. Do, M.-D. Tao, and H.-V. Tran, “A graphbased financial report representation for interpretable retrieval and
analytical reasoning,” in Proceedings of this venue, 2026, forthcoming;
bibliographic details to be updated after acceptance.
[4] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,
L. Kaiser, and I. Polosukhin, “Attention is all you need,” in Advances in
Neural Information Processing Systems, vol. 30, 2017, pp. 5998–6008.
[5] A. G. Kim, M. Muhn, and V. V. Nikolaev, “Financial statement analysis
with large language models,” arXiv preprint arXiv:2407.17866, 2024.
[6] P. Pedinotti, P. Baumann, N. Jessurun, L. Barrett, and E. Santus,
“Prompting the market? a large-scale meta-analysis of GenAI in finance
NLP (2022–2025),” CoRR, vol. abs/2509.09544, 2025.
[7] Z. Chen, W. Chen, C. Smiley, S. Shah, I. Borova, D. Langdon,
R. Moussa, M. Beane, T.-H. Huang, B. R. Routledge, and W. Y. Wang,
“FinQA: A dataset of numerical reasoning over financial data,” in
Proceedings of the 2021 Conference on Empirical Methods in Natural
Language Processing. Association for Computational Linguistics, 2021,
pp. 3697–3711.
[8] F. Zhu, W. Lei, Y. Huang, C. Wang, S. Zhang, J. Lv, F. Feng, and
T.-S. Chua, “TAT-QA: A question answering benchmark on a hybrid
of tabular and textual content in finance,” in Proceedings of the 59th
Annual Meeting of the Association for Computational Linguistics and the
11th International Joint Conference on Natural Language Processing.
Association for Computational Linguistics, 2021, pp. 3277–3287.
[9] Z. Chen, S. Li, C. Smiley, Z. Ma, S. Shah, and W. Y. Wang, “ConvFinQA: Exploring the chain of numerical reasoning in conversational
finance question answering,” in Proceedings of the 2022 Conference on
Empirical Methods in Natural Language Processing.
Association for
Computational Linguistics, 2022, pp. 6279–6292.
[10] Y. Zhao, Y. Li, C. Li, and R. Zhang, “MultiHiertt: Numerical reasoning
over multi hierarchical tabular and textual data,” in Proceedings of the
60th Annual Meeting of the Association for Computational Linguistics.
Association for Computational Linguistics, 2022, pp. 6588–6600.
[11] V. Reddy, R. Koncel-Kedziorski, V. D. Lai, M. Krumdick, C. Lovering,
and C. Tanner, “DocFinQA: A long-context financial reasoning dataset,”
in Proceedings of the 62nd Annual Meeting of the Association for
Computational Linguistics.
Association for Computational Linguistics,
2024, pp. 445–458.
[12] A. Asai, Z. Wu, Y. Wang, A. Sil, and H. Hajishirzi, “Self-RAG: Learning
to retrieve, generate, and critique through self-reflection,” in Proceedings
of the International Conference on Learning Representations, 2024.
[13] S. Es, J. James, L. Espinosa-Anke, and S. Schockaert, “RAGAS:
Automated evaluation of retrieval augmented generation,” arXiv preprint
arXiv:2309.15217, 2023.
