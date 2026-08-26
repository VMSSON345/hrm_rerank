# Neuro-Symbolic HRM cho bao cao tai chinh Viet Nam

Du an tra loi cau hoi so hoc tu PDF/TXT bao cao tai chinh. He thong truy xuat
bang chung, dung **Hierarchical Reasoning Model (HRM)** de sinh chuong trinh
`operation + operand pointers`, sau do thuc thi bang `Decimal` va tra citation.

## Kien truc hien tai

```text
Cau hoi + PDF/TXT
  -> persistent page index (native + MinerU artifact cache)
  -> MinerU mot lan cho page range native bi loi/scan
  -> BM25 page + page lan can
  -> chunk BM25 + chunk lan can
  -> numeric candidates
  -> semantic H/L HRM: chon operation
  -> lexical H/L HRM: de xuat operand pointers
  -> Top-K executable program search
  -> symbolic verifier: operation/metric/period/unit/entity/operand role
  -> symbolic Decimal executor
  -> calibration / abstention
  -> answer + citation
```

Checkpoint neural mặc định là `models/document_hrm_V8_candidate.pt`; runtime
nghiên cứu hiện tại là V8 kết hợp V12 typed program synthesis:

- Kiến trúc `source-grounded-semantic-HRM-v10-warmup-end-to-end-joint-pair`, có `843,097` tham số
  trainable.
- Encoder semantic frozen
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 chiều).
- H/L recurrent reasoning dùng RoPE attention, RMSNorm, SwiGLU và tối đa ba
  reasoning segments.
- Operation head chọn phép toán; joint-pair head chọn đồng thời hai operand để
  giữ quan hệ tử số/mẫu số hoặc hai kỳ so sánh.
- Grounding runtime giới hạn candidate theo period, metric và operand role khi
  câu hỏi cung cấp đủ bằng chứng. `SUBTRACT` được canonicalize thành độ chênh
  lệch không âm, nhất quán với dữ liệu train và cách diễn đạt “tăng/giảm X”.
- Decoder không còn phụ thuộc duy nhất vào pointer `argmax`: nó tìm trong nhiều
  chương trình ứng viên, loại chương trình vi phạm ràng buộc rồi mới chọn
  chương trình hợp lệ có grounding tốt nhất.
- Executor dùng `Decimal`; calibration có thể trả `ABSTAIN` khi evidence/pointer
  chưa đủ chắc chắn.

## Cài đặt

Môi trường chạy pipeline chính trong WSL là `venv`; MinerU dùng môi trường riêng
`.venv-mineru` vì dependency nặng và có thể xung đột:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

## Danh gia

Chạy unit test và validation V12 mà không đọc blind test:

```bash
bash scripts/evaluate_document_hrm_v12_synthesis.sh
```

Blind V8 đã được sử dụng và đóng băng. Không chạy lại hoặc dùng nó để chọn rule,
threshold hay model cho V12.

Chay holdout that duoc luu trong checkpoint:

```bash
python main.py evaluate-document-hrm
```

In bao cao day du dang JSON (gia tri metric nam trong khoang 0-1):

```bash
python main.py evaluate-document-hrm --json
```

Luu bao cao JSON de dung cho bang/plot trong nghien cuu:

```bash
python main.py evaluate-document-hrm \
  --output-report reports/document_hrm_evaluation.json
```

Bao cao JSON gom metadata model, so tham so, split, metric tong, pointer tung
slot, calibration va Precision/Recall/F1/support cua tung operation.

Kết quả checkpoint mặc định trên fixed real holdout 46 mẫu, tách theo
company/report:

| Metric | Score |
|---|---:|
| Operation accuracy | 95.65% |
| Operation macro F1 | 70.81% |
| Program exact accuracy | 45.65% |
| Execution accuracy | 47.83% |
| Pointer pair accuracy | 45.65% |
| Accepted coverage | 10.87% |
| Accepted precision | 100.00% |

Quality gate so với v7 đã PASS. Bộ MinerU regression ABI/ABR/ABS đạt operation,
execution và page provenance `12/12`; accepted coverage `5/12` với precision
`5/5`. Đây vẫn là checkpoint nghiên cứu: execution tổng quát chưa đủ cao để tự
động hóa quyết định tài chính, vì vậy phải giữ abstention và citation review.

Chan doan bottleneck:

```bash
python main.py diagnose-document-hrm
```

Xem cau hinh, metadata va so tham so:

```bash
python main.py document-hrm-info
```

## Tra loi PDF

`answer-document` mac dinh dung SQLite page index tai
`cache/document_page_index.sqlite3` va chay native-first. He thong trich native
text, rank chunk, tao numeric candidate va thu reasoning truoc. MinerU chi duoc
bat khi native retrieval khong match, khong co numeric candidate, thieu ky duoc
hoi hoac program khong thuc thi duoc. Confidence thap don thuan khong kich hoat
OCR. PDF scan toan phan van phai duoc MinerU doc mot lan de co searchable text;
cac cau hoi sau chi doc SQLite/artifact cache. Ket qua native/MinerU duoc luu
theo fingerprint path/size/mtime.

He thong tu dong chon cach doc theo tung trang:

- text layer sach va du y nghia: dung truc tiep, khong render va khong OCR;
- khong co text, text qua ngan, loi ma `(cid:...)`, ky tu thay the hoac text bi vo:
  dung MinerU structured OCR; runtime tu sinh artifact neu chua co;
- PDF hon hop co the dung native text o trang nay va OCR o trang khac.
- neu native evidence da du tra loi, cac trang scan con lai khong bi OCR trong
  request do.

MinerU duoc cai rieng khoi project venv:

```bash
bash scripts/setup_mineru_venv.sh
```

PDF co text layer:

```bash
python main.py answer-document \
  --document "pdf/HPG_2025.pdf" \
  "Tinh ty le loi nhuan sau thue tren von chu so huu nam 2025"
```

PDF scan tu dong fallback sang OCR, khong can them `--ocr`:

```bash
python main.py answer-document \
  --document "pdf/HPG_2025_scan.pdf" \
  "Tong tai san cua HPG cuoi nam 2025 la bao nhieu?"
```

`--ocr` van duoc giu de tuong thich voi lenh cu nhung automatic fallback da la
mac dinh. Dung `--no-ocr` de chi cho phep native text (ke ca khi index da co OCR);
dung `--force-ocr` de bo qua text layer va OCR moi page. Dung
`--rebuild-page-index` de rebuild thu cong, hoac `--no-page-index` de quay ve
duong runtime cu. Dung `--mineru-output-dir` neu muon doc artifact da tao truoc,
hoac `--no-mineru-auto` de cam runtime tu khoi chay MinerU. Output bao cao tong
page, native page, MinerU page,
cache hit, page da rank va page thuc su dua vao HRM. Citation co nhan `[MinerU OCR]`;
can doi chieu lai chu so voi anh scan.

Smoke test OCR dong goi:

```bash
python main.py answer-document \
  --document examples/sample_ratio_report_scan.pdf \
  "Tinh ty le tai san co dinh huu hinh tren tong tai san dai han vao cuoi nam 2024."
```

Benchmark page index cold/warm trong cache tam rieng (khong xoa cache runtime):

```bash
python scripts/benchmark_document_page_index.py \
  examples/sample_ratio_report_scan.pdf \
  "Tinh ty le tai san co dinh huu hinh tren tong tai san dai han vao cuoi nam 2024." \
  --mineru-api-url http://127.0.0.1:8000 \
  --output reports/page_index_benchmark.json
```

Bao cao gom wall time, so page MinerU, cache hit, page dua vao HRM va peak Python
memory theo `tracemalloc`. Memory cua MinerU subprocess khong nam trong so do
`tracemalloc` cua parent process. Bo `--mineru-api-url` neu can do ca model
startup subprocess; giu flag nay de do production cold-index voi MinerU model da warm.

## Persistent HRM service va background MinerU

CLI load Torch/checkpoint lai cho moi process. Production dung hai process song
lau dai: `mineru-api` giu OCR model trong memory, con Document HRM service load
checkpoint dung mot lan. Mo terminal thu nhat:

```bash
bash scripts/run_mineru_api.sh
```

Service mac dinh gioi han `MINERU_PROCESSING_WINDOW_SIZE=4` va
`MINERU_API_MAX_CONCURRENT_REQUESTS=1` de tranh giu toan bo PDF scan dai trong
RAM. Co the override tren server nhieu RAM. Vi du RTX 4060 8 GB:

```bash
MINERU_DEVICE_MODE=cuda \
MINERU_HYBRID_BATCH_RATIO=1 \
MINERU_PROCESSING_WINDOW_SIZE=4 \
MINERU_API_MAX_CONCURRENT_REQUESTS=1 \
bash scripts/run_mineru_api.sh
```

Mo terminal thu hai trong project `venv`:

```bash
python scripts/serve_document_hrm.py
```

Mac dinh ca hai chi bind loopback: MinerU tai `127.0.0.1:8000`, HRM tai
`127.0.0.1:8765`. Ingest PDF scan chay background va request tra ve ngay:

```bash
curl -sS -X POST http://127.0.0.1:8765/v1/ingest \
  -H 'Content-Type: application/json' \
  -d '{"document":"examples/sample_ratio_report_scan.pdf","rebuild":true}'
```

Lay `job_id` trong response, sau do xem trang thai:

```bash
curl -sS http://127.0.0.1:8765/v1/jobs/JOB_ID
```

Khi job la `ready`, hoi nhieu lan ma khong load lai checkpoint hay khoi dong
MinerU trong request:

```bash
curl -sS -X POST http://127.0.0.1:8765/v1/answer \
  -H 'Content-Type: application/json' \
  -d '{"document":"examples/sample_ratio_report_scan.pdf","question":"Tính tỷ lệ tài sản cố định hữu hình trên tổng tài sản dài hạn vào cuối năm 2024."}'
```

PDF native co the duoc tra loi ngay. PDF scan chua ready tra HTTP `409`; client
phai submit `/v1/ingest` va poll job, khong giu request hoi dap trong nhieu phut.
Chi cac path nam trong `pdf/`, `examples/` hoac `--document-root` duoc phep.
Service chua co authentication, vi vay khong bind public interface.

HTTP service xu ly diagnostics tren thread rieng, nhung model inference van duoc
serialize de bao ve checkpoint va semantic encoder. Vi vay `/health` va
`/v1/jobs/JOB_ID` van phan hoi trong luc mot cau hoi lon dang chay;
`inference_busy=true` cho biet model dang ban. Response `/v1/answer` co
`stage_timings`, va terminal service log tung stage de tach retrieval,
candidate extraction, semantic prediction va execution latency.

Do latency khi checkpoint da nam san trong memory:

```bash
python scripts/benchmark_persistent_service.py \
  examples/sample_ratio_report_scan.pdf \
  "Tính tỷ lệ tài sản cố định hữu hình trên tổng tài sản dài hạn vào cuối năm 2024." \
  --repeats 3
```

Chay bo ABI/ABR/ABS regression qua checkpoint da load san (cac document scan
phai ingest thanh `ready` truoc):

```bash
python scripts/ingest_documents.py \
  pdf/ABI_25CN_BCTN.pdf \
  pdf/ABR_25CN_BCTN.pdf \
  pdf/ABS_25CN_BCTN.pdf

DOCUMENT_SERVICE_URL=http://127.0.0.1:8765 \
REPORT_PATH=reports/document_hrm_external_pdf_persistent.txt \
bash scripts/run_external_pdf_smoke_v1.sh
```

Neu may khong du RAM de giu MinerU va HRM/Torch cung luc, dung workflow tach
process: tat HRM, chi bat MinerU, build runtime index bang script nhe, sau do tat
MinerU va bat lai HRM. Script nay khong import Torch:

```bash
python scripts/ingest_page_index.py \
  pdf/ABI_25CN_BCTN.pdf \
  pdf/ABR_25CN_BCTN.pdf \
  pdf/ABS_25CN_BCTN.pdf
```

Client page-index cung chia PDF scan thanh API request 4 trang
(`--mineru-api-page-batch-size 4`). Day la gioi han khac voi processing window
noi bo cua MinerU: no ngan cac stage nhu OCR-det nhin thay toan bo PDF dai trong
mot request, va moi range hoan tat duoc cache rieng de retry tiep tuc thay vi
lam lai tu dau.

Checklist server day du, gom RSS/cgroup/OOM evidence, freeze gate va nhanh xu ly
`Terminated`, nam tai muc `Checklist xac minh tren server sau khi push` trong
`PROJECT_CONTEXT.md`.

## Train semantic HRM

Tao program dataset:

```bash
python main.py prepare-document-hrm-data \
  --source data/hrm_instruction_1k.jsonl \
  --output data/hrm_programs_clean_v5.jsonl \
  --audit-report reports/hrm_programs_clean_v5_audit.json \
  --synthetic-per-operation 60
```

Mặc định, bước chuẩn bị chỉ nhận task `QA`, loại câu không thể trả lời/câu định
tính, loại `SELECT` mơ hồ có nhiều giá trị và yêu cầu mọi operand phải được dẫn
chứng trong answer. File `*_real.jsonl` chỉ chứa mẫu real đã qua lọc; audit JSON
ghi số lượng và example ID theo từng lý do loại. Chỉ dùng `--allowed-task` khi
chủ động xây một output schema khác; không thêm `SA`/`NER` vào program dataset.

Train semantic H/L HRM tren CPU:

```bash
python main.py \
  --document-model-path models/document_hrm_semantic_new.pt \
  train-document-hrm \
  --dataset data/hrm_programs_clean_v5.jsonl \
  --epochs 6 --batch-size 64 \
  --embedding-dim 64 --hidden-size 96 \
  --high-cycles 1 --low-steps 2 --max-segments 3 \
  --attention-heads 4 --ff-multiplier 3 \
  --input-encoder semantic \
  --semantic-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --semantic-dim 384 --device cpu
```

Embedding duoc cache trong `models/embedding_cache`; lan train sau khong can
encode lai toan bo data. Synthetic examples chi nam trong train, validation chi
gom real examples.

## V11 safe-residual frozen-HRM reranker — ablation lịch sử

V11 giu nguyen checkpoint V8 va train mot sidecar residual nho. Legacy symbolic
winner la fallback bat buoc; sidecar chi duoc override khi correction vuot margin
da calibration va khong lam giam program exact/execution tren calibration.
Pipeline tu choi `test.jsonl`/blind input, tach calibration theo company/report
group va xac minh SHA-256 cua V8 truoc/sau.

Sidecar và báo cáo V11 chỉ được giữ để đối chiếu. V12-A không gắn sidecar này
vào runtime. Nếu cần tái lập đầy đủ, script lịch sử nằm trong
`archive/pre_v12_cleanup_20260802/scripts/`.

Đọc sidecar ablation thủ công:

```bash
venv/bin/python main.py \
  --document-model-path models/document_hrm_V8_candidate.pt \
  --program-reranker-path models/document_hrm_V11_safe_residual.pt \
  evaluate-document-hrm \
  --dataset data/validation.jsonl --all-examples --device cpu
```

Chi tiet kien truc, data split va acceptance gate nam trong `strategy.md`.

## V12 constraint-guided program synthesis (thử nghiệm)

V12 không chỉ lọc chương trình sau HRM. Nó biên dịch câu hỏi thành Typed Query
IR, tạo operand slots từ Typed Evidence Graph, sau đó dùng operation/pointer
score của V8 để chấm các assignment trước khi verifier chọn chương trình. V8
vẫn frozen và tính năng mặc định tắt cho demo đến khi đạt validation gate.

Chạy unit test và ablation chỉ trên validation:

```bash
bash scripts/evaluate_document_hrm_v12_synthesis.sh
```

Thử trên web local sau khi validation được chấp nhận:

```bash
TYPED_PROGRAM_SYNTHESIS=1 bash scripts/start_demo.sh
```

Catalog công thức tài chính YAML V12.2 là một ablation riêng và mặc định tắt.
Chỉ bật để thử nghiệm cùng typed synthesis:

```bash
TYPED_PROGRAM_SYNTHESIS=1 FINANCIAL_IDENTITY_CATALOG=1 \
  bash scripts/start_demo.sh
```

Catalog nằm tại `config/financial_identities.yaml`. Safe overlay V12.3 đã loại
bỏ regression của V12.2 và trở lại đúng các chỉ số V12.1, nhưng chưa vượt mốc
đó và accepted precision vẫn dưới 90%; vì vậy chưa bật cờ này cho demo chính.

Báo cáo được lưu tại
`reports/document_hrm_V12_typed_synthesis_validation.json`. Script từ chối
`RUN_BLIND=1` và các đường dẫn có tên test/blind. Kiến trúc, positive taxonomy,
gate tổng quát hóa và phạm vi CAS/SMT được mô tả trong `strategy.md`.
