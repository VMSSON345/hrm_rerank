# BÁO CÁO TỔNG HỢP KIẾN TRÚC & TRIỂN KHAI CHI TIẾT HỆ THỐNG FINHRM

---

## 1. TỔNG QUAN HỆ THỐNG & NGUYÊN TẮC THIẾT KẾ

**FinHRM** là hệ thống hỏi đáp và phân tích tài chính tự động hóa dành riêng cho Báo cáo tài chính (BCTC) Việt Nam, ứng dụng phương pháp tiếp cận **Neuro-Symbolic kết hợp giữa Mạng nơ-ron hồi quy phân cấp (Hierarchical Reasoning Model - HRM)** và **Động cơ kiểm chứng hợp đồng dữ liệu & kế toán (Post-HRM Contract Engine)**.

### Mục tiêu cốt lõi:
1. **Loại bỏ hiện tượng ảo giác (Zero Hallucination):** Không dùng LLM sinh văn bản tự do để tính toán số liệu tài chính; thay vào đó, hệ thống sử dụng mạng nơ-ron để sinh **chương trình tính toán (Executable AST/DAG Program)** gồm `phép toán + con trỏ toán hạng (pointers)`.
2. **Tính toán chính xác tuyệt đối:** Mọi phép tính số học được thực thi bằng thư viện số học chính xác (`Decimal`, `Fraction`), không sử dụng số thực dấu phẩy động (float).
3. **Minh bạch dẫn chứng (Provenance & Citation):** Mọi số liệu đầu vào đều có con trỏ truy vết ngược về số trang, tọa độ bảng, dòng văn bản của file PDF gốc thông qua Markdown OCR.
4. **Cơ chế Fail-Closed (Từ chối trả lời an toàn):** Khi dữ liệu bị thiếu, mơ hồ, không cùng kỳ, không cùng phạm vi hợp nhất, hoặc phát hiện mâu thuẫn công bố trong tài liệu, hệ thống bắt buộc phát lệnh `ABSTAIN` (tạm dừng/từ chối) hoặc `REJECT` thay vì tự ý chọn một con số.

---

## 2. SƠ ĐỒ KIẾN TRÚC TỔNG THỂ (SYSTEM ARCHITECTURE)

```
[ PDF Báo cáo tài chính ]
         │
         ▼ (Bước 1: OCR & Trích xuất cấu trúc)
┌────────────────────────────────────────────────────────┐
│  MinerU OCR Service (127.0.0.1:8000 - .venv-mineru)     │
│  - Pipeline OCR & Table Reconstruction                 │
│  - Tạo Markdown có đánh dấu phân trang `## Page N`     │
│  - Kiểm tra & gán nhãn SHA-256 file gốc                │
└────────────────────────────────────────────────────────┘
         │
         ▼ (Bước 2: Ingestion & Lập chỉ mục)
┌────────────────────────────────────────────────────────┐
│  Document Ingestion Manager & SQLite Page Index        │
│  - Lưu cache tại cache/document_page_index.sqlite3     │
│  - Khôi phục ngữ cảnh cột, bảng, dòng, đơn vị          │
└────────────────────────────────────────────────────────┘
         │
         ▼ (Bước 3: Lập kế hoạch & Truy hồi bằng chứng)
┌────────────────────────────────────────────────────────┐
│  Question Planner & Evidence Extraction                │
│  - Phân tích câu hỏi: Chỉ tiêu, kỳ, vai trò, đơn vị    │
│  - BM25 Chunks + Mở rộng lân cận (Neighbor Chunks)     │
│  - Trích xuất ứng viên số (Numeric Candidates)         │
│  - Snapshot thực thể & dữ kiện nguồn (SourceFacts)     │
└────────────────────────────────────────────────────────┘
         │
         ▼ (Bước 4: Mạng nơ-ron phân cấp dự đoán chương trình)
┌────────────────────────────────────────────────────────┐
│  Document HRM Bundle (models/*.pt)                     │
│  - Semantic Encoder: BKAI Bi-Encoder / MiniLM-L12      │
│  - Recurrent Segments (High-level: H / Low-level: L)   │
│  - Operation Head: Chọn phép toán (SELECT, RATIO,...)  │
│  - Joint-pair Pointer Head: Chọn đồng thời các toán hạng│
└────────────────────────────────────────────────────────┘
         │
         ▼ (Bước 5: Kiểm định 4 tầng - Post-HRM Contract Runtime)
┌────────────────────────────────────────────────────────┐
│  Post-HRM Contract Engine (post_hrm_contract_v1)       │
│  ├── Tầng 1: Data Contract (Nguồn, đơn vị, format số)  │
│  ├── Tầng 2: Accounting Contract (Cân đối kế toán)     │
│  ├── Tầng 3: Computation Contract (Thực thi DAG)       │
│  └── Tầng 4: Cross-Statement Contract (Đối chiếu chéo) │
└────────────────────────────────────────────────────────┘
         │
         ▼ (Bước 6: Tổng hợp kết quả & Hiển thị)
┌────────────────────────────────────────────────────────┐
│  FinHRM Web Gateway & UI (FastAPI 8765 + React 19)     │
│  - Trả về Rendered Answer + Citation + Giấy chứng nhận │
│  - Quản lý hồ sơ doanh nghiệp, phê duyệt, stress test  │
└────────────────────────────────────────────────────────┘
```

---

## 3. CÁC THÀNH PHẦN CHI TIẾT CỦA KIẾN TRÚC

### 3.1. Tầng Tiền xử lý & OCR (Document Preparation)
* **Công cụ độc quyền:** **MinerU** (`mineru-api` / `mineru` CLI). Không sử dụng PyMuPDF hay trích xuất text thuần từ PDF để tránh sai lệch cấu trúc cột và bảng.
* **Môi trường:** Được cô lập hoàn toàn trong môi trường ảo `.venv-mineru` nhằm cách ly các thư viện nặng (Torch Vision, PDFium, v.v.).
* **Định dạng đầu ra:** Toàn bộ văn bản và bảng biểu được chuyển đổi thành Markdown chuẩn UTF-8, có định dạng header trang `## Page N`.
* **Cơ chế bộ nhớ đệm (Cache & Integrity Guard):**
  * Lưu trữ tại `cache/mineru/` và `.mineru/` cạnh file gốc.
  * Hệ thống luôn tính toán băm SHA-256 kép (`source_sha256` và `artifact_sha256`). Nếu PDF hoặc Markdown bị sửa đổi, hệ thống sẽ từ chối xử lý và yêu cầu chạy lại OCR.

### 3.2. Mô hình Nơ-ron Phân cấp (Document HRM Model)
Được định nghĩa tại `src/hrm_finance_qa/document_hrm.py`:
* **Kiến trúc H/L Recurrent:**
  * **High-level (H):** Quản lý trạng thái chiến lược cấp cao (phép toán số học, chu kỳ tư duy tổng thể).
  * **Low-level (L):** Tập trung truy vấn và tương tác lặp đi lặp lại với danh sách các ứng viên số (numeric candidates) để tinh chỉnh con trỏ toán hạng.
  * **Cơ chế Attention & Kích hoạt:** Sử dụng RoPE (Rotary Position Embedding), chuẩn hóa RMSNorm, hàm kích hoạt SwiGLU và tối đa 3 reasoning segments.
* **Tầng mã hóa ngữ nghĩa (Semantic Encoder):**
  * Checkpoint hiện hành (`models/document_hrm_V14_1_bkai_scratch_seed31_v1.pt`) sử dụng `bkai-foundation-models/vietnamese-bi-encoder` (768 chiều), kết hợp tiền xử lý tách từ bằng `pyvi`.
  * Có tầng chuyển đổi thích ứng ngữ nghĩa (Semantic Adapter): `layernorm_linear_l2_v1` để chiếu đặc trưng vào không gian ẩn của HRM.
* **Đầu ra của mô hình:**
  * `operation_logits`: Phân loại phép tính (`SELECT`, `ADD`, `SUBTRACT`, `DIVIDE`, `PERCENT_CHANGE`, `RATIO_PERCENT`, `SUM_N`).
  * `pair_logits`: Trực tiếp chọn cặp con trỏ đồng thời (ví dụ: tử số & mẫu số, năm nay & năm trước) nhằm bảo toàn mối quan hệ ngữ nghĩa chặt chẽ.

### 3.3. Động cơ Kiểm định Hợp đồng (Post-HRM Contract Engine v1)
Nằm tại `src/hrm_finance_qa/post_hrm_engine.py` và `src/hrm_finance_qa/post_hrm_contract.py`, đây là chốt chặn quyết định việc câu trả lời có được công bố hay không qua 4 tầng kiểm định:

1. **Tầng 1 - Hợp đồng Dữ liệu (`data_contract`):**
   * Đối chiếu chỉ tiêu câu hỏi với nhãn chỉ tiêu thực tế (`metric_semantics.py`).
   * Kiểm tra năm tài chính (`period`), phạm vi kế toán (`scope`: riêng lẻ hay hợp nhất), vai trò số liệu (`role`: thực hiện `actual` hay kế hoạch `plan`).
   * Kiểm tra định dạng số học (phân biệt dấu chấm hàng nghìn và dấu phẩy thập phân theo chuẩn Việt Nam).
   * Phát hiện xung đột công bố (`disclosure_conflict`): Nếu có nhiều số liệu khác nhau cùng một chỉ tiêu trên các trang khác nhau mà không có căn cứ ưu tiên, hệ thống dừng lại và liệt kê các mâu thuẫn để người dùng đối chiếu.
2. **Tầng 2 - Hợp đồng Kế toán (`accounting`):**
   * Kiểm tra tính toàn vẹn của báo cáo: Tài sản ngắn hạn + Tài sản dài hạn = Tổng tài sản; Nợ phải trả + Vốn chủ sở hữu = Tổng nguồn vốn.
3. **Tầng 3 - Hợp đồng Tính toán (`computation`):**
   * Biên dịch chương trình thành DAG và chạy qua `ExactDAGExecutor`.
   * Tính toán bằng `Decimal` chính xác, sinh ra chứng chỉ thực thi (`execution_certificate`).
4. **Tầng 4 - Đối chiếu Chéo Báo cáo (`cross_statement`):**
   * Đối chiếu tính nhất quán giữa Bảng Cân đối kế toán, Báo cáo Kết quả kinh doanh và Báo cáo Lưu chuyển tiền tệ.

### 3.4. Tầng Ứng dụng & Dịch vụ Web (FinHRM Web Workspace)
Nằm tại `src/finhrm_web/`:
* **API Gateway & Business Backend:** Xây dựng bằng **FastAPI** (`app.py`), phục vụ các nghiệp vụ ngân hàng/doanh nghiệp:
  * Quản lý người dùng, phân quyền RBAC (`can_bo`, `kiem_soat`, `truong_phong`, `quan_tri`), mã hóa mật khẩu bằng `bcrypt`, xác thực qua `JWT`.
  * Quản lý vòng đời hồ sơ thẩm định (`ho_so`) và lịch sử phê duyệt (`approvals`).
  * Quản lý hội thoại hỏi đáp theo từng hồ sơ/tài liệu (`conversations`, `messages`).
  * Tính toán chỉ số tài chính (Current Ratio, Quick Ratio, Debt/Equity, ROE, ROA...) và chạy mô phỏng kịch bản căng thẳng (**Stress Test**: sốc lãi suất, tỷ giá, sụt giảm dòng tiền, chu kỳ thu nợ) tại `src/finhrm_web/calculations.py`.
* **Giao diện người dùng (Frontend SPA):**
  * Xây dựng bằng **React 19 + Vite 6**, thư viện icon `lucide-react`.
  * Bản build sẵn nằm tại `web_demo/dist/` và được FastAPI phục vụ trực tiếp dưới dạng static files, không phụ thuộc vào Node.js khi chạy production.

---

## 4. MÔ HÌNH TRIỂN KHAI & TIẾN TRÌNH HỆ THỐNG (DEPLOYMENT TOPOLOGY)

Hệ thống được thiết kế theo kiến trúc **Microservices nội bộ đa tiến trình (Multi-Process Local Microservice)**, bảo đảm an toàn bằng cách chỉ bind vào loopback `127.0.0.1`:

```
                 [ Trình duyệt Web Người dùng ]
                              │
                    HTTP Port: 8765 (PORT)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Tiến trình 1: FinHRM Web Gateway & Web Static (FastAPI)      │
│  - File thực thi: src/finhrm_web/app.py                     │
│  - Database: tmp/finhrm_web/workspace.sqlite3               │
│  - Static Mount: web_demo/dist/                             │
└─────────────────────────────────────────────────────────────┘
          │ (Forward HTTP nội bộ)
          ▼
┌─────────────────────────────────────────────────────────────┐
│ Tiến trình 2: Persistent Document HRM Service (Port: 8766)   │
│  - File thực thi: src/hrm_finance_qa/document_service.py    │
│  - Model PyTorch nạp cố định vào RAM: document_hrm_V14...pt │
│  - Quản lý hàng đợi Ingestion & Cache SQLite Page Index     │
│  - Đơn luồng suy luận (Serialized Inference Lock)           │
└─────────────────────────────────────────────────────────────┘
          │ (Gọi HTTP Ingest PDF khi tải tệp)
          ▼
┌─────────────────────────────────────────────────────────────┐
│ Tiến trình 3: Background MinerU OCR Engine (Port: 8000)      │
│  - File thực thi: .venv-mineru/bin/mineru-api               │
│  - Môi trường riêng: .venv-mineru                           │
│  - Giới hạn: Window=4 trang, Max Concurrency=1              │
└─────────────────────────────────────────────────────────────┘
```

### 4.1. Cơ chế Quản lý Tài nguyên & Bộ nhớ
1. **Cô lập môi trường Python:**
   * Môi trường chính (`.venv-demo` hoặc `venv`): Chạy Web Gateway và Document HRM (PyTorch CPU, Transformers, FastAPI).
   * Môi trường OCR (`.venv-mineru`): Chạy MinerU API riêng biệt để tránh xung đột phiên bản CUDA/PyTorch và các thư viện xử lý hình ảnh nặng.
2. **Kiểm soát CPU & Thread:**
   * Hệ thống xuất rõ các biến môi trường `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `torch.set_num_threads(4)` nhằm tránh hiện tượng chiếm dụng 100% CPU của máy chủ khi chạy các mô hình embedding và tensor.
3. **Phòng chống OOM (Out Of Memory):**
   * MinerU API được cấu hình mặc định `MINERU_PROCESSING_WINDOW_SIZE=4` và `MINERU_API_MAX_CONCURRENT_REQUESTS=1`. Điều này giúp xử lý các file scan tài chính dày hàng trăm trang mà không làm cạn kiệt RAM/VRAM.
   * Tiến trình HRM nạp mô hình một lần duy nhất vào bộ nhớ (`PersistentDocumentHRMService`), tránh tải lại trọng số cho mỗi truy vấn.
   * `Inference Lock`: Các tác vụ chẩn đoán (`/health`, `/v1/jobs/...`) chạy trên luồng riêng phản hồi ngay lập tức, nhưng tác vụ suy luận mô hình (`/v1/answer`) được xếp hàng tuần tự để bảo vệ an toàn luồng của Torch.

---

## 5. QUẢN LÝ DỮ LIỆU & LƯU TRỮ (PERSISTENCE LAYER)

Hệ thống sử dụng hệ sinh thái lưu trữ tệp tin và cơ sở dữ liệu SQLite cục bộ:

| Thành phần | Đường dẫn | Mục đích sử dụng |
|---|---|---|
| **Cơ sở dữ liệu nghiệp vụ** | `tmp/finhrm_web/workspace.sqlite3` | Lưu trữ tài khoản người dùng, hồ sơ doanh nghiệp (`ho_so`), văn bản tải lên, luồng phê duyệt, lịch sử đoạn chat và các chỉ số nhập thủ công. |
| **Chỉ mục trang tài liệu** | `cache/document_page_index.sqlite3` | Lưu trữ kết quả phân đoạn trang và chỉ mục tra cứu văn bản của các tài liệu đã được OCR. |
| **Kho lưu Markdown OCR** | `cache/mineru/` hoặc `<pdf_dir>/.mineru/` | Lưu trữ các tệp Markdown kết quả sau khi MinerU quét PDF, giữ nguyên cấu trúc bảng và trang. |
| **Kho tải lên tạm thời** | `tmp/demo_uploads/` | Nơi tiếp nhận các file PDF tải lên từ Web trước khi đưa vào hàng đợi Ingestion. |
| **Trọng số & Embedding Cache**| `models/` & `models/embedding_cache/` | Lưu trữ các checkpoint nơ-ron (`.pt`) và bộ đệm nhúng vector của các câu hỏi/ứng viên. |

---

## 6. QUY TRÌNH VẬN HÀNH & KỊCH BẢN KHỞI CHẠY (OPERATIONAL RUNBOOKS)

### 6.1. Khởi động hệ thống đầy đủ (Khuyến nghị)
* **Khởi chạy trên Windows:** Nhấp đúp hoặc chạy từ PowerShell:
  ```powershell
  .\start_demo.cmd
  ```
  *(Tự động kích hoạt script `scripts/start_demo.ps1`, điều hướng vào môi trường WSL tương ứng và thực thi `scripts/start_demo.sh`)*.
* **Khởi chạy trực tiếp trong Linux/WSL:**
  ```bash
  bash scripts/start_demo.sh
  ```
  *(Launcher `src/finhrm_web/launcher.py` sẽ tự động kiểm tra xem MinerU API tại port 8000 đã chạy chưa, nếu chưa sẽ tự fork một tiến trình ngầm, sau đó nạp Model HRM lên port 8766 và mở Web Gateway tại port 8765)*.

### 6.2. Tài khoản mặc định hệ thống
Khi khởi tạo cơ sở dữ liệu lần đầu, hệ thống tạo sẵn 4 tài khoản theo vai trò nghiệp vụ với mật khẩu mặc định `FinHRM@123`:
* `admin`: Quản trị hệ thống.
* `nvpt01`: Cán bộ phân tích.
* `ksv01`: Kiểm soát viên.
* `tp01`: Trưởng phòng phê duyệt.

---

## 7. ĐÁNH GIÁ TÌNH TRẠNG & CÁC GIỚI HẠN KỸ THUẬT HIỆN HÀNH

Theo các biên bản kiểm nghiệm kỹ thuật gần nhất (`finhrm_B_remaining_limitations_v1.md`):

* **Ưu điểm đã đạt được:**
  * Toàn bộ 583 bài kiểm thử đơn vị & kiểm thử tích hợp của nhánh `finhrm_B` đều đạt (PASS).
  * Khả năng đối chiếu nguồn gốc, bảo vệ đơn vị mặc định (`đồng`) và liên kết thực thể sở hữu hoạt động tin cậy.
  * Khi gặp các câu hỏi không đủ dữ kiện hoặc mâu thuẫn giữa các bảng, hệ thống đã dừng đúng quy chuẩn `ABSTAIN/REJECT` thay vì sinh dữ liệu giả.
* **Các giới hạn còn tồn tại cần lưu ý:**
  1. **Các chỉ tiêu phụ/dòng "khác":** Các hàng như *"Tài sản ngắn hạn khác"* đôi khi bị thất thoát trong khâu trích lọc ứng viên hàng đầu do đặc trưng ngữ nghĩa bảng kéo dài qua nhiều trang.
  2. **Bố cục bảng phức tạp:** Một số báo cáo quét scan có độ nghiêng hoặc mất đường viền bảng khiến bảng bị nối dòng lệch, dẫn đến việc thiếu ngữ cảnh cột năm.
  3. **Độ phân giải số học:** Cần tiếp tục hoàn thiện cơ chế nhận diện tự động giữa dấu chấm và dấu phẩy ở các báo cáo không ghi rõ quy chuẩn kế toán số học.
