# FinHRM — snapshot demo V14.1

Nhánh này lưu trạng thái demo ngày 27/09/2026, trước khi thử hướng HRM tự chọn chương trình. Checkpoint đang dùng là V14.1 scratch seed 31; backend trả lời là post_hrm_contract_v1. Đây là bản nghiên cứu chạy cục bộ, chưa phải hệ thống trả lời đúng mọi báo cáo.

## Luồng đang chạy

1. PDF được MinerU chuyển thành Markdown có phân trang. Khi hỏi, hệ thống đọc Markdown đã chuẩn bị và giữ hash, dòng, span để đối chiếu citation.
2. Parser tạo catalog ứng viên và SourceFact, gắn chỉ tiêu, kỳ, phạm vi, đơn vị và bằng chứng từ bảng hoặc văn bản. Không suy ra đơn vị/kỳ khi nguồn không chứng minh được.
3. Context resolver và QuestionPlan hiểu yêu cầu, xác định phép toán cùng các slot toán hạng. Truy hồi dùng BM25; khi thiếu slot, có thể mở rộng bằng encoder BKAI và EvidenceGraph.
4. Checkpoint HRM V14.1 xếp hạng các ứng viên số. Ràng buộc nguồn quyết định fact nào được phép dùng. Chương trình trong luồng mặc định lấy từ QuestionPlan; Decimal executor tính kết quả.
5. Post-HRM contract kiểm lại chỉ tiêu, kỳ, phạm vi, đơn vị, mâu thuẫn nguồn và citation. Thiếu bằng chứng thì hỏi làm rõ, công bố số gốc có giới hạn hoặc dừng có giải thích.

Một số tra cứu văn bản, tỷ lệ in sẵn và thuyết minh đi qua nhánh nguồn riêng. Tra cứu span văn bản hiện dựa vào nhãn và chứng cứ nguồn, chưa có bộ chọn span được huấn luyện riêng.

**Ranh giới quan trọng:** chế độ thử HRM đề xuất phép toán/con trỏ có trong mã (numeric_program_mode=hrm), nhưng demo khởi tạo với numeric_program_mode=off. Vì thế kết quả demo này không chứng minh HRM đã chọn chương trình tốt hơn baseline. EvidenceGraph hỗ trợ truy hồi; nó chưa là đầu vào đồ thị trực tiếp của checkpoint.

Mã chính: [launcher](src/finhrm_web/launcher.py), [dịch vụ](src/hrm_finance_qa/document_service.py), [engine hiện hành](src/hrm_finance_qa/post_hrm_engine.py), [parser nguồn](src/hrm_finance_qa/post_hrm_source.py), [truy hồi](src/hrm_finance_qa/hybrid_retrieval.py). Xem [tổng quan](tongquan.md) để biết lịch sử và giới hạn đánh giá.

## Model và dữ liệu được lưu

- Checkpoint demo: [models/document_hrm_V14_1_bkai_scratch_seed31_v1.pt](models/document_hrm_V14_1_bkai_scratch_seed31_v1.pt), SHA-256: E6641D13F169DC674728C32AAFB508946495027985CE1FDE6E2B1095A8D55413.
- Encoder ngoài: bkai-foundation-models/vietnamese-bi-encoder tại revision 84f9d9ada0d1a3c37557398b9ae9fcedcdf40be0. Runtime yêu cầu snapshot này có sẵn trong models/embedding_cache và chỉ đọc local.
- Mã nguồn, cấu hình, giao diện React và unit test trong data/test/unit được lưu ở nhánh này.
- PDF/Markdown báo cáo, cache encoder/vector, database hội thoại, output chạy thử, gold và kết quả holdout cục bộ **không** được đưa lên GitHub. Không dùng bản sao GitHub này làm bằng chứng để chấm lại holdout đã khóa.

## Chạy demo cục bộ

Môi trường đã kiểm tra: Windows + WSL, Python 3.12. Cần Node.js để build giao diện. Các thư mục venv và cache không nằm trong Git.

~~~bash
python3.12 -m venv venv
venv/bin/python -m pip install -r requirements.txt
cd web_demo
npm ci
npm run build
cd ..
~~~

Tải đúng revision BKAI vào cache trước khi chạy, vì checkpoint đặt local_files_only=true:

~~~bash
venv/bin/python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="bkai-foundation-models/vietnamese-bi-encoder", revision="84f9d9ada0d1a3c37557398b9ae9fcedcdf40be0", cache_dir="models/embedding_cache")'
~~~

Nếu chỉ hỏi trên Markdown đã chuẩn bị, có thể chạy không khởi động MinerU:

~~~bash
DEMO_PYTHON="$PWD/venv/bin/python" bash scripts/start_demo.sh --no-mineru-start
~~~

Mở http://127.0.0.1:8765; dịch vụ HRM nội bộ dùng cổng 8766. Với PDF upload, cần cài MinerU riêng để có .venv-mineru/bin/mineru và .venv-mineru/bin/mineru-api, rồi chạy lệnh trên không có --no-mineru-start. MinerU và model OCR không được đóng gói trong repo. Trên máy đã có các venv cục bộ, start_demo.cmd gọi cùng launcher qua WSL.

Chạy unit test:

~~~bash
venv/bin/python -m pytest data/test/unit -q
~~~

Một số test và phép đánh giá toàn luồng cần fixture hoặc báo cáo nằm trong data/ cục bộ; các nguồn đó không được xuất bản cùng snapshot.

## Giới hạn đánh giá

Bản này đã cải thiện xử lý nguồn và khả năng dừng khi thiếu chứng cứ, nhưng OCR Markdown vẫn có thể mất nhãn, đơn vị hoặc quan hệ cột/kỳ. Lượt holdout 40 câu đầu chủ yếu chẩn đoán lỗi trước fact pool; không cho phép kết luận V14.1 hơn baseline. Muốn kết luận hiệu quả HRM chọn chương trình cần pool fact thật có ít nhất hai chương trình cạnh tranh cùng qua verifier, nguồn mới và gold được duyệt độc lập.
