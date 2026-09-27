> **Bản hiện hành đã thêm tra cứu thông tin doanh nghiệp/nhân sự và giải thích công bố mâu thuẫn.** [Câu thử và kết quả](data/test/suites/report_information_v1/delivery.md). 27/27 lượt backend đã kiểm tra; kiểm tra lại gateway có đăng nhập còn chờ quyền dùng credential. Ctrl+F5 để nhận giao diện mới. Các ghi chú dưới là lịch sử.

> **Bản demo hiện hành đã phục hồi ABW 2024:** làm rõ loại lợi nhuận, tính LNST và so sánh doanh thu thuần; đã kiểm tra 15/15 lượt demo và hồi quy nhiều báo cáo. [Bàn giao/giới hạn](data/test/suites/semantic_transfer_v1/delivery.md). Ctrl+F5 để nhận giao diện mới. Các ghi chú dưới về lỗi ABW là lịch sử trước sửa.

> **Giới hạn đã tái hiện trên demo:** ABW 2024 còn tự chọn trước thuế khi câu mơ hồ và không trả được LNST/doanh thu thuần dù Markdown có số. Bốn tầng mới đang chạy nhưng chưa đạt yêu cầu đa báo cáo. [Kiểm toán và hướng sửa](data/test/suites/cross_report_audit_v1/analysis.md).

> Runtime hiện hành: `post_hrm_contract_v1`, bốn tầng kiểm tra nguồn/kế toán/tính toán/đối chiếu; symbolic cũ không còn nằm trong đường trả lời. [Bàn giao và kiểm tra demo](data/test/suites/post_hrm_contract_v1/delivery.md).

> Yêu cầu hiện hành: giữ nền trắng, chữ đen/xám và font system UI của bản cũ; chỉ lấy bố cục, màn hình và luồng thao tác từ mẫu. Màu đỏ/vàng trong mô tả trước đã được loại bỏ.

# Giao diện FinHRM theo mẫu tham khảo

Giao diện sử dụng các màn hình và bố cục từ `tham_khao/thamkhao_int/frontend/src`: đăng nhập, tổng quan, hồ sơ, tài liệu/OCR, phân tích, thống kê/báo cáo, trợ lý, phê duyệt, quản lý. Giữ font system UI cũ; dùng màu đỏ rượu, nền hồng nhạt và điểm nhấn vàng của mẫu. Tên sản phẩm là FinHRM.

## Chạy trong WSL đang dùng venv hiện có

```bash
cd /mnt/d/reasoning_model_V4/reasoning_model
venv/bin/python -m pip install -r requirements-web.txt
bash scripts/start_demo.sh
```

Mở http://127.0.0.1:8765. Bản giao diện đã build trong `web_demo/dist`; không cần Node để chạy. HRM dùng cổng nội bộ 8766. Ctrl+C dừng cả hai tiến trình. Nếu đổi `PORT`, HRM dùng `PORT + 1`.

Tài khoản **demo local** được tạo khi cơ sở dữ liệu còn trống:

| Tài khoản | Vai trò |
|---|---|
| admin | Quản trị |
| nvpt01 | Cán bộ phân tích |
| ksv01 | Kiểm soát viên |
| tp01 | Trưởng phòng |

Mật khẩu demo mặc định: `FinHRM@123`. Có thể đặt `FINHRM_WEB_SEED_PASSWORD` trước lần khởi động đầu. JWT dùng khóa ngẫu nhiên khi khởi động, nên phải đăng nhập lại sau khi restart. Đây là workspace local, chỉ bind `127.0.0.1`.

Trên Windows: chạy `start_demo.cmd` trong workspace đã có môi trường WSL hiện tại. Không chạy `.cmd` như lệnh Bash trong WSL.

## Luồng thao tác

1. Đăng nhập → Hồ sơ phân tích → Tạo hồ sơ.
2. Tài liệu & OCR → chọn hồ sơ, kỳ báo cáo → tải PDF. Gateway chuyển upload và ingest sang dịch vụ HRM hiện có; PDF chỉ qua MinerU. Xem nội dung từ cache MinerU, không dùng parser PDF của mẫu.
3. Trợ lý phân tích → chọn tài liệu đã xử lý → hỏi. Phản hồi là trường `rendered` của HRM, giữ nguyên quyết định công bố/từ chối. Hội thoại lưu SQLite theo người dùng. Khi chọn hai tài liệu, trả kết quả riêng từng tài liệu và ghi rõ chưa tính toán chéo nguồn.
4. Phân tích → chọn hồ sơ/tài liệu → nhập số đã đối chiếu, đơn vị triệu đồng → lưu. Các tỷ số và kịch bản tính từ dữ liệu nhập, có nhãn rõ ràng. Bảng này **chưa tự điền số liệu HRM**; không giả vờ đã trích xuất hoặc đã thẩm định tín dụng. Thiếu đầu vào thì không tính kịch bản.
5. Hồ sơ → gửi kiểm soát → kiểm soát viên chấp thuận/từ chối → trưởng phòng phê duyệt/từ chối. Lưu người thao tác và lịch sử. Phê duyệt hồ sơ là quy trình người dùng, không thay thế proof của HRM.
6. Thống kê & báo cáo → xuất CSV từ hồ sơ đã lưu. Quản lý → tạo/khóa tài khoản theo quyền.

Tài liệu lỗi/chưa chạy có thể gỡ khỏi danh sách. Tài liệu đang xử lý không bị xóa giữa lúc MinerU còn chạy. File nguồn/cache được giữ lại, đúng thông báo xác nhận trên UI.

## Mã nguồn và dữ liệu

- `web_demo/src/App.jsx`, `web_demo/src/styles.css`: giao diện đang dùng.
- `web_demo/dist`: bản build đang phục vụ.
- `src/finhrm_web`: gateway, SQLite, xác thực và các luồng nghiệp vụ.
- `src/finhrm_web/launcher.py`: chạy gateway cùng dịch vụ HRM hiện có.
- `tmp/finhrm_web/workspace.sqlite3`: hồ sơ/tài khoản/hội thoại/số nhập tay. Không xóa thư mục này nếu cần giữ dữ liệu.
- `web_demo/app.js`, `web_demo/styles.css`: bản giao diện trước, không được bản build mới nạp.

Sửa giao diện rồi build lại:

```bash
cd web_demo
npm ci
npm run build
```

Không dùng Qwen/Ollama hoặc phản hồi mô phỏng để thay HRM. MinerU phải chạy riêng theo cấu hình hiện có nếu cần OCR PDF mới.


## Cập nhật upload MinerU

Launcher tự khởi động API MinerU local đã cài nếu chưa chạy. PDF upload được chuẩn bị thành Markdown trước xem trước/hỏi đáp. Tài liệu lỗi có nút Chạy lại OCR, không cần tải lại. Xem reports/finhrm_upload_mineru_fix_v1.md cho kết quả kiểm tra thực tế.
