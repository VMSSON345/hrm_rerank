# FinHRM local demo

Giao diện demo dùng trực tiếp checkpoint:

`models/document_hrm_V8_candidate.pt`

V12-A là runtime neuro-symbolic chạy trên checkpoint V8 frozen. Tính năng vẫn
được bật tường minh để phân biệt với baseline V8.

Ứng dụng chạy tại `http://127.0.0.1:8765`. Mô hình, API và giao diện nằm trong
cùng một tiến trình local; tài liệu tải lên được lưu dưới
`tmp/demo_uploads/`.

## Chạy trên Windows

Thiết lập môi trường một lần:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_demo.ps1
```

Lần thiết lập đầu sẽ tải và cache semantic encoder 384 chiều để lúc trình diễn
không phải chờ tải mô hình phụ.

Sau đó nhấp đúp `start_demo.cmd`, hoặc chạy:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1 `
  -TypedProgramSynthesis
```

Trình duyệt sẽ tự mở sau khi checkpoint đã được nạp. Nhấn `Ctrl+C` trong cửa
sổ terminal để dừng dịch vụ.

## Chạy trên WSL/Linux

```bash
bash scripts/setup_demo.sh
TYPED_PROGRAM_SYNTHESIS=1 bash scripts/start_demo.sh
```

Thử nghiệm thêm catalog công thức YAML (không phải cấu hình demo mặc định):

```bash
TYPED_PROGRAM_SYNTHESIS=1 FINANCIAL_IDENTITY_CATALOG=1 \
  bash scripts/start_demo.sh
```

Mở `http://127.0.0.1:8765` trong trình duyệt.

## Luồng trình diễn đề xuất

1. Tải một báo cáo PDF có lớp chữ.
2. Chờ trạng thái `Tài liệu đã sẵn sàng`.
3. Hỏi một số liệu trực tiếp, ví dụ: `Doanh thu thuần năm nay là bao nhiêu?`
4. Hỏi tiếp một phép tính, ví dụ:
   `Lợi nhuận sau thuế tăng bao nhiêu phần trăm so với năm trước?`
5. Mở phần `Bằng chứng nguồn` và `Chi tiết kỹ thuật` để trình bày cách mô hình
   lập chương trình, chọn toán hạng và dẫn trang.
6. Dùng `Xuất kết quả` để lưu toàn bộ phiên thành JSON.

## PDF scan và MinerU

PDF có lớp chữ hoạt động ngay cả khi MinerU không chạy. Đối với PDF scan, khởi
động MinerU API trong WSL ở một terminal khác:

```bash
bash scripts/setup_mineru_venv.sh
bash scripts/run_mineru_api.sh
```

Hộp `Trạng thái hệ thống` trong giao diện cho biết OCR đang sẵn sàng hay không.

## Tùy chọn

Chạy trên GPU CUDA và đổi cổng:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1 -Device cuda -Port 8877
```

Không tự mở trình duyệt:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1 -NoBrowser
```

Checkpoint chỉ được nạp một lần khi khởi động. Các câu hỏi tiếp theo dùng lại
cùng mô hình và semantic cache.

Chạy baseline V8 mà không bật typed synthesis:

```bash
bash scripts/start_demo.sh
```
