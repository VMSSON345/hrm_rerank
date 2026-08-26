# QP4 page-index byte-diff diagnostic

Kết luận: **PASS — cơ chế đã được chứng minh trên bản sao cô lập**.

- SQLite SELECT thuần giữ nguyên hash.
- `DocumentPageIndex._connect()` và `load_or_build()` cùng đổi đúng hai byte tại offsets `27` và `95`.
- Hai SQLite header fields `change_counter` và `version_valid_for` cùng tăng `2996 → 2997`.
- Logical digest của schema, 24 documents và 1511 pages giữ nguyên; `documents_rebuilt=0`.
- Frozen page index gốc giữ nguyên hash `df8be071…84fd7` trước/sau.

Code path cụ thể là `DocumentPageIndex._connect()`: mỗi lần mở chạy `CREATE … IF NOT EXISTS` và gán `PRAGMA user_version`, tạo write-capable transaction dù logical content không đổi. Diagnostic này giải thích integrity failure trước; không sửa lại kết quả replay cũ và không cho phép runtime repair.

Next allowed action: chỉ tạo precommit positive-target instrumentation cohort cố định; chưa chạy replay trong bước này.
