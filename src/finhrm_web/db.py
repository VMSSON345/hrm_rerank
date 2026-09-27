"""
db.py — Lớp lưu trữ SQLite cho toàn bộ dữ liệu nghiệp vụ (không phải dữ liệu
vector — vector nằm riêng ở vectorstore.py/thư mục storage).

Bảng:
  users            — người dùng + vai trò (Cán bộ phân tích / Kiểm soát viên /
                      Trưởng phòng / Quản trị hệ thống)
  ho_so            — hồ sơ phân tích doanh nghiệp (liên kết tới nhiều document)
  documents        — metadata tài liệu đã tải lên (OCR status, số trang...)
  approval_events  — nhật ký từng bước trong luồng phê duyệt của 1 hồ sơ
"""
from __future__ import annotations

import datetime
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

DB_PATH = os.environ.get("FINHRM_WEB_DB_PATH", "tmp/finhrm_web/workspace.sqlite3")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    full_name TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('can_bo', 'kiem_soat', 'truong_phong', 'quan_tri')),
    password_hash TEXT NOT NULL DEFAULT '',
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ho_so (
    id TEXT PRIMARY KEY,
    ten_doanh_nghiep TEXT NOT NULL,
    ma_so_thue TEXT,
    nganh_nghe TEXT,
    nguoi_phu_trach_id TEXT,
    trang_thai TEXT NOT NULL DEFAULT 'moi_tao'
        CHECK(trang_thai IN ('moi_tao','cho_kiem_soat','cho_phe_duyet','da_duyet','tu_choi')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(nguoi_phu_trach_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    ho_so_id TEXT,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    reporting_period TEXT,
    total_pages INTEGER,
    ocr_pages TEXT,           -- JSON list số trang đã OCR
    status TEXT NOT NULL DEFAULT 'uploaded'
        CHECK(status IN ('uploaded','processing','indexed','failed')),
    chunk_count INTEGER DEFAULT 0,
    uploaded_at TEXT NOT NULL,
    FOREIGN KEY(ho_so_id) REFERENCES ho_so(id)
);

CREATE TABLE IF NOT EXISTS approval_events (
    id TEXT PRIMARY KEY,
    ho_so_id TEXT NOT NULL,
    step TEXT NOT NULL,       -- 'submit' | 'kiem_soat' | 'phe_duyet' | 'tu_choi'
    actor_id TEXT,
    comment TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(ho_so_id) REFERENCES ho_so(id),
    FOREIGN KEY(actor_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user','assistant')),
    text TEXT NOT NULL,
    source TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS financial_analysis_cache (
    document_id TEXT PRIMARY KEY,
    file_mtime_ns INTEGER NOT NULL,
    file_size INTEGER NOT NULL,
    payload TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
);
"""


@contextmanager
def get_conn(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


DEFAULT_SEED_PASSWORD = os.environ.get("FINHRM_WEB_SEED_PASSWORD", "FinHRM@123")  # chỉ dùng cho tài khoản demo/seed ban đầu


def _ensure_password_column(conn: sqlite3.Connection):
    """Migration nhẹ: nếu đang chạy trên DB cũ tạo trước khi có cột
    password_hash, tự thêm cột vào thay vì lỗi khi truy vấn."""
    cols = [r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()]
    if "password_hash" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN password_hash TEXT NOT NULL DEFAULT ''")


def _ensure_document_columns(conn: sqlite3.Connection):
    """Migration nhẹ cho các DB đã tạo trước khi có kỳ báo cáo."""
    cols = [r["name"] for r in conn.execute("PRAGMA table_info(documents)").fetchall()]
    if "reporting_period" not in cols:
        conn.execute("ALTER TABLE documents ADD COLUMN reporting_period TEXT")


def init_db(db_path: str = DB_PATH):
    from finhrm_web import auth  # import trễ để tránh phụ thuộc vòng nếu auth.py sau này cần db.py

    with get_conn(db_path) as conn:
        conn.executescript(SCHEMA)
        _ensure_password_column(conn)
        _ensure_document_columns(conn)
        # seed vài user mặc định nếu bảng rỗng
        row = conn.execute("SELECT COUNT(*) c FROM users").fetchone()
        if row["c"] == 0:
            seed_users = [
                ("admin", "Quản trị hệ thống", "quan_tri"),
                ("nvpt01", "Nguyễn Văn Phân Tích", "can_bo"),
                ("ksv01", "Trần Thị Kiểm Soát", "kiem_soat"),
                ("tp01", "Lê Văn Trưởng Phòng", "truong_phong"),
            ]
            now = _now()
            pw_hash = auth.hash_password(DEFAULT_SEED_PASSWORD)
            for username, full_name, role in seed_users:
                conn.execute(
                    "INSERT INTO users (id, username, full_name, role, password_hash, active, created_at) VALUES (?,?,?,?,?,1,?)",
                    (str(uuid.uuid4()), username, full_name, role, pw_hash, now),
                )


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return dict(row) if row else None


PUBLIC_USER_COLUMNS = "id, username, full_name, role, active, created_at"  # không lộ password_hash


# ---------------- Users ----------------

def list_users(db_path: str = DB_PATH) -> List[Dict]:
    with get_conn(db_path) as conn:
        rows = conn.execute(f"SELECT {PUBLIC_USER_COLUMNS} FROM users ORDER BY created_at").fetchall()
        return [_row_to_dict(r) for r in rows]


def get_user_by_username(username: str, db_path: str = DB_PATH) -> Optional[Dict]:
    """Dùng riêng cho luồng đăng nhập — CÓ kèm password_hash để verify.
    Không dùng hàm này cho bất kỳ endpoint trả dữ liệu công khai nào."""
    with get_conn(db_path) as conn:
        row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        return _row_to_dict(row)


def get_user_by_id(user_id: str, db_path: str = DB_PATH) -> Optional[Dict]:
    with get_conn(db_path) as conn:
        row = conn.execute(f"SELECT {PUBLIC_USER_COLUMNS} FROM users WHERE id=?", (user_id,)).fetchone()
        return _row_to_dict(row)


def create_user(username: str, full_name: str, role: str, password: str, db_path: str = DB_PATH) -> Dict:
    from finhrm_web import auth  # import trễ, tránh phụ thuộc vòng

    uid = str(uuid.uuid4())
    pw_hash = auth.hash_password(password)
    with get_conn(db_path) as conn:
        conn.execute(
            "INSERT INTO users (id, username, full_name, role, password_hash, active, created_at) VALUES (?,?,?,?,?,1,?)",
            (uid, username, full_name, role, pw_hash, _now()),
        )
    return {"id": uid, "username": username, "full_name": full_name, "role": role, "active": 1}


def set_user_active(user_id: str, active: bool, db_path: str = DB_PATH):
    with get_conn(db_path) as conn:
        conn.execute("UPDATE users SET active=? WHERE id=?", (1 if active else 0, user_id))


# ---------------- Hồ sơ phân tích ----------------

def list_ho_so(db_path: str = DB_PATH) -> List[Dict]:
    with get_conn(db_path) as conn:
        rows = conn.execute("""
            SELECT h.*, u.full_name AS nguoi_phu_trach_ten
            FROM ho_so h LEFT JOIN users u ON h.nguoi_phu_trach_id = u.id
            ORDER BY h.updated_at DESC
        """).fetchall()
        return [_row_to_dict(r) for r in rows]


def get_ho_so(ho_so_id: str, db_path: str = DB_PATH) -> Optional[Dict]:
    with get_conn(db_path) as conn:
        row = conn.execute("SELECT * FROM ho_so WHERE id=?", (ho_so_id,)).fetchone()
        return _row_to_dict(row)


def create_ho_so(ten_doanh_nghiep: str, ma_so_thue: str, nganh_nghe: str,
                  nguoi_phu_trach_id: Optional[str], db_path: str = DB_PATH) -> Dict:
    hid = str(uuid.uuid4())
    now = _now()
    with get_conn(db_path) as conn:
        conn.execute(
            """INSERT INTO ho_so (id, ten_doanh_nghiep, ma_so_thue, nganh_nghe,
               nguoi_phu_trach_id, trang_thai, created_at, updated_at)
               VALUES (?,?,?,?,?, 'moi_tao', ?, ?)""",
            (hid, ten_doanh_nghiep, ma_so_thue, nganh_nghe, nguoi_phu_trach_id, now, now),
        )
    return get_ho_so(hid, db_path)


def update_ho_so_status(ho_so_id: str, trang_thai: str, db_path: str = DB_PATH):
    with get_conn(db_path) as conn:
        conn.execute(
            "UPDATE ho_so SET trang_thai=?, updated_at=? WHERE id=?",
            (trang_thai, _now(), ho_so_id),
        )


# ---------------- Documents ----------------

def create_document(ho_so_id: Optional[str], filename: str, file_path: str,
                    db_path: str = DB_PATH, reporting_period: Optional[str] = None) -> Dict:
    did = str(uuid.uuid4())
    with get_conn(db_path) as conn:
        conn.execute(
            """INSERT INTO documents (id, ho_so_id, filename, file_path, reporting_period, status, uploaded_at)
               VALUES (?,?,?,?,?, 'uploaded', ?)""",
            (did, ho_so_id, filename, file_path, reporting_period, _now()),
        )
    return get_document(did, db_path)


def get_document(document_id: str, db_path: str = DB_PATH) -> Optional[Dict]:
    with get_conn(db_path) as conn:
        row = conn.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
        return _row_to_dict(row)


def list_documents(ho_so_id: Optional[str] = None, db_path: str = DB_PATH) -> List[Dict]:
    with get_conn(db_path) as conn:
        if ho_so_id:
            rows = conn.execute("SELECT * FROM documents WHERE ho_so_id=? ORDER BY uploaded_at DESC", (ho_so_id,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM documents ORDER BY uploaded_at DESC").fetchall()
        return [_row_to_dict(r) for r in rows]


def update_document_status(document_id: str, status: str, total_pages: int = None,
                            ocr_pages: str = None, chunk_count: int = None, db_path: str = DB_PATH):
    fields, values = ["status=?"], [status]
    if total_pages is not None:
        fields.append("total_pages=?"); values.append(total_pages)
    if ocr_pages is not None:
        fields.append("ocr_pages=?"); values.append(ocr_pages)
    if chunk_count is not None:
        fields.append("chunk_count=?"); values.append(chunk_count)
    values.append(document_id)
    with get_conn(db_path) as conn:
        conn.execute(f"UPDATE documents SET {', '.join(fields)} WHERE id=?", values)


def delete_document(document_id: str, db_path: str = DB_PATH) -> bool:
    """Xoá metadata tài liệu và cache phân tích liên quan.

    File PDF và vector index được router dọn trước; hàm này chỉ quản lý dữ liệu
    SQLite để không để cache mồ côi với các DB cũ chưa có cascade đầy đủ.
    """
    with get_conn(db_path) as conn:
        conn.execute("DELETE FROM financial_analysis_cache WHERE document_id=?", (document_id,))
        result = conn.execute("DELETE FROM documents WHERE id=?", (document_id,))
    return result.rowcount > 0


def update_document_reporting_period(document_id: str, reporting_period: Optional[str], db_path: str = DB_PATH) -> Optional[Dict]:
    with get_conn(db_path) as conn:
        conn.execute(
            "UPDATE documents SET reporting_period=? WHERE id=?",
            (reporting_period or None, document_id),
        )
    return get_document(document_id, db_path)


# ---------------- Cache phân tích tài chính ----------------

def get_financial_analysis_cache(document_id: str, file_mtime_ns: int,
                                 file_size: int, db_path: str = DB_PATH) -> Optional[Dict]:
    """Chỉ trả cache khi đúng phiên bản file PDF hiện tại.

    Khi file bị thay thế, mtime/size đổi và kết quả cũ tự động bị bỏ qua.
    """
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT payload FROM financial_analysis_cache "
            "WHERE document_id=? AND file_mtime_ns=? AND file_size=?",
            (document_id, file_mtime_ns, file_size),
        ).fetchone()
    if not row:
        return None
    try:
        return json.loads(row["payload"])
    except (TypeError, json.JSONDecodeError):
        return None


def save_financial_analysis_cache(document_id: str, file_mtime_ns: int,
                                  file_size: int, payload: Dict,
                                  db_path: str = DB_PATH):
    with get_conn(db_path) as conn:
        conn.execute(
            """INSERT INTO financial_analysis_cache
               (document_id, file_mtime_ns, file_size, payload, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(document_id) DO UPDATE SET
                 file_mtime_ns=excluded.file_mtime_ns,
                 file_size=excluded.file_size,
                 payload=excluded.payload,
                 updated_at=excluded.updated_at""",
            (document_id, file_mtime_ns, file_size,
             json.dumps(payload, ensure_ascii=False), _now()),
        )


# ---------------- Hội thoại trợ lý ----------------

def create_conversation(user_id: str, title: str = "Cuộc hội thoại mới", db_path: str = DB_PATH) -> Dict:
    conversation_id = str(uuid.uuid4())
    now = _now()
    with get_conn(db_path) as conn:
        conn.execute(
            "INSERT INTO conversations (id, user_id, title, created_at, updated_at) VALUES (?,?,?,?,?)",
            (conversation_id, user_id, title[:200], now, now),
        )
    return get_conversation(conversation_id, user_id, db_path)


def list_conversations(user_id: str, db_path: str = DB_PATH) -> List[Dict]:
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM conversations WHERE user_id=? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
        return [_row_to_dict(row) for row in rows]


def get_conversation(conversation_id: str, user_id: str, db_path: str = DB_PATH) -> Optional[Dict]:
    with get_conn(db_path) as conn:
        conversation = conn.execute(
            "SELECT * FROM conversations WHERE id=? AND user_id=?",
            (conversation_id, user_id),
        ).fetchone()
        if not conversation:
            return None
        result = _row_to_dict(conversation)
        result["messages"] = [
            _row_to_dict(row) for row in conn.execute(
                "SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at",
                (conversation_id,),
            ).fetchall()
        ]
        return result


def add_message(conversation_id: str, role: str, text: str, source: str = None,
                db_path: str = DB_PATH) -> Dict:
    message_id = str(uuid.uuid4())
    now = _now()
    with get_conn(db_path) as conn:
        conn.execute(
            "INSERT INTO messages (id, conversation_id, role, text, source, created_at) VALUES (?,?,?,?,?,?)",
            (message_id, conversation_id, role, text, source, now),
        )
        conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now, conversation_id))
        row = conn.execute("SELECT * FROM messages WHERE id=?", (message_id,)).fetchone()
        return _row_to_dict(row)


# ---------------- Phê duyệt ----------------

def add_approval_event(ho_so_id: str, step: str, actor_id: Optional[str], comment: str = "", db_path: str = DB_PATH) -> Dict:
    eid = str(uuid.uuid4())
    now = _now()
    with get_conn(db_path) as conn:
        conn.execute(
            "INSERT INTO approval_events (id, ho_so_id, step, actor_id, comment, created_at) VALUES (?,?,?,?,?,?)",
            (eid, ho_so_id, step, actor_id, comment, now),
        )
    return {"id": eid, "ho_so_id": ho_so_id, "step": step, "actor_id": actor_id, "comment": comment, "created_at": now}


def list_approval_events(ho_so_id: str, db_path: str = DB_PATH) -> List[Dict]:
    with get_conn(db_path) as conn:
        rows = conn.execute("""
            SELECT e.*, u.full_name AS actor_ten
            FROM approval_events e LEFT JOIN users u ON e.actor_id = u.id
            WHERE e.ho_so_id=? ORDER BY e.created_at
        """, (ho_so_id,)).fetchall()
        return [_row_to_dict(r) for r in rows]
