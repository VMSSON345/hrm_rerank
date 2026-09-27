"""
auth.py — Tiện ích xác thực: hash/verify mật khẩu bằng bcrypt (dùng thư viện
bcrypt trực tiếp, KHÔNG dùng passlib — passlib đã ngừng bảo trì và có lỗi
tương thích đã biết với các bản bcrypt >=4.1), và tạo/giải mã JWT bằng
python-jose.

Biến môi trường:
  FINHRM_WEB_JWT_SECRET — khoá ký JWT. BẮT BUỘC đặt giá trị riêng khi triển
    khai thật; giá trị mặc định dưới đây CHỈ dùng cho môi trường phát triển/
    demo cục bộ, không được dùng khi expose ra internet.
"""
from __future__ import annotations

import datetime
import os
import secrets
from typing import Optional

import bcrypt
from jose import JWTError, jwt

JWT_SECRET = os.environ.get("FINHRM_WEB_JWT_SECRET", secrets.token_urlsafe(48))
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("FINHRM_WEB_TOKEN_EXPIRE_MINUTES", "480"))  # 8 giờ


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_access_token(subject: str, extra_claims: Optional[dict] = None) -> str:
    now = datetime.datetime.now(datetime.timezone.utc)
    payload = {
        "sub": subject,
        "iat": now,
        "exp": now + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        **(extra_claims or {}),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """Ném JWTError nếu token không hợp lệ hoặc đã hết hạn — caller (deps.py)
    chịu trách nhiệm bắt lỗi và trả 401 cho client."""
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
