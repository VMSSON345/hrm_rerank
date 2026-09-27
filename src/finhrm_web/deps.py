"""
deps.py — FastAPI dependencies dùng chung: bắt buộc đăng nhập (JWT hợp lệ)
cho mọi router nghiệp vụ, và kiểm tra vai trò cho các hành động nhạy cảm
(duyệt hồ sơ, quản lý người dùng...).
"""
from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError

from finhrm_web import auth
from finhrm_web import db

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


def get_current_user(token: str | None = Depends(oauth2_scheme)) -> dict:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Chưa đăng nhập hoặc token không hợp lệ.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not token:
        raise unauthorized
    try:
        payload = auth.decode_access_token(token)
    except JWTError:
        raise unauthorized

    user_id = payload.get("sub")
    if not user_id:
        raise unauthorized

    user = db.get_user_by_id(user_id)
    if not user:
        raise unauthorized
    if not user["active"]:
        raise HTTPException(status_code=403, detail="Tài khoản đã bị khoá.")
    return user


def require_role(*allowed_roles: str):
    """Tạo dependency chỉ cho phép các vai trò cụ thể (quan_tri luôn được
    phép, đóng vai trò quản trị viên có thể thay thế mọi vai trò khác)."""

    def _checker(current_user: dict = Depends(get_current_user)) -> dict:
        if current_user["role"] == "quan_tri" or current_user["role"] in allowed_roles:
            return current_user
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Thao tác này yêu cầu vai trò thuộc {allowed_roles}, tài khoản hiện tại là '{current_user['role']}'.",
        )

    return _checker
