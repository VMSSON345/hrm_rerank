"""routers/users.py — CRUD người dùng + vai trò cho trang Quản lý.
Chỉ quan_tri (quản trị hệ thống) mới được tạo/khoá người dùng."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from finhrm_web import db
from finhrm_web.deps import get_current_user, require_role

router = APIRouter(prefix="/users", tags=["users"])

VALID_ROLES = {"can_bo", "kiem_soat", "truong_phong", "quan_tri"}


class CreateUserPayload(BaseModel):
    username: str
    full_name: str
    role: str
    password: str


class SetActivePayload(BaseModel):
    active: bool


@router.get("")
def list_users(current_user: dict = Depends(get_current_user)):
    return db.list_users()


@router.post("")
def create_user(payload: CreateUserPayload, current_user: dict = Depends(require_role("quan_tri"))):
    if payload.role not in VALID_ROLES:
        raise HTTPException(400, f"Vai trò không hợp lệ, phải thuộc {VALID_ROLES}")
    if len(payload.password) < 6:
        raise HTTPException(400, "Mật khẩu cần tối thiểu 6 ký tự.")
    return db.create_user(payload.username, payload.full_name, payload.role, payload.password)


@router.patch("/{user_id}/active")
def set_active(user_id: str, payload: SetActivePayload, current_user: dict = Depends(require_role("quan_tri"))):
    db.set_user_active(user_id, payload.active)
    return {"id": user_id, "active": payload.active}
