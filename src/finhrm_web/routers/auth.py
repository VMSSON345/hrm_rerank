"""routers/auth.py — Đăng nhập cấp JWT, endpoint /auth/me."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from finhrm_web import auth
from finhrm_web import db
from finhrm_web.deps import get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginPayload(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


@router.post("/login", response_model=LoginResponse)
def login(payload: LoginPayload):
    user = db.get_user_by_username(payload.username)
    if not user or not auth.verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Sai tên đăng nhập hoặc mật khẩu.")
    if not user["active"]:
        raise HTTPException(status_code=403, detail="Tài khoản đã bị khoá, liên hệ quản trị hệ thống.")

    token = auth.create_access_token(subject=user["id"], extra_claims={"role": user["role"]})
    public_user = {k: v for k, v in user.items() if k != "password_hash"}
    return LoginResponse(access_token=token, user=public_user)


@router.get("/me")
def me(current_user: dict = Depends(get_current_user)):
    return {k: v for k, v in current_user.items() if k != "password_hash"}
