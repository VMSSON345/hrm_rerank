"""routers/ho_so.py — CRUD hồ sơ phân tích doanh nghiệp."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from finhrm_web import db

router = APIRouter(prefix="/ho-so", tags=["ho_so"])


class CreateHoSoPayload(BaseModel):
    ten_doanh_nghiep: str
    ma_so_thue: str | None = None
    nganh_nghe: str | None = None
    nguoi_phu_trach_id: str | None = None


@router.get("")
def list_ho_so():
    return db.list_ho_so()


@router.post("")
def create_ho_so(payload: CreateHoSoPayload):
    return db.create_ho_so(
        payload.ten_doanh_nghiep, payload.ma_so_thue,
        payload.nganh_nghe, payload.nguoi_phu_trach_id,
    )


@router.get("/{ho_so_id}")
def get_ho_so(ho_so_id: str):
    hs = db.get_ho_so(ho_so_id)
    if not hs:
        raise HTTPException(404, "Không tìm thấy hồ sơ")
    hs["documents"] = db.list_documents(ho_so_id=ho_so_id)
    hs["approval_events"] = db.list_approval_events(ho_so_id)
    return hs
