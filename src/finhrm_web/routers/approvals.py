"""
routers/approvals.py — Luồng phê duyệt 3 cấp cho hồ sơ phân tích:

    moi_tao --submit--> cho_kiem_soat --kiểm soát duyệt--> cho_phe_duyet
                                       (kiểm soát từ chối--> tu_choi)
    cho_phe_duyet --trưởng phòng duyệt--> da_duyet
                  (trưởng phòng từ chối--> tu_choi)

Mỗi bước đều kiểm tra đúng trạng thái hiện tại trước khi cho chuyển tiếp
(tránh việc phê duyệt "nhảy cóc" hoặc phê duyệt 2 lần), và ghi lại đầy đủ vào
approval_events để tra soát sau này — đúng tinh thần "lưu vết đầy đủ" đã nêu
trong tài liệu tập huấn Chuyên đề 10 (hộ chiếu đầu ra AI / audit trail).

Từ khi có xác thực JWT: người thực hiện (actor) KHÔNG còn do client tự khai
(actor_id trong body) — luôn được backend lấy từ danh tính đã xác thực
trong token, để không ai có thể "hành động thay" người khác. Mỗi bước cũng
yêu cầu đúng vai trò tương ứng (kiểm soát viên mới được duyệt bước kiểm
soát, trưởng phòng mới được duyệt bước cuối), quan_tri được phép ở mọi bước
với vai trò quản trị/dự phòng.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from finhrm_web import db
from finhrm_web.deps import require_role

router = APIRouter(prefix="/approvals", tags=["approvals"])


class ActionPayload(BaseModel):
    comment: str = ""


class ReviewPayload(ActionPayload):
    approve: bool


def _require_status(ho_so_id: str, expected: str) -> dict:
    hs = db.get_ho_so(ho_so_id)
    if not hs:
        raise HTTPException(404, "Không tìm thấy hồ sơ")
    if hs["trang_thai"] != expected:
        raise HTTPException(
            409,
            f"Hồ sơ đang ở trạng thái '{hs['trang_thai']}', cần ở trạng thái "
            f"'{expected}' để thực hiện thao tác này.",
        )
    return hs


@router.post("/{ho_so_id}/submit")
def submit(ho_so_id: str, payload: ActionPayload, current_user: dict = Depends(require_role("can_bo"))):
    _require_status(ho_so_id, "moi_tao")
    db.update_ho_so_status(ho_so_id, "cho_kiem_soat")
    return db.add_approval_event(ho_so_id, "submit", current_user["id"], payload.comment)


@router.post("/{ho_so_id}/kiem-soat")
def kiem_soat(ho_so_id: str, payload: ReviewPayload, current_user: dict = Depends(require_role("kiem_soat"))):
    _require_status(ho_so_id, "cho_kiem_soat")
    if payload.approve:
        db.update_ho_so_status(ho_so_id, "cho_phe_duyet")
        return db.add_approval_event(ho_so_id, "kiem_soat", current_user["id"], payload.comment)
    else:
        db.update_ho_so_status(ho_so_id, "tu_choi")
        return db.add_approval_event(ho_so_id, "tu_choi", current_user["id"], payload.comment)


@router.post("/{ho_so_id}/phe-duyet")
def phe_duyet(ho_so_id: str, payload: ReviewPayload, current_user: dict = Depends(require_role("truong_phong"))):
    _require_status(ho_so_id, "cho_phe_duyet")
    if payload.approve:
        db.update_ho_so_status(ho_so_id, "da_duyet")
        return db.add_approval_event(ho_so_id, "phe_duyet", current_user["id"], payload.comment)
    else:
        db.update_ho_so_status(ho_so_id, "tu_choi")
        return db.add_approval_event(ho_so_id, "tu_choi", current_user["id"], payload.comment)


@router.get("/{ho_so_id}/history")
def history(ho_so_id: str):
    if not db.get_ho_so(ho_so_id):
        raise HTTPException(404, "Không tìm thấy hồ sơ")
    return db.list_approval_events(ho_so_id)


@router.get("/queue/{role}")
def queue(role: str):
    """Danh sách hồ sơ đang chờ vai trò `role` xử lý — dùng cho trang Phê duyệt
    hiển thị đúng việc cần làm theo từng vai trò đăng nhập."""
    status_map = {"kiem_soat": "cho_kiem_soat", "truong_phong": "cho_phe_duyet"}
    target_status = status_map.get(role)
    if not target_status:
        raise HTTPException(400, "Vai trò không hợp lệ (kiem_soat | truong_phong)")
    return [h for h in db.list_ho_so() if h["trang_thai"] == target_status]
