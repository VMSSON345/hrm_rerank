"""Persistent assistant conversations owned by the authenticated user."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from finhrm_web import db
from finhrm_web.deps import get_current_user

router = APIRouter(prefix="/conversations", tags=["conversations"])


class ConversationCreate(BaseModel):
    title: str = Field(default="Cuộc hội thoại mới", max_length=200)


class MessageCreate(BaseModel):
    role: str
    text: str
    source: str | None = None


@router.get("")
def list_conversations(current_user: dict = Depends(get_current_user)):
    return db.list_conversations(current_user["id"])


@router.post("")
def create_conversation(payload: ConversationCreate, current_user: dict = Depends(get_current_user)):
    return db.create_conversation(current_user["id"], payload.title)


@router.get("/{conversation_id}")
def get_conversation(conversation_id: str, current_user: dict = Depends(get_current_user)):
    conversation = db.get_conversation(conversation_id, current_user["id"])
    if not conversation:
        raise HTTPException(404, "Không tìm thấy hội thoại")
    return conversation


@router.post("/{conversation_id}/messages")
def add_message(conversation_id: str, payload: MessageCreate, current_user: dict = Depends(get_current_user)):
    if payload.role not in {"user", "assistant"}:
        raise HTTPException(400, "Role tin nhắn không hợp lệ")
    if not db.get_conversation(conversation_id, current_user["id"]):
        raise HTTPException(404, "Không tìm thấy hội thoại")
    return db.add_message(conversation_id, payload.role, payload.text, payload.source)