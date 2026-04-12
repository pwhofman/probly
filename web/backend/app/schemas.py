"""Pydantic models for the chat API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Role = Literal["user", "assistant", "system"]


class ChatMessage(BaseModel):
    role: Role
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    mode: Literal["probly", "gemma"] = "probly"


class ChatResponse(BaseModel):
    message: ChatMessage
