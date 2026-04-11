"""Chat endpoints backed by the locally cached Gemma 4 model."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.schemas import ChatMessage, ChatRequest, ChatResponse

if TYPE_CHECKING:
    from collections.abc import Iterator

router = APIRouter()


def _require_user_message(request: ChatRequest) -> None:
    if not any(m.role == "user" for m in request.messages):
        raise HTTPException(status_code=400, detail="At least one user message is required.")


@router.post("/chat", response_model=ChatResponse)
def post_chat(request: ChatRequest, http_request: Request) -> ChatResponse:
    """Run the user's chat history through Gemma and return the full reply."""
    _require_user_message(request)
    gemma = http_request.app.state.gemma
    reply = gemma.reply([m.model_dump() for m in request.messages])
    return ChatResponse(message=ChatMessage(role="assistant", content=reply))


@router.post("/chat/stream")
def post_chat_stream(request: ChatRequest, http_request: Request) -> StreamingResponse:
    """Stream the assistant reply as newline-delimited JSON chunks.

    Each line is one JSON object:

    - ``{"delta": "...", "confidence": 0.83}`` — a new piece of assistant
      text together with the model's per-token confidence (top-1 softmax
      probability for real Gemma; hand-authored floats for the mock).
    - ``{"done": true}``  — generation finished cleanly
    - ``{"error": "..."}`` — generation failed mid-stream

    The duck-typed ``reply_stream`` contract on the backing chat object
    yields ``(text, confidence)`` tuples, so both ``MockChat`` and
    ``GemmaChat`` use the same wire format.
    """
    _require_user_message(request)
    gemma = http_request.app.state.gemma
    messages = [m.model_dump() for m in request.messages]

    def iter_ndjson() -> Iterator[str]:
        try:
            for chunk, confidence in gemma.reply_stream(messages):
                yield json.dumps({"delta": chunk, "confidence": confidence}) + "\n"
            yield json.dumps({"done": True}) + "\n"
        except Exception as exc:  # noqa: BLE001 - surface any generation error to the client
            yield json.dumps({"error": str(exc)}) + "\n"

    return StreamingResponse(iter_ndjson(), media_type="application/x-ndjson")
