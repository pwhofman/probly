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


def _get_chat_backend(request: ChatRequest, http_request: Request):
    """Return the appropriate chat backend based on the request mode."""
    if request.mode == "gemma":
        gemma = http_request.app.state.gemma
        if gemma is None:
            raise HTTPException(
                status_code=503,
                detail="Gemma model not available. The model cache was not found at startup.",
            )
        return gemma
    return http_request.app.state.mock


@router.post("/chat", response_model=ChatResponse)
def post_chat(request: ChatRequest, http_request: Request) -> ChatResponse:
    """Run the user's chat history through Gemma and return the full reply."""
    _require_user_message(request)
    backend = _get_chat_backend(request, http_request)
    reply = backend.reply([m.model_dump() for m in request.messages])
    return ChatResponse(message=ChatMessage(role="assistant", content=reply))


@router.post("/chat/stream")
def post_chat_stream(request: ChatRequest, http_request: Request) -> StreamingResponse:
    """Stream the assistant reply as newline-delimited JSON chunks.

    Each line is one JSON object:

    - ``{"delta": "..."}`` — a new piece of assistant text. Deltas carry
      text only; per-word, per-concept, and per-line confidences are
      deferred to the final payload below so the frontend never has to
      compute anything itself.
    - ``{"confidence": {...}}`` — the single final confidence payload for
      the whole reply. Emitted exactly once, after the last ``delta`` and
      before ``done``. Omitted entirely if the backing chat object
      returns ``None`` from ``reply_confidence`` (real Gemma case), so
      the frontend knows to keep its confidence toggle disabled.
      Shape: ``{"words": [{"text", "confidence"}], "concepts":
      [{"first_word", "last_word", "confidence"}], "full": float}``
      — ``full`` is the single "whole response" confidence the frontend
      paints across every visual line in full mode.
    - ``{"done": true}`` — generation finished cleanly.
    - ``{"error": "..."}`` — generation failed mid-stream.

    Both ``MockChat`` and ``GemmaChat`` implement the duck-typed pair
    ``reply_stream(messages) -> Iterator[str]`` plus
    ``reply_confidence(messages) -> dict | None``.
    """
    _require_user_message(request)
    backend = _get_chat_backend(request, http_request)
    messages = [m.model_dump() for m in request.messages]

    def iter_ndjson() -> Iterator[str]:
        try:
            for chunk in backend.reply_stream(messages):
                yield json.dumps({"delta": chunk}) + "\n"
            payload = backend.reply_confidence(messages)
            if payload is not None:
                yield json.dumps({"confidence": payload}) + "\n"
            yield json.dumps({"done": True}) + "\n"
        except Exception as exc:  # noqa: BLE001 - surface any generation error to the client
            yield json.dumps({"error": str(exc)}) + "\n"

    return StreamingResponse(iter_ndjson(), media_type="application/x-ndjson")
