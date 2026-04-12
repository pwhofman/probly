"""FastAPI entry point for the probly Gemma chat UI."""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.gemma import GemmaChat
from app.mock import MockChat
from app.routes import chat

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_MOCK_ENV_VALUES = {"1", "true", "yes"}


def _mock_mode_enabled() -> bool:
    return os.environ.get("MOCK_MODE", "").strip().lower() in _MOCK_ENV_VALUES


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the chat backend once at startup and attach it to the app state.

    When ``MOCK_MODE`` is set to a truthy value, a scripted :class:`MockChat`
    is used instead of the real Gemma model. If mock mode is disabled but
    the local Gemma cache is unavailable, startup falls back to a fixed
    "Model not available." mock response.
    """
    if _mock_mode_enabled():
        app.state.gemma = MockChat()
    else:
        try:
            app.state.gemma = GemmaChat()
        except FileNotFoundError:
            app.state.gemma = MockChat(model_unavailable=True)
    yield


app = FastAPI(title="Probly Gemma Chat", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api")


@app.get("/api/health")
def health() -> dict[str, str]:
    """Liveness probe for the backend."""
    return {"status": "ok"}
