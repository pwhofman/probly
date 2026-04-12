"""FastAPI entry point for the probly Gemma chat UI."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.gemma import GemmaChat
from app.mock import MockChat
from app.routes import chat

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load both chat backends at startup.

    ``app.state.mock`` is always available (scripted demo responses).
    ``app.state.gemma`` is the real Gemma model when the local cache
    exists, or ``None`` if the cache is missing.
    """
    app.state.mock = MockChat()
    try:
        app.state.gemma = GemmaChat()
    except FileNotFoundError:
        app.state.gemma = None
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
