# Probly Gemma Chat

A minimal FastAPI + React + Tailwind chat interface backed by the
locally cached `google/gemma-4-E2B-it` model. Generation mirrors the
REPL in `experiments/gemma/interaction.py`.

## Layout

```
web/
├── backend/    FastAPI app (uv-managed, isolated from main probly)
└── frontend/   Vite + React + TypeScript + Tailwind
```

## Model cache

The backend loads Gemma from `data/model_cache/` at the repo root (git
ignored). Populate it once before starting the backend, either by
copying an existing cache:

```bash
cp -c -R experiments/gemma/model_cache data/model_cache
```

or by running `experiments/gemma/download.py` against `data/model_cache/`.

## Backend

Run from `web/backend/`:

```bash
uv sync
uv run uvicorn app.main:app --reload --port 8000
```

Startup will load the model into memory (tens of seconds). Endpoints:

- `GET  /api/health` — liveness probe
- `POST /api/chat`   — `{ "messages": [{"role": "user", "content": "hi"}] }`
                      returns `{ "message": {"role": "assistant", "content": "..."} }`

## Frontend

Run from `web/frontend/`:

```bash
npm install
npm run dev
```

Then open <http://localhost:5173>. The Vite dev server proxies `/api`
to `http://localhost:8000`, so start the backend first.

## Next steps

- Add streaming (SSE or WebSocket) so tokens arrive as they are
  generated.
- Surface token-level uncertainty in the chat bubbles.
- Expose generation parameters (temperature, max tokens) in the UI.
