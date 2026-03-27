---
name: Project commands and conventions
description: Key commands and conventions for the probly project
type: project
---

Pre-commit: `uv run prek run --all-files` (fast, ~1-2s)
Tests: `uv run pytest tests/`
Type check only: `ty check <path>`
Docs build: `rm -rf docs/source/api && uv run sphinx-build -b html docs/source docs/build/html`

Docstyle: Google-style, NO type info in docstrings (inferred from code).

Ruff TC001: The project enforces TC001 — imports only needed for type annotations must go under `TYPE_CHECKING`. With `from __future__ import annotations`, all annotation imports (including those used only in function signatures) qualify. Move them to the `TYPE_CHECKING` block to avoid the lint error.

**Why:** Described in AGENTS.md / CLAUDE.md as project conventions.
**How to apply:** Always follow when writing or reviewing code in this repo.
