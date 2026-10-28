"""Inline the Milton demo's assets into one page, to look at the demo without building the docs.

Writes ``cache/preview.html``: the demo on a page with probly's brand tokens, in light or dark mode
as the system is set. Run from the repository's root::

    uv run python docs/source/demos/milton/compute/preview.py
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "assets"
OUT = HERE / "cache" / "preview.html"

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Milton demo preview</title>
<style>
:root {{
  scrollbar-gutter: stable; /* a scrollbar that comes and goes would change the demo's width */
  --probly-red: #ff2763;
  --probly-blue: #285bff;
  --page: #ffffff;
}}
body {{
  --probly-surface: #ffffff;
  --probly-surface-raised: #f7f8fc;
  --probly-border: #e2e6f3;
  --probly-text-muted: #5b6275;
  --color-foreground-primary: #1a1c23;
  margin: 0;
  background: var(--page);
  color: var(--color-foreground-primary);
  font: 16px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
}}
@media (prefers-color-scheme: dark) {{
  :root {{ --page: #131416; color-scheme: dark; }}
  body {{
    --probly-surface: #1a1c23;
    --probly-surface-raised: #22252e;
    --probly-border: #343846;
    --probly-text-muted: #a3a9ba;
    --color-foreground-primary: #e6e8ee;
  }}
}}
main {{ max-width: 68rem; margin: 0 auto; padding: 1.5rem 1rem 3rem; }}
</style>
<style>
{css}
</style>
</head>
<body>
<main>
<h2>Forecasting a Hurricane</h2>
<div class="milton-demo" data-root=""></div>
</main>
<script>
{data}
</script>
<script>
{js}
</script>
</body>
</html>
"""


def main() -> None:
    """Write the preview page."""
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(
        PAGE.format(
            css=(ASSETS / "demo.css").read_text(encoding="utf-8"),
            data=(ASSETS / "data.js").read_text(encoding="utf-8"),
            js=(ASSETS / "demo.js").read_text(encoding="utf-8"),
        ),
        encoding="utf-8",
    )
    print(f"wrote {OUT}")  # noqa: T201


if __name__ == "__main__":
    main()
