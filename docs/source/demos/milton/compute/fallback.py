"""Photograph the Milton demo for its fallback image, ``docs/source/_static/demos/milton.png``.

The image shows where the demo cannot run (without JavaScript, or when its data does not load), and
on the card of the Demos page. Run it after the demo or its data changed: it writes the preview
(``preview.py``) and photographs the demo in it, as it opens, in light mode at 1200 pixels wide.

It needs Playwright for Python and its Chromium. From the repository's root::

    uv run --with playwright playwright install chromium
    uv run --with playwright python docs/source/demos/milton/compute/fallback.py

``--browser`` takes another Chromium to drive, such as one already installed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright
from preview import OUT as PREVIEW, main as write_preview

HERE = Path(__file__).resolve().parent
IMAGE = HERE.parents[2] / "_static" / "demos" / "milton.png"


def main() -> None:
    """Write the preview and photograph the demo in it."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--browser", type=Path, help="a Chromium executable, instead of Playwright's")
    args = parser.parse_args()
    write_preview()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=args.browser)
        page = browser.new_page(viewport={"width": 1200, "height": 900}, color_scheme="light")
        page.emulate_media(reduced_motion="reduce")  # no transitions under way
        page.goto(PREVIEW.as_uri())
        page.locator(".milton-demo-ready").wait_for()
        page.wait_for_timeout(300)
        page.locator(".milton-demo").screenshot(path=IMAGE)
        browser.close()
    print(f"wrote {IMAGE}")  # noqa: T201


if __name__ == "__main__":
    main()
