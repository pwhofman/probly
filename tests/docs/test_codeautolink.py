"""Incremental code-linking must leave cached HTML alone."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("sphinx")


def test_only_rendered_pages_are_relinked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "docs" / "source"))
    track_codeautolink_written_pages = import_module("_sphinx_helpers").track_codeautolink_written_pages
    app = Mock()
    state = SimpleNamespace(outdated_docs={"cached/already-linked", "removed-page"})
    track_codeautolink_written_pages(app, state)
    reset_call, render_call = app.connect.call_args_list
    assert reset_call.args[0] == "builder-inited"
    assert reset_call.kwargs == {"priority": 501}
    assert render_call.args[0] == "doctree-resolved"
    reset = reset_call.args[1]
    render = render_call.args[1]

    reset(app)
    assert state.outdated_docs == set()
    render(app, object(), "api/new-page")
    render(app, object(), "index")
    assert state.outdated_docs == {str(Path("api/new-page")), "index"}

    # A subsequent no-op build must not process the previous build's output.
    reset(app)
    assert state.outdated_docs == set()


def test_parallel_and_cached_builds_preserve_links(tmp_path: Path) -> None:
    pytest.importorskip("sphinx_codeautolink")
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "html"
    helpers = Path(__file__).resolve().parents[2] / "docs" / "source"
    config = f"""
import sys
sys.path.insert(0, {str(helpers)!r})
from _sphinx_helpers import track_codeautolink_written_pages
extensions = ["sphinx_codeautolink"]
def setup(app):
    for listener in app.events.listeners["build-finished"]:
        if getattr(listener.handler, "__name__", "") == "apply_links":
            track_codeautolink_written_pages(app, listener.handler.__self__)
"""
    (source / "conf.py").write_text(config)
    (source / "index.rst").write_text(
        "Code links\n==========\n\n.. py:function:: len(obj)\n\n.. toctree::\n\n   first\n   second\n"
    )
    for page in ("first", "second"):
        (source / f"{page}.rst").write_text(f"{page}\n======\n\n.. code-block:: python\n\n   len([1, 2])\n")

    def build() -> None:
        result = subprocess.run(  # noqa: S603 - fixed command and pytest-owned paths
            [sys.executable, "-m", "sphinx", "-j", "2", "-b", "html", "-W", str(source), str(output)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        for page in ("first", "second"):
            assert 'class="sphinx-codeautolink-a"' in (output / f"{page}.html").read_text()

    build()
    # A changed HTML config can make codeautolink select already-linked pages.
    (source / "conf.py").write_text(config + '\nhtml_title = "Changed title"\n')
    build()
    build()
