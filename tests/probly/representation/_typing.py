"""Check representation contracts against source rather than generated partial stubs."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def check_types(source: str, tmp_path: Path) -> None:
    executable = shutil.which("ty")
    if executable is None:
        pytest.skip("ty is required for static representation checks")
    path = tmp_path / "check_representations.py"
    path.write_text(source, encoding="utf-8")
    result = subprocess.run(  # noqa: S603
        [
            executable,
            "check",
            str(path),
            "--project",
            str(tmp_path),
            "--python",
            sys.prefix,
            "--extra-search-path",
            str(Path(__file__).resolve().parents[3] / "src"),
            "--error-on-warning",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
