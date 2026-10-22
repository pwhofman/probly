"""Tests for decomposition lookup typing."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

import pytest

_TYPING_CHECK = """
from typing import assert_type

from probly.quantification.decomposition.decomposition import Decomposition
from probly.quantification.notion import AleatoricUncertainty, Notion

def check_decomposition_lookup(decomposition: Decomposition) -> None:
    assert_type(decomposition[AleatoricUncertainty], AleatoricUncertainty)
    assert_type(decomposition["aleatoric"], Notion)
"""


def test_lookup_types(tmp_path: Path) -> None:
    executable = shutil.which("ty")
    if executable is None:
        pytest.skip("ty is required for the static decomposition lookup checks")
    source = tmp_path / "check_decomposition.py"
    source.write_text(_TYPING_CHECK, encoding="utf-8")
    # Check source directly: generated partial stubs shadow imports in tests.
    source_root = Path(__file__).resolve().parents[4] / "src"
    result = subprocess.run(  # noqa: S603
        [
            executable,
            "check",
            str(source),
            "--project",
            str(tmp_path),
            "--python",
            sys.prefix,
            "--extra-search-path",
            str(source_root),
            "--error-on-warning",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
