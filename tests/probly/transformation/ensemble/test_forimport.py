from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize("mod", [
    "probly.transformation.ensemble",
    "probly.transformation.ensemble.common",
    "probly.transformation.ensemble.torch",
])
def test_modules_import(mod: str) -> None:
    """Check if each ensemble module can be imported."""
    try:
        importlib.import_module(mod)
    except ModuleNotFoundError as exc:
        pytest.skip(f"Optional module not available: {mod} ({exc})")
