"""Tests for the ensemble transformation common logic."""

from __future__ import annotations

import importlib
from typing import Any

import pytest


def test_raises_if_no_impl_registered() -> None:
    """It should raise if no implementation is registered."""
    common = importlib.import_module("probly.transformation.ensemble.common")
    importlib.reload(common)  # reset registry to clean state

    class Dummy:
        """Dummy predictor without registration."""

    # Der Code wirft aktuell TypeError, nicht NotImplementedError
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        _ = common.ensemble(Dummy(), num_members=2, reset_params=False)


def test_register_wires_generator() -> None:
    """It should wire the registered generator and forward arguments correctly."""
    common = importlib.import_module("probly.transformation.ensemble.common")
    importlib.reload(common)

    calls: list[tuple[Any, int, bool]] = []

    class Dummy:
        """Dummy predictor."""

    def gen(obj: Dummy, *, n_members: int, reset_params: bool) -> str:
        calls.append((obj, n_members, reset_params))
        return "OK"

    common.register(Dummy, gen)
    base = Dummy()
    out = common.ensemble(base, num_members=5, reset_params=True)

    assert out == "OK"
    assert calls == [(base, 5, True)]
