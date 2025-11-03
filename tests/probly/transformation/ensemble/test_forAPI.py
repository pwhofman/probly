from __future__ import annotations

import importlib


def test_ensemble_has_public_api() -> None:
    """Check if the ensemble module exposes the correct public API."""
    ens_mod = importlib.import_module("probly.transformation.ensemble")

    # Must have these attributes
    assert hasattr(ens_mod, "ensemble")
    assert hasattr(ens_mod, "register")

    # Ensure 'ensemble' is callable
    assert callable(ens_mod.ensemble)

