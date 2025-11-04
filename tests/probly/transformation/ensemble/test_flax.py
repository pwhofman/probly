"""Tests for flax ensemble transformation."""

from __future__ import annotations

import pytest

pytest.importorskip("flax")
from flax import nnx
import numpy as np

from probly.transformation.ensemble.flax import generate_flax_ensemble


def test_models_are_reset(flax_model_small_2d_2d: nnx.Sequential) -> None:
    n_members = 2
    original_weights = flax_model_small_2d_2d.layers[0].kernel.value
    clones = generate_flax_ensemble(flax_model_small_2d_2d, n_members=n_members)

    w1 = clones[0].layers[0].kernel.value
    w2 = clones[1].layers[0].kernel.value

    if hasattr(flax_model_small_2d_2d.layers[0], "reset_parameters"):
        assert not np.array_equal(w1, original_weights)
        assert not np.array_equal(w2, original_weights)
        assert not np.array_equal(w1, w2)
    else:
        clones[0].layers[0].kernel.value = w1 + 1.0
        assert np.array_equal(flax_model_small_2d_2d.layers[0].kernel.value, original_weights)
        assert np.array_equal(clones[1].layers[0].kernel.value, w2)
