"""Torch and jax compute the same convex-hull upper entropies.

Both backends run the same iteration from the same start, so on the same inputs they
agree up to floating-point rounding. This module needs both backends and is skipped
otherwise; each backend is also checked against the same numpy references in its own
test module.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("jax")

import jax
import jax.numpy as jnp
import numpy as np
import torch

from probly.quantification.measure.credal_set import upper_entropy
from probly.representation.credal_set.jax import JaxConvexCredalSet
from probly.representation.credal_set.torch import TorchConvexCredalSet
from probly.representation.distribution.jax_categorical import JaxProbabilityCategoricalDistribution
from probly.representation.distribution.torch_categorical import TorchProbabilityCategoricalDistribution

from ._convex_suite import DTYPES, random_vertices


@pytest.fixture(autouse=True)
def _jax_float64():
    """Let float64 inputs keep their dtype in jax."""
    previous = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@DTYPES
@pytest.mark.parametrize(("n_vertices", "n_classes"), [(2, 3), (5, 10), (10, 10), (20, 5), (8, 50)])
def test_convex_upper_entropy_parity(dtype: type[np.floating], n_vertices: int, n_classes: int) -> None:
    rng = np.random.default_rng(10 * n_vertices + n_classes)
    kinds = ["ensemble", "dirichlet", "sparse", "repeated"] * 3
    vertices = np.stack([random_vertices(rng, kind, n_vertices, n_classes) for kind in kinds]).astype(dtype)

    torch_value, torch_p = upper_entropy(
        TorchConvexCredalSet(tensor=TorchProbabilityCategoricalDistribution(torch.as_tensor(vertices))),
        return_distribution=True,
    )
    jax_value, jax_p = upper_entropy(
        JaxConvexCredalSet(tensor=JaxProbabilityCategoricalDistribution(jnp.asarray(vertices))),
        return_distribution=True,
    )

    atol = 1e-10 if dtype == np.float64 else 1e-5
    assert jax_value.dtype == torch_value.numpy().dtype == dtype
    np.testing.assert_allclose(np.asarray(jax_value), torch_value.numpy(), atol=atol)
    np.testing.assert_allclose(np.asarray(jax_p), torch_p.numpy(), atol=10 * atol)
