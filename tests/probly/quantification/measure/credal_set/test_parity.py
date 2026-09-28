"""Torch and jax compute the same credal-set entropies.

Both backends implement the same algorithms, so on the same inputs they agree up to
floating-point rounding. Minimizers can tie, for example when two classes have the same
bounds, and rounding may break such ties differently, so distributions are compared up to
a permutation of the classes. This module needs both backends and is skipped otherwise;
each backend is also checked against the same numpy references in its own test module.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("jax")

import jax
import jax.numpy as jnp
import numpy as np
import torch

from probly.quantification.measure.credal_set import lower_entropy, upper_entropy
from probly.representation.credal_set.jax import JaxDistanceBasedCredalSet, JaxProbabilityIntervalsCredalSet
from probly.representation.credal_set.torch import TorchDistanceBasedCredalSet, TorchProbabilityIntervalsCredalSet
from probly.representation.distribution.jax_categorical import JaxProbabilityCategoricalDistribution
from probly.representation.distribution.torch_categorical import TorchProbabilityCategoricalDistribution

from ._entropy_suite import DTYPES, EXACT_MAX_CLASSES, random_intervals, random_tv_ball


@pytest.fixture(autouse=True)
def _jax_float64():
    """Let float64 inputs keep their dtype in jax."""
    previous = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _parity_tolerance(dtype: type[np.floating]) -> float:
    return 1e-12 if dtype == np.float64 else 1e-6


@DTYPES
@pytest.mark.parametrize("n_classes", [3, 6, EXACT_MAX_CLASSES, EXACT_MAX_CLASSES + 1, 40])
def test_intervals_lower_entropy_parity(dtype: type[np.floating], n_classes: int) -> None:
    rng = np.random.default_rng(n_classes)
    bounds = [random_intervals(rng, kind, n_classes) for kind in ["ensemble", "dirichlet", "sparse", "box"] * 5]
    lower = np.stack([b[0] for b in bounds]).astype(dtype)
    upper = np.stack([b[1] for b in bounds]).astype(dtype)

    approximate = n_classes > EXACT_MAX_CLASSES
    torch_value, torch_p = lower_entropy(
        TorchProbabilityIntervalsCredalSet(torch.as_tensor(lower), torch.as_tensor(upper)),
        return_distribution=True,
        approximate=approximate,
    )
    jax_value, jax_p = lower_entropy(
        JaxProbabilityIntervalsCredalSet(jnp.asarray(lower), jnp.asarray(upper)),
        return_distribution=True,
        approximate=approximate,
    )

    assert jax_value.dtype == torch_value.numpy().dtype == dtype
    np.testing.assert_allclose(np.asarray(jax_value), torch_value.numpy(), atol=_parity_tolerance(dtype))
    np.testing.assert_allclose(np.sort(np.asarray(jax_p)), np.sort(torch_p.numpy()), atol=10 * _parity_tolerance(dtype))


@DTYPES
@pytest.mark.parametrize("n_classes", [2, 4, 10, 50])
@pytest.mark.parametrize("measure", [upper_entropy, lower_entropy], ids=["upper", "lower"])
def test_distance_based_entropy_parity(dtype: type[np.floating], n_classes: int, measure) -> None:
    rng = np.random.default_rng(n_classes)
    balls = [random_tv_ball(rng, kind, n_classes) for kind in ["softmax", "dirichlet", "sparse"] * 5]
    nominal = np.stack([b[0] for b in balls]).astype(dtype)
    radius = np.array([b[1] for b in balls], dtype=dtype)

    torch_value, torch_p = measure(
        TorchDistanceBasedCredalSet(
            TorchProbabilityCategoricalDistribution(torch.as_tensor(nominal)), torch.as_tensor(radius)
        ),
        return_distribution=True,
    )
    jax_value, jax_p = measure(
        JaxDistanceBasedCredalSet(JaxProbabilityCategoricalDistribution(jnp.asarray(nominal)), jnp.asarray(radius)),
        return_distribution=True,
    )

    assert jax_value.dtype == torch_value.numpy().dtype == dtype
    np.testing.assert_allclose(np.asarray(jax_value), torch_value.numpy(), atol=_parity_tolerance(dtype))
    np.testing.assert_allclose(np.sort(np.asarray(jax_p)), np.sort(torch_p.numpy()), atol=10 * _parity_tolerance(dtype))
