"""Numpy, torch and jax compute the same credal-set entropies.

The backends share one implementation, so on the same inputs they agree up to floating-point
rounding. Minimizers can tie, for example when two classes have the same bounds, and rounding
may break such ties differently, so distributions are compared up to a permutation of the
classes. This module needs torch and jax and is skipped otherwise; each backend is also checked
against the same numpy references in its own test module.
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
from probly.representation.credal_set.jax import JaxProbabilityIntervalsCredalSet
from probly.representation.credal_set.numpy import NumpyProbabilityIntervalsCredalSet
from probly.representation.credal_set.torch import TorchProbabilityIntervalsCredalSet

from ._entropy_suite import DTYPES, EXACT_MAX_CLASSES, random_intervals


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
    # The backends sum the entropy terms in different orders, which costs a few float32 roundings for many classes.
    return 1e-12 if dtype == np.float64 else 5e-6


def _assert_parity(results: dict[str, tuple[np.ndarray, np.ndarray]], dtype: type[np.floating]) -> None:
    """Compare the entropies and the sorted distributions of torch and jax with those of numpy."""
    numpy_value, numpy_p = results["numpy"]
    for backend in ("torch", "jax"):
        value, p = results[backend]
        assert value.dtype == numpy_value.dtype == dtype, backend
        np.testing.assert_allclose(value, numpy_value, atol=_parity_tolerance(dtype), err_msg=backend)
        np.testing.assert_allclose(np.sort(p), np.sort(numpy_p), atol=10 * _parity_tolerance(dtype), err_msg=backend)


@DTYPES
@pytest.mark.parametrize("n_classes", [3, 6, EXACT_MAX_CLASSES, EXACT_MAX_CLASSES + 1, 40])
def test_intervals_lower_entropy_parity(dtype: type[np.floating], n_classes: int) -> None:
    rng = np.random.default_rng(n_classes)
    bounds = [random_intervals(rng, kind, n_classes) for kind in ["ensemble", "dirichlet", "sparse", "box"] * 5]
    lower = np.stack([b[0] for b in bounds]).astype(dtype)
    upper = np.stack([b[1] for b in bounds]).astype(dtype)
    credal_sets = {
        "numpy": NumpyProbabilityIntervalsCredalSet(lower, upper),
        "torch": TorchProbabilityIntervalsCredalSet(torch.as_tensor(lower), torch.as_tensor(upper)),
        "jax": JaxProbabilityIntervalsCredalSet(jnp.asarray(lower), jnp.asarray(upper)),
    }

    approximate = n_classes > EXACT_MAX_CLASSES
    results = {}
    for backend, credal_set in credal_sets.items():
        value, p = lower_entropy(credal_set, return_distribution=True, approximate=approximate)
        results[backend] = (np.asarray(value), np.asarray(p))

    _assert_parity(results, dtype)


@DTYPES
@pytest.mark.parametrize("n_classes", [3, 6, 40])
def test_intervals_upper_entropy_parity(dtype: type[np.floating], n_classes: int) -> None:
    rng = np.random.default_rng(n_classes)
    bounds = [random_intervals(rng, kind, n_classes) for kind in ["ensemble", "dirichlet", "sparse", "box"] * 5]
    lower = np.stack([b[0] for b in bounds]).astype(dtype)
    upper = np.stack([b[1] for b in bounds]).astype(dtype)
    credal_sets = {
        "numpy": NumpyProbabilityIntervalsCredalSet(lower, upper),
        "torch": TorchProbabilityIntervalsCredalSet(torch.as_tensor(lower), torch.as_tensor(upper)),
        "jax": JaxProbabilityIntervalsCredalSet(jnp.asarray(lower), jnp.asarray(upper)),
    }

    results = {}
    for backend, credal_set in credal_sets.items():
        value, p = upper_entropy(credal_set, return_distribution=True)
        results[backend] = (np.asarray(value), np.asarray(p))

    _assert_parity(results, dtype)
