"""Basic sanity checks for the ARF -> probly bridge.

Run with: ``uv run pytest``
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from probly.quantification import SecondOrderEntropyDecomposition
from river import forest

from river_uncertainty import make_synthetic_stream, river_arf_to_probly_sample


@pytest.fixture()
def trained_arf():
    arf = forest.ARFClassifier(n_models=4, seed=0)
    stream = make_synthetic_stream("agrawal", n_samples=150, seed=0)
    for x, y in itertools.islice(stream, 100):
        arf.learn_one(x, y)
    # leave the iterator for callers that need a fresh point
    return arf, stream


def test_representation_shape(trained_arf):
    arf, stream = trained_arf
    x, _ = next(iter(stream))
    rep = river_arf_to_probly_sample(arf, x)

    assert rep.n_trees == 4
    assert rep.sample.array.probabilities.shape == (4, rep.n_classes)
    # Each row should sum to 1.
    np.testing.assert_allclose(rep.sample.array.probabilities.sum(axis=-1), 1.0, atol=1e-8)


def test_bma_matches_unweighted_mean(trained_arf):
    arf, stream = trained_arf
    x, _ = next(iter(stream))
    rep = river_arf_to_probly_sample(arf, x)

    bma = rep.bma().probabilities
    expected = rep.sample.array.probabilities.mean(axis=0)
    np.testing.assert_allclose(bma, expected, atol=1e-12)


def test_entropy_decomposition_additive(trained_arf):
    arf, stream = trained_arf
    x, _ = next(iter(stream))
    rep = river_arf_to_probly_sample(arf, x)

    decomp = SecondOrderEntropyDecomposition(rep.sample)
    total = float(decomp.total)
    aleatoric = float(decomp.aleatoric)
    epistemic = float(decomp.epistemic)

    # AdditiveDecomposition: total == aleatoric + epistemic (up to numerical error)
    assert total == pytest.approx(aleatoric + epistemic, abs=1e-10)
    assert epistemic >= -1e-10  # mutual information is non-negative
    assert aleatoric >= -1e-10


def test_metric_weights_plumb_through(trained_arf):
    arf, stream = trained_arf
    x, _ = next(iter(stream))
    rep = river_arf_to_probly_sample(arf, x, use_metric_weights=True)

    assert rep.weights is not None
    assert rep.weights.shape == (arf.n_models,)
    # BMA with weights should still be a valid categorical distribution.
    bma = rep.bma().probabilities
    assert bma.sum() == pytest.approx(1.0)
    assert (bma >= -1e-12).all()
