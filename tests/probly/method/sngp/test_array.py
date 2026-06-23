"""Tests for ``SNGPDecomposition`` on numpy Gaussian representations."""

from __future__ import annotations

import math

import numpy as np
import pytest

from probly.method.sngp import SNGPDecomposition
from probly.quantification import AleatoricUncertainty, EpistemicUncertainty, TotalUncertainty
from probly.quantification.measure.distribution import dempster_shafer_uncertainty
from probly.representation.distribution.array_gaussian import ArrayGaussianDistribution


def _gaussian() -> ArrayGaussianDistribution:
    mean = np.array(
        [
            [0.0, 0.0, 0.0],  # uniform-zero -> u = 1/2
            [10.0, -10.0, 0.0],  # peaked but high var (set below)
            [5.0, 5.0, 5.0],  # peaked + low var -> u close to 0
            [0.0, 0.0, 0.0],  # zero mean again, but with high var
        ],
        dtype=float,
    )
    var = np.array(
        [
            [1.0, 1.0, 1.0],
            [100.0, 100.0, 100.0],  # high var shrinks logits via mean field
            [0.01, 0.01, 0.01],
            [100.0, 100.0, 100.0],
        ],
        dtype=float,
    )
    return ArrayGaussianDistribution(mean=mean, var=var)


def test_array_decomposition_epistemic_matches_measure() -> None:
    distribution = _gaussian()

    decomposition = SNGPDecomposition(distribution)

    np.testing.assert_allclose(
        decomposition.epistemic, dempster_shafer_uncertainty(distribution), rtol=1e-12, atol=1e-12
    )


def test_array_decomposition_components_only_epistemic() -> None:
    decomposition = SNGPDecomposition(_gaussian())

    assert decomposition.components == [EpistemicUncertainty]
    assert len(decomposition) == 1


def test_array_decomposition_canonical_notion_is_epistemic() -> None:
    decomposition = SNGPDecomposition(_gaussian())

    assert decomposition.canonical_notion is EpistemicUncertainty
    np.testing.assert_array_equal(decomposition.get_canonical(), decomposition.epistemic)


def test_array_decomposition_does_not_expose_aleatoric_or_total() -> None:
    """SNGP paper has no aleatoric / total measures; decomposition reflects that."""
    decomposition = SNGPDecomposition(_gaussian())

    with pytest.raises(KeyError):
        decomposition[AleatoricUncertainty]
    with pytest.raises(KeyError):
        decomposition[TotalUncertainty]
    with pytest.raises(KeyError):
        decomposition["au"]
    with pytest.raises(KeyError):
        decomposition["tu"]


def test_array_decomposition_caches_component() -> None:
    decomposition = SNGPDecomposition(_gaussian())

    epistemic = decomposition.epistemic

    assert decomposition.epistemic is epistemic
    assert decomposition[EpistemicUncertainty] is epistemic


def test_array_decomposition_returns_ndarray() -> None:
    decomposition = SNGPDecomposition(_gaussian())

    assert isinstance(decomposition.epistemic, np.ndarray)


def test_array_decomposition_uniform_zero_logits_gives_one_half() -> None:
    """h=0 with default mean-field correction: u = K / (K + K * exp(0)) = 1/2."""
    distribution = ArrayGaussianDistribution(mean=np.zeros((1, 4), dtype=float), var=np.ones((1, 4), dtype=float))

    decomposition = SNGPDecomposition(distribution)

    np.testing.assert_allclose(decomposition.epistemic, 0.5, rtol=1e-12, atol=1e-12)


def test_array_decomposition_high_variance_increases_uncertainty() -> None:
    """Mean-field correction shrinks logits when variance is large -> vacuity goes up."""
    mean = np.array([[10.0, -10.0, 0.0, 0.0]], dtype=float)
    low_var = np.full_like(mean, 1e-3)
    high_var = np.full_like(mean, 1000.0)

    low_score = SNGPDecomposition(ArrayGaussianDistribution(mean=mean, var=low_var)).epistemic
    high_score = SNGPDecomposition(ArrayGaussianDistribution(mean=mean, var=high_var)).epistemic

    assert high_score[0] > low_score[0]


def test_array_decomposition_mean_field_factor_is_configurable() -> None:
    distribution = _gaussian()

    default_score = SNGPDecomposition(distribution).epistemic
    no_mf_score = SNGPDecomposition(distribution, mean_field_factor=0.0).epistemic
    custom_score = SNGPDecomposition(distribution, mean_field_factor=1.0).epistemic

    # All three should differ on at least the rows where variance is not dominated by mean.
    assert not np.allclose(default_score, no_mf_score)
    assert not np.allclose(default_score, custom_score)


def test_array_decomposition_default_mean_field_factor_is_pi_over_eight() -> None:
    distribution = _gaussian()

    default_score = SNGPDecomposition(distribution).epistemic
    explicit_pi_over_8 = SNGPDecomposition(distribution, mean_field_factor=math.pi / 8.0).epistemic

    np.testing.assert_array_equal(default_score, explicit_pi_over_8)
