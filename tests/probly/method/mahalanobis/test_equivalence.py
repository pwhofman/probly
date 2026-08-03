"""Tests verifying equivalence between the NumPy and Torch Mahalanobis heads."""

from __future__ import annotations

import pytest

pytest.importorskip("sklearn")
torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402
from sklearn.covariance import EmpiricalCovariance  # noqa: E402

from probly.layers.array import ArrayMahalanobisHead  # noqa: E402
from probly.layers.torch import MahalanobisHead  # noqa: E402

NUM_CLASSES = 3
FEATURE_DIM = 5
NUM_SAMPLES = 200


@pytest.fixture
def labelled_features() -> tuple[np.ndarray, np.ndarray]:
    """Well-conditioned float64 features with every class populated."""
    rng = np.random.default_rng(123)
    return rng.standard_normal((NUM_SAMPLES, FEATURE_DIM)), rng.integers(0, NUM_CLASSES, size=NUM_SAMPLES)


def test_numpy_and_torch_heads_agree(labelled_features: tuple[np.ndarray, np.ndarray]) -> None:
    """Both backends estimate the same Gaussian parameters and confidences.

    The torch head is promoted to float64 because ``pinv`` picks its rank cutoff
    from the dtype, which would otherwise dominate the comparison.
    """
    features, labels = labelled_features

    array_head = ArrayMahalanobisHead(NUM_CLASSES, FEATURE_DIM)
    array_head.fit(features, labels)

    torch_head = MahalanobisHead(NUM_CLASSES, FEATURE_DIM).double()
    torch_head.fit(torch.tensor(features), torch.tensor(labels))

    np.testing.assert_allclose(array_head.means, torch_head.means.numpy(), atol=1e-10)
    np.testing.assert_allclose(array_head.precision, torch_head.precision.numpy(), atol=1e-8)

    with torch.no_grad():
        torch_scores = torch_head(torch.tensor(features)).numpy()
    np.testing.assert_allclose(array_head.score(features), torch_scores, atol=1e-8)


def test_empirical_covariance_matches_the_pseudo_inverse(
    labelled_features: tuple[np.ndarray, np.ndarray],
) -> None:
    """Sklearn's covariance estimator reproduces the default pseudo-inverse precision.

    The reference implementation of :cite:`leeSimpleUnifiedFramework2018` derives
    the tied precision from ``EmpiricalCovariance``; this pins that path to the
    backend-agnostic default.
    """
    features, labels = labelled_features

    default_head = ArrayMahalanobisHead(NUM_CLASSES, FEATURE_DIM)
    default_head.fit(features, labels)

    sklearn_head = ArrayMahalanobisHead(NUM_CLASSES, FEATURE_DIM)
    sklearn_head.fit(features, labels, covariance_estimator=EmpiricalCovariance(assume_centered=True))

    np.testing.assert_allclose(sklearn_head.precision, default_head.precision, atol=1e-10)
