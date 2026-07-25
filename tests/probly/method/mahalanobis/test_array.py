"""Tests for the numpy Mahalanobis representation and head."""

from __future__ import annotations

import numpy as np
import pytest

from probly.decider import categorical_from_mean
from probly.layers.array import ArrayMahalanobisHead
from probly.method.mahalanobis import create_mahalanobis_representation
from probly.method.mahalanobis.array import ArrayMahalanobisRepresentation
from probly.quantification import decompose
from probly.quantification.notion import AleatoricUncertainty, EpistemicUncertainty
from probly.representation.distribution.array_categorical import (
    ArrayCategoricalDistribution,
    ArrayProbabilityCategoricalDistribution,
)

NUM_CLASSES = 3
FEATURE_DIM = 4
NUM_LAYERS = 2


@pytest.fixture
def softmax() -> ArrayProbabilityCategoricalDistribution:
    """Softmax probabilities for three samples over two classes."""
    return ArrayProbabilityCategoricalDistribution(np.array([[0.2, 0.8], [0.5, 0.5], [0.1, 0.9]]))


@pytest.fixture
def representation(softmax: ArrayProbabilityCategoricalDistribution) -> ArrayMahalanobisRepresentation:
    """A two-layer Mahalanobis representation over three samples."""
    return ArrayMahalanobisRepresentation(
        softmax,
        np.array([[1.0, 2.0], [3.0, 4.0], [0.5, 0.5]]),
        -np.ones(NUM_LAYERS),
        np.zeros(()),
    )


@pytest.fixture
def labelled_features() -> tuple[np.ndarray, np.ndarray]:
    """Well-conditioned features with every class populated."""
    rng = np.random.default_rng(0)
    return rng.normal(size=(120, FEATURE_DIM)), rng.integers(0, NUM_CLASSES, size=120)


class _StubCovarianceEstimator:
    """Minimal duck-typed covariance estimator returning a fixed precision matrix."""

    def __init__(self, precision: np.ndarray) -> None:
        self.precision_ = precision
        self.seen: np.ndarray | None = None

    def fit(self, features: np.ndarray, /) -> _StubCovarianceEstimator:
        """Record the features it was fitted on."""
        self.seen = features
        return self


class TestRepresentation:
    """The ArrayMahalanobisRepresentation dataclass."""

    def test_fields_are_preserved(
        self, representation: ArrayMahalanobisRepresentation, softmax: ArrayProbabilityCategoricalDistribution
    ) -> None:
        """The dataclass stores the values it was built from unchanged."""
        assert representation.softmax is softmax
        assert representation.layer_scores.shape == (3, NUM_LAYERS)
        assert representation.weight.shape == (NUM_LAYERS,)

    def test_factory_dispatches_to_array_backend(self, softmax: ArrayProbabilityCategoricalDistribution) -> None:
        """A numpy-backed softmax selects the numpy representation implementation."""
        created = create_mahalanobis_representation(
            softmax, np.zeros((3, NUM_LAYERS)), -np.ones(NUM_LAYERS), np.zeros(())
        )
        assert isinstance(created, ArrayMahalanobisRepresentation)

    def test_categorical_from_mean_returns_softmax(self, representation: ArrayMahalanobisRepresentation) -> None:
        """The categorical mean decider reduces the representation to its softmax."""
        single = categorical_from_mean(representation)
        assert isinstance(single, ArrayCategoricalDistribution)
        assert single is representation.softmax

    def test_indexing_preserves_combiner(self, representation: ArrayMahalanobisRepresentation) -> None:
        """Indexing batches the per-sample fields but keeps the shared combiner weights."""
        sub = representation[:2]
        assert sub.layer_scores.shape == (2, NUM_LAYERS)
        assert np.array_equal(sub.weight, representation.weight)
        assert np.array_equal(sub.bias, representation.bias)

    def test_decomposition_is_epistemic_only(self, representation: ArrayMahalanobisRepresentation) -> None:
        """The decomposition exposes the combined OOD score as epistemic uncertainty only."""
        decomposition = decompose(representation)
        expected = representation.layer_scores @ representation.weight + representation.bias
        np.testing.assert_allclose(decomposition[EpistemicUncertainty], expected)
        with pytest.raises(KeyError):
            _ = decomposition[AleatoricUncertainty]


class TestArrayMahalanobisHead:
    """The numpy class-conditional Gaussian head."""

    def test_fit_populates_parameters(self, labelled_features: tuple[np.ndarray, np.ndarray]) -> None:
        """Fitting sets non-trivial class means and a non-identity precision."""
        features, labels = labelled_features
        head = ArrayMahalanobisHead(NUM_CLASSES, FEATURE_DIM)
        head.fit(features, labels)
        assert head.means.shape == (NUM_CLASSES, FEATURE_DIM)
        assert np.any(head.means != 0)
        assert not np.allclose(head.precision, np.eye(FEATURE_DIM))

    def test_score_shape_and_sign(self, labelled_features: tuple[np.ndarray, np.ndarray]) -> None:
        """Scores are non-positive per-class confidences of shape (N, num_classes)."""
        features, labels = labelled_features
        head = ArrayMahalanobisHead(NUM_CLASSES, FEATURE_DIM)
        head.fit(features, labels)
        scores = head.score(features)
        assert scores.shape == (len(features), NUM_CLASSES)
        assert np.all(scores <= 0)
        np.testing.assert_allclose(head(features), scores)

    def test_far_samples_score_lower(self, labelled_features: tuple[np.ndarray, np.ndarray]) -> None:
        """Samples far from every class centroid get a lower confidence."""
        features, labels = labelled_features
        head = ArrayMahalanobisHead(NUM_CLASSES, FEATURE_DIM)
        head.fit(features, labels)
        far = features * 5 + 20
        assert head.score(far).max(axis=-1).mean() < head.score(features).max(axis=-1).mean()

    def test_fit_without_matching_labels_raises(self, labelled_features: tuple[np.ndarray, np.ndarray]) -> None:
        """Fitting when no sample matches any class index raises a clear error."""
        features, _ = labelled_features
        head = ArrayMahalanobisHead(NUM_CLASSES, FEATURE_DIM)
        out_of_range = np.full(len(features), NUM_CLASSES)
        with pytest.raises(ValueError, match="no labelled samples"):
            head.fit(features, out_of_range)

    def test_covariance_estimator_overrides_precision(self, labelled_features: tuple[np.ndarray, np.ndarray]) -> None:
        """A supplied covariance estimator provides the precision matrix."""
        features, labels = labelled_features
        precision = np.full((FEATURE_DIM, FEATURE_DIM), 0.25)
        estimator = _StubCovarianceEstimator(precision)
        head = ArrayMahalanobisHead(NUM_CLASSES, FEATURE_DIM)
        head.fit(features, labels, covariance_estimator=estimator)
        np.testing.assert_allclose(head.precision, precision)
        # The estimator sees the per-class-centered features, not the raw ones.
        assert estimator.seen is not None
        np.testing.assert_allclose(estimator.seen.mean(axis=0), 0.0, atol=1e-12)
