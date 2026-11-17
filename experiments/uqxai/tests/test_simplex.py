"""Tests of the class-contrastive analysis of attributions."""

from __future__ import annotations

import numpy as np
import pytest
from uqxai import simplex


@pytest.fixture
def attributions() -> np.ndarray:
    return np.random.default_rng(0).normal(size=(40, 3))


def test_magnitude_splits_into_common_and_contrast(attributions: np.ndarray) -> None:
    squared = simplex.common_magnitude(attributions) ** 2 + simplex.contrast_magnitude(attributions) ** 2
    assert np.allclose(squared, simplex.magnitude(attributions) ** 2)
    assert np.allclose(simplex.contrast_magnitude(attributions) ** 2, 3 * attributions.var(axis=1))
    assert np.allclose(simplex.contrast(attributions).sum(axis=1), 0)


def test_embeddings_are_points_of_the_simplex(attributions: np.ndarray) -> None:
    for mode in ("contrast", "minmax"):
        points = simplex.simplex_embedding(attributions, mode=mode)
        assert points.min() >= 0
        assert np.allclose(points.sum(axis=1), 1)


def test_embeddings_agree_for_attributions_that_sum_to_zero(attributions: np.ndarray) -> None:
    zero_sum = simplex.contrast(attributions)
    assert np.allclose(simplex.simplex_embedding(zero_sum, "contrast"), simplex.simplex_embedding(zero_sum, "minmax"))


def test_contrast_embedding_distance_is_proportional_to_the_standard_deviation(attributions: np.ndarray) -> None:
    distance = np.linalg.norm(simplex.simplex_embedding(attributions) - 1 / 3, axis=1)
    ratio = distance / attributions.std(axis=1)
    assert np.allclose(ratio, ratio[0])


def test_supported_classes_and_contrast_share() -> None:
    examples = np.array([[2.0, -1.0, -1.0], [1.0, 1.0, -2.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    assert np.allclose(simplex.supported_classes(examples), [1, 2, 0, 0])
    assert np.allclose(simplex.contrast_share(examples), [1, 1, 0, 0])


def test_regimes() -> None:
    examples = np.array([[0.01, -0.01, 0.0], [2.0, -1.0, -1.0], [1.0, 1.1, 0.9]])
    assert simplex.regimes(examples, background_threshold=0.1).tolist() == [
        simplex.BACKGROUND,
        simplex.SPECIFIC,
        simplex.SHARED,
    ]


def test_structure_separates_conflict_from_ambiguity() -> None:
    # Conflict: half of the features favor the first class, half the second.
    conflict = np.array([[2.0, -1.0, -1.0], [-1.0, 2.0, -1.0], [0.01, 0.0, -0.01]])
    # Ambiguity: every feature favors the second and third class alike.
    ambiguous = np.array([[-2.0, 1.0, 1.0], [-2.0, 1.0, 1.0], [0.01, 0.0, -0.01]])
    summary = simplex.structure(conflict, background_threshold=0.1)
    assert np.allclose(summary["magnitude_share_per_class"], [0.5, 0.5, 0])
    assert np.isclose(summary["supported_classes"], 1)
    assert np.isclose(summary["multi_class_share"], 0)
    summary = simplex.structure(ambiguous, background_threshold=0.1)
    assert np.isclose(summary["supported_classes"], 2)
    assert np.isclose(summary["multi_class_share"], 1)


@pytest.mark.parametrize("scale", ["linear", "power", "softmax", "direction"])
def test_scales_keep_directions_and_stay_in_the_simplex(attributions: np.ndarray, scale: str) -> None:
    floor = np.median(simplex.contrast_magnitude(attributions))
    points = simplex.simplex_embedding(attributions, scale=scale, noise_floor=floor)
    assert points.min() >= -1e-12
    assert np.allclose(points.sum(axis=1), 1)
    below = simplex.contrast_magnitude(attributions) <= floor
    assert np.allclose(points[below], 1 / 3)
    # Features above the floor keep the order of their classes.
    order = np.argsort(simplex.contrast(attributions)[~below], axis=1)
    assert np.array_equal(np.argsort(points[~below], axis=1), order)
    if scale == "direction":
        assert np.allclose(points[~below].min(axis=1), 0)


def test_noise_floor_of_two_noisy_estimates() -> None:
    rng = np.random.default_rng(0)
    first, second = (simplex.contrast(rng.normal(scale=0.1, size=(20_000, 3))) for _ in range(2))
    floor = simplex.noise_floor(first, second, quantile=0.99)
    mean = (first + second) / 2
    assert np.isclose(np.mean(simplex.contrast_magnitude(mean) < floor), 0.99, atol=0.005)
    # Without a second estimate, the bound is close for pure noise.
    assert np.isclose(simplex.robust_noise_floor(mean), floor, rtol=0.05)
