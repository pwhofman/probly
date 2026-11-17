"""Class-contrastive analysis of attributions: where a feature points, and how strongly.

A feature ``i`` has one attribution per class, a vector ``phi_i`` of ``C`` numbers. It splits into

- its *common* part ``mean_c(phi_i)``, the amount by which the feature moves all classes alike, and
- its *contrast* part ``phi_i - mean_c(phi_i)``, which sums to zero over the classes and says
  between which classes the feature discriminates.

The two parts are orthogonal, so the squared magnitude splits as
``|phi_i|^2 = C * mean_c(phi_i)^2 + C * var_c(phi_i)``: the variance of a feature's attributions
across the classes *is* its squared contrast. This gives three kinds of features:

1. background: small magnitude,
2. class-specific: large magnitude that is mostly contrast (high variance across the classes),
3. shared: large magnitude that is mostly common (low variance across the classes).

The contrast part lives in the plane of the probability simplex, so for three classes it is drawn
in a ternary plot; see :func:`simplex_embedding`.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi

BACKGROUND, SPECIFIC, SHARED = 0, 1, 2
REGIME_NAMES = ("background", "class-specific", "shared")


def common(attributions: np.ndarray) -> np.ndarray:
    """Return the common part of each feature: the mean of its attributions over the classes.

    Args:
        attributions: Attributions of shape ``(n_features, n_classes)``.

    Returns:
        The common parts of shape ``(n_features,)``.
    """
    return attributions.mean(axis=1)


def contrast(attributions: np.ndarray) -> np.ndarray:
    """Return the contrast part of each feature: its attributions minus their mean over the classes.

    Args:
        attributions: Attributions of shape ``(n_features, n_classes)``.

    Returns:
        The contrast parts of shape ``(n_features, n_classes)``; every row sums to zero.
    """
    return attributions - attributions.mean(axis=1, keepdims=True)


def magnitude(attributions: np.ndarray) -> np.ndarray:
    """Return the Euclidean norm of each feature's attributions over the classes."""
    return np.linalg.norm(attributions, axis=1)


def common_magnitude(attributions: np.ndarray) -> np.ndarray:
    """Return the signed norm of the common part, ``sqrt(C) * mean_c(phi_i)``."""
    return np.sqrt(attributions.shape[1]) * common(attributions)


def contrast_magnitude(attributions: np.ndarray) -> np.ndarray:
    """Return the norm of the contrast part, ``sqrt(C * var_c(phi_i))``."""
    return magnitude(contrast(attributions))


def contrast_share(attributions: np.ndarray) -> np.ndarray:
    """Return the fraction of each feature's squared magnitude that is contrast.

    This is the variance across the classes, normalized to ``[0, 1]``: ``1`` for a feature that only
    discriminates between classes, ``0`` for a feature that moves all classes alike. Features
    without any attribution get ``0``.
    """
    squared = magnitude(attributions) ** 2
    return np.divide(contrast_magnitude(attributions) ** 2, squared, out=np.zeros_like(squared), where=squared > 0)


def supported_classes(attributions: np.ndarray) -> np.ndarray:
    """Return the effective number of classes each feature speaks for.

    This is the participation ratio ``sum(p)^2 / sum(p^2)`` of the positive parts ``p`` of the
    contrast: ``1`` if the feature favors one class over all others, ``k`` if it favors ``k``
    classes equally. Features without contrast get ``0``.
    """
    positive = np.clip(contrast(attributions), 0, None)
    squares = np.sum(positive**2, axis=1)
    return np.divide(positive.sum(axis=1) ** 2, squares, out=np.zeros_like(squares), where=squares > 0)


def regimes(attributions: np.ndarray, background_threshold: float, shared_threshold: float = 0.5) -> np.ndarray:
    """Assign each feature to one of the three regimes.

    Args:
        attributions: Attributions of shape ``(n_features, n_classes)``.
        background_threshold: Features with a magnitude below this value are background.
        shared_threshold: Among the other features, those with a contrast share below this value
            are shared, and the rest class-specific.

    Returns:
        The regime of each feature: ``BACKGROUND``, ``SPECIFIC`` or ``SHARED``.
    """
    regime = np.where(contrast_share(attributions) < shared_threshold, SHARED, SPECIFIC)
    regime[magnitude(attributions) < background_threshold] = BACKGROUND
    return regime


def structure(attributions: np.ndarray, background_threshold: float) -> dict[str, object]:
    """Summarize where the features of an explanation point, weighted by their magnitude.

    Two explanations of equally uncertain predictions can differ in this summary: if different
    features favor different classes, the magnitude is split over the classes but each feature
    supports one class; if the same features favor several classes, the features themselves support
    more than one class.

    Args:
        attributions: Attributions of shape ``(n_features, n_classes)``.
        background_threshold: Features with a magnitude below this value are left out.

    Returns:
        ``magnitude_share_per_class``: the share of the magnitude of the features that favor each
        class most; ``supported_classes``: the mean of :func:`supported_classes`;
        ``multi_class_share``: the share of the magnitude of features that support at least 1.5
        classes. All are weighted by the features' magnitude. ``effective_features`` is the
        participation ratio of the magnitudes: the number of features that carry the explanation.
    """
    weights = np.where(magnitude(attributions) >= background_threshold, magnitude(attributions), 0.0)
    effective_features = weights.sum() ** 2 / max(np.sum(weights**2), 1e-12)
    weights = weights / max(weights.sum(), 1e-12)
    favored = contrast(attributions).argmax(axis=1)
    supported = supported_classes(attributions)
    multi_class = 1.5
    return {
        "magnitude_share_per_class": np.bincount(favored, weights=weights, minlength=attributions.shape[1]).tolist(),
        "supported_classes": float(weights @ supported),
        "multi_class_share": float(weights[supported >= multi_class].sum()),
        "effective_features": float(effective_features),
    }


def noise_floor(first: np.ndarray, second: np.ndarray, quantile: float = 0.99) -> float:
    """Measure the magnitude below which a feature of the mean of two estimates is noise.

    The two estimates come from independent coalition samples. The noise of the contrast part is
    taken as Gaussian with one standard deviation for all features and directions, estimated from
    the difference of the estimates.

    Args:
        first: Attributions of shape ``(n_features, n_classes)`` from one coalition sample.
        second: Attributions of the same game from an independent coalition sample.
        quantile: Share of pure-noise features whose contrast magnitude lies below the floor.

    Returns:
        The noise floor of the contrast magnitude of ``(first + second) / 2``.
    """
    dimensions = first.shape[1] - 1  # the contrast part lives in a space of C - 1 dimensions
    difference = contrast(first) - contrast(second)
    deviation = np.sqrt(np.mean(np.sum(difference**2, axis=1)) / (2 * dimensions))  # of one estimate
    return float(deviation / np.sqrt(2) * chi(dimensions).ppf(quantile))


def robust_noise_floor(attributions: np.ndarray, quantile: float = 0.99) -> float:
    """Bound the noise floor of a single estimate from above, without a second coalition sample.

    Takes the median contrast magnitude as the median of pure noise, which holds if at least half of
    the features have no effect. Features with a small true effect raise the median, so the floor
    comes out too high: about twice the measured floor in the explanations of this experiment.

    Args:
        attributions: Attributions of shape ``(n_features, n_classes)``.
        quantile: Share of pure-noise features whose contrast magnitude lies below the floor.

    Returns:
        An upper bound of the noise floor of the contrast magnitude.
    """
    noise = chi(attributions.shape[1] - 1)
    return float(np.median(contrast_magnitude(attributions)) / noise.median() * noise.ppf(quantile))


def simplex_embedding(
    attributions: np.ndarray,
    mode: str = "contrast",
    *,
    scale: str = "linear",
    noise_floor: float = 0.0,
    exponent: float = 0.5,
    sharpness: float = 3.0,
) -> np.ndarray:
    """Map each feature's attributions to a point of the probability simplex.

    Args:
        attributions: Attributions of shape ``(n_features, n_classes)``.
        mode: ``"contrast"`` places a feature at the barycenter, displaced in the direction of its
            contrast part by a distance that ``scale`` computes from the contrast magnitude.
            ``"minmax"`` rescales all attributions jointly to ``[0, 1]`` and divides each feature's
            by their sum; it ignores the other arguments. Both agree for the defaults and
            attributions that sum to zero over the classes, as those of class probabilities do.
        scale: How the contrast magnitude becomes a distance to the barycenter.
            ``"linear"``: proportional, so the distance shows the standard deviation across the
            classes; the feature that leaves the simplex first touches its boundary.
            ``"power"``: proportional to the relative magnitude to the power of ``exponent``.
            ``"softmax"``: the softmax of the contrast, so that the strongest feature in any
            direction lies near the boundary; features for one class reach its corner.
            ``"direction"``: every feature lies on the boundary; only its direction is shown.
        noise_floor: Contrast magnitude that counts as noise. It is subtracted from every magnitude,
            so features below it lie at the barycenter and the others start there.
        exponent: Exponent of the ``"power"`` scale: below one spreads weak features outwards,
            above one pulls them to the barycenter.
        sharpness: Sharpness of the ``"softmax"`` scale: the logit gap of the strongest feature.

    Returns:
        Barycentric coordinates of shape ``(n_features, n_classes)``; every row is non-negative and
        sums to one.

    Raises:
        ValueError: If ``mode`` or ``scale`` is not known.
    """
    n_classes = attributions.shape[1]
    barycenter = np.full_like(attributions, 1 / n_classes, dtype=float)
    if mode == "minmax":
        lowest, highest = attributions.min(), attributions.max()
        if highest == lowest:
            return barycenter
        scaled = (attributions - lowest) / (highest - lowest)
        total = scaled.sum(axis=1, keepdims=True)
        return np.divide(scaled, total, out=barycenter, where=total > 0)
    if mode != "contrast":
        msg = f"Unknown mode {mode!r}; use 'contrast' or 'minmax'."
        raise ValueError(msg)

    directions = contrast(attributions)
    radius = np.linalg.norm(directions, axis=1, keepdims=True)
    above = np.clip(radius - noise_floor, 0, None)
    if above.max() == 0:
        return barycenter
    unit = np.divide(directions, radius, out=np.zeros_like(directions, dtype=float), where=radius > 0)
    relative = above / above.max()
    match scale:
        case "linear":
            vectors = relative * unit
        case "power":
            vectors = relative**exponent * unit
        case "softmax":
            logits = sharpness * relative * unit
            exponentials = np.exp(logits - logits.max(axis=1, keepdims=True))
            return exponentials / exponentials.sum(axis=1, keepdims=True)
        case "direction":
            lowest = unit.min(axis=1, keepdims=True)  # negative for every feature with a direction
            on_boundary = barycenter - unit / (n_classes * np.where(lowest < 0, lowest, -1))
            return np.where(above > 0, on_boundary, barycenter)
        case _:
            msg = f"Unknown scale {scale!r}; use 'linear', 'power', 'softmax' or 'direction'."
            raise ValueError(msg)
    return barycenter - vectors / (n_classes * vectors.min())
