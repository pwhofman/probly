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


def simplex_embedding(attributions: np.ndarray, mode: str = "contrast") -> np.ndarray:
    """Map each feature's attributions to a point of the probability simplex.

    Args:
        attributions: Attributions of shape ``(n_features, n_classes)``.
        mode: ``"contrast"`` places a feature at the barycenter plus its scaled contrast part. The
            scale is the same for all features and the largest that keeps every point in the
            simplex, so the distance to the barycenter is proportional to the feature's standard
            deviation across the classes. ``"minmax"`` rescales all attributions jointly to
            ``[0, 1]`` and divides each feature's by their sum. Both agree for attributions that
            sum to zero over the classes, as those of class probabilities do.

    Returns:
        Barycentric coordinates of shape ``(n_features, n_classes)``; every row is non-negative and
        sums to one.

    Raises:
        ValueError: If ``mode`` is not known.
    """
    n_classes = attributions.shape[1]
    barycenter = np.full_like(attributions, 1 / n_classes, dtype=float)
    match mode:
        case "contrast":
            directions = contrast(attributions)
            lowest = directions.min()
            return barycenter if lowest >= 0 else barycenter - directions / (n_classes * lowest)
        case "minmax":
            lowest, highest = attributions.min(), attributions.max()
            if highest == lowest:
                return barycenter
            scaled = (attributions - lowest) / (highest - lowest)
            total = scaled.sum(axis=1, keepdims=True)
            return np.divide(scaled, total, out=barycenter, where=total > 0)
        case _:
            msg = f"Unknown mode {mode!r}; use 'contrast' or 'minmax'."
            raise ValueError(msg)
