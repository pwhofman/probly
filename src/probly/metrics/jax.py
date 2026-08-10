"""JAX implementation of Metrics.

Note:
    There is no ``JaxSingletonCredalSet`` or ``JaxDiscreteCredalSet`` in
    :mod:`probly.representation.credal_set.jax`; for those semantics, use the
    numpy-side ``ArraySingletonCredalSet`` / ``ArrayDiscreteCredalSet`` types.
    The remaining jax credal sets (Convex, DistanceBased, ProbabilityIntervals,
    DirichletLevelSet) all use the interval-dominance rule via their
    ``lower()`` / ``upper()`` envelopes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from probly.metrics._common import CREDAL_ROUND_DECIMALS
from probly.metrics.array import _convex_hull_lp_coverage
from probly.representation.credal_set.jax import (
    JaxConvexCredalSet,
    JaxDirichletLevelSetCredalSet,
    JaxDistanceBasedCredalSet,
    JaxProbabilityIntervalsCredalSet,
)

from ._common import (
    accuracy,
    auc,
    average_interval_width,
    average_precision_score,
    convex_hull_coverage,
    coverage,
    efficiency,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

if TYPE_CHECKING:
    from probly.representation.distribution.jax_categorical import JaxCategoricalDistribution


@accuracy.register(jax.Array)
def accuracy_jax(y_pred: jax.Array, y_true: jax.Array) -> jax.Array:
    """Compute top-1 classification accuracy for JAX arrays."""
    labels = y_true.reshape(-1)
    predicted = y_pred
    if predicted.ndim == 2:
        predicted = jnp.argmax(predicted, axis=-1)
    elif predicted.ndim != 1:
        msg = f"accuracy expects predictions of shape (n,) or (n, k), got shape {predicted.shape}."
        raise ValueError(msg)
    if predicted.shape[0] != labels.shape[0]:
        msg = (
            "accuracy labels must match predictions batch size. "
            f"Got {labels.shape[0]} labels for {predicted.shape[0]} predictions."
        )
        raise ValueError(msg)
    return (predicted == labels).astype(jnp.float32).mean()


@classwise_ece.register(jax.Array)
def classwise_ece_jax(y_prob: jax.Array, y_true: jax.Array, *, num_bins: int = 15) -> jax.Array:
    """Compute the classwise expected calibration error for JAX arrays."""
    probs = y_prob.astype(jnp.float32)
    if probs.ndim != 2:
        msg = f"classwise_ece expects probabilities of shape (n, k), got shape {probs.shape}."
        raise ValueError(msg)
    labels = y_true.reshape(-1)
    n, k = probs.shape
    if labels.shape[0] != n:
        msg = f"classwise_ece labels must match probabilities batch size. Got {labels.shape[0]} labels for {n} rows."
        raise ValueError(msg)
    if num_bins < 1:
        msg = f"classwise_ece expects num_bins >= 1, got {num_bins}."
        raise ValueError(msg)

    one_hot = (labels[:, None] == jnp.arange(k)[None, :]).astype(jnp.float32)
    bin_idx = jnp.minimum((probs * num_bins).astype(jnp.int32), num_bins - 1)

    total = jnp.float32(0.0)
    for b in range(num_bins):
        mask = (bin_idx == b).astype(jnp.float32)
        count = mask.sum(axis=0)
        safe_count = jnp.where(count > 0, count, 1.0)
        mean_prob = (probs * mask).sum(axis=0) / safe_count
        freq = (one_hot * mask).sum(axis=0) / safe_count
        total = total + (count / n * jnp.abs(freq - mean_prob)).sum()
    return total / k


@expected_calibration_error.register(jax.Array)
def expected_calibration_error_jax(y_prob: jax.Array, y_true: jax.Array, *, num_bins: int = 15) -> jax.Array:
    """Compute the confidence expected calibration error for JAX arrays."""
    probs = y_prob.astype(jnp.float32)
    if probs.ndim != 2:
        msg = f"expected_calibration_error expects probabilities of shape (n, k), got shape {probs.shape}."
        raise ValueError(msg)
    labels = y_true.reshape(-1)
    n = probs.shape[0]
    if labels.shape[0] != n:
        msg = (
            "expected_calibration_error labels must match probabilities batch size. "
            f"Got {labels.shape[0]} labels for {n} rows."
        )
        raise ValueError(msg)
    if num_bins < 1:
        msg = f"expected_calibration_error expects num_bins >= 1, got {num_bins}."
        raise ValueError(msg)

    conf = probs.max(axis=-1)
    correct = (probs.argmax(axis=-1) == labels).astype(jnp.float32)
    bin_idx = jnp.minimum((conf * num_bins).astype(jnp.int32), num_bins - 1)

    total = jnp.float32(0.0)
    for b in range(num_bins):
        mask = (bin_idx == b).astype(jnp.float32)
        count = mask.sum()
        safe_count = jnp.where(count > 0, count, 1.0)
        acc_bin = (correct * mask).sum() / safe_count
        conf_bin = (conf * mask).sum() / safe_count
        total = total + count / n * jnp.abs(acc_bin - conf_bin)
    return total


@false_positive_rate.register(jax.Array)
def false_positive_rate_jax(y_pred: jax.Array, y_true: jax.Array) -> jax.Array:
    """Compute the false positive rate for JAX arrays."""
    y = y_true.reshape(-1)
    p = y_pred.reshape(-1)
    if p.shape[0] != y.shape[0]:
        msg = f"false_positive_rate y_true and y_pred must match in batch size. Got {y.shape[0]} and {p.shape[0]}."
        raise ValueError(msg)
    negatives = (y == 0).astype(jnp.float32)
    false_positives = (p == 1).astype(jnp.float32) * negatives
    # 0/0 yields NaN when there are no negative samples.
    return false_positives.sum() / negatives.sum()


@false_negative_rate.register(jax.Array)
def false_negative_rate_jax(y_pred: jax.Array, y_true: jax.Array) -> jax.Array:
    """Compute the false negative rate for JAX arrays."""
    y = y_true.reshape(-1)
    p = y_pred.reshape(-1)
    if p.shape[0] != y.shape[0]:
        msg = f"false_negative_rate y_true and y_pred must match in batch size. Got {y.shape[0]} and {p.shape[0]}."
        raise ValueError(msg)
    positives = (y == 1).astype(jnp.float32)
    false_negatives = (p == 0).astype(jnp.float32) * positives
    # 0/0 yields NaN when there are no positive samples.
    return false_negatives.sum() / positives.sum()


@auc.register(jax.Array)
def auc_jax(x: jax.Array, y: jax.Array) -> jax.Array:
    """Compute area under a curve using the trapezoid rule."""
    return jnp.trapezoid(y, x, axis=-1)


@average_precision_score.register(jax.Array)
def average_precision_score_jax(y_true: jax.Array, y_score: jax.Array) -> jax.Array:
    """Compute average precision for JAX arrays."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return -jnp.sum(jnp.diff(recall, axis=-1) * precision[..., :-1], axis=-1)  # ty:ignore[invalid-argument-type, not-subscriptable]


@precision_recall_curve.register(jax.Array)
def precision_recall_curve_jax(y_true: jax.Array, y_score: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute precision-recall curve along the last axis."""
    y_true = y_true.astype(jnp.float32)
    y_score = y_score.astype(jnp.float32)
    n = y_score.shape[-1]

    desc_idx = jnp.flip(jnp.argsort(y_score, axis=-1, stable=True), axis=-1)
    y_score_sorted = jnp.take_along_axis(y_score, desc_idx, axis=-1)
    y_true_sorted = jnp.take_along_axis(y_true, desc_idx, axis=-1)

    tps = jnp.cumsum(y_true_sorted, axis=-1)
    predicted_pos = jnp.arange(1, n + 1, dtype=jnp.float32)
    total_pos = tps[..., -1:]

    precision = tps / predicted_pos
    recall = jnp.where(total_pos > 0, tps / jnp.where(total_pos > 0, total_pos, 1.0), 0.0)

    ones = jnp.ones((*y_score.shape[:-1], 1), dtype=jnp.float32)
    zeros = jnp.zeros((*y_score.shape[:-1], 1), dtype=jnp.float32)
    precision = jnp.concatenate([jnp.flip(precision, axis=-1), ones], axis=-1)
    recall = jnp.concatenate([jnp.flip(recall, axis=-1), zeros], axis=-1)

    return precision, recall, y_score_sorted


@roc_curve.register(jax.Array)
def roc_curve_jax(y_true: jax.Array, y_score: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute ROC curve along the last axis."""
    y_true = y_true.astype(jnp.float32)
    y_score = y_score.astype(jnp.float32)
    n = y_score.shape[-1]

    desc_idx = jnp.flip(jnp.argsort(y_score, axis=-1, stable=True), axis=-1)
    y_score_sorted = jnp.take_along_axis(y_score, desc_idx, axis=-1)
    y_true_sorted = jnp.take_along_axis(y_true, desc_idx, axis=-1)

    tps = jnp.cumsum(y_true_sorted, axis=-1)
    fps = jnp.arange(1, n + 1, dtype=jnp.float32) - tps

    total_pos = tps[..., -1:]
    total_neg = fps[..., -1:]

    tpr = jnp.where(total_pos > 0, tps / jnp.where(total_pos > 0, total_pos, 1.0), 0.0)
    fpr = jnp.where(total_neg > 0, fps / jnp.where(total_neg > 0, total_neg, 1.0), 0.0)

    zeros = jnp.zeros((*y_score.shape[:-1], 1), dtype=jnp.float32)
    tpr = jnp.concatenate([zeros, tpr], axis=-1)
    fpr = jnp.concatenate([zeros, fpr], axis=-1)
    thresholds = jnp.concatenate([y_score_sorted[..., :1] + 1, y_score_sorted], axis=-1)

    return fpr, tpr, thresholds


@roc_auc_score.register(jax.Array)
def roc_auc_score_jax(y_true: jax.Array, y_score: jax.Array) -> jax.Array:
    """Compute area under the ROC curve for JAX arrays."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)  # ty:ignore[invalid-return-type]


# --- Predicted-set metrics ----------------------------------------------------


def _interval_dominance_mask(lower: jax.Array, upper: jax.Array) -> jax.Array:
    """Return the boolean class mask selected by the interval-dominance rule.

    Args:
        lower: Lower probability envelope of shape ``(..., C)``.
        upper: Upper probability envelope of shape ``(..., C)``.

    Returns:
        Boolean array of shape ``(..., C)``.
    """
    threshold = jnp.max(lower, axis=-1, keepdims=True)
    return upper >= threshold


def _onehot_membership(mask: jax.Array, y_true: object) -> jax.Array:
    """Look up the membership flag for the true class along the last axis."""
    indices = jnp.expand_dims(jnp.asarray(y_true).astype(jnp.int32), axis=-1)
    return jnp.squeeze(jnp.take_along_axis(mask, indices, axis=-1), axis=-1)


def _envelope_coverage(lower: jax.Array, upper: jax.Array, y_true: object) -> float:
    mask = _interval_dominance_mask(lower, upper)
    return float(jnp.mean(_onehot_membership(mask, y_true).astype(jnp.float32)))


def _envelope_efficiency(lower: jax.Array, upper: jax.Array) -> float:
    mask = _interval_dominance_mask(lower, upper)
    return float(jnp.mean(jnp.sum(mask.astype(jnp.float32), axis=-1)))


def _envelope_average_interval_width(lower: jax.Array, upper: jax.Array) -> float:
    return float(jnp.mean((upper - lower).astype(jnp.float32)))


def _is_first_order_target(y_true: object, num_classes: int) -> bool:
    """Detect a first-order target array (probability vectors) vs class indices."""
    y = jnp.asarray(y_true)
    return y.ndim >= 2 and y.shape[-1] == num_classes


def _credal_containment_coverage_jax(lower: jax.Array, upper: jax.Array, y_true: object) -> float:
    """Fraction of instances whose target lies inside the credal set's envelope.

    Dispatches on the shape of ``y_true``:

    * **Integer class labels** (``y_true.ndim == lower.ndim - 1``): covered when
      the true class is selected by the interval-dominance rule, i.e.
      ``upper[y] >= max_k lower[k]``.
    * **Probability vectors** (``y_true.ndim == lower.ndim``): covered when
      ``lower[k] <= y_true[k] <= upper[k]`` for all classes ``k``.

    Args:
        lower: Lower probability envelope of shape ``(..., C)``.
        upper: Upper probability envelope of shape ``(..., C)``.
        y_true: Integer class labels of shape ``(...)`` or target probability
            arrays of shape ``(..., C)``.

    Returns:
        Mean containment indicator as a Python float.
    """
    y = jnp.asarray(y_true)
    if y.ndim < lower.ndim:
        covered = _onehot_membership(_interval_dominance_mask(lower, upper), y)
    else:
        scale = 10**CREDAL_ROUND_DECIMALS
        lower_r = jnp.round(lower * scale) / scale
        upper_r = jnp.round(upper * scale) / scale
        y = y.astype(lower.dtype)
        covered = jnp.all((lower_r <= y) & (y <= upper_r), axis=-1)
    return float(jnp.mean(covered.astype(jnp.float32)))


def _credal_interval_efficiency_jax(lower: jax.Array, upper: jax.Array) -> float:
    """Efficiency of a credal set as ``1 - mean(upper - lower)``.

    Bounds are rounded to ``CREDAL_ROUND_DECIMALS`` decimals before subtracting
    so floating-point residuals (e.g. softmax outputs near 0) do not perturb
    the width.

    Args:
        lower: Lower probability envelope of shape ``(N, C)``.
        upper: Upper probability envelope of shape ``(N, C)``.

    Returns:
        Scalar in ``(-inf, 1]``; higher means a tighter (more efficient) credal set.
    """
    scale = 10**CREDAL_ROUND_DECIMALS
    lower_r = jnp.round(lower * scale) / scale
    upper_r = jnp.round(upper * scale) / scale
    return float(1.0 - jnp.mean((upper_r - lower_r).astype(jnp.float32)))


@coverage.register(JaxConvexCredalSet)
def _coverage_jax_convex(y_pred: JaxConvexCredalSet, y_true: object) -> float:
    """Containment coverage for a convex credal set.

    Args:
        y_pred: Convex credal set.
        y_true: Target probability arrays of shape ``(N, C)`` or class indices.

    Returns:
        Fraction of instances where the target lies in ``[lower, upper]`` for all classes.
    """
    return _credal_containment_coverage_jax(y_pred.lower(), y_pred.upper(), y_true)


@efficiency.register(JaxConvexCredalSet)
def _efficiency_jax_convex(y_pred: JaxConvexCredalSet) -> float:
    """Interval-width efficiency for a convex credal set: ``1 - mean(upper - lower)``.

    Returns:
        Scalar efficiency; higher means a tighter credal set.
    """
    return _credal_interval_efficiency_jax(y_pred.lower(), y_pred.upper())


@coverage.register(JaxDistanceBasedCredalSet)
def _coverage_jax_distance(y_pred: JaxDistanceBasedCredalSet, y_true: object) -> float:
    """Coverage for a distance-based (TV-ball) credal set.

    With class-index targets: interval-dominance coverage on the envelope.

    With first-order targets ``y_true`` of shape ``(..., C)``: the paper's
    distribution-membership coverage ``mean(lambda_i in Q_i)``, computed as
    ``mean(TV(nominal_i, lambda_i) <= radius_i)``.
    """
    if _is_first_order_target(y_true, y_pred.num_classes):
        nominal = y_pred.nominal.probabilities
        target = jnp.asarray(y_true).astype(nominal.dtype)
        tv = 0.5 * jnp.sum(jnp.abs(nominal - target), axis=-1)
        radius = jnp.asarray(y_pred.radius).astype(nominal.dtype)
        return float(jnp.mean((tv <= radius).astype(jnp.float32)))
    return _envelope_coverage(y_pred.lower(), y_pred.upper(), y_true)


@efficiency.register(JaxDistanceBasedCredalSet)
def _efficiency_jax_distance(y_pred: JaxDistanceBasedCredalSet) -> float:
    """Interval-width efficiency for a distance-based credal set: ``1 - mean(upper - lower)``.

    Same semantic as ``ConvexCredalSet`` and ``ProbabilityIntervalsCredalSet``:
    higher = tighter credal set. For TV (L1) distance-based credal sets the
    per-class envelope bounds ``[max(0, p_k - r), min(1, p_k + r)]`` are tight.
    """
    return _credal_interval_efficiency_jax(y_pred.lower(), y_pred.upper())


@coverage.register(JaxProbabilityIntervalsCredalSet)
def _coverage_jax_probability_intervals(y_pred: JaxProbabilityIntervalsCredalSet, y_true: object) -> float:
    """Containment coverage for a probability-intervals credal set.

    Args:
        y_pred: Probability-intervals credal set.
        y_true: Target probability arrays of shape ``(N, C)`` or class indices.

    Returns:
        Fraction of instances where the target lies in ``[lower, upper]`` for all classes.
    """
    return _credal_containment_coverage_jax(y_pred.lower(), y_pred.upper(), y_true)


@efficiency.register(JaxProbabilityIntervalsCredalSet)
def _efficiency_jax_probability_intervals(y_pred: JaxProbabilityIntervalsCredalSet) -> float:
    """Interval-width efficiency for a probability-intervals credal set: ``1 - mean(upper - lower)``.

    Returns:
        Scalar efficiency; higher means a tighter credal set.
    """
    return _credal_interval_efficiency_jax(y_pred.lower(), y_pred.upper())


@coverage.register(JaxDirichletLevelSetCredalSet)
def _coverage_jax_dirichlet_level_set(y_pred: JaxDirichletLevelSetCredalSet, y_true: object) -> float:
    """Interval-dominance coverage for a Dirichlet-level-set credal set.

    The lower/upper envelopes are estimated by Monte-Carlo sampling from a fixed
    default PRNG key, so repeated calls are deterministic.
    """
    return _envelope_coverage(y_pred.lower(), y_pred.upper(), y_true)


@efficiency.register(JaxDirichletLevelSetCredalSet)
def _efficiency_jax_dirichlet_level_set(y_pred: JaxDirichletLevelSetCredalSet) -> float:
    """Interval-dominance prediction-set cardinality for a Dirichlet-level-set credal set.

    The lower/upper envelopes are estimated by Monte-Carlo sampling from a fixed
    default PRNG key, so repeated calls are deterministic.
    """
    return _envelope_efficiency(y_pred.lower(), y_pred.upper())


@average_interval_width.register(JaxConvexCredalSet)
def _average_interval_width_jax_convex(y_pred: JaxConvexCredalSet) -> float:
    """Mean per-class width of the vertex-derived envelope of a convex credal set."""
    return _envelope_average_interval_width(y_pred.lower(), y_pred.upper())


@average_interval_width.register(JaxDistanceBasedCredalSet)
def _average_interval_width_jax_distance(y_pred: JaxDistanceBasedCredalSet) -> float:
    """Mean per-class width of the L1-clip envelope of a distance-based credal set."""
    return _envelope_average_interval_width(y_pred.lower(), y_pred.upper())


@average_interval_width.register(JaxProbabilityIntervalsCredalSet)
def _average_interval_width_jax_probability_intervals(y_pred: JaxProbabilityIntervalsCredalSet) -> float:
    """Mean per-class interval width of a probability-intervals credal set."""
    return _envelope_average_interval_width(y_pred.lower(), y_pred.upper())


@average_interval_width.register(JaxDirichletLevelSetCredalSet)
def _average_interval_width_jax_dirichlet_level_set(y_pred: JaxDirichletLevelSetCredalSet) -> float:
    """Mean per-class width of the MC-sampled envelope of a Dirichlet-level-set credal set."""
    return _envelope_average_interval_width(y_pred.lower(), y_pred.upper())


@convex_hull_coverage.register(JaxConvexCredalSet)
def _convex_hull_coverage_jax_convex(
    y_pred: JaxConvexCredalSet,
    y_true: JaxCategoricalDistribution,
    *,
    epsilon: float = 0.0005,
    **linprog_kwargs: object,
) -> object:
    """Hull coverage for a jax convex credal set; routes through the numpy LP solver.

    ``scipy.linprog`` is numpy-only, so the vertex array and target array are
    materialized onto the host as numpy arrays first.
    """
    vertices = jax.device_get(y_pred.tensor.probabilities)
    targets = jax.device_get(y_true.probabilities)
    return _convex_hull_lp_coverage(vertices, targets, epsilon, **linprog_kwargs)
