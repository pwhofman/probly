"""JAX implementation of Metrics.

Note:
    There is no ``JaxSingletonCredalSet`` or ``JaxDiscreteCredalSet`` in
    :mod:`probly.representation.credal_set.jax`; for those semantics, use the
    numpy-side ``NumpySingletonCredalSet`` / ``NumpyDiscreteCredalSet`` types.
    The remaining jax credal sets (Convex, DistanceBased, ProbabilityIntervals,
    DirichletLevelSet) all use the interval-dominance rule via their
    ``lower()`` / ``upper()`` envelopes.
"""

from __future__ import annotations

import jax
from jax.core import Tracer
import jax.numpy as jnp

from probly.metrics._common import CREDAL_COVERAGE_EPSILON
from probly.metrics.numpy import _numpy_convex_hull_lp_coverage
from probly.representation.credal_set.jax import JaxConvexCredalSet
from probly.representation.distribution._common import CategoricalDistribution
from probly.representation.distribution.jax_categorical import (
    JaxCategoricalDistribution,
    JaxProbabilityCategoricalDistribution,
)

from ._common import (
    _asarray_like,
    _expected_loss,
    _floating,
    _label_values,
    _log_probabilities,
    _machine_epsilon,
    _one_hot_argmax,
    _probabilities,
    _validate_labels,
    classwise_ece,
    convex_hull_coverage,
    expected_calibration_error,
    log_loss_per_instance,
    precision_recall_curve,
    roc_curve,
)


@classwise_ece.register((jax.Array, Tracer))
def jax_classwise_ece(y_prob: jax.Array, y_true: jax.Array, *, num_bins: int = 15) -> jax.Array:
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


@expected_calibration_error.register((jax.Array, Tracer))
def jax_expected_calibration_error(y_prob: jax.Array, y_true: jax.Array, *, num_bins: int = 15) -> jax.Array:
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


def _binary_clf_curve(y_true: jax.Array, y_score: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Count false and true positives at every score threshold, along the last axis.

    Adapted from scikit-learn's ``sklearn.metrics._ranking._binary_clf_curve`` (BSD-3-Clause). Samples are
    ranked by decreasing score and the counts after accepting each sample are recorded. Equal scores form a
    single threshold: sklearn keeps only the last index of every run of equal scores, which makes the number
    of thresholds depend on the data, whereas this batched version maps every sample to the last index of its
    run and keeps the ``(..., n)`` shape. For scores ``[0.9, 0.5, 0.5, 0.5, 0.2]`` the threshold indices are
    ``[0, 3, 3, 3, 4]``. Reading the counts per sample instead would let the sort order among tied samples
    decide the curve, and identical scores could then yield an AUROC of 1.0 or 0.0 instead of 0.5.

    Args:
        y_true: Binary labels of shape ``(..., n)``.
        y_score: Scores of shape ``(..., n)``.

    Returns:
        fps: False positives after each threshold, shape ``(..., n)``.
        tps: True positives after each threshold, shape ``(..., n)``.
        thresholds: Scores in decreasing order, shape ``(..., n)``.
    """
    n = y_score.shape[-1]
    # Sort scores and corresponding truth values.
    desc_score_indices = jnp.flip(jnp.argsort(y_score, axis=-1, stable=True), axis=-1)
    y_score = jnp.take_along_axis(y_score, desc_score_indices, axis=-1)
    y_true = jnp.take_along_axis(y_true, desc_score_indices, axis=-1)

    # y_score typically has many tied values. A distinct value ends where the next score differs, and the end
    # of the curve is always a threshold. The running minimum from the right turns these ends into the
    # threshold index of every position.
    end = jnp.ones((*y_score.shape[:-1], 1), dtype=bool)
    is_distinct_value = jnp.concatenate([y_score[..., 1:] != y_score[..., :-1], end], axis=-1)
    threshold_idxs = jnp.where(is_distinct_value, jnp.arange(n), n - 1)
    threshold_idxs = jax.lax.cummin(threshold_idxs, axis=threshold_idxs.ndim - 1, reverse=True)

    # Accumulate the true positives with decreasing threshold.
    tps = jnp.take_along_axis(jnp.cumsum(y_true, axis=-1), threshold_idxs, axis=-1)
    fps = 1 + threshold_idxs - tps
    return fps, tps, y_score


@precision_recall_curve.register((jax.Array, Tracer))
def jax_precision_recall_curve(y_true: jax.Array, y_score: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute precision-recall curve along the last axis."""
    y_true = y_true.astype(jnp.float32)
    y_score = y_score.astype(jnp.float32)

    fps, tps, thresholds = _binary_clf_curve(y_true, y_score)
    total_pos = tps[..., -1:]

    precision = tps / (tps + fps)
    recall = jnp.where(total_pos > 0, tps / jnp.where(total_pos > 0, total_pos, 1.0), 0.0)

    ones = jnp.ones((*y_score.shape[:-1], 1), dtype=jnp.float32)
    zeros = jnp.zeros((*y_score.shape[:-1], 1), dtype=jnp.float32)
    precision = jnp.concatenate([jnp.flip(precision, axis=-1), ones], axis=-1)
    recall = jnp.concatenate([jnp.flip(recall, axis=-1), zeros], axis=-1)

    return precision, recall, thresholds


@roc_curve.register((jax.Array, Tracer))
def jax_roc_curve(y_true: jax.Array, y_score: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute ROC curve along the last axis."""
    y_true = y_true.astype(jnp.float32)
    y_score = y_score.astype(jnp.float32)

    fps, tps, thresholds = _binary_clf_curve(y_true, y_score)
    total_pos = tps[..., -1:]
    total_neg = fps[..., -1:]

    tpr = jnp.where(total_pos > 0, tps / jnp.where(total_pos > 0, total_pos, 1.0), 0.0)
    fpr = jnp.where(total_neg > 0, fps / jnp.where(total_neg > 0, total_neg, 1.0), 0.0)

    zeros = jnp.zeros((*y_score.shape[:-1], 1), dtype=jnp.float32)
    tpr = jnp.concatenate([zeros, tpr], axis=-1)
    fpr = jnp.concatenate([zeros, fpr], axis=-1)
    thresholds = jnp.concatenate([thresholds[..., :1] + 1, thresholds], axis=-1)

    return fpr, tpr, thresholds


# --- Predicted-set metrics ----------------------------------------------------


@_asarray_like.register((jax.Array, Tracer))
def _jax_asarray_like(_: jax.Array, values: object) -> jax.Array:
    """Convert values to a JAX array."""
    return jnp.asarray(values)


@_label_values.register((jax.Array, Tracer))
def _jax_label_values(values: jax.Array, y_true: object) -> jax.Array:
    """Look up the entry at the class label of each instance along the last axis."""
    indices = jnp.expand_dims(jnp.asarray(y_true).astype(jnp.int32), axis=-1)
    return jnp.squeeze(jnp.take_along_axis(values, indices, axis=-1), axis=-1)


@_floating.register((jax.Array, Tracer))
def _jax_floating(values: jax.Array) -> jax.Array:
    """Cast a JAX array to a floating dtype of at least single precision."""
    return values.astype(jnp.promote_types(values.dtype, jnp.float32))


@_machine_epsilon.register((jax.Array, Tracer))
def _jax_machine_epsilon(values: jax.Array) -> float:
    """Return the machine epsilon of the dtype of a floating JAX array."""
    return float(jnp.finfo(values.dtype).eps)


@convex_hull_coverage.register(JaxConvexCredalSet)
def _jax_convex_convex_hull_coverage(
    y_pred: JaxConvexCredalSet,
    y_true: JaxCategoricalDistribution,
    *,
    epsilon: float = CREDAL_COVERAGE_EPSILON,
    **linprog_kwargs: object,
) -> object:
    """Hull coverage for a jax convex credal set; routes through the numpy LP solver.

    ``scipy.linprog`` is numpy-only, so the vertex array and target array are
    materialized onto the host as numpy arrays first.
    """
    vertices = jax.device_get(y_pred.tensor.probabilities)
    targets = jax.device_get(y_true.probabilities)
    return _numpy_convex_hull_lp_coverage(vertices, targets, epsilon, **linprog_kwargs)


@_probabilities.register((jax.Array, Tracer))
def jax_probabilities(probabilities: jnp.ndarray) -> jnp.ndarray:
    """Return a JAX array of probabilities unchanged."""
    return probabilities


@_log_probabilities.register((jax.Array, Tracer))
def jax_log_probabilities(probabilities: jnp.ndarray) -> jnp.ndarray:
    """Compute the log probabilities of a JAX array."""
    return jnp.log(probabilities)


def _jax_scoring_target(y_true: object, reference: jax.Array) -> jax.Array:
    """Convert targets and validate labels without changing their dtype."""
    target = jnp.asarray(y_true.probabilities if isinstance(y_true, CategoricalDistribution) else y_true)
    if target.shape != reference.shape:
        _validate_labels(target.shape, reference.shape, is_integer=bool(jnp.issubdtype(target.dtype, jnp.integer)))
    return target


@log_loss_per_instance.register((jax.Array, Tracer))
def jax_log_loss_per_instance(y_prob: jax.Array, y_true: object) -> jax.Array:
    """Compute log loss without logging zero probabilities of classes with no target mass."""
    target = _jax_scoring_target(y_true, y_prob)
    if target.shape == y_prob.shape:
        safe_probabilities = jnp.where(target > 0, y_prob, jnp.ones_like(y_prob))
        return -(target * jnp.log(safe_probabilities)).sum(axis=-1)
    selected = jnp.take_along_axis(y_prob, target[..., None], axis=-1).squeeze(-1)
    return -jnp.log(selected)


@log_loss_per_instance.register(JaxCategoricalDistribution)
def jax_categorical_log_loss_per_instance(y_prob: JaxCategoricalDistribution, y_true: object) -> jax.Array:
    """Score categorical log probabilities directly, preserving extreme logits."""
    return _jax_expected_loss(-y_prob.log_probabilities, y_true)


@log_loss_per_instance.register(JaxProbabilityCategoricalDistribution)
def jax_probability_categorical_log_loss_per_instance(
    y_prob: JaxProbabilityCategoricalDistribution, y_true: object
) -> jax.Array:
    """Score normalized categorical probabilities with target-aware logarithms."""
    return jax_log_loss_per_instance(y_prob.probabilities, y_true)


@_one_hot_argmax.register((jax.Array, Tracer))
def jax_one_hot_argmax(probabilities: jnp.ndarray) -> jnp.ndarray:
    """One-hot encode the first maximum of each probability vector of a JAX array."""
    num_classes = probabilities.shape[-1]
    argmax = jnp.argmax(probabilities, axis=-1)
    return jnp.eye(num_classes, dtype=probabilities.dtype)[argmax]


@_expected_loss.register((jax.Array, Tracer))
def _jax_expected_loss(loss_vector: jax.Array, y_true: object) -> jax.Array:
    """Reduce a per-label loss vector to the loss under integer labels or target probabilities."""
    target = _jax_scoring_target(y_true, loss_vector)
    if target.shape == loss_vector.shape:
        # A class without target mass never occurs, so it adds no loss even where that loss is infinite.
        return jnp.where(target > 0, target * loss_vector, 0.0).sum(axis=-1)
    return jnp.take_along_axis(loss_vector, target[..., None], axis=-1).squeeze(-1)
