"""NumPy implementation of Metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

import numpy as np
from scipy.optimize import linprog

from probly.representation.credal_set.numpy import (
    NumpyConvexCredalSet,
    NumpyDiscreteCredalSet,
    NumpySingletonCredalSet,
)
from probly.representation.distribution._common import CategoricalDistribution
from probly.representation.sample import Sample

from ._common import (
    CREDAL_COVERAGE_EPSILON,
    _asarray_like,
    _expected_loss,
    _floating,
    _label_values,
    _log_probabilities,
    _machine_epsilon,
    _one_hot_argmax,
    _probabilities,
    _validate_epsilon,
    _validate_labels,
    classwise_ece,
    convex_hull_coverage,
    expected_calibration_error,
    precision_recall_curve,
    roc_curve,
)

if TYPE_CHECKING:
    from probly.representation.distribution import NumpyCategoricalDistribution


def _binary_clf_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    desc_score_indices = np.flip(np.argsort(y_score, axis=-1, kind="mergesort"), axis=-1)
    y_score = np.take_along_axis(y_score, desc_score_indices, axis=-1)
    y_true = np.take_along_axis(y_true, desc_score_indices, axis=-1)

    # y_score typically has many tied values. A distinct value ends where the next score differs, and the end
    # of the curve is always a threshold. The running minimum from the right turns these ends into the
    # threshold index of every position.
    end = np.ones((*y_score.shape[:-1], 1), dtype=bool)
    is_distinct_value = np.concatenate([y_score[..., 1:] != y_score[..., :-1], end], axis=-1)
    threshold_idxs = np.where(is_distinct_value, np.arange(n), n - 1)
    threshold_idxs = np.flip(np.minimum.accumulate(np.flip(threshold_idxs, axis=-1), axis=-1), axis=-1)

    # Accumulate the true positives with decreasing threshold.
    tps = np.take_along_axis(np.cumsum(y_true, axis=-1), threshold_idxs, axis=-1)
    fps = 1 + threshold_idxs - tps
    return fps, tps, y_score


@precision_recall_curve.register(np.ndarray)
def numpy_precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute precision-recall curve along the last axis."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    fps, tps, thresholds = _binary_clf_curve(y_true, y_score)
    total_pos = tps[..., -1:]

    precision = tps / (tps + fps)
    recall = np.where(total_pos > 0, tps / np.where(total_pos > 0, total_pos, 1.0), 0.0)

    ones = np.ones((*y_score.shape[:-1], 1))
    zeros = np.zeros((*y_score.shape[:-1], 1))
    precision = np.concatenate([np.flip(precision, axis=-1), ones], axis=-1)
    recall = np.concatenate([np.flip(recall, axis=-1), zeros], axis=-1)

    return precision, recall, thresholds


@classwise_ece.register(np.ndarray)
def numpy_classwise_ece(y_prob: np.ndarray, y_true: np.ndarray, *, num_bins: int = 15) -> np.floating:
    """Compute the classwise expected calibration error for NumPy arrays."""
    probs = np.asarray(y_prob, dtype=float)
    if probs.ndim != 2:
        msg = f"classwise_ece expects probabilities of shape (n, k), got shape {probs.shape}."
        raise ValueError(msg)
    labels = np.asarray(y_true).reshape(-1)
    n, k = probs.shape
    if labels.shape[0] != n:
        msg = f"classwise_ece labels must match probabilities batch size. Got {labels.shape[0]} labels for {n} rows."
        raise ValueError(msg)
    if num_bins < 1:
        msg = f"classwise_ece expects num_bins >= 1, got {num_bins}."
        raise ValueError(msg)

    one_hot = (labels[:, None] == np.arange(k)[None, :]).astype(float)
    bin_idx = np.minimum((probs * num_bins).astype(int), num_bins - 1)

    total = np.float64(0.0)
    for b in range(num_bins):
        mask = (bin_idx == b).astype(float)
        count = mask.sum(axis=0)
        safe_count = np.where(count > 0, count, 1.0)
        mean_prob = (probs * mask).sum(axis=0) / safe_count
        freq = (one_hot * mask).sum(axis=0) / safe_count
        total += (count / n * np.abs(freq - mean_prob)).sum()
    return total / k


@classwise_ece.register(CategoricalDistribution)
def numpy_categorical_classwise_ece(y_prob: CategoricalDistribution, y_true: object, *, num_bins: int = 15) -> object:
    """Compute the classwise ECE for a categorical distribution via its class probabilities."""
    return classwise_ece(y_prob.probabilities, y_true, num_bins=num_bins)


@expected_calibration_error.register(np.ndarray)
def numpy_expected_calibration_error(y_prob: np.ndarray, y_true: np.ndarray, *, num_bins: int = 15) -> np.floating:
    """Compute the confidence expected calibration error for NumPy arrays."""
    probs = np.asarray(y_prob, dtype=float)
    if probs.ndim != 2:
        msg = f"expected_calibration_error expects probabilities of shape (n, k), got shape {probs.shape}."
        raise ValueError(msg)
    labels = np.asarray(y_true).reshape(-1)
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
    correct = (probs.argmax(axis=-1) == labels).astype(float)
    bin_idx = np.minimum((conf * num_bins).astype(int), num_bins - 1)

    total = np.float64(0.0)
    for b in range(num_bins):
        mask = (bin_idx == b).astype(float)
        count = mask.sum()
        safe_count = np.where(count > 0, count, 1.0)
        acc_bin = (correct * mask).sum() / safe_count
        conf_bin = (conf * mask).sum() / safe_count
        total += count / n * np.abs(acc_bin - conf_bin)
    return total


@expected_calibration_error.register(CategoricalDistribution)
def numpy_categorical_expected_calibration_error(
    y_prob: CategoricalDistribution, y_true: object, *, num_bins: int = 15
) -> object:
    """Compute the confidence ECE for a categorical distribution via its class probabilities."""
    return expected_calibration_error(y_prob.probabilities, y_true, num_bins=num_bins)


@roc_curve.register(np.ndarray)
def numpy_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC curve along the last axis."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    fps, tps, thresholds = _binary_clf_curve(y_true, y_score)
    total_pos = tps[..., -1:]
    total_neg = fps[..., -1:]

    tpr = np.where(total_pos > 0, tps / np.where(total_pos > 0, total_pos, 1.0), 0.0)
    fpr = np.where(total_neg > 0, fps / np.where(total_neg > 0, total_neg, 1.0), 0.0)

    zeros = np.zeros((*y_score.shape[:-1], 1))
    tpr = np.concatenate([zeros, tpr], axis=-1)
    fpr = np.concatenate([zeros, fpr], axis=-1)
    thresholds = np.concatenate([thresholds[..., :1] + 1, thresholds], axis=-1)

    return fpr, tpr, thresholds


# --- Predicted-set metrics ----------------------------------------------------


@_asarray_like.register(np.ndarray)
def _numpy_asarray_like(_: np.ndarray, values: object) -> np.ndarray:
    """Convert values to a NumPy array."""
    return np.asarray(values)


@_label_values.register(np.ndarray)
def _numpy_label_values(values: np.ndarray, y_true: object) -> np.ndarray:
    """Look up the entry at the class label of each instance along the last axis."""
    indices = np.asarray(y_true, dtype=np.int64)[..., None]
    return np.take_along_axis(values, indices, axis=-1).squeeze(-1)


@_floating.register(np.ndarray)
def _numpy_floating(values: np.ndarray) -> np.ndarray:
    """Cast a NumPy array to a floating dtype of at least single precision."""
    return np.asarray(values, dtype=np.result_type(values, np.float32))


@_machine_epsilon.register(np.ndarray)
def _numpy_machine_epsilon(values: np.ndarray) -> float:
    """Return the machine epsilon of the dtype of a floating NumPy array."""
    return float(np.finfo(values.dtype).eps)


# --- Convex-hull coverage ----------------------------------------------------


def _numpy_convex_hull_lp_coverage(
    vertices: np.ndarray,
    targets: np.ndarray,
    epsilon: float,
    **linprog_kwargs: object,
) -> np.floating:
    """Convex-hull membership coverage via per-instance LP feasibility.

    Solves one linear program per instance. The strict variant
    (``epsilon == 0``) tests feasibility of
    ``V^T lambda = t, sum(lambda) = 1, lambda in [0, 1]``. The relaxed
    variant (``epsilon > 0``) introduces L1 slack variables ``s+`` and
    ``s-`` and minimizes their sum; an instance counts as covered iff the
    LP is feasible and the optimal slack sum is at most ``epsilon``.

    Args:
        vertices: Array of shape ``(N, V, K)`` holding ``V`` vertex
            distributions over ``K`` classes for each of ``N`` instances.
        targets: Array of shape ``(N, K)`` holding the target distribution
            for each instance.
        epsilon: L1 tolerance. ``0.0`` selects the strict LP.
        **linprog_kwargs: Forwarded to :func:`scipy.optimize.linprog`
            (e.g. ``method`` or solver tolerances).

    Returns:
        Fraction of instances whose target lies in (or within ``epsilon`` of)
        the hull, as ``np.float64``.
    """
    _validate_epsilon(epsilon)
    if vertices.ndim != 3:
        msg = f"vertices must be 3D (N, V, K); got shape {vertices.shape}."
        raise ValueError(msg)
    if targets.ndim != 2:
        msg = f"targets must be 2D (N, K); got shape {targets.shape}."
        raise ValueError(msg)
    if vertices.shape[0] != targets.shape[0]:
        msg = f"vertices and targets must agree on N; got {vertices.shape[0]} and {targets.shape[0]}."
        raise ValueError(msg)
    if vertices.shape[2] != targets.shape[1]:
        msg = f"vertices and targets must agree on K; got {vertices.shape[2]} and {targets.shape[1]}."
        raise ValueError(msg)

    n_instances, n_vertices, n_classes = vertices.shape
    relaxed = epsilon > 0.0

    if relaxed:
        c = np.concatenate([np.zeros(n_vertices), np.ones(2 * n_classes)])
        bounds: list[tuple[float, float | None]] = [(0.0, 1.0)] * n_vertices + [(0.0, None)] * (2 * n_classes)
    else:
        c = np.zeros(n_vertices)
        bounds = [(0.0, 1.0)] * n_vertices

    covered = 0
    # Per-instance LP loop (Python-level). For very large N (~10^6) consider
    # joblib.Parallel; not implemented here to keep the dependency surface small.
    for i in range(n_instances):
        v = vertices[i]
        t = targets[i]
        if relaxed:
            a_eq_top = np.hstack([v.T, np.eye(n_classes), -np.eye(n_classes)])
            a_eq_bot = np.concatenate([np.ones(n_vertices), np.zeros(2 * n_classes)])
            a_eq = np.vstack([a_eq_top, a_eq_bot])
            b_eq = np.concatenate([t, [1.0]])
        else:
            a_eq = np.vstack([v.T, np.ones(n_vertices)])
            b_eq = np.concatenate([t, [1.0]])

        res = linprog(c=c, A_eq=a_eq, b_eq=b_eq, bounds=bounds, **linprog_kwargs)
        if relaxed:
            covered += int(bool(res.success) and float(res.fun) <= epsilon)
        else:
            covered += int(bool(res.success))

    return np.float64(covered / n_instances) if n_instances > 0 else np.float64("nan")


@convex_hull_coverage.register(NumpyConvexCredalSet)
def _numpy_convex_convex_hull_coverage(
    y_pred: NumpyConvexCredalSet,
    y_true: NumpyCategoricalDistribution,
    *,
    epsilon: float = CREDAL_COVERAGE_EPSILON,
    **linprog_kwargs: object,
) -> np.floating:
    """LP-based hull coverage for a convex credal set."""
    return _numpy_convex_hull_lp_coverage(
        np.asarray(y_pred.array.probabilities),
        np.asarray(y_true.probabilities),
        epsilon,
        **linprog_kwargs,
    )


@convex_hull_coverage.register(NumpyDiscreteCredalSet)
def _numpy_discrete_convex_hull_coverage(
    y_pred: NumpyDiscreteCredalSet,
    y_true: NumpyCategoricalDistribution,
    *,
    epsilon: float = CREDAL_COVERAGE_EPSILON,
    **linprog_kwargs: object,
) -> np.floating:
    """LP-based hull coverage for a discrete credal set (same vertex structure as Convex)."""
    return _numpy_convex_hull_lp_coverage(
        np.asarray(y_pred.array.probabilities),
        np.asarray(y_true.probabilities),
        epsilon,
        **linprog_kwargs,
    )


@convex_hull_coverage.register(NumpySingletonCredalSet)
def _numpy_singleton_convex_hull_coverage(
    y_pred: NumpySingletonCredalSet,
    y_true: NumpyCategoricalDistribution,
    *,
    epsilon: float = CREDAL_COVERAGE_EPSILON,
    **_linprog_kwargs: object,
) -> np.floating:
    """Hull degenerates to a point; coverage is closed-form L1 distance test.

    The singleton handler does not call ``linprog`` and is therefore unaffected
    by solver tolerances. ``epsilon=0.0`` performs strict element-wise equality
    of ``predicted == target`` (subject to float arithmetic), which can produce
    slightly different verdicts than the LP path on numerically-tight inputs.
    """
    _validate_epsilon(epsilon)
    predicted = np.asarray(y_pred.array.probabilities)
    targets = np.asarray(y_true.probabilities)
    l1_dist = np.abs(predicted - targets).sum(axis=-1)
    return np.mean(l1_dist <= epsilon)


@_probabilities.register
def numpy_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Return a NumPy array of probabilities unchanged."""
    return probabilities


@_log_probabilities.register
def numpy_log_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Compute the log probabilities of a NumPy array."""
    # A zero probability has an infinite log loss; the inf is the exact value, so the warning is noise.
    with np.errstate(divide="ignore"):
        return np.log(probabilities)


@_one_hot_argmax.register
def numpy_one_hot_argmax(probabilities: np.ndarray) -> np.ndarray:
    """One-hot encode the first maximum of each probability vector of a NumPy array."""
    num_classes = probabilities.shape[-1]
    argmax = np.argmax(probabilities, axis=-1)
    return np.eye(num_classes, dtype=probabilities.dtype)[argmax]


@_probabilities.register(CategoricalDistribution)
def _(y_prob: CategoricalDistribution) -> object:
    """Unwrap the normalized probabilities of a categorical distribution."""
    return y_prob.probabilities


@_log_probabilities.register(CategoricalDistribution)
def _(y_prob: CategoricalDistribution) -> object:
    """Unwrap the log probabilities of a categorical distribution, exact for logit-parameterized ones."""
    return y_prob.log_probabilities


@_log_probabilities.register(Sample)
@_probabilities.register(Sample)
def _(y_prob: Sample) -> NoReturn:
    """Reject samples, whose members would otherwise be scored as separate predictions."""
    msg = (
        f"Expected one prediction per instance, got a {type(y_prob).__name__}. Reduce the sample "
        "first, e.g. with sample_mean()."
    )
    raise TypeError(msg)


@_expected_loss.register(np.ndarray)
def _numpy_expected_loss(loss_vector: np.ndarray, y_true: object) -> np.ndarray:
    """Reduce a per-label loss vector to the loss under integer labels or target probabilities."""
    target = np.asarray(y_true.probabilities if isinstance(y_true, CategoricalDistribution) else y_true)
    if target.shape == loss_vector.shape:
        # A class without target mass never occurs, so it adds no loss even where that loss is infinite.
        with np.errstate(invalid="ignore"):
            weighted = target * loss_vector
        return np.where(target > 0, weighted, 0.0).sum(axis=-1)
    _validate_labels(target.shape, loss_vector.shape, is_integer=np.issubdtype(target.dtype, np.integer))
    return np.take_along_axis(loss_vector, target[..., np.newaxis], axis=-1).squeeze(-1)
