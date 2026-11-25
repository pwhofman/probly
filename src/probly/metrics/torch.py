"""PyTorch implementation of Metrics.

Note:
    There is no ``TorchSingletonCredalSet`` or ``TorchDiscreteCredalSet`` in
    :mod:`probly.representation.credal_set.torch`; for those semantics, use
    the numpy-side ``NumpySingletonCredalSet`` / ``NumpyDiscreteCredalSet``
    types. The remaining torch credal sets (Convex, DistanceBased,
    ProbabilityIntervals, DirichletLevelSet) all use the interval-dominance
    rule via their ``lower()`` / ``upper()`` envelopes.
"""

from __future__ import annotations

import torch

from probly.metrics import (
    classwise_ece,
    convex_hull_coverage,
    expected_calibration_error,
    precision_recall_curve,
    roc_curve,
)
from probly.metrics._common import (
    CREDAL_COVERAGE_EPSILON,
    _asarray_like,
    _expected_loss,
    _floating,
    _label_values,
    _log_probabilities,
    _machine_epsilon,
    _one_hot_argmax,
    _probabilities,
    _validate_labels,
    log_loss_per_instance,
)
from probly.metrics.numpy import _numpy_convex_hull_lp_coverage
from probly.representation.credal_set.torch import TorchConvexCredalSet
from probly.representation.distribution._common import CategoricalDistribution
from probly.representation.distribution.torch_categorical import (
    TorchCategoricalDistribution,
    TorchProbabilityCategoricalDistribution,
)


def _binary_clf_curve(y_true: torch.Tensor, y_score: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    desc_score_indices = torch.argsort(y_score, dim=-1, stable=True, descending=True)
    y_score = torch.take_along_dim(y_score, desc_score_indices, dim=-1)
    y_true = torch.take_along_dim(y_true, desc_score_indices, dim=-1)

    # y_score typically has many tied values. A distinct value ends where the next score differs, and the end
    # of the curve is always a threshold. The running minimum from the right turns these ends into the
    # threshold index of every position.
    end = torch.ones((*y_score.shape[:-1], 1), dtype=torch.bool, device=y_score.device)
    is_distinct_value = torch.cat([y_score[..., 1:] != y_score[..., :-1], end], dim=-1)
    threshold_idxs = torch.where(is_distinct_value, torch.arange(n, device=y_score.device), n - 1)
    threshold_idxs = torch.cummin(threshold_idxs.flip(-1), dim=-1).values.flip(-1)

    # Accumulate the true positives with decreasing threshold.
    tps = torch.take_along_dim(torch.cumsum(y_true, dim=-1), threshold_idxs, dim=-1)
    fps = 1 + threshold_idxs - tps
    return fps, tps, y_score


@precision_recall_curve.register(torch.Tensor)
def torch_precision_recall_curve(
    y_true: torch.Tensor, y_score: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute precision-recall curve along the last axis."""
    y_true = y_true.float()
    y_score = y_score.float()

    fps, tps, thresholds = _binary_clf_curve(y_true, y_score)
    total_pos = tps[..., -1:]

    precision = tps / (tps + fps)
    recall = torch.where(
        total_pos > 0, tps / torch.where(total_pos > 0, total_pos, torch.ones_like(total_pos)), torch.zeros_like(tps)
    )

    ones = torch.ones((*y_score.shape[:-1], 1), device=y_true.device, dtype=y_true.dtype)
    zeros = torch.zeros((*y_score.shape[:-1], 1), device=y_true.device, dtype=y_true.dtype)
    precision = torch.cat([precision.flip(-1), ones], dim=-1)
    recall = torch.cat([recall.flip(-1), zeros], dim=-1)

    return precision, recall, thresholds


@classwise_ece.register(torch.Tensor)
def torch_classwise_ece(y_prob: torch.Tensor, y_true: torch.Tensor, *, num_bins: int = 15) -> torch.Tensor:
    """Compute the classwise expected calibration error for PyTorch tensors."""
    probs = y_prob.float()
    if probs.ndim != 2:
        msg = f"classwise_ece expects probabilities of shape (n, k), got shape {tuple(probs.shape)}."
        raise ValueError(msg)
    labels = y_true.reshape(-1)
    n, k = probs.shape
    if labels.shape[0] != n:
        msg = f"classwise_ece labels must match probabilities batch size. Got {labels.shape[0]} labels for {n} rows."
        raise ValueError(msg)
    if num_bins < 1:
        msg = f"classwise_ece expects num_bins >= 1, got {num_bins}."
        raise ValueError(msg)

    one_hot = (labels[:, None] == torch.arange(k, device=probs.device)[None, :]).float()
    bin_idx = torch.clamp((probs * num_bins).long(), max=num_bins - 1)

    total = probs.new_zeros(())
    for b in range(num_bins):
        mask = (bin_idx == b).float()
        count = mask.sum(dim=0)
        safe_count = torch.where(count > 0, count, torch.ones_like(count))
        mean_prob = (probs * mask).sum(dim=0) / safe_count
        freq = (one_hot * mask).sum(dim=0) / safe_count
        total = total + (count / n * (freq - mean_prob).abs()).sum()
    return total / k


@expected_calibration_error.register(torch.Tensor)
def torch_expected_calibration_error(y_prob: torch.Tensor, y_true: torch.Tensor, *, num_bins: int = 15) -> torch.Tensor:
    """Compute the confidence expected calibration error for PyTorch tensors."""
    probs = y_prob.float()
    if probs.ndim != 2:
        msg = f"expected_calibration_error expects probabilities of shape (n, k), got shape {tuple(probs.shape)}."
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

    conf, predicted = torch.max(probs, dim=-1)
    correct = (predicted == labels).float()
    bin_idx = torch.clamp((conf * num_bins).long(), max=num_bins - 1)

    total = probs.new_zeros(())
    for b in range(num_bins):
        mask = (bin_idx == b).float()
        count = mask.sum()
        safe_count = torch.where(count > 0, count, torch.ones_like(count))
        acc_bin = (correct * mask).sum() / safe_count
        conf_bin = (conf * mask).sum() / safe_count
        total = total + count / n * (acc_bin - conf_bin).abs()
    return total


@roc_curve.register(torch.Tensor)
def torch_roc_curve(y_true: torch.Tensor, y_score: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute ROC curve along the last axis."""
    y_true = y_true.float()
    y_score = y_score.float()

    fps, tps, thresholds = _binary_clf_curve(y_true, y_score)
    total_pos = tps[..., -1:]
    total_neg = fps[..., -1:]

    tpr = torch.where(
        total_pos > 0, tps / torch.where(total_pos > 0, total_pos, torch.ones_like(total_pos)), torch.zeros_like(tps)
    )
    fpr = torch.where(
        total_neg > 0, fps / torch.where(total_neg > 0, total_neg, torch.ones_like(total_neg)), torch.zeros_like(fps)
    )

    zeros = torch.zeros((*y_score.shape[:-1], 1), device=y_true.device, dtype=y_true.dtype)
    tpr = torch.cat([zeros, tpr], dim=-1)
    fpr = torch.cat([zeros, fpr], dim=-1)
    thresholds = torch.cat([thresholds[..., :1] + 1, thresholds], dim=-1)

    return fpr, tpr, thresholds


# --- Predicted-set metrics ----------------------------------------------------


@_asarray_like.register(torch.Tensor)
def _torch_asarray_like(reference: torch.Tensor, values: object) -> torch.Tensor:
    """Convert values to a torch tensor on the device of the reference."""
    return torch.as_tensor(values, device=reference.device)


@_label_values.register(torch.Tensor)
def _torch_label_values(values: torch.Tensor, y_true: object) -> torch.Tensor:
    """Look up the entry at the class label of each instance along the last axis."""
    indices = torch.as_tensor(y_true, device=values.device).to(dtype=torch.int64).unsqueeze(-1)
    return torch.gather(values, -1, indices).squeeze(-1)


@_floating.register(torch.Tensor)
def _torch_floating(values: torch.Tensor) -> torch.Tensor:
    """Cast a torch tensor to a floating dtype of at least single precision."""
    return values.to(torch.promote_types(values.dtype, torch.float32))


@_machine_epsilon.register(torch.Tensor)
def _torch_machine_epsilon(values: torch.Tensor) -> float:
    """Return the machine epsilon of the dtype of a floating torch tensor."""
    return torch.finfo(values.dtype).eps


@convex_hull_coverage.register(TorchConvexCredalSet)
def _torch_convex_convex_hull_coverage(
    y_pred: TorchConvexCredalSet,
    y_true: TorchCategoricalDistribution,
    *,
    epsilon: float = CREDAL_COVERAGE_EPSILON,
    **linprog_kwargs: object,
) -> object:
    """Hull coverage for a torch convex credal set; routes through the numpy LP solver.

    ``scipy.linprog`` is numpy-only, so the vertex tensor and target tensor
    are detached, moved to CPU, and converted to numpy arrays.
    """
    vertices = y_pred.tensor.probabilities.detach().cpu().numpy()
    targets = y_true.probabilities.detach().cpu().numpy()
    return _numpy_convex_hull_lp_coverage(vertices, targets, epsilon, **linprog_kwargs)


@_probabilities.register
def torch_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    """Return a torch tensor of probabilities unchanged."""
    return probabilities


@_log_probabilities.register
def torch_log_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    """Compute the log probabilities of a torch tensor."""
    return torch.log(probabilities)


def _torch_scoring_target(y_true: object, reference: torch.Tensor) -> torch.Tensor:
    """Convert targets and validate labels without changing their dtype."""
    target = y_true.probabilities if isinstance(y_true, CategoricalDistribution) else y_true
    target = torch.as_tensor(target, device=reference.device)
    if target.shape != reference.shape:
        _validate_labels(
            tuple(target.shape),
            tuple(reference.shape),
            is_integer=not (target.is_floating_point() or target.is_complex()),
        )
    return target


@log_loss_per_instance.register(torch.Tensor)
def torch_log_loss_per_instance(y_prob: torch.Tensor, y_true: object) -> torch.Tensor:
    """Compute log loss without logging zero probabilities of classes with no target mass."""
    target = _torch_scoring_target(y_true, y_prob)
    if target.shape == y_prob.shape:
        safe_probabilities = torch.where(target > 0, y_prob, torch.ones_like(y_prob))
        return -(target * torch.log(safe_probabilities)).sum(dim=-1)
    selected = torch.gather(y_prob, -1, target.long().unsqueeze(-1)).squeeze(-1)
    return -torch.log(selected)


@log_loss_per_instance.register(TorchCategoricalDistribution)
def torch_categorical_log_loss_per_instance(y_prob: TorchCategoricalDistribution, y_true: object) -> torch.Tensor:
    """Score categorical log probabilities directly, preserving extreme logits."""
    return _torch_expected_loss(-y_prob.log_probabilities, y_true)


@log_loss_per_instance.register(TorchProbabilityCategoricalDistribution)
def torch_probability_categorical_log_loss_per_instance(
    y_prob: TorchProbabilityCategoricalDistribution, y_true: object
) -> torch.Tensor:
    """Score normalized categorical probabilities with target-aware logarithms."""
    return torch_log_loss_per_instance(y_prob.probabilities, y_true)


@_one_hot_argmax.register
def torch_one_hot_argmax(probabilities: torch.Tensor) -> torch.Tensor:
    """One-hot encode the first maximum of each probability vector of a torch tensor."""
    num_classes = probabilities.shape[-1]
    argmax = torch.argmax(probabilities, dim=-1)
    return torch.nn.functional.one_hot(argmax, num_classes).to(probabilities.dtype)


@_expected_loss.register(torch.Tensor)
def _torch_expected_loss(loss_vector: torch.Tensor, y_true: object) -> torch.Tensor:
    """Reduce a per-label loss vector to the loss under integer labels or target probabilities."""
    target = _torch_scoring_target(y_true, loss_vector)
    if target.shape == loss_vector.shape:
        # A class without target mass never occurs, so it adds no loss even where that loss is infinite.
        return torch.where(target > 0, target * loss_vector, loss_vector.new_zeros(())).sum(dim=-1)
    return torch.gather(loss_vector, -1, target.long().unsqueeze(-1)).squeeze(-1)
