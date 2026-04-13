"""Common metrics for conformal prediction."""

from __future__ import annotations

import numpy as np

from lazy_dispatch import lazydispatch
from probly.representation.conformal_set.array import ArrayIntervalConformalSet, ArrayOneHotConformalSet
from probly.representation.conformal_set.torch import TorchIntervalConformalSet, TorchOneHotConformalSet


@lazydispatch
def empirical_coverage_classification[T](y_pred: T, y_true: T) -> float:
    """Calculate the empirical coverage for classification."""
    msg = f"Empirical coverage for classification is not implemented for this type {type(y_pred)}."
    raise NotImplementedError(msg)


@lazydispatch
def empirical_coverage_regression[T](y_pred: T, y_true: T) -> float:
    """Calculate the empirical coverage for regression."""
    msg = f"Empirical coverage for regression is not implemented for this type {type(y_pred)}."
    raise NotImplementedError(msg)


@lazydispatch
def average_set_size[T](y_pred: T) -> float:
    """Calculate the average prediction set size for classification."""
    msg = f"Average set size for classification is not implemented for this type {type(y_pred)}."
    raise NotImplementedError(msg)


@lazydispatch
def average_interval_size[T](y_pred: T) -> float:
    """Calculate the average interval size for regression."""
    msg = f"Average interval size for regression is not implemented for this type {type(y_pred)}."
    raise NotImplementedError(msg)


@empirical_coverage_classification.register(np.ndarray)
def _empirical_coverage_classification_numpy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    contained = y_pred[np.arange(len(y_true)), y_true.astype(int)]
    return contained.mean()


@empirical_coverage_classification.register(ArrayOneHotConformalSet)
def _empirical_coverage_classification_array_onehot[T](y_pred: ArrayOneHotConformalSet, y_true: T) -> float:
    return empirical_coverage_classification(y_pred.array, y_true)


@empirical_coverage_classification.register(TorchOneHotConformalSet)
def _empirical_coverage_classification_torch_onehot[T](y_pred: TorchOneHotConformalSet, y_true: T) -> float:
    return empirical_coverage_classification(y_pred.tensor.cpu(), y_true)


@empirical_coverage_regression.register(ArrayIntervalConformalSet)
def _empirical_coverage_regression_array_onehot[T](y_pred: ArrayIntervalConformalSet, y_true: T) -> float:
    return empirical_coverage_regression(y_pred.array, y_true)


@empirical_coverage_regression.register(TorchIntervalConformalSet)
def _empirical_coverage_regression_torch_interval[T](y_pred: TorchIntervalConformalSet, y_true: T) -> float:
    return empirical_coverage_regression(y_pred.tensor.cpu(), y_true)


@empirical_coverage_regression.register(np.ndarray)
def _empirical_coverage_regression_numpy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return ((y_true >= y_pred[:, 0]) & (y_true <= y_pred[:, 1])).mean()


@average_set_size.register(np.ndarray)
def _average_set_size_numpy(y_pred: np.ndarray) -> float:
    return np.mean(y_pred.sum(axis=1))


@average_set_size.register(ArrayOneHotConformalSet)
def _average_set_size_array_onehot(y_pred: ArrayOneHotConformalSet) -> float:
    return average_set_size(y_pred.array)


@average_set_size.register(TorchOneHotConformalSet)
def _average_set_size_torch_onehot(y_pred: TorchOneHotConformalSet) -> float:
    return average_set_size(y_pred.tensor.cpu().numpy())


@average_interval_size.register(np.ndarray)
def _average_interval_size_numpy(y_pred: np.ndarray) -> float:
    return np.mean(y_pred[:, 1] - y_pred[:, 0])


@average_interval_size.register(ArrayIntervalConformalSet)
def _average_interval_size_array_interval(y_pred: ArrayIntervalConformalSet) -> float:
    return average_interval_size(y_pred.array)


@average_interval_size.register(TorchIntervalConformalSet)
def _average_interval_size_torch_interval(y_pred: TorchIntervalConformalSet) -> float:
    return average_interval_size(y_pred.tensor.cpu())
