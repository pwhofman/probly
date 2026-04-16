"""Absolute Error Score implementation."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from lazy_dispatch import lazydispatch
from probly.representation.sample.array import ArraySample


@lazydispatch
def absolute_error_score_func[T](
    y_pred: T,
    y_true: T,
) -> npt.NDArray[np.floating]:
    """Compute the absolute error nonconformity score."""
    msg = "Absolute error score computation not implemented for this type."
    raise NotImplementedError(msg)


@absolute_error_score_func.register(np.ndarray)
def compute_absolute_error_score_numpy(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Absolute error for numpy arrays."""
    if y_pred.ndim > 2:
        msg = (
            "y_pred must have shape (n_evaluations, n_samples) or (n_samples,), "
            f"got {y_pred.shape}. The n_evaluations dimension is optional and "
            "will be averaged over if present."
        )
        raise ValueError(msg)
    if y_pred.ndim == 2:
        y_pred = y_pred.mean(axis=0)
    return np.abs(y_true - y_pred)


@absolute_error_score_func.register(ArraySample)
def _(y_pred: ArraySample, y_true: np.ndarray) -> np.ndarray:
    """Absolute error for ArraySamples."""
    return absolute_error_score_func(y_pred.array, y_true)
