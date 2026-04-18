"""Nonconformity scores for conformalized credal set prediction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from lazy_dispatch import ProtocolRegistry, lazydispatch

if TYPE_CHECKING:
    from collections.abc import Callable
import torch


@lazydispatch
def tv_score_func[T](probs: T, y_cal: T | None = None) -> T:
    """Compute Total Variation Score."""
    msg = "Total Variation score not implemented for this type."
    raise NotImplementedError(msg)


@tv_score_func.register(np.ndarray)
def compute_tv_score_numpy(probs: np.ndarray, y_cal: np.ndarray | None = None) -> np.ndarray:
    """Computes the Total Variation score using NumPy Arrays.

    Args:
        probs: Probabilities.
        y_cal: Calibration prediction
    """
    probs = np.atleast_2d(probs)
    if y_cal is None:
        msg = "y_cal cannot be None for TV score computation."
        raise ValueError(msg)
    y_cal_arr = np.atleast_2d(y_cal)

    if y_cal_arr.ndim == 1:
        y_one_hot = np.zeros_like(probs)
        y_one_hot[np.arange(len(y_cal_arr)), y_cal_arr.astype(int)] = 1.0
        y_cal_arr = y_one_hot
    return 0.5 * np.sum(np.abs(probs - y_cal_arr), axis=1)


@tv_score_func.register(torch.Tensor)
def compute_tv_score_torch(probs: torch.Tensor, y_cal: torch.Tensor | None = None) -> torch.Tensor:
    """Computes the Total Variation score using Torch Tensor.

    Args:
        probs: Probabilities.
        y_cal: Calibration prediction
    """
    probs = torch.atleast_2d(probs)
    if y_cal is None:
        msg = "y_cal cannot be None for TV score computation."
        raise ValueError(msg)
    y_cal_t = torch.atleast_2d(y_cal)

    if y_cal_t.ndim == 1:
        y_one_hot = torch.zeros_like(probs)
        y_one_hot[torch.arange(len(y_cal_t)), y_cal_t.long()] = 1.0
        y_cal_t = y_one_hot
    return 0.5 * torch.sum(torch.abs(probs - y_cal_t), dim=1)


@runtime_checkable
class NonConformityFunction[In, Out](ProtocolRegistry, Protocol, structural_checking=False):
    """Base protocol for nonconformity functions."""

    non_conformity_score: Callable[..., Out]

    def __call__(self, y_pred: In, y_true: In | None = None, **kwargs: dict[str, Any]) -> Out:
        """Obtain the nonconformity score for the calibration data."""
        return self.non_conformity_score(y_pred, y_true, **kwargs)


class CredalSetNonConformityScore[In](NonConformityFunction[In, Any]):
    """Protocol for Conformal Credal Set Prediction nonconformity score."""


class TVScore[T](CredalSetNonConformityScore[T]):
    """Total Variation score class."""

    non_conformity_score = tv_score_func
