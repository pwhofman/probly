"""Absolute Error Score implementation."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from lazy_dispatch import lazydispatch
from probly.conformal.scores._common import RegressionNonConformityScore


@lazydispatch
def absolute_error_score_func[T](y_true: T, y_pred: T) -> npt.NDArray[np.floating]:
    """Compute the absolute error nonconformity score."""
    msg = "Absolute error score computation not implemented for this type."
    raise NotImplementedError(msg)


@absolute_error_score_func.register(np.ndarray)
def _(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Absolute error for numpy arrays."""
    return np.abs(y_true - y_pred)


class AbsoluteErrorScore[T](RegressionNonConformityScore[T]):
    """Absolute Error Nonconformity Score."""

    non_conformity_score = absolute_error_score_func

    def __init__(self) -> None:
        super().__init__()

    def weight(self, _: T) -> tuple[T, T]:
        return 1, 1
