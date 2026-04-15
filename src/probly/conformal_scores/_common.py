"""Contract definitions for nonconformity scores."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from lazy_dispatch.registry_meta import ProtocolRegistry
from probly.conformal_scores.absolute_error._common import absolute_error_score_func
from probly.conformal_scores.aps._common import aps_score_func
from probly.conformal_scores.cqr._common import cqr_score_func
from probly.conformal_scores.cqr_r._common import cqr_r_score_func
from probly.conformal_scores.lac._common import lac_score_func
from probly.conformal_scores.raps._common import raps_score_func
from probly.conformal_scores.saps._common import saps_score_func
from probly.conformal_scores.uacqr._common import uacqr_score_func

if TYPE_CHECKING:
    from collections.abc import Callable


@runtime_checkable
class NonConformityScore[In, Out](ProtocolRegistry, Protocol, structural_checking=False):
    """Base protocol for nonconformity scores."""

    non_conformity_score: Callable[..., Out]

    def __call__(self, y_pred: In, y_true: In | None = None, **kwargs: dict[str, Any]) -> Out:
        """Obtain the nonconformity score for the calibration data."""
        return self.non_conformity_score(y_pred, y_true, **kwargs)

    @classmethod
    def __subclasshook__(cls, subclass: type) -> bool:
        compute_method = getattr(subclass, "non_conformity_score_func", None)
        if compute_method and callable(compute_method) and callable(subclass):
            return True
        return NotImplemented


class ClassificationNonConformityScore[In](NonConformityScore[In, Any]):
    """Protocol for classification nonconformity scores."""


class QuantileNonConformityScore[In](NonConformityScore[In, Any]):
    """Protocol for quantile regression nonconformity scores."""


class RegressionNonConformityScore[In](NonConformityScore[In, Any]):
    """Protocol for regression nonconformity scores."""


class LACScore[T](ClassificationNonConformityScore[T]):
    """LAC nonconformity score class."""

    non_conformity_score = lac_score_func


class APSScore[T](ClassificationNonConformityScore[T]):
    """APS nonconformity score class."""

    non_conformity_score = aps_score_func


class SAPSScore[T](ClassificationNonConformityScore[T]):
    """SAPS nonconformity score class."""

    non_conformity_score = saps_score_func


class RAPSScore[T](ClassificationNonConformityScore[T]):
    """RAPS nonconformity score class."""

    non_conformity_score = raps_score_func


class CQRScore[T](QuantileNonConformityScore[T]):
    """CQR nonconformity score class."""

    non_conformity_score = cqr_score_func


class CQRrScore[T](QuantileNonConformityScore[T]):
    """CQR-r nonconformity score class."""

    non_conformity_score = cqr_r_score_func


class UACQRScore[T](QuantileNonConformityScore[T]):
    """UACQR nonconformity score class."""

    non_conformity_score = uacqr_score_func


class AbsoluteErrorScore[T](RegressionNonConformityScore[T]):
    """Absolute error nonconformity score class."""

    non_conformity_score = absolute_error_score_func
