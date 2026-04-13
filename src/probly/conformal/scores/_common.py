"""Contract definitions for nonconformity scores."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from lazy_dispatch.registry_meta import ProtocolRegistry

if TYPE_CHECKING:
    from collections.abc import Callable


@runtime_checkable
class NonConformityScore[T](ProtocolRegistry, Protocol, structural_checking=False):
    """Base protocol for nonconformity scores."""

    non_conformity_score: Callable[..., T]

    def __call__(self, y_pred: T, y_true: T | None = None, **kwargs: dict[str, Any]) -> T:
        """Compute the nonconformity score."""
        return self.non_conformity_score(y_pred, y_true, **kwargs)

    @classmethod
    def __subclasshook__(cls, subclass: type) -> bool:
        compute_method = getattr(subclass, "compute", None)
        if compute_method is not None and callable(compute_method):
            return True
        return NotImplemented


@runtime_checkable
class ClassificationNonConformityScore[T](NonConformityScore[T], Protocol):
    """Protocol for classification nonconformity scores."""


@runtime_checkable
class QuantileNonConformityScore[T](NonConformityScore[T], Protocol):
    """Protocol for quantile regression nonconformity scores."""

    def weight(self, y_pred: T) -> tuple[T, T]:
        """Compute the weight for the nonconformity score."""
        ...


@runtime_checkable
class RegressionNonConformityScore[T](NonConformityScore[T], Protocol):
    """Protocol for regression nonconformity scores."""

    def weight(self, y_pred: T) -> tuple[T, T]:
        """Compute the weight for the nonconformity score."""
        ...
