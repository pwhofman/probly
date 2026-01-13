"""Sampling preparation for sklearn."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.base import BaseEstimator

from . import sampler

if TYPE_CHECKING:
    from lazy_dispatch.isinstance import LazyType
    from pytraverse import State

def _enforce_fitted_already(obj: BaseEstimator, state: State) -> tuple[BaseEstimator, State]:
    """Should check that the sklearn estimator is fitted already.
    
    There is no standard way to check if a sklearn estimator is fitted.

    See: https://scikit-learn.org/stable/glossary.html#term-fitted
    """
    
    return obj, state


def register_forced_fitted_already_mode(cls: LazyType) -> None:
    """Register a class to be forced into fitted already mode during sampling."""
    sampler.sampling_preparation_traverser.register(
        cls,
        _enforce_fitted_already,
    )


register_forced_fitted_already_mode(
    BaseEstimator,
)