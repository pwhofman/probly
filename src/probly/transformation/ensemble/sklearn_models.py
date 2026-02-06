"""Implementation for sklearn ensemble models."""

from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator, clone

from .common import register


def generate_sklearn_ensemble(
    obj: BaseEstimator,
    num_members: int,
    reset_params: bool,
    key: Any = None,  # noqa: ANN401, ARG001
) -> BaseEstimator:
    """Generates an ensemble model from a sklearn base estimator."""
    if reset_params:
        obj.__setattr__("random_state", None)
    return [clone(obj, safe=not reset_params) for _ in range(num_members)]


register(BaseEstimator, generate_sklearn_ensemble)
