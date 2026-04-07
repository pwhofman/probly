"""Implementation for sklearn ensemble models."""

from __future__ import annotations

from sklearn.base import BaseEstimator, clone

from ._common import ensemble_generator


@ensemble_generator.register(BaseEstimator)
def generate_sklearn_ensemble(obj: BaseEstimator, num_members: int, reset_params: bool, **_kwargs) -> list[object]:  # noqa: ANN003
    """Generates an ensemble model from a sklearn base estimator."""
    if reset_params:
        obj.__setattr__("random_state", None)
    return [clone(obj, safe=not reset_params) for _ in range(num_members)]
