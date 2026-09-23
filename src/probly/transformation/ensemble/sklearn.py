"""Implementation for sklearn ensemble models."""

from __future__ import annotations

from sklearn.base import BaseEstimator, clone

from ._common import ensemble_generator


@ensemble_generator.register(BaseEstimator)
def generate_sklearn_ensemble(obj: BaseEstimator, num_members: int, reset_params: bool) -> list[object]:
    """Generates an ensemble model from a sklearn base estimator.

    The base estimator is left untouched: every member is a clone, and with ``reset_params`` the clones get an
    unset ``random_state`` so that each member is fitted with its own randomness.
    """
    members = [clone(obj) for _ in range(num_members)]
    if reset_params:
        for member in members:
            # Also unset the seeds of nested estimators, such as the steps of a pipeline.
            random_states = [param for param in member.get_params(deep=True) if param.split("__")[-1] == "random_state"]
            member.set_params(**dict.fromkeys(random_states))
    return members
