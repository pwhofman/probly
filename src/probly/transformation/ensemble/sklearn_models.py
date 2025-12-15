"""Implementation for sklearn ensemble models."""

from __future__ import annotations

from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .common import register


def copy_imp_params(base: BaseEstimator, forest: BaseEstimator, reset_params: bool) -> dict[str, object]:
    """Copies the parameters of a DecisionTree that are important to create a RandomForest."""
    if reset_params:
        base.__setattr__("random_state", None)

    if (base == DecisionTreeClassifier):
        forest = RandomForestClassifier()
    elif (base == DecisionTreeRegressor):
        forest = RandomForestRegressor()

    base_params = base.get_params(deep=False)
    forest_params = forest.get_params(deep=False)

    copied = {k: v for k, v in base_params.items() if k in forest_params}

    return copied


def sklearn_list_ensemble(base: BaseEstimator, num_members: int, reset_params: bool) -> list[BaseEstimator]:
    """Generates a list of sklearn estimators by copying the base model num_members times."""
    if reset_params:
        base.__setattr__("random_state", None)
    return [clone(base, safe=not reset_params) for _ in range(num_members)]


def generate_sklearn_ensemble(obj: BaseEstimator, num_members: int, reset_params: bool) -> BaseEstimator:
    """Generates an ensemble model from a sklearn base estimator.

    By either creating a RandomForest model (for decision trees) or a list of models (for other sklearn estimators).
    If reset_params is True, the random_state parameter of the base model is set to None before creating the ensemble.
    Resetting the random_state ensures that the ensemble members are different.

    """
    if isinstance(obj, DecisionTreeClassifier):
        given_params = copy_imp_params(base=obj, forest=RandomForestClassifier(), reset_params=reset_params)
        sklearn_ensemble = RandomForestClassifier(**given_params, n_estimators=num_members)
    elif isinstance(obj, DecisionTreeRegressor):
        given_params = copy_imp_params(base=obj, forest=RandomForestRegressor(), reset_params=reset_params)
        sklearn_ensemble = RandomForestRegressor(**given_params, n_estimators=num_members)
    elif isinstance(obj, BaseEstimator):
        sklearn_ensemble = sklearn_list_ensemble(base=obj, num_members=num_members, reset_params=reset_params)

    else:
        msg = "Cannot create sklearn ensemble for object of type {type(obj)}"
        raise TypeError(msg)
    return sklearn_ensemble


register(BaseEstimator, generate_sklearn_ensemble)
