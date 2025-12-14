"""Implementation for sklearn ensemble models"""

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from lazy_dispatch import lazydispatch
from typing import TYPE_CHECKING

from .common import register

if TYPE_CHECKING:
    from collections.abc import Callable

    from lazy_dispatch.isinstance import LazyType
    from probly.predictor import Predictor

def copy_imp_params(base: BaseEstimator, forest: BaseEstimator) -> list:
    """Copies the parameters of a DecisionTree that are important to create a RandomForest"""
    
    if (base == DecisionTreeClassifier):
        forest = RandomForestClassifier()
    if (base == DecisionTreeRegressor):
        forest = RandomForestRegressor()
     
    base_params = base.get_params(deep=False)
    forest_params = forest.get_params(deep=False)
    
    copied = {k: v for k, v in base_params.items() if k in forest_params}
            
    return copied

def sklearn_list_ensemble(base: BaseEstimator, num_members: int, reset_params: bool) -> list[BaseEstimator]:
    """Generates a list of sklearn estimators bt copying the base model num_members times."""
    return [base for _ in range(num_members)]

def generate_sklearn_ensemble(obj: BaseEstimator, num_members: int, reset_params: bool) -> BaseEstimator:
    if isinstance(obj, DecisionTreeClassifier):
        given_params = copy_imp_params(base=obj, forest=RandomForestClassifier())
        forest = RandomForestClassifier(**given_params, n_estimators=num_members)
    elif isinstance(obj, DecisionTreeRegressor):
        given_params = copy_imp_params(base=obj, forest=RandomForestRegressor())
        forest = RandomForestRegressor(**given_params, n_estimators=num_members)
    elif isinstance(obj, BaseEstimator):
        members = sklearn_list_ensemble(base=obj, num_members=num_members, reset_params=reset_params)
        forest = members # type: ignore[assignment]
    else:
        raise TypeError(f"Cannot create sklearn ensemble for object of type {type(obj)}")
    return forest

register(BaseEstimator, generate_sklearn_ensemble)
    