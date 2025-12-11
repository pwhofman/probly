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
     
    copied = []
    base_params = base.get_params(deep=False)
    forest_params = forest.get_params()
    
    for x in forest_params:
        if x in base_params:
            copied.append(x)
            
    return copied

class RandomForests:
    
    @staticmethod
    def generate_random_forest_classifier(obj: DecisionTreeClassifier
                                          , num_members: int
                                          , reset_params: bool) -> RandomForestClassifier:
        given_params = copy_imp_params(base=obj, forest=RandomForestClassifier())
        
        return RandomForestClassifier(**given_params, n_estimators=num_members)
    
    @staticmethod
    def generate_random_forest_regressor(base: DecisionTreeRegressor, num_members: int) -> RandomForestRegressor:
        given_params = copy_imp_params(base=base, forest=RandomForestRegressor())
        
        return RandomForestRegressor(**given_params, n_estimators=num_members)
    
register(DecisionTreeClassifier, RandomForests.generate_random_forest_classifier)
register(DecisionTreeRegressor, RandomForests.generate_random_forest_regressor)
    
        