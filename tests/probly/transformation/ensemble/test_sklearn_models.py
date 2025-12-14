"""Test for implementation of sklearn models."""

from __future__ import annotations
import pytest

from probly.transformation import ensemble

pytest.importorskip("sklearn")
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.base import BaseEstimator

class TestModelGeneration:
    
    def test_classifier(self) -> None:
        "Tests the correct generation of RandomForestClassifier."
        num_members = 3
        model = DecisionTreeClassifier()
        forest = ensemble(model, num_members=num_members, reset_params=False)
        
        assert isinstance(forest, RandomForestClassifier)
        assert num_members==forest.__getattribute__("n_estimators")
        assert model.__getattribute__("max_depth")==forest.__getattribute__("max_depth")
        assert model.__getattribute__("min_samples_leaf")==forest.__getattribute__("min_samples_leaf")
        
    def test_regressor(self, sklearn_decision_tree_regressor: BaseEstimator) -> None:
        """Tests the correct generation of RandomForestRegressor."""
        num_members = 3
        model = sklearn_decision_tree_regressor
        forest = ensemble(model, num_members=num_members, reset_params=False)
        
        assert isinstance(forest, RandomForestRegressor)
        assert num_members==forest.__getattribute__("n_estimators")
        assert model.__getattribute__("max_depth")==forest.__getattribute__("max_depth")
        assert model.__getattribute__("min_samples_leaf")==forest.__getattribute__("min_samples_leaf")
        
    def sklearn_model_mlp_regressor(self, sklearn_mlp_regressor_2d_1d: BaseEstimator) -> None:
        """Tests that an error is raised when trying to ensemble unsupported models."""
        num_members = 4
        model = sklearn_mlp_regressor_2d_1d
        
        ens = ensemble(model, num_members=num_members, reset_params=False)
        
        assert len(ens) == num_members
        for member in ens:
            assert isinstance(member, type(model))
        