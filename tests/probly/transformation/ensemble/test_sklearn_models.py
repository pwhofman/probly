"""Test for implementation of sklearn models."""

from __future__ import annotations
import pytest

from probly.transformation import ensemble

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

class TestRandomForestGeneration:
    
    def test_classifier(self) -> None:
        """Tests the correct generation of RandomForestClaassifier."""
        num_members = 3
        forest = ensemble(DecisionTreeClassifier, num_members=num_members, reset_params=False)
        
        assert isinstance(forest, RandomForestClassifier)
        