"""Backend-independent predictor transformation contracts."""

from __future__ import annotations

import pytest

from probly.predictor import Predictor
from probly.transformation.ensemble._common import EnsemblePredictor, register_ensemble_members
from probly.transformation.transformation import current_predictor_type, predictor_transformation


class PlainPredictor:
    def predict(self, value: object) -> object:
        return value


def test_transformation_rejects_non_registry_predictor_type():
    @predictor_transformation(None)
    def identity(base):
        return base

    with pytest.raises(TypeError, match="flextype instance registration"):
        identity(PlainPredictor(), predictor_type=PlainPredictor)


def test_ensemble_registration_rejects_non_registry_predictor_type():
    members = [PlainPredictor()]
    assert isinstance(members, EnsemblePredictor)
    with pytest.raises(TypeError, match="flextype instance registration"):
        register_ensemble_members(members, PlainPredictor)


@pytest.mark.parametrize("fail_in_post_transform", [False, True])
def test_transformation_restores_context_on_error(fail_in_post_transform):
    initial = current_predictor_type.get()

    def post_transform(base, _predictor_type):
        if fail_in_post_transform:
            msg = "post-transform failed"
            raise ValueError(msg)
        return base

    @predictor_transformation(None, post_transform=post_transform)
    def failing(base):
        assert current_predictor_type.get()[0] is base
        if not fail_in_post_transform:
            msg = "transform failed"
            raise ValueError(msg)
        return base

    with pytest.raises(ValueError, match="failed"):
        failing(PlainPredictor(), predictor_type=Predictor)
    assert current_predictor_type.get() == initial
