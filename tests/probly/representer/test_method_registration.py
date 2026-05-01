"""Tests for method-owned representer registrations."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from probly.method.credal_bnn import CredalBNNPredictor
from probly.method.credal_ensembling import CredalEnsemblingPredictor
from probly.method.credal_relative_likelihood import CredalRelativeLikelihoodPredictor
from probly.method.credal_wrapper import CredalWrapperPredictor
from probly.method.efficient_credal_prediction import EfficientCredalRepresenter, efficient_credal_prediction
from probly.representation.credal_set import ProbabilityIntervalsCredalSet
from probly.representation.distribution import ArrayCategoricalDistribution
from probly.representer import (
    ConvexCredalSetRepresenter,
    ProbabilityIntervalsRepresenter,
    RepresentativeConvexCredalSetRepresenter,
    representer,
)


class _DummyEnsemble(list[Callable[[], ArrayCategoricalDistribution]]):
    __slots__ = ("__weakref__",)


def _categorical_member(probabilities: list[float]) -> Callable[[], ArrayCategoricalDistribution]:
    def predict() -> ArrayCategoricalDistribution:
        return ArrayCategoricalDistribution(np.asarray(probabilities))

    return predict


def _ensemble() -> list[Callable[[], ArrayCategoricalDistribution]]:
    return _DummyEnsemble(
        [
            _categorical_member([0.8, 0.2]),
            _categorical_member([0.3, 0.7]),
        ]
    )


def test_credal_bnn_uses_convex_credal_set_representer() -> None:
    predictor = CredalBNNPredictor.register_instance(_ensemble())

    rep = representer(predictor)

    assert isinstance(rep, ConvexCredalSetRepresenter)


def test_credal_ensembling_uses_representative_convex_credal_set_representer() -> None:
    predictor = CredalEnsemblingPredictor.register_instance(_ensemble())

    rep = representer(predictor)

    assert isinstance(rep, RepresentativeConvexCredalSetRepresenter)


def test_credal_wrapper_uses_probability_interval_representer() -> None:
    predictor = CredalWrapperPredictor.register_instance(_ensemble())

    rep = representer(predictor)

    assert isinstance(rep, ProbabilityIntervalsRepresenter)


def test_credal_relative_likelihood_uses_probability_interval_representer() -> None:
    predictor = CredalRelativeLikelihoodPredictor.register_instance(_ensemble())

    rep = representer(predictor)

    assert isinstance(rep, ProbabilityIntervalsRepresenter)


def test_efficient_credal_prediction_uses_method_local_representer() -> None:
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("torch.nn")

    predictor = efficient_credal_prediction(nn.Linear(2, 2), predictor_type="logit_classifier")
    predictor.lower = torch.zeros(2)
    predictor.upper = torch.zeros(2)

    rep = representer(predictor)
    output = rep.predict(torch.ones(1, 2))

    assert isinstance(rep, EfficientCredalRepresenter)
    assert isinstance(output, ProbabilityIntervalsCredalSet)
