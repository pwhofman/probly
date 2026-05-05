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
from probly.representation.distribution.array_categorical import (
    ArrayCategoricalDistribution,
    ArrayProbabilityCategoricalDistribution,
)
from probly.representer import (
    ConvexCredalSetRepresenter,
    ProbabilityIntervalsRepresenter,
    RepresentativeConvexCredalSetRepresenter,
    SampleMeanConvexCredalSetRepresenter,
    representer,
)


class _DummyEnsemble(list[Callable[[], ArrayCategoricalDistribution]]):
    __slots__ = ("__weakref__",)


def _categorical_member(probabilities: list[float]) -> Callable[[], ArrayCategoricalDistribution]:
    def predict() -> ArrayCategoricalDistribution:
        return ArrayProbabilityCategoricalDistribution(np.asarray(probabilities))

    return predict


def _ensemble() -> list[Callable[[], ArrayCategoricalDistribution]]:
    return _DummyEnsemble(
        [
            _categorical_member([0.8, 0.2]),
            _categorical_member([0.3, 0.7]),
        ]
    )


def test_credal_bnn_uses_sample_mean_convex_credal_set_representer() -> None:
    predictor = CredalBNNPredictor.register_instance(_ensemble())

    rep = representer(predictor)

    assert isinstance(rep, SampleMeanConvexCredalSetRepresenter)
    assert isinstance(rep, ConvexCredalSetRepresenter)
    assert rep.num_samples == 20
    assert len(rep.sub_samplers) == len(predictor)
    assert all(s.num_samples == 20 for s in rep.sub_samplers)


def test_credal_bnn_representer_honors_custom_num_samples() -> None:
    predictor = CredalBNNPredictor.register_instance(_ensemble())

    rep = SampleMeanConvexCredalSetRepresenter(predictor, num_samples=7)

    assert rep.num_samples == 7
    assert all(s.num_samples == 7 for s in rep.sub_samplers)


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


def test_credal_bnn_representer_yields_k_vertex_credal_set_torch() -> None:
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("torch.nn")

    from probly.method.credal_bnn import credal_bnn  # noqa: PLC0415
    from probly.representation.credal_set._common import ConvexCredalSet  # noqa: PLC0415

    torch.manual_seed(0)
    base = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 3))
    predictor = credal_bnn(base, num_members=4, predictor_type="logit_classifier")

    rep = representer(predictor)
    assert isinstance(rep, SampleMeanConvexCredalSetRepresenter)
    assert len(rep.sub_samplers) == 4

    cset = rep.represent(torch.randn(2, 4))

    assert isinstance(cset, ConvexCredalSet)
    # Vertex axis is the second-to-last; classes is the last.
    vertex_probs = cset.tensor.probabilities
    assert vertex_probs.shape[-2] == 4
    assert vertex_probs.shape[-1] == 3
