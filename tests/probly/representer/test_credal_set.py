"""Tests for ``probly.representer.credal_set``.

Behavioural coverage of the generic credal-set representers and the
``compute_representative_sample`` flexdispatch fallback.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from probly.method.credal_ensembling import CredalEnsemblingPredictor
from probly.representation.credal_set._common import ConvexCredalSet
from probly.representation.distribution.array_categorical import (
    ArrayCategoricalDistribution,
    ArrayProbabilityCategoricalDistribution,
)
from probly.representation.sample import ArraySample
from probly.representer import (
    ConvexCredalSetRepresenter,
    ProbabilityIntervalsRepresenter,
    RepresentativeConvexCredalSetRepresenter,
)
from probly.representer.credal_set import compute_representative_sample


class _DummyEnsemble(list[Callable[[], ArrayCategoricalDistribution]]):
    __slots__ = ("__weakref__",)


def _categorical_member(probabilities: list[float]) -> Callable[[], ArrayCategoricalDistribution]:
    def predict() -> ArrayCategoricalDistribution:
        return ArrayProbabilityCategoricalDistribution(np.asarray(probabilities))

    return predict


def _ensemble() -> _DummyEnsemble:
    return _DummyEnsemble(
        [
            _categorical_member([0.8, 0.2]),
            _categorical_member([0.3, 0.7]),
        ]
    )


class TestComputeRepresentativeSampleDispatch:
    """``compute_representative_sample`` raises for unsupported sample backends."""

    def test_array_sample_without_handler_raises(self) -> None:
        """An ArraySample (no array handler registered) hits the default branch (lines 29-30)."""
        sample = ArraySample(
            array=ArrayProbabilityCategoricalDistribution(np.array([[0.5, 0.5], [0.7, 0.3]])),
            sample_axis=0,
        )

        with pytest.raises(NotImplementedError, match="No representative-sample computation"):
            compute_representative_sample(sample, alpha=0.5, distance="euclidean")


class TestConvexCredalSetRepresenter:
    """Direct tests against ``ConvexCredalSetRepresenter`` (covers lines 50-51, 56)."""

    def test_represent_returns_convex_credal_set(self) -> None:
        predictor = CredalEnsemblingPredictor.register_instance(_ensemble())
        rep = ConvexCredalSetRepresenter(predictor)

        cset = rep.represent()

        assert isinstance(cset, ConvexCredalSet)

    def test_predict_returns_array_sample(self) -> None:
        """``_predict`` wraps the iterable predictions in an ArraySample (lines 50-51)."""
        predictor = CredalEnsemblingPredictor.register_instance(_ensemble())
        rep = ConvexCredalSetRepresenter(predictor)

        sample = rep._predict()  # noqa: SLF001

        assert isinstance(sample, ArraySample)
        assert sample.sample_size == len(predictor)


class TestProbabilityIntervalsRepresenter:
    """``ProbabilityIntervalsRepresenter`` for the torch backend (lines 77-78, 83-84)."""

    def test_predict_returns_a_sample(self) -> None:
        """``_predict`` produces a sample wrapping the categorical predictions (lines 77-78)."""
        from probly.method.credal_wrapper import CredalWrapperPredictor  # noqa: PLC0415

        predictor = CredalWrapperPredictor.register_instance(_ensemble())
        rep = ProbabilityIntervalsRepresenter(predictor)

        sample = rep._predict()  # noqa: SLF001

        assert isinstance(sample, ArraySample)
        assert sample.sample_size == len(predictor)

    def test_represent_returns_probability_intervals_torch(self) -> None:
        """``represent`` for a torch ensemble produces a ProbabilityIntervalsCredalSet (lines 83-84)."""
        torch = pytest.importorskip("torch")
        from probly.method.credal_wrapper import CredalWrapperPredictor  # noqa: PLC0415
        from probly.representation.credal_set import ProbabilityIntervalsCredalSet  # noqa: PLC0415
        from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
            TorchProbabilityCategoricalDistribution,
        )

        class _TorchEnsemble(list[Callable[[], TorchProbabilityCategoricalDistribution]]):
            __slots__ = ("__weakref__",)

        def _torch_member(probs: list[float]) -> Callable[[], TorchProbabilityCategoricalDistribution]:
            def predict() -> TorchProbabilityCategoricalDistribution:
                return TorchProbabilityCategoricalDistribution(tensor=torch.tensor(probs))

            return predict

        ensemble = _TorchEnsemble([_torch_member([0.8, 0.2]), _torch_member([0.3, 0.7])])
        predictor = CredalWrapperPredictor.register_instance(ensemble)
        rep = ProbabilityIntervalsRepresenter(predictor)

        cset = rep.represent()

        assert isinstance(cset, ProbabilityIntervalsCredalSet)


class TestRepresentativeConvexCredalSetRepresenter:
    """``RepresentativeConvexCredalSetRepresenter`` (lines 50-51 via super, 110-111)."""

    def test_predict_alpha_zero_passes_through(self) -> None:
        """``alpha=0`` calls super()._predict (lines 50-51) then short-circuits in the torch handler.

        For ArraySample inputs there is no registered handler for ``compute_representative_sample``,
        so we call the parent's ``_predict`` directly to cover lines 50-51.
        """
        predictor = CredalEnsemblingPredictor.register_instance(_ensemble())
        rep = RepresentativeConvexCredalSetRepresenter(predictor, alpha=0.0, distance="euclidean")

        # Parent _predict (lines 50-51) returns an ArraySample.
        sample = ConvexCredalSetRepresenter._predict(rep)  # type: ignore[arg-type]  # noqa: SLF001

        assert isinstance(sample, ArraySample)

    def test_alpha_zero_falls_back_to_unfiltered(self) -> None:
        """alpha=0 short-circuits the torch handler — direct call test."""
        torch = pytest.importorskip("torch")
        from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
            TorchProbabilityCategoricalDistribution,
        )
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415
        from probly.representer.credal_set_torch import torch_compute_representative_sample  # noqa: PLC0415

        dist = TorchProbabilityCategoricalDistribution(tensor=torch.tensor([[[0.5, 0.5]], [[0.3, 0.7]]]))
        sample = TorchSample(tensor=dist, sample_dim=0)
        result = torch_compute_representative_sample(sample, alpha=0.0, distance="euclidean")
        assert result is sample


class TestLazyTorchRegistration:
    """The torch lazy registration (line 151) triggers when a TorchSample is passed."""

    def test_torch_handler_registered(self) -> None:
        """Importing the torch sub-module exposes the torch dispatcher to the registry."""
        pytest.importorskip("torch")
        import probly.representer.credal_set_torch as _torch_mod  # noqa: PLC0415

        # The module wires the torch dispatch into ``compute_representative_sample``.
        assert _torch_mod.torch_compute_representative_sample is not None

    def test_torch_handler_short_circuit_alpha_zero(self) -> None:
        """Calling the torch handler directly with alpha=0 returns the sample unchanged."""
        torch = pytest.importorskip("torch")
        from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
            TorchProbabilityCategoricalDistribution,
        )
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415
        from probly.representer.credal_set_torch import torch_compute_representative_sample  # noqa: PLC0415

        dist = TorchProbabilityCategoricalDistribution(tensor=torch.tensor([[[0.4, 0.4, 0.2]], [[0.6, 0.3, 0.1]]]))
        sample = TorchSample(tensor=dist, sample_dim=0)

        result = torch_compute_representative_sample(sample, alpha=0.0, distance="euclidean")
        assert result is sample


class TestMLEProbabilityIntervalsRepresenter:
    """The MLE interval representer wraps an iterable predictor and produces an MLE credal set."""

    def test_represent_returns_mle_probability_intervals_credal_set(self) -> None:
        from probly.method.credal_relative_likelihood import CredalRelativeLikelihoodPredictor  # noqa: PLC0415
        from probly.representation.credal_set._common import MLEProbabilityIntervalsCredalSet  # noqa: PLC0415
        from probly.representer import MLEProbabilityIntervalsRepresenter  # noqa: PLC0415

        predictor = CredalRelativeLikelihoodPredictor.register_instance(_ensemble())
        rep = MLEProbabilityIntervalsRepresenter(predictor)

        cset = rep.represent()

        assert isinstance(cset, MLEProbabilityIntervalsCredalSet)
        # MLE corresponds to the first ensemble member [0.8, 0.2]
        np.testing.assert_allclose(cset.mle.probabilities, [0.8, 0.2])
        # Bounds span min/max over [0.8, 0.2] and [0.3, 0.7]
        np.testing.assert_allclose(cset.lower_bounds, [0.3, 0.2])
        np.testing.assert_allclose(cset.upper_bounds, [0.8, 0.7])
