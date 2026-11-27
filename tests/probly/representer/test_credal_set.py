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
from probly.representation.distribution.numpy_categorical import (
    NumpyCategoricalDistribution,
    NumpyProbabilityCategoricalDistribution,
)
from probly.representation.sample import NumpySample
from probly.representer import (
    ConvexCredalSetRepresenter,
    ProbabilityIntervalsRepresenter,
    RepresentativeConvexCredalSetRepresenter,
)
from probly.representer.credal_set import compute_representative_sample


class _DummyEnsemble(list[Callable[[], NumpyCategoricalDistribution]]):
    __slots__ = ("__weakref__",)


def _categorical_member(probabilities: list[float]) -> Callable[[], NumpyCategoricalDistribution]:
    def predict() -> NumpyCategoricalDistribution:
        return NumpyProbabilityCategoricalDistribution(np.asarray(probabilities))

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

    def test_unsupported_sample_type_raises(self) -> None:
        """A sample type without a registered handler hits the default branch."""
        with pytest.raises(NotImplementedError, match="No representative-sample computation"):
            compute_representative_sample(object(), alpha=0.5, distance="euclidean")


def _numpy_sample(probabilities: list, sample_axis: int = 0) -> NumpySample[NumpyCategoricalDistribution]:
    return NumpySample(
        array=NumpyProbabilityCategoricalDistribution(np.asarray(probabilities)),
        sample_axis=sample_axis,
    )


class TestNumpyComputeRepresentativeSample:
    """``compute_representative_sample`` for NumPy samples of categorical distributions."""

    # Four members, one instance, three classes; members 1 and 0 are closest to the mean.
    members = (
        [[0.4, 0.4, 0.2]],
        [[0.5, 0.3, 0.2]],
        [[0.1, 0.1, 0.8]],
        [[0.8, 0.1, 0.1]],
    )

    def test_alpha_zero_returns_input(self) -> None:
        sample = _numpy_sample(list(self.members))

        assert compute_representative_sample(sample, alpha=0.0, distance="euclidean") is sample

    def test_keeps_members_closest_to_mean(self) -> None:
        sample = _numpy_sample(list(self.members))

        result = compute_representative_sample(sample, alpha=0.5, distance="euclidean")

        assert result.sample_size == 2
        np.testing.assert_allclose(result.samples.probabilities, [self.members[1], self.members[0]])

    def test_selects_per_instance_along_sample_axis(self) -> None:
        """With the sample axis in second position each instance keeps its own closest members."""
        first = [m[0] for m in self.members]
        second = first[::-1]
        sample = _numpy_sample([first, second], sample_axis=1)

        result = compute_representative_sample(sample, alpha=0.5, distance="euclidean")

        assert result.sample_axis == 1
        np.testing.assert_allclose(
            result.array.probabilities,
            [[first[1], first[0]], [first[1], first[0]]],
        )

    def test_alpha_one_keeps_one_member(self) -> None:
        sample = _numpy_sample(list(self.members))

        result = compute_representative_sample(sample, alpha=1.0, distance="euclidean")

        assert result.sample_size == 1

    def test_unsupported_distance_raises(self) -> None:
        sample = _numpy_sample(list(self.members))

        with pytest.raises(NotImplementedError, match="not implemented"):
            compute_representative_sample(sample, alpha=0.5, distance="manhattan")

    def test_agrees_with_torch(self) -> None:
        torch = pytest.importorskip("torch")
        from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
            TorchProbabilityCategoricalDistribution,
        )
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        rng = np.random.default_rng(0)
        probabilities = rng.dirichlet(np.ones(4), size=(5, 7))  # (instances, members, classes)
        numpy_sample = _numpy_sample(probabilities.tolist(), sample_axis=1)
        torch_sample = TorchSample(
            tensor=TorchProbabilityCategoricalDistribution(tensor=torch.as_tensor(probabilities)), sample_dim=1
        )

        numpy_result = compute_representative_sample(numpy_sample, alpha=0.4, distance="euclidean")
        torch_result = compute_representative_sample(torch_sample, alpha=0.4, distance="euclidean")

        np.testing.assert_allclose(
            numpy_result.array.probabilities, torch_result.tensor.probabilities.numpy(), rtol=1e-12
        )


class TestConvexCredalSetRepresenter:
    """Direct tests against ``ConvexCredalSetRepresenter`` (covers lines 50-51, 56)."""

    def test_represent_returns_convex_credal_set(self) -> None:
        predictor = CredalEnsemblingPredictor.register_instance(_ensemble())
        rep = ConvexCredalSetRepresenter(predictor)

        cset = rep.represent()

        assert isinstance(cset, ConvexCredalSet)

    def test_predict_returns_array_sample(self) -> None:
        """``_predict`` wraps the iterable predictions in an NumpySample (lines 50-51)."""
        predictor = CredalEnsemblingPredictor.register_instance(_ensemble())
        rep = ConvexCredalSetRepresenter(predictor)

        sample = rep._predict()  # noqa: SLF001

        assert isinstance(sample, NumpySample)
        assert sample.sample_size == len(predictor)


class TestProbabilityIntervalsRepresenter:
    """``ProbabilityIntervalsRepresenter`` for the torch backend (lines 77-78, 83-84)."""

    def test_predict_returns_a_sample(self) -> None:
        """``_predict`` produces a sample wrapping the categorical predictions (lines 77-78)."""
        from probly.method.credal_wrapper import CredalWrapperPredictor  # noqa: PLC0415

        predictor = CredalWrapperPredictor.register_instance(_ensemble())
        rep = ProbabilityIntervalsRepresenter(predictor)

        sample = rep._predict()  # noqa: SLF001

        assert isinstance(sample, NumpySample)
        assert sample.sample_size == len(predictor)

    def test_represent_returns_probability_intervals_numpy(self) -> None:
        """``represent`` for a NumPy ensemble produces the envelope of the member predictions."""
        from probly.method.credal_wrapper import CredalWrapperPredictor  # noqa: PLC0415
        from probly.representation.credal_set import ProbabilityIntervalsCredalSet  # noqa: PLC0415

        predictor = CredalWrapperPredictor.register_instance(_ensemble())
        rep = ProbabilityIntervalsRepresenter(predictor)

        cset = rep.represent()

        assert isinstance(cset, ProbabilityIntervalsCredalSet)
        np.testing.assert_allclose(cset.lower(), [0.3, 0.2])
        np.testing.assert_allclose(cset.upper(), [0.8, 0.7])

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
        """``alpha=0`` keeps every member of a NumPy ensemble."""
        predictor = CredalEnsemblingPredictor.register_instance(_ensemble())
        rep = RepresentativeConvexCredalSetRepresenter(predictor, alpha=0.0, distance="euclidean")

        sample = rep._predict()  # noqa: SLF001

        assert isinstance(sample, NumpySample)
        assert sample.sample_size == len(predictor)

    def test_represent_drops_distant_members_numpy(self) -> None:
        """``alpha=0.5`` on a two-member NumPy ensemble leaves a single vertex."""
        predictor = CredalEnsemblingPredictor.register_instance(_ensemble())
        rep = RepresentativeConvexCredalSetRepresenter(predictor, alpha=0.5, distance="euclidean")

        cset = rep.represent()

        assert isinstance(cset, ConvexCredalSet)
        np.testing.assert_allclose(cset.lower(), cset.upper())

    def test_alpha_zero_falls_back_to_unfiltered(self) -> None:
        """alpha=0 short-circuits the torch handler — direct call test."""
        torch = pytest.importorskip("torch")
        from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
            TorchProbabilityCategoricalDistribution,
        )
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415
        from probly.representer.torch_credal_set import torch_compute_representative_sample  # noqa: PLC0415

        dist = TorchProbabilityCategoricalDistribution(tensor=torch.tensor([[[0.5, 0.5]], [[0.3, 0.7]]]))
        sample = TorchSample(tensor=dist, sample_dim=0)
        result = torch_compute_representative_sample(sample, alpha=0.0, distance="euclidean")
        assert result is sample


class TestLazyTorchRegistration:
    """The torch lazy registration (line 151) triggers when a TorchSample is passed."""

    def test_torch_handler_registered(self) -> None:
        """Importing the torch sub-module exposes the torch dispatcher to the registry."""
        pytest.importorskip("torch")
        import probly.representer.torch_credal_set as _torch_mod  # noqa: PLC0415

        # The module wires the torch dispatch into ``compute_representative_sample``.
        assert _torch_mod.torch_compute_representative_sample is not None

    def test_torch_handler_short_circuit_alpha_zero(self) -> None:
        """Calling the torch handler directly with alpha=0 returns the sample unchanged."""
        torch = pytest.importorskip("torch")
        from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
            TorchProbabilityCategoricalDistribution,
        )
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415
        from probly.representer.torch_credal_set import torch_compute_representative_sample  # noqa: PLC0415

        dist = TorchProbabilityCategoricalDistribution(tensor=torch.tensor([[[0.4, 0.4, 0.2]], [[0.6, 0.3, 0.1]]]))
        sample = TorchSample(tensor=dist, sample_dim=0)

        result = torch_compute_representative_sample(sample, alpha=0.0, distance="euclidean")
        assert result is sample
