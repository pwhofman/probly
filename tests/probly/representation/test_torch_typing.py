"""Static contracts for Torch representations."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from tests.probly.representation._typing import check_types

_TYPING_CHECK = """
from typing import Any, assert_type
from ndlike.array_like import WeakArrayLike
from probly.method.efficient_credal_prediction._common import EfficientCredalPredictor
import torch
from probly.representation.distribution.torch_gaussian import TorchGaussianDistribution
from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution
from probly.representation.conformal_set.torch import TorchOneHotConformalSet
from probly.representation.conformal_set.torch import TorchIntervalConformalSet
from probly.representation.credal_set.torch import TorchConvexCredalSet
from probly.representation.credal_set.torch import TorchDistanceBasedCredalSet
from probly.representation.credal_set.torch import TorchProbabilityIntervalsCredalSet
from probly.representation.credal_set.torch import TorchDirichletLevelSetCredalSet
from probly.representation.embedding.torch import TorchEmbedding
from probly.representation.text_generation.torch import TorchTextGeneration
from probly.representation.text_generation.torch import TorchTokenGeneration
from probly.representation.distribution.torch_sparse_log_categorical import TorchSparseLogCategoricalDistribution
from probly.representation.distribution.torch_categorical import TorchProbabilityCategoricalDistribution
from probly.representation.distribution.torch_bernoulli import TorchProbabilityBernoulliDistribution
from probly.representation.distribution.torch_categorical import TorchLogitCategoricalDistribution
from probly.representation.distribution.torch_bernoulli import TorchLogitBernoulliDistribution
from probly.representation.sample.torch import TorchSample
from probly.representation.distribution.torch_mixture import TorchDirichletMixtureDistribution

def check_mixture(value: TorchDirichletMixtureDistribution) -> None:
    assert_type(value[0], TorchDirichletMixtureDistribution)
    assert_type(value.clone(), TorchDirichletMixtureDistribution)

def check_torchgaussiandistribution(value: TorchGaussianDistribution) -> None:
    assert_type(value[0], TorchGaussianDistribution)
    assert_type(value.reshape((1, -1)), TorchGaussianDistribution)
    assert_type(value.clone(), TorchGaussianDistribution)

def check_torchdirichletdistribution(value: TorchDirichletDistribution) -> None:
    assert_type(value[0], TorchDirichletDistribution)
    assert_type(value.reshape((1, -1)), TorchDirichletDistribution)
    assert_type(value.clone(), TorchDirichletDistribution)

def check_torchonehotconformalset(value: TorchOneHotConformalSet) -> None:
    assert_type(value[0], TorchOneHotConformalSet)
    assert_type(value.reshape((1, -1)), TorchOneHotConformalSet)
    assert_type(value.clone(), TorchOneHotConformalSet)

def check_torchintervalconformalset(value: TorchIntervalConformalSet) -> None:
    assert_type(value[0], TorchIntervalConformalSet)
    assert_type(value.reshape((1, -1)), TorchIntervalConformalSet)
    assert_type(value.clone(), TorchIntervalConformalSet)

def check_torchconvexcredalset(value: TorchConvexCredalSet) -> None:
    assert_type(value[0], TorchConvexCredalSet)
    assert_type(value.reshape((1, -1)), TorchConvexCredalSet)
    assert_type(value.clone(), TorchConvexCredalSet)

def check_torchdistancebasedcredalset(value: TorchDistanceBasedCredalSet) -> None:
    assert_type(value[0], TorchDistanceBasedCredalSet)
    assert_type(value.reshape((1, -1)), TorchDistanceBasedCredalSet)
    assert_type(value.clone(), TorchDistanceBasedCredalSet)

def check_torchprobabilityintervalscredalset(value: TorchProbabilityIntervalsCredalSet) -> None:
    assert_type(value[0], TorchProbabilityIntervalsCredalSet)
    assert_type(value.reshape((1, -1)), TorchProbabilityIntervalsCredalSet)
    assert_type(value.clone(), TorchProbabilityIntervalsCredalSet)

def check_torchdirichletlevelsetcredalset(value: TorchDirichletLevelSetCredalSet) -> None:
    assert_type(value[0], TorchDirichletLevelSetCredalSet)
    assert_type(value.reshape((1, -1)), TorchDirichletLevelSetCredalSet)
    assert_type(value.clone(), TorchDirichletLevelSetCredalSet)

def check_torchembedding(value: TorchEmbedding) -> None:
    assert_type(value[0], TorchEmbedding)
    assert_type(value.reshape((1, -1)), TorchEmbedding)
    assert_type(value.clone(), TorchEmbedding)

def check_torchtextgeneration(value: TorchTextGeneration) -> None:
    assert_type(value[0], TorchTextGeneration)
    assert_type(value.reshape((1, -1)), TorchTextGeneration)
    assert_type(value.clone(), TorchTextGeneration)

def check_torchtokengeneration(value: TorchTokenGeneration) -> None:
    assert_type(value[0], TorchTokenGeneration)
    assert_type(value.reshape((1, -1)), TorchTokenGeneration)
    assert_type(value.clone(), TorchTokenGeneration)

def check_torchsparselogcategoricaldistribution(value: TorchSparseLogCategoricalDistribution) -> None:
    assert_type(value[0], TorchSparseLogCategoricalDistribution)
    assert_type(value.reshape((1, -1)), TorchSparseLogCategoricalDistribution)
    assert_type(value.clone(), TorchSparseLogCategoricalDistribution)

def check_torchprobabilitycategoricaldistribution(value: TorchProbabilityCategoricalDistribution) -> None:
    assert_type(value[0], TorchProbabilityCategoricalDistribution[Any])
    assert_type(value.clone(), TorchProbabilityCategoricalDistribution[Any])

def check_torchprobabilitybernoullidistribution(value: TorchProbabilityBernoulliDistribution) -> None:
    assert_type(value[0], TorchProbabilityBernoulliDistribution)
    assert_type(value.clone(), TorchProbabilityBernoulliDistribution)

def check_torchlogitcategoricaldistribution(value: TorchLogitCategoricalDistribution) -> None:
    assert_type(value[0], TorchLogitCategoricalDistribution[Any])
    assert_type(value.clone(), TorchLogitCategoricalDistribution[Any])

def check_torchlogitbernoullidistribution(value: TorchLogitBernoulliDistribution) -> None:
    assert_type(value[0], TorchLogitBernoulliDistribution)
    assert_type(value.clone(), TorchLogitBernoulliDistribution)

def check_sample(value: TorchSample[torch.Tensor]) -> None:
    assert_type(value[0], TorchSample[torch.Tensor] | torch.Tensor)
    assert_type(value.sample_mean(), torch.Tensor)
    assert_type(value + 1, TorchSample[torch.Tensor] | torch.Tensor)
    assert_type(value.clone(), TorchSample[torch.Tensor])

def check_gaussian(value: TorchGaussianDistribution) -> None:
    assert_type(value + 1.0, TorchGaussianDistribution)

from probly.method.duq.torch import TorchDUQRepresentation
def check_duq(value: TorchDUQRepresentation) -> None:
    assert_type(value[0], TorchDUQRepresentation)

from probly.method.deup.torch import TorchDEUPRepresentation
def check_deup(value: TorchDEUPRepresentation) -> None:
    assert_type(value[0], TorchDEUPRepresentation)

from probly.method.ddu.torch import TorchDDURepresentation
def check_ddu(value: TorchDDURepresentation) -> None:
    assert_type(value[0], TorchDDURepresentation)

from probly.method.mahalanobis.torch import TorchMahalanobisRepresentation
def check_mahalanobis(value: TorchMahalanobisRepresentation) -> None:
    assert_type(value[0], TorchMahalanobisRepresentation)

def check_credal_bounds(predictor: EfficientCredalPredictor) -> None:
    assert_type(predictor.lower, WeakArrayLike[Any])
    assert_type(predictor.upper, WeakArrayLike[Any])
    predictor.lower = torch.ones(2)
    predictor.upper = torch.ones(2)
"""


def test_torch_representation_types(tmp_path: Path) -> None:
    check_types(_TYPING_CHECK, tmp_path)
