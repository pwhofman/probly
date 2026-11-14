"""Static contracts for Numpy representations."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("numpy")

from tests.probly.representation._typing import check_types

_TYPING_CHECK = """
from typing import Any, assert_type
import numpy as np
from probly.representation.distribution.numpy_gaussian import NumpyGaussianDistribution
from probly.representation.distribution.numpy_dirichlet import NumpyDirichletDistribution
from probly.representation.conformal_set.numpy import NumpyOneHotConformalSet
from probly.representation.conformal_set.numpy import NumpyIntervalConformalSet
from probly.representation.credal_set.numpy import NumpyConvexCredalSet
from probly.representation.credal_set.numpy import NumpyDistanceBasedCredalSet
from probly.representation.credal_set.numpy import NumpyProbabilityIntervalsCredalSet
from probly.representation.credal_set.numpy import NumpyDiscreteCredalSet
from probly.representation.credal_set.numpy import NumpySingletonCredalSet
from probly.representation.distribution.numpy_categorical import NumpyProbabilityCategoricalDistribution
from probly.representation.distribution.numpy_bernoulli import NumpyProbabilityBernoulliDistribution
from probly.representation.distribution.numpy_categorical import NumpyLogitCategoricalDistribution
from probly.representation.distribution.numpy_bernoulli import NumpyLogitBernoulliDistribution
from probly.representation.sample.numpy import NumpySample

def check_numpygaussiandistribution(value: NumpyGaussianDistribution) -> None:
    assert_type(value[0], NumpyGaussianDistribution)
    assert_type(value.reshape((1, -1)), NumpyGaussianDistribution)
    assert_type(value.copy(), NumpyGaussianDistribution)

def check_numpydirichletdistribution(value: NumpyDirichletDistribution) -> None:
    assert_type(value[0], NumpyDirichletDistribution)
    assert_type(value.reshape((1, -1)), NumpyDirichletDistribution)
    assert_type(value.copy(), NumpyDirichletDistribution)

def check_numpyonehotconformalset(value: NumpyOneHotConformalSet) -> None:
    assert_type(value[0], NumpyOneHotConformalSet)
    assert_type(value.reshape((1, -1)), NumpyOneHotConformalSet)
    assert_type(value.copy(), NumpyOneHotConformalSet)

def check_numpyintervalconformalset(value: NumpyIntervalConformalSet) -> None:
    assert_type(value[0], NumpyIntervalConformalSet)
    assert_type(value.reshape((1, -1)), NumpyIntervalConformalSet)
    assert_type(value.copy(), NumpyIntervalConformalSet)

def check_numpyconvexcredalset(value: NumpyConvexCredalSet) -> None:
    assert_type(value[0], NumpyConvexCredalSet)
    assert_type(value.reshape((1, -1)), NumpyConvexCredalSet)
    assert_type(value.copy(), NumpyConvexCredalSet)

def check_numpydistancebasedcredalset(value: NumpyDistanceBasedCredalSet) -> None:
    assert_type(value[0], NumpyDistanceBasedCredalSet)
    assert_type(value.reshape((1, -1)), NumpyDistanceBasedCredalSet)
    assert_type(value.copy(), NumpyDistanceBasedCredalSet)

def check_numpyprobabilityintervalscredalset(value: NumpyProbabilityIntervalsCredalSet) -> None:
    assert_type(value[0], NumpyProbabilityIntervalsCredalSet)
    assert_type(value.reshape((1, -1)), NumpyProbabilityIntervalsCredalSet)
    assert_type(value.copy(), NumpyProbabilityIntervalsCredalSet)

def check_numpydiscretecredalset(value: NumpyDiscreteCredalSet) -> None:
    assert_type(value[0], NumpyDiscreteCredalSet)
    assert_type(value.reshape((1, -1)), NumpyDiscreteCredalSet)
    assert_type(value.copy(), NumpyDiscreteCredalSet)

def check_numpysingletoncredalset(value: NumpySingletonCredalSet) -> None:
    assert_type(value[0], NumpySingletonCredalSet)
    assert_type(value.reshape((1, -1)), NumpySingletonCredalSet)
    assert_type(value.copy(), NumpySingletonCredalSet)

def check_numpyprobabilitycategoricaldistribution(value: NumpyProbabilityCategoricalDistribution) -> None:
    assert_type(value[0], NumpyProbabilityCategoricalDistribution[Any])
    assert_type(value.copy(), NumpyProbabilityCategoricalDistribution[Any])

def check_numpyprobabilitybernoullidistribution(value: NumpyProbabilityBernoulliDistribution) -> None:
    assert_type(value[0], NumpyProbabilityBernoulliDistribution)
    assert_type(value.copy(), NumpyProbabilityBernoulliDistribution)

def check_numpylogitcategoricaldistribution(value: NumpyLogitCategoricalDistribution) -> None:
    assert_type(value[0], NumpyLogitCategoricalDistribution[Any])
    assert_type(value.copy(), NumpyLogitCategoricalDistribution[Any])

def check_numpylogitbernoullidistribution(value: NumpyLogitBernoulliDistribution) -> None:
    assert_type(value[0], NumpyLogitBernoulliDistribution)
    assert_type(value.copy(), NumpyLogitBernoulliDistribution)

def check_sample(value: NumpySample[np.ndarray]) -> None:
    assert_type(value[0], NumpySample[np.ndarray] | np.ndarray)
    assert_type(value.sample_mean(), np.ndarray)
    assert_type(value + 1, NumpySample[np.ndarray] | np.ndarray)
    assert_type(value.copy(), NumpySample[np.ndarray])

def check_gaussian(value: NumpyGaussianDistribution) -> None:
    assert_type(value + 1.0, NumpyGaussianDistribution)

from probly.method.mahalanobis.numpy import NumpyMahalanobisRepresentation
def check_mahalanobis(value: NumpyMahalanobisRepresentation) -> None:
    assert_type(value[0], NumpyMahalanobisRepresentation)
"""


def test_numpy_representation_types(tmp_path: Path) -> None:
    check_types(_TYPING_CHECK, tmp_path)
