"""Static contracts for Jax representations."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("jax")

from tests.probly.representation._typing import check_types

_TYPING_CHECK = """
from typing import Any, assert_type
import jax
from probly.representation.distribution.jax_gaussian import JaxGaussianDistribution
from probly.representation.distribution.jax_dirichlet import JaxDirichletDistribution
from probly.representation.conformal_set.jax import JaxOneHotConformalSet
from probly.representation.conformal_set.jax import JaxIntervalConformalSet
from probly.representation.credal_set.jax import JaxConvexCredalSet
from probly.representation.credal_set.jax import JaxDistanceBasedCredalSet
from probly.representation.credal_set.jax import JaxProbabilityIntervalsCredalSet
from probly.representation.credal_set.jax import JaxDirichletLevelSetCredalSet
from probly.representation.distribution.jax_categorical import JaxProbabilityCategoricalDistribution
from probly.representation.distribution.jax_bernoulli import JaxProbabilityBernoulliDistribution
from probly.representation.distribution.jax_categorical import JaxLogitCategoricalDistribution
from probly.representation.distribution.jax_bernoulli import JaxLogitBernoulliDistribution
from probly.representation.sample.jax import JaxSample

def check_jaxgaussiandistribution(value: JaxGaussianDistribution) -> None:
    assert_type(value[0], JaxGaussianDistribution)
    assert_type(value.reshape((1, -1)), JaxGaussianDistribution)
    assert_type(value.copy(), JaxGaussianDistribution)

def check_jaxdirichletdistribution(value: JaxDirichletDistribution) -> None:
    assert_type(value[0], JaxDirichletDistribution)
    assert_type(value.reshape((1, -1)), JaxDirichletDistribution)
    assert_type(value.copy(), JaxDirichletDistribution)

def check_jaxonehotconformalset(value: JaxOneHotConformalSet) -> None:
    assert_type(value[0], JaxOneHotConformalSet)
    assert_type(value.reshape((1, -1)), JaxOneHotConformalSet)
    assert_type(value.copy(), JaxOneHotConformalSet)

def check_jaxintervalconformalset(value: JaxIntervalConformalSet) -> None:
    assert_type(value[0], JaxIntervalConformalSet)
    assert_type(value.reshape((1, -1)), JaxIntervalConformalSet)
    assert_type(value.copy(), JaxIntervalConformalSet)

def check_jaxconvexcredalset(value: JaxConvexCredalSet) -> None:
    assert_type(value[0], JaxConvexCredalSet)
    assert_type(value.reshape((1, -1)), JaxConvexCredalSet)
    assert_type(value.copy(), JaxConvexCredalSet)

def check_jaxdistancebasedcredalset(value: JaxDistanceBasedCredalSet) -> None:
    assert_type(value[0], JaxDistanceBasedCredalSet)
    assert_type(value.reshape((1, -1)), JaxDistanceBasedCredalSet)
    assert_type(value.copy(), JaxDistanceBasedCredalSet)

def check_jaxprobabilityintervalscredalset(value: JaxProbabilityIntervalsCredalSet) -> None:
    assert_type(value[0], JaxProbabilityIntervalsCredalSet)
    assert_type(value.reshape((1, -1)), JaxProbabilityIntervalsCredalSet)
    assert_type(value.copy(), JaxProbabilityIntervalsCredalSet)

def check_jaxdirichletlevelsetcredalset(value: JaxDirichletLevelSetCredalSet) -> None:
    assert_type(value[0], JaxDirichletLevelSetCredalSet)
    assert_type(value.reshape((1, -1)), JaxDirichletLevelSetCredalSet)
    assert_type(value.copy(), JaxDirichletLevelSetCredalSet)

def check_jaxprobabilitycategoricaldistribution(value: JaxProbabilityCategoricalDistribution) -> None:
    assert_type(value[0], JaxProbabilityCategoricalDistribution[Any])
    assert_type(value.copy(), JaxProbabilityCategoricalDistribution[Any])

def check_jaxprobabilitybernoullidistribution(value: JaxProbabilityBernoulliDistribution) -> None:
    assert_type(value[0], JaxProbabilityBernoulliDistribution)
    assert_type(value.copy(), JaxProbabilityBernoulliDistribution)

def check_jaxlogitcategoricaldistribution(value: JaxLogitCategoricalDistribution) -> None:
    assert_type(value[0], JaxLogitCategoricalDistribution[Any])
    assert_type(value.copy(), JaxLogitCategoricalDistribution[Any])

def check_jaxlogitbernoullidistribution(value: JaxLogitBernoulliDistribution) -> None:
    assert_type(value[0], JaxLogitBernoulliDistribution)
    assert_type(value.copy(), JaxLogitBernoulliDistribution)

def check_sample(value: JaxSample[jax.Array]) -> None:
    assert_type(value[0], JaxSample[jax.Array] | jax.Array)
    assert_type(value.sample_mean(), jax.Array)
    assert_type(value + 1, JaxSample[jax.Array] | jax.Array)
    assert_type(value.copy(), JaxSample[jax.Array])

def check_gaussian(value: JaxGaussianDistribution) -> None:
    assert_type(value + 1.0, JaxGaussianDistribution)
"""


def test_jax_representation_types(tmp_path: Path) -> None:
    check_types(_TYPING_CHECK, tmp_path)
