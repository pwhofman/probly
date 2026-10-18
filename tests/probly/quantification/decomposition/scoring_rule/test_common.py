"""Tests for the dispatch of the loss-induced decomposition."""

from __future__ import annotations

import numpy as np
import pytest

from probly.quantification import (
    BrierLoss,
    LogLoss,
    SecondOrderScoringRuleDecomposition,
    SecondOrderZeroOneDecomposition,
    ZeroOneLoss,
    scoring_rule_decomposition,
)
from probly.quantification.decomposition.decomposition import ConstantTotalDecomposition
from probly.representation.distribution import (
    create_categorical_distribution,
    create_dirichlet_distribution_from_alphas,
)
from probly.representation.sample import create_sample

_PROBS = [np.array([[0.7, 0.3], [0.4, 0.6]]), np.array([[0.5, 0.5], [0.2, 0.8]])]


def _sample():
    return create_sample([create_categorical_distribution(p) for p in _PROBS])


def test_sample_with_zero_one_loss_gives_the_zero_one_decomposition() -> None:
    assert isinstance(scoring_rule_decomposition(_sample(), ZeroOneLoss()), SecondOrderZeroOneDecomposition)


@pytest.mark.parametrize("loss", [LogLoss(), BrierLoss()])
def test_sample_with_other_losses_gives_the_scoring_rule_decomposition(loss) -> None:
    decomposition = scoring_rule_decomposition(_sample(), loss)

    assert isinstance(decomposition, SecondOrderScoringRuleDecomposition)
    assert decomposition.scoring_rule is loss


def test_single_distribution_gives_a_total_only_decomposition() -> None:
    decomposition = scoring_rule_decomposition(create_categorical_distribution(_PROBS[0]), LogLoss())

    assert isinstance(decomposition, ConstantTotalDecomposition)
    np.testing.assert_allclose(decomposition.total, -(_PROBS[0] * np.log(_PROBS[0])).sum(axis=-1))


def test_dirichlet_supports_only_the_zero_one_loss() -> None:
    dirichlet = create_dirichlet_distribution_from_alphas(np.array([[1.0, 2.0], [3.0, 1.0]]))

    assert isinstance(scoring_rule_decomposition(dirichlet, ZeroOneLoss()), SecondOrderZeroOneDecomposition)
    with pytest.raises(NotImplementedError, match="Dirichlet"):
        scoring_rule_decomposition(dirichlet, LogLoss())


def test_unregistered_representation_raises() -> None:
    with pytest.raises(NotImplementedError, match="not supported for object"):
        scoring_rule_decomposition(object(), ZeroOneLoss())
