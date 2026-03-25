"""Shared CredalBNN implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from probly.transformation.bayesian.common import bayesian
from probly.transformation.ensemble.common import ensemble
from pytraverse import GlobalVariable

if TYPE_CHECKING:
    from probly.predictor import EnsemblePredictor, Predictor

USE_BASE_WEIGHTS = GlobalVariable[bool]("USE_BASE_WEIGHTS", default=False)
POSTERIOR_STD = GlobalVariable[list[float]]("POSTERIOR_STD", default=[0.05] * 5)
PRIOR_MEAN = GlobalVariable[list[float]]("PRIOR_MEAN", default=[-1.0, -0.5, 0.0, 0.5, 1.0])
PRIOR_STD = GlobalVariable[list[float]]("PRIOR_STD", default=[0.1, 0.325, 0.55, 0.775, 1.0])
NUM_MEMBERS = GlobalVariable[int]("NUM_MEMBERS", default=5)


def credalbnn[**In, Out](
    base: Predictor[In, Out],
    use_base_weights: bool = USE_BASE_WEIGHTS.default,
    posterior_std: list[float] = POSTERIOR_STD.default,
    prior_mean: list[float] = PRIOR_MEAN.default,
    prior_std: list[float] = PRIOR_STD.default,
    num_members: int = NUM_MEMBERS.default,
) -> EnsemblePredictor[In, Out]:
    """Create a CredalBNN predictor from a base predictor based on :cite:`caprio2023credalbnn`.

    Args:
        base: The base model to be used for the CredalBNN ensemble.
        use_base_weights: If True, the weights of the base model are used as the prior mean.
        posterior_std: The list of initial posterior standard deviations.
        prior_mean: The list of prior means.
        prior_std: The list of prior standard deviations.
        num_members: The number of members in the ensemble.

    Returns:
        The CredalBNN predictor.
    """
    if min(posterior_std) <= 0:
        msg = (
            "Any initial posterior standard deviation posterior_std must be greater than 0, "
            f"but got one value of {posterior_std} instead."
        )
        raise ValueError(msg)
    if min(prior_std) <= 0:
        msg = (
            f"Any prior standard deviation prior_std must be greater than 0, but got  one value of {prior_std} instead."
        )
        raise ValueError(msg)
    if len(posterior_std) != num_members or len(prior_std) != num_members or len(prior_mean) != num_members:
        msg = (
            f"posterior_std, prior_std and prior_mean must have length of {num_members} (num_members), but got"
            f" {len(posterior_std)}, {len(prior_std)} and {len(prior_mean)} respectively."
        )

    bnn_members = [
        bayesian(
            base,
            use_base_weights=use_base_weights,
            posterior_std=posterior_std[i],
            prior_mean=prior_mean[i],
            prior_std=prior_std[i],
        )
        for i in range(num_members)
    ]

    # TODO(tloehr): fix ensemble based on list of bnn members instead of just one  # noqa: TD003
    return ensemble(bnn_members[0], num_members=num_members, reset_params=False)
