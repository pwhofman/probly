"""Shared CredalBNN implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from probly.method.bayesian._common import bayesian
from probly.method.ensemble import EnsemblePredictor, register_ensemble_members
from probly.method.method import predictor_transformation
from probly.predictor import ProbabilisticClassifier
from pytraverse import GlobalVariable

if TYPE_CHECKING:
    from probly.predictor import Predictor

USE_BASE_WEIGHTS = GlobalVariable[bool]("USE_BASE_WEIGHTS", default=False)
POSTERIOR_STD = GlobalVariable[float]("POSTERIOR_STD", default=0.05)
PRIOR_MEAN = GlobalVariable[float]("PRIOR_MEAN", default=0.0)
PRIOR_STD = GlobalVariable[float]("PRIOR_STD", default=1.0)
NUM_MEMBERS = GlobalVariable[int]("NUM_MEMBERS", default=5)


@runtime_checkable
class CredalBNNPredictor[**In, Out](EnsemblePredictor[In, Out], Protocol):
    """A predictor that applies the Credal Bayesian Neural Network transformation."""


def _resolve_param(
    value: float | list[float] | None,
    num_members: int,
    default: float,
    name: str,
) -> list[float]:
    """Resolve a scalar, list, or None parameter to a list of length num_members."""
    if value is None:
        return [default] * num_members
    if isinstance(value, (int, float)):
        return [float(value)] * num_members
    if len(value) != num_members:
        msg = f"'{name}' has length {len(value)}, expected {num_members} (num_members)."
        raise ValueError(msg)
    return list(value)


@predictor_transformation(
    permitted_predictor_types=(ProbabilisticClassifier,), post_transform=register_ensemble_members
)
@CredalBNNPredictor.register_factory
def credal_bnn[**In, Out](
    base: Predictor[In, Out],
    use_base_weights: bool = USE_BASE_WEIGHTS.default,
    posterior_std: float | list[float] | None = POSTERIOR_STD.default,
    prior_mean: float | list[float] | None = PRIOR_MEAN.default,
    prior_std: float | list[float] | None = PRIOR_STD.default,
    num_members: int = NUM_MEMBERS.default,
) -> CredalBNNPredictor[In, Out]:
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
    posterior_stds = _resolve_param(posterior_std, num_members, POSTERIOR_STD.default, "posterior_std")
    prior_means = _resolve_param(prior_mean, num_members, PRIOR_MEAN.default, "prior_mean")
    prior_stds = _resolve_param(prior_std, num_members, PRIOR_STD.default, "prior_std")

    bnn_members = [
        bayesian(
            base,
            use_base_weights=use_base_weights,
            posterior_std=posterior_stds[i],
            prior_mean=prior_means[i],
            prior_std=prior_stds[i],
        )
        for i in range(num_members)
    ]

    return bnn_members  # ty:ignore[invalid-return-type]
