"""CredalBNN method compatibility layer."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from probly.representer import SampleMeanConvexCredalSetRepresenter, representer
from probly.transformation.bayesian_ensemble import BayesianEnsemblePredictor, bayesian_ensemble


@runtime_checkable
class CredalBNNPredictor[**In, Out](BayesianEnsemblePredictor[In, Out], Protocol):
    """A predictor routed through the CredalBNN representer."""


credal_bnn = CredalBNNPredictor.register_factory(bayesian_ensemble, autocast_builtins=True)
_credal_bnn_doc = """\
Create a Credal BNN predictor from a base predictor based on :cite:`caprioCredalBayesian2024`.

Args:
    base: The base model to replicate into a Bayesian ensemble.
    use_base_weights: If True, the weights of the base model are used as the prior mean. Default is False.
    posterior_std: Initial posterior standard deviation(s). Default is 0.05.
    prior_mean: Prior mean(s). Default is 0.0.
    prior_std: Prior standard deviation(s). Default is 1.0.
    num_members: Number of BNN ensemble members. Default is 5.

Returns:
    The credal BNN predictor outputting a convex credal set.
"""
credal_bnn.__doc__ = _credal_bnn_doc
representer.register(CredalBNNPredictor, SampleMeanConvexCredalSetRepresenter)


__all__ = ["CredalBNNPredictor", "credal_bnn"]
