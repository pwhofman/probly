"""Bayesian Neural Network (BNN) implementation."""

from __future__ import annotations

import torch
from torch import nn

from probly.representation.layers import BayesConv2d, BayesLinear
from probly.representation.predictor_torch import TorchSamplingRepresentationPredictor
from probly.traverse_nn import nn_compose
from pytraverse import (
    CLONE,
    GlobalVariable,
    State,
    TraverserResult,
    singledispatch_traverser,
    traverse,
    traverse_with_state,
)

USE_BASE_WEIGHTS = GlobalVariable[bool]("USE_BASE_WEIGHTS", default=False)
POSTERIOR_STD = GlobalVariable[float]("POSTERIOR_STD", default=0.05)
PRIOR_MEAN = GlobalVariable[float]("PRIOR_MEAN", default=0.0)
PRIOR_STD = GlobalVariable[float]("PRIOR_STD", default=1.0)

KL_DIVERGENCE = GlobalVariable[torch.Tensor | float]("KL_DIVERGENCE", default=0.0)

bayesian_traverser = singledispatch_traverser[object](name="bayesian_traverser")


@bayesian_traverser.register(
    vars={
        "use_base_weights": USE_BASE_WEIGHTS,
        "posterior_std": POSTERIOR_STD,
        "prior_mean": PRIOR_MEAN,
        "prior_std": PRIOR_STD,
    },
)
def _(
    obj: nn.Linear,
    *,
    use_base_weights: bool,
    posterior_std: float,
    prior_mean: float,
    prior_std: float,
) -> BayesLinear:
    return BayesLinear(
        obj.in_features,
        obj.out_features,
        obj.bias is not None,
        posterior_std,
        prior_mean,
        prior_std,
        obj if use_base_weights else None,
    )


@bayesian_traverser.register(
    vars={
        "use_base_weights": USE_BASE_WEIGHTS,
        "posterior_std": POSTERIOR_STD,
        "prior_mean": PRIOR_MEAN,
        "prior_std": PRIOR_STD,
    },
)
def _(
    obj: nn.Conv2d,
    *,
    use_base_weights: bool,
    posterior_std: float,
    prior_mean: float,
    prior_std: float,
) -> BayesConv2d:
    return BayesConv2d(
        obj.in_channels,
        obj.out_channels,
        obj.kernel_size,
        obj.stride,
        obj.padding,
        obj.dilation,
        obj.groups,
        obj.bias is not None,
        posterior_std,
        prior_mean,
        prior_std,
        obj if use_base_weights else None,
    )


@singledispatch_traverser[object]
def kl_divergence_traverser(
    obj: BayesLinear | BayesConv2d,
    state: State,
) -> TraverserResult[BayesLinear | BayesConv2d]:
    """Traverser to compute the KL divergence of a Bayesian layer."""
    state[KL_DIVERGENCE] += obj.kl_divergence
    return obj, state


class Bayesian[In, KwIn](TorchSamplingRepresentationPredictor[In, KwIn]):
    """Implementation of a Bayesian neural network to be used for uncertainty quantification.

    Implementation is based on :cite:`blundellWeightUncertainty2015`.

    Attributes:
        model: torch.nn.Module, The transformed model with Bayesian layers.
    """

    def __init__(
        self,
        base: nn.Module,
        use_base_weights: bool = USE_BASE_WEIGHTS.default,
        posterior_std: float = POSTERIOR_STD.default,
        prior_mean: float = PRIOR_MEAN.default,
        prior_std: float = PRIOR_STD.default,
    ) -> None:
        """Initialize an instance of the Bayesian class.

        Convert the base model into a Bayesian model by replacing suitable layers by Bayesian layers.

        Args:
            base: torch.nn.Module, The base model.
            use_base_weights: bool, If True, the weights of the base model are used as the prior mean.
            posterior_std: float, The initial posterior standard deviation.
            prior_mean: float, The prior mean.
            prior_std: float, The prior standard deviation.
        """
        self.use_base_weights = use_base_weights
        self.posterior_std = posterior_std
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        super().__init__(base)

    def _convert(self, base: nn.Module) -> nn.Module:
        """Converts the base model to a Bayesian model by replacing all layers by Bayesian layers.

        Args:
            base: torch.nn.Module, The base model to be used for dropout.
        """
        return traverse(
            base,
            nn_compose(bayesian_traverser),
            init={
                CLONE: True,
                USE_BASE_WEIGHTS: self.use_base_weights,
                POSTERIOR_STD: self.posterior_std,
                PRIOR_MEAN: self.prior_mean,
                PRIOR_STD: self.prior_std,
            },
        )

    @property
    def kl_divergence(self) -> torch.Tensor | float:
        """Collects the KL divergence of the model by summing the KL divergence of each layer."""
        _, state = traverse_with_state(self.model, nn_compose(kl_divergence_traverser))

        return state[KL_DIVERGENCE]
