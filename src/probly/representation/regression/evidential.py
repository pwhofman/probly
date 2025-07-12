"""Evidential model class implementation."""

from __future__ import annotations

import torch
from torch import nn

from probly.representation.layers import NormalInverseGammaLinear
from probly.representation.predictor import RepresentationPredictor
from probly.traverse_nn import nn_compose
from pytraverse import TRAVERSE_REVERSED, GlobalVariable, State, TraverserResult, singledispatch_traverser, traverse

REPLACED_LAST_LINEAR = GlobalVariable[bool](
    "REPLACED_LAST_LINEAR",
    "Whether the last linear layer has been replaced with a NormalInverseGammaLinear layer.",
    default=False,
)


evidential_traverser = singledispatch_traverser[object](name="evidential_traverser")


@evidential_traverser.register(skip_if=lambda s: s[REPLACED_LAST_LINEAR])
def _(layer: nn.Linear, state: State) -> TraverserResult:
    state[REPLACED_LAST_LINEAR] = True
    return NormalInverseGammaLinear(
        layer.in_features,
        layer.out_features,
        device=layer.weight.device,
        bias=layer.bias is not None,
    ), state


class Evidential[In, KwIn](nn.Module, RepresentationPredictor[In, KwIn, torch.Tensor, dict[str, torch.Tensor]]):
    """This class implements an evidential deep learning model for regression.

    Attributes:
        model: torch.nn.Module, The evidential model with a normal inverse gamma layer suitable
        for evidential regression.

    """

    def __init__(self, base: nn.Module) -> None:
        """Initialize the Evidential model.

        Convert the base model into an evidential deep learning regression model.

        Args:
            base: torch.nn.Module, The base model to be used.
        """
        super().__init__()
        self.model = self._convert(base)

    def forward(self, *args: In, **kwargs: KwIn) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            *args: Input data.
            **kwargs: Additional keyword arguments.

        Returns:
            Model output.

        """
        return self.model(*args, **kwargs)

    def predict_pointwise(self, *args: In, **kwargs: KwIn) -> torch.Tensor:
        """Forward pass of the model for point-wise prediction.

        Args:
            *args: Input data.
            **kwargs: Additional keyword arguments.

        Returns:
            Model output.
        """
        return self.model(*args, **kwargs)["gamma"]

    def predict_representation(self, *args: In, **kwargs: KwIn) -> dict[str, torch.Tensor]:
        """Forward pass of the model for uncertainty representation.

        Args:
            *args: Input data.
            **kwargs: Additional keyword arguments.

        Returns:
            Model output.
        """
        return dict(self.model(*args, **kwargs))

    def _convert(self, base: nn.Module) -> nn.Module:
        """Convert a model into an evidential deep learning regression model.

        Replace the last (linear) layer by a layer parameterizing a normal inverse gamma distribution.

        Args:
            base: The base model to be used.

        """
        return traverse(base, nn_compose(evidential_traverser), init={TRAVERSE_REVERSED: True})

    def sample(self, *args: In, n_samples: int, **kwargs: KwIn) -> torch.Tensor:
        """Sample from the predicted distribution for a given input x.

        Returns a tensor of shape (n_instances, n_samples, 2) representing the parameters of the sampled normal
        distributions. The mean of the normal distribution is the gamma parameter and the variance is sampled from
        an inverse gamma distribution and divided by the nu parameter. The first dimension is the mean and
        the second dimension is the variance.

        Args:
            *args: Input data.
            n_samples: Number of samples.
            **kwargs: Additional keyword arguments.

        Returns:
            Samples from the normal-inverse-gamma distribution.
        """
        x = self.model(*args, **kwargs)
        inverse_gamma = torch.distributions.InverseGamma(x["alpha"], x["beta"])
        sigma2 = torch.stack([inverse_gamma.sample() for _ in range(n_samples)]).swapaxes(0, 1)
        normal_mu = x["gamma"].unsqueeze(-1).expand(-1, n_samples, 1)
        normal_sigma2 = sigma2 / x["nu"].unsqueeze(2)
        x = torch.cat((normal_mu, normal_sigma2), dim=2)
        return x
