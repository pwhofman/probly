"""torch layer implementations."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F


class BatchEnsembleLinear(nn.Module):
    """Implements a BatchEnsemble linear layer.

    Attributes:
        in_features: int, number of input features
        out_features: int, number of output features
        num_members: int, number of batch ensemble members
        weight: torch.Tensor, shared weight matrix
        bias: torch.Tensor, shared bias vector
        s: torch.Tensor, rank-one factor for input features
        r: torch.Tensor, rank-one factor for output features
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        num_members: int = 1,
        use_base_weights: bool = False,
        s_mean: float = 1.0,
        s_std: float = 0.01,
        r_mean: float = 1.0,
        r_std: float = 0.01,
    ) -> None:
        """Initializes the BatchEnsemble linear layer.

        Args:
            base_layer (nn.Linear): The original linear layer to be used.
            num_members (int): number of ensemble members
            use_base_weights (bool): Whether to use the weights of the base layer as prior means. Default is False.
            s_mean (float): mean of a normal distribution to initialize s
            s_std (float): standard deviation of a normal distribution to initialize s
            r_mean (float): mean of a normal distribution to initialize r
            r_std (float): standard deviation of a normal distribution to initialize r
        """
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.num_members = num_members

        if use_base_weights:
            self.weight = nn.Parameter(base_layer.weight.detach().clone())
        else:
            self.weight = nn.Parameter(torch.empty((self.out_features, self.in_features)))

        if base_layer.bias is not None:
            self.bias = nn.Parameter(base_layer.bias.detach().clone())
        else:
            self.bias = nn.Parameter(torch.zeros(self.out_features))

        self.s = nn.Parameter(torch.Tensor(self.num_members, self.in_features))
        self.r = nn.Parameter(torch.Tensor(self.num_members, self.out_features))

        nn.init.normal_(self.s, s_mean, s_std)
        nn.init.normal_(self.r, r_mean, r_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the BatchEnsemble linear layer.

        Args:
            x: torch.Tensor, Input tensor of shape [B, in_features] or [E, B, in_features],
                            where B is the batch size and E is the ensemble size.

        Returns:
            torch.Tensor: Output tensor of shape [E, B, out_features].
        """
        # TODO @<jnpippert>: maybe use buffers for some parameters? r,s, and their mu and std? # noqa: TD003
        if x.dim() == 2:
            # If this is the first layer, expand to ensemble dimension
            x = x.unsqueeze(0).expand(self.num_members, -1, -1)
        elif x.dim() == 3 and x.size(0) != self.num_members:
            msg = f"Expected first dim={self.num_members}, got {x.size(0)}"
            raise ValueError(msg)

        x = x * self.s.clone().unsqueeze(1)
        y = F.linear(x, self.weight, bias=None)
        y = y * self.r.clone().unsqueeze(1)
        y = y + self.bias
        return y

    def extra_repr(self) -> str:
        """Expose description of in- and out-features, num_members and bias of this layer."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features},"
            f" num_members={self.num_members}, bias={self.bias is not None}"
        )


class BatchEnsembleConv2d(nn.Module):
    """Implements a BatchEnsemble convolutional layer.

    Attributes:
        in_channels: int, number of input channels
        out_channels: int, number of output channels
        kernel_size: int or tuple, size of the convolutional kernel
        stride: int or tuple, stride of the convolution
        padding: int or tuple, padding of the convolution
        dilation: int or tuple, dilation of the convolution
        groups: int, number of groups for grouped convolution
        num_members: int, number of batch ensemble members
        weight: torch.Tensor, shared weight matrix
        bias: torch.Tensor, shared bias vector
        s: torch.Tensor, rank-one factor for input features
        r: torch.Tensor, rank-one factor for output features
    """

    def __init__(
        self,
        base_layer: nn.Conv2d,
        num_members: int = 1,
        use_base_weights: bool = False,
        s_mean: float = 1.0,
        s_std: float = 0.01,
        r_mean: float = 1.0,
        r_std: float = 0.01,
    ) -> None:
        """Initializes the BatchEnsemble linear layer.

        Args:
            base_layer (nn.Linear): The original linear layer to be used.
            num_members (int): number of ensemble members
            use_base_weights (bool): Whether to use the weights of the base layer as prior means. Default is False.
            s_mean (float): mean of a normal distribution to initialize s
            s_std (float): standard deviation of a normal distribution to initialize s
            r_mean (float): mean of a normal distribution to initialize r
            r_std (float): standard deviation of a normal distribution to initialize r
        """
        super().__init__()

        self.in_channels = base_layer.in_channels
        self.out_channels = base_layer.out_channels
        self.kernel_size = base_layer.kernel_size
        self.stride = base_layer.stride
        self.padding = base_layer.padding
        self.dilation = base_layer.dilation
        self.groups = base_layer.groups

        self.num_members = num_members

        if use_base_weights:
            self.weight = nn.Parameter(base_layer.weight.detach().clone())
        else:
            self.weight = nn.Parameter(
                torch.empty((self.out_channels, self.in_channels // self.groups, *self.kernel_size)),
            )

        if base_layer.bias is not None:
            self.bias = nn.Parameter(base_layer.bias.detach().clone())
        else:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))

        self.s = nn.Parameter(torch.Tensor(num_members, self.in_channels))
        self.r = nn.Parameter(torch.Tensor(num_members, self.out_channels))

        nn.init.normal_(self.s, s_mean, s_std)
        nn.init.normal_(self.r, r_mean, r_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the BatchEnsemble Conv2d layer.

        Args:
            x: torch.Tensor, Input tensor of shape [B, in_channels, H, W] or [E, B, in_channels, H, W],
                            where B is the batch size, E is the ensemble size, H is height, and W is width.

        Returns:
            torch.Tensor: Output tensor of shape [E, B, out_channels, H_out, W_out].
        """
        if x.dim() == 4:
            # If this is the first layer, expand to ensemble dimension
            x = x.unsqueeze(0).expand(self.num_members, -1, -1, -1, -1)
        elif x.dim() == 5 and x.size(0) != self.num_members:
            msg = f"Expected ensemble dim {self.num_members}, got {x.size(0)}"
            raise ValueError(msg)

        x = x.clone()
        x *= self.s[:, None, :, None, None]
        e, b, c, h, w = x.shape
        x = x.reshape(e * b, c, h, w)

        y = F.conv2d(
            x,
            self.weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        _, c, h, w = y.shape
        y = y.view(e, b, c, h, w)
        y *= self.r[:, None, :, None, None]
        y += self.bias[None, None, :, None, None]

        return y

    def extra_repr(self) -> str:
        """Expose description of in- and out-features, kernel size, stride and num_members of this layer."""
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size},"
            f" stride={self.stride}, num_members={self.num_members}"
        )


# ======================================================================================================================


class BayesLinear(nn.Module):
    """Implements a Bayesian linear layer based on :cite:`blundellWeightUncertainty2015`.

    Attributes:
        in_features: int, number of input features
        out_features: int, number of output features
        bias: bool, whether to use a bias term
        weight_mu: torch.Tensor, mean of the posterior weights
        weight_rho: torch.Tensor, transformed standard deviation of the posterior weights
        weight_prior_mu: torch.Tensor, mean of the prior weights
        weight_prior_sigma: torch.Tensor, standard deviation of the prior weights
        bias_mu: torch.Tensor, mean of the posterior bias
        bias_rho: torch.Tensor, transformed standard deviation of the posterior bias
        bias_prior_mu: torch.Tensor, mean of the prior bias
        bias_prior_sigma: torch.Tensor, standard deviation of the prior bias
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        use_base_weights: bool = False,
        posterior_std: float = 0.05,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
    ) -> None:
        """Initializes the Bayesian linear layer.

        Reparameterize the standard deviation of the posterior weights using the re-parameterization trick.

        Args:
            base_layer: The original linear layer to be used.
            use_base_weights: Whether to use the weights of the base layer as prior means. Default is False.
            posterior_std: float, initial standard deviation of the posterior
            prior_mean: float, mean of the prior
            prior_std: float, standard deviation of the prior
        """
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.bias = base_layer.bias is not None

        # transform standard deviation for the re-parametrization trick
        rho = cast("float", _inverse_softplus(torch.tensor(posterior_std)))

        # posterior weights
        if not use_base_weights:
            self.weight_mu = nn.Parameter(torch.empty((self.out_features, self.in_features)))
        else:
            self.weight_mu = nn.Parameter(base_layer.weight.data)
        self.weight_rho = nn.Parameter(torch.full((self.out_features, self.in_features), rho))

        # prior weights
        if not use_base_weights:
            self.register_buffer(
                "weight_prior_mu",
                torch.full((self.out_features, self.in_features), prior_mean),
            )
        else:
            self.register_buffer(
                "weight_prior_mu",
                base_layer.weight.data,
            )
        self.register_buffer(
            "weight_prior_sigma",
            torch.full((self.out_features, self.in_features), prior_std),
        )

        if self.bias:
            # posterior bias
            if not use_base_weights:
                self.bias_mu = nn.Parameter(torch.empty((self.out_features,)))
            else:
                self.bias_mu = nn.Parameter(base_layer.bias.data)
            self.bias_rho = nn.Parameter(
                torch.full((self.out_features,), rho),
            )

            # prior bias
            if not use_base_weights:
                self.register_buffer(
                    "bias_prior_mu",
                    torch.full((self.out_features,), prior_mean),
                )
            else:
                self.register_buffer(
                    "bias_prior_mu",
                    base_layer.bias.data,
                )
            self.register_buffer(
                "bias_prior_sigma",
                torch.full((self.out_features,), prior_std),
            )

        if not use_base_weights:
            self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Bayesian linear layer.

        Args:
            x: torch.Tensor, input data
        Returns:
            torch.Tensor, layer output
        """
        eps_weight = torch.randn_like(self.weight_mu)
        weight = self.weight_mu + torch.log1p(torch.exp(self.weight_rho)) * eps_weight
        if self.bias:
            eps_bias = torch.randn_like(self.bias_mu)
            bias = self.bias_mu + torch.log1p(torch.exp(self.bias_rho)) * eps_bias
            x = F.linear(x, weight, bias)
        else:
            x = F.linear(x, weight)
        return x

    def reset_parameters(self) -> None:
        """Reset the parameters of the Bayesian conv2d layer.

        Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        uniform(-1/sqrt(k), 1/sqrt(k)), where ``k = weight.size(1) * prod(*kernel_size)``
        For more details see: https://github.com/pytorch/pytorch/issues/15314
        """
        init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        if self.bias is not False:
            fan_in, _ = init._calculate_fan_in_and_fan_out(  # noqa: SLF001
                self.weight_mu,
            )
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias_mu, -bound, bound)

    @property
    def kl_divergence(self) -> torch.Tensor:
        """Computes the KL-divergence between the posterior and prior."""
        kl = torch.sum(
            _kl_divergence_gaussian(
                self.weight_mu,
                torch.log1p(torch.exp(self.weight_rho)) ** 2,
                cast("torch.Tensor", self.weight_prior_mu),
                cast("torch.Tensor", self.weight_prior_sigma) ** 2,
            ),
        )
        if self.bias:
            kl += torch.sum(
                _kl_divergence_gaussian(
                    self.bias_mu,
                    torch.log1p(torch.exp(self.bias_rho)) ** 2,
                    cast("torch.Tensor", self.bias_prior_mu),
                    cast("torch.Tensor", self.bias_prior_sigma) ** 2,
                ),
            )
        return kl


class BayesConv2d(nn.Module):
    """Implementation of a Bayesian convolutional layer based on :cite:`blundellWeightUncertainty2015`.

    Attributes:
        in_channels: int, number of input channels
        out_channels: int, number of output channels
        kernel_size: int or tuple, size of the convolutional kernel
        stride: int or tuple, stride of the convolution
        padding: int or tuple, padding of the convolution
        dilation: int or tuple, dilation of the convolution
        groups: int, number of groups for grouped convolution
        bias: bool, whether to use a bias term
        weight_mu: torch.Tensor, mean of the posterior weights
        weight_rho: torch.Tensor, transformed standard deviation of the posterior weights
        weight_prior_mu: torch.Tensor, mean of the prior weights
        weight_prior_sigma: torch.Tensor, standard deviation of the prior weights
        bias_mu: torch.Tensor, mean of the posterior bias
        bias_rho: torch.Tensor, transformed standard deviation of the posterior bias
        bias_prior_mu: torch.Tensor, mean of the prior bias
        bias_prior_sigma: torch.Tensor, standard deviation of the prior bias
    """

    def __init__(
        self,
        base_layer: nn.Conv2d,
        use_base_weights: bool = False,
        posterior_std: float = 0.05,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
    ) -> None:
        """Initializes the Bayesian convolutional layer.

        Reparameterize the standard deviation of the posterior weights using the re-parameterization trick.

        Args:
            base_layer: The original conv2d layer to be used.
            use_base_weights: Whether to use the weights of the base layer as prior means. Default is False.
            posterior_std: float, initial standard deviation of the posterior
            prior_mean: float, mean of the prior
            prior_std: float, standard deviation of the prior
        """
        super().__init__()
        self.in_channels = base_layer.in_channels
        self.out_channels = base_layer.out_channels
        self.kernel_size = base_layer.kernel_size
        self.stride = base_layer.stride
        self.padding = base_layer.padding
        self.dilation = base_layer.dilation
        self.groups = base_layer.groups
        self.bias = base_layer.bias is not None

        # transform standard deviation for the re-parametrization trick
        rho = cast("float", _inverse_softplus(torch.tensor(posterior_std)))

        # posterior weights
        if not use_base_weights:
            self.weight_mu = nn.Parameter(
                torch.empty((self.out_channels, self.in_channels // self.groups, *self.kernel_size)),
            )
        else:
            self.weight_mu = nn.Parameter(base_layer.weight.data)
        self.weight_rho = nn.Parameter(
            torch.full((self.out_channels, self.in_channels // self.groups, *self.kernel_size), rho),
        )

        # prior weights
        if not use_base_weights:
            self.register_buffer(
                "weight_prior_mu",
                torch.full(
                    (self.out_channels, self.in_channels // self.groups, *self.kernel_size),
                    prior_mean,
                ),
            )
        else:
            self.register_buffer(
                "weight_prior_mu",
                base_layer.weight.data,
            )

        self.register_buffer(
            "weight_prior_sigma",
            torch.full((self.out_channels, self.in_channels // self.groups, *self.kernel_size), prior_std),
        )

        if self.bias:
            # posterior bias
            if not use_base_weights:
                self.bias_mu = nn.Parameter(torch.empty((self.out_channels,)))
            else:
                self.bias_mu = nn.Parameter(
                    base_layer.bias.data if base_layer.bias is not None else torch.empty((self.out_channels,))
                )
            self.bias_rho = nn.Parameter(torch.full((self.out_channels,), rho))

            # prior bias
            if not use_base_weights:
                self.register_buffer(
                    "bias_prior_mu",
                    torch.full((self.out_channels,), prior_mean),
                )
            else:
                self.register_buffer(
                    "bias_prior_mu",
                    base_layer.bias.data
                    if base_layer.bias is not None
                    else torch.full((self.out_channels,), prior_mean),
                )
            self.register_buffer(
                "bias_prior_sigma",
                torch.full((self.out_channels,), prior_std),
            )

        if not use_base_weights:
            self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Bayesian conv2d layer.

        Args:
            x: torch.Tensor, input data
        Returns:
            torch.Tensor, layer output
        """
        eps_weight = torch.randn_like(self.weight_mu)
        weight = self.weight_mu + torch.log1p(torch.exp(self.weight_rho)) * eps_weight
        if self.bias:
            eps_bias = torch.randn_like(self.bias_mu)
            bias = self.bias_mu + torch.log1p(torch.exp(self.bias_rho)) * eps_bias
            x = F.conv2d(
                x,
                weight,
                bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        else:
            x = F.conv2d(
                x,
                weight,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        return x

    def reset_parameters(self) -> None:
        """Reset the parameters of the Bayesian conv2d layer.

        Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        uniform(-1/sqrt(k), 1/sqrt(k)), where ``k = weight.size(1) * prod(*kernel_size)``
        For more details see: https://github.com/pytorch/pytorch/issues/15314
        """
        init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        if self.bias is not False:
            fan_in, _ = init._calculate_fan_in_and_fan_out(  # noqa: SLF001
                self.weight_mu,
            )
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias_mu, -bound, bound)

    @property
    def kl_divergence(self) -> torch.Tensor:
        """Compute the KL-divergence between the posterior and prior."""
        kl = torch.sum(
            _kl_divergence_gaussian(
                self.weight_mu,
                torch.log1p(torch.exp(self.weight_rho)) ** 2,
                cast("torch.Tensor", self.weight_prior_mu),
                cast("torch.Tensor", self.weight_prior_sigma) ** 2,
            ),
        )
        if self.bias:
            kl += torch.sum(
                _kl_divergence_gaussian(
                    self.bias_mu,
                    torch.log1p(torch.exp(self.bias_rho)) ** 2,
                    cast("torch.Tensor", self.bias_prior_mu),
                    cast("torch.Tensor", self.bias_prior_sigma) ** 2,
                ),
            )
        return kl


def _kl_divergence_gaussian(
    mu1: torch.Tensor,
    sigma21: torch.Tensor,
    mu2: torch.Tensor,
    sigma22: torch.Tensor,
) -> torch.Tensor:
    """Compute the KL-divergence between two Gaussian distributions.

    https://en.wikipedia.org/wiki/Kullback-Leibler_divergence#Examples
    Args:
        mu1: torch.Tensor, mean of the first Gaussian distribution
        sigma21: torch.Tensor, variance of the first Gaussian distribution
        mu2: torch.Tensor, mean of the second Gaussian distribution
        sigma22: torch.Tensor, variance of the second Gaussian distribution
    Returns:
        kl_div: float or numpy.ndarray shape (n_instances,), KL-divergence between the two Gaussian distributions
    """
    kl_div: torch.Tensor = 0.5 * torch.log(sigma22 / sigma21) + (sigma21 + (mu1 - mu2) ** 2) / (2 * sigma22) - 0.5
    return kl_div


def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    """Compute the inverse softplus function.

    Args:
        x: Input tensor.

    Returns:
        Output tensor after applying the inverse softplus function.
    """
    return torch.log(torch.exp(x) - 1)


# ======================================================================================================================


class DropConnectLinear(nn.Module):
    """Custom Linear layer with DropConnect applied to weights during training based on :cite:`mobiny2021dropconnect`.

    Attributes:
        in_features: int, number of input features.
        out_features: int, number of output features.
        p: float, probability of dropping individual weights.
        weight: torch.Tensor, weight matrix of the layer
        bias: torch.Tensor, bias of the layer

    """

    def __init__(self, base_layer: nn.Linear, p: float = 0.25) -> None:
        """Initialize a DropConnectLinear layer based on given linear base layer.

        Args:
            base_layer: nn.Linear, The original linear layer to be wrapped.
            p: float, The probability of dropping individual weights.
        """
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.p = p
        self.weight = nn.Parameter(base_layer.weight.clone().detach())
        self.bias = nn.Parameter(base_layer.bias.clone().detach()) if base_layer.bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the DropConnect layer.

        Args:
            x: torch.Tensor, input data
        Returns:
            torch.Tensor, layer output

        """
        if self.training:
            mask = (torch.rand_like(self.weight) > self.p).float()
            weight = self.weight * mask  # Apply DropConnect
        else:
            weight = self.weight * (1 - self.p)  # Scale weights at inference time

        return F.linear(x, weight, self.bias)

    def extra_repr(self) -> str:
        """Expose description of in- and out-features of this layer."""
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


# ======================================================================================================================


class NormalInverseGammaLinear(nn.Module):
    """Custom Linear layer for a normal-inverse-gamma-distribution based on :cite:`aminiDeepEvidential2020`.

    Attributes:
        gamma: torch.Tensor, shape (out_features, in_features), the mean of the normal distribution.
        nu: torch.Tensor, shape (out_features, in_features), parameter of the normal distribution.
        alpha: torch.Tensor, shape (out_features, in_features), parameter of the inverse-gamma distribution.
        beta: torch.Tensor, shape (out_features, in_features), parameter of the inverse-gamma distribution.
        gamma_bias: torch.Tensor, shape (out_features), the mean of the normal distribution for the bias.
        nu_bias: torch.Tensor, shape (out_features), parameter of the normal distribution for the bias.
        alpha_bias: torch.Tensor, shape (out_features), parameter of the inverse-gamma distribution for the bias.
        beta_bias: torch.Tensor, shape (out_features), parameter of the inverse-gamma distribution for the bias.
        bias: bool, whether to include bias in the layer.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        *,
        bias: bool = True,
    ) -> None:
        """Initialize an instance of the NormalInverseGammaLinear layer.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            device: Device to initialize the parameters on.
            bias: Whether to include bias in the layer. Defaults to True
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.empty((out_features, in_features), device=device))
        self.nu = nn.Parameter(torch.empty((out_features, in_features), device=device))
        self.alpha = nn.Parameter(torch.empty((out_features, in_features), device=device))
        self.beta = nn.Parameter(torch.empty((out_features, in_features), device=device))
        if bias:
            self.gamma_bias = nn.Parameter(torch.empty(out_features, device=device))
            self.nu_bias = nn.Parameter(torch.empty(out_features, device=device))
            self.alpha_bias = nn.Parameter(torch.empty(out_features, device=device))
            self.beta_bias = nn.Parameter(torch.empty(out_features, device=device))
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass of the NormalInverseGamma layer.

        Args:
            x: torch.Tensor, input data
        Returns:
            dict[str, torch.Tensor], layer output containing the parameters of the normal-inverse-gamma distribution
        """
        gamma = F.linear(x, self.gamma, self.gamma_bias)
        nu = F.softplus(F.linear(x, self.nu, self.nu_bias))
        alpha = F.softplus(F.linear(x, self.alpha, self.alpha_bias)) + 1
        beta = F.softplus(F.linear(x, self.beta, self.beta_bias))
        return {"gamma": gamma, "nu": nu, "alpha": alpha, "beta": beta}

    def reset_parameters(self) -> None:
        """Reset the parameters of the NormalInverseGamma layer.

        Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        https://github.com/pytorch/pytorch/issues/57109.
        """
        init.kaiming_uniform_(self.gamma, a=math.sqrt(5))
        init.kaiming_uniform_(self.nu, a=math.sqrt(5))
        init.kaiming_uniform_(self.alpha, a=math.sqrt(5))
        init.kaiming_uniform_(self.beta, a=math.sqrt(5))
        if self.gamma_bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.gamma)  # noqa: SLF001
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.gamma_bias, -bound, bound)
            init.uniform_(self.nu_bias, -bound, bound)
            init.uniform_(self.alpha_bias, -bound, bound)
            init.uniform_(self.beta_bias, -bound, bound)


# radial flows
class BatchedRadialFlowLayer(nn.Module):
    """Single radial flow transformation shared across all classes."""

    def __init__(self, latent_dim: int, num_classes: int) -> None:
        """Initialize parameters for a radial flow transform."""
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        self.x0 = nn.Parameter(torch.zeros(self.num_classes, self.latent_dim))
        self.alpha_prime = nn.Parameter(torch.zeros(self.num_classes))
        self.beta_prime = nn.Parameter(torch.zeros(self.num_classes))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset learnable parameters with a small uniform init."""
        stdv = 1.0 / math.sqrt(self.latent_dim)
        self.x0.data.uniform_(-stdv, stdv)
        self.alpha_prime.data.uniform_(-stdv, stdv)
        self.beta_prime.data.uniform_(-stdv, stdv)

    def forward(self, zc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the radial flow to latent inputs zc."""
        alpha = torch.nn.functional.softplus(self.alpha_prime)
        beta = -alpha + torch.nn.functional.softplus(self.beta_prime)

        x0 = self.x0.unsqueeze(1)
        diff = zc - x0
        r = diff.norm(dim=-1)

        h = 1.0 / (alpha.unsqueeze(1) + r)
        h_prime = -h * h
        beta_h = beta.unsqueeze(1) * h

        z_new = zc + beta_h.unsqueeze(-1) * diff

        term1 = (self.latent_dim - 1) * torch.log1p(beta_h)
        term2 = torch.log1p(beta_h + beta.unsqueeze(1) * h_prime * r)
        log_abs_det = term1 + term2

        return z_new, log_abs_det


class BatchedRadialFlowDensity(nn.Module):
    """Radial-flow density estimator that computes P(z|c) for all classes."""

    def __init__(self, latent_dim: int, num_classes: int, flow_length: int = 6) -> None:
        """Create a sequence of radial flow layers and base distribution."""
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.layers = nn.ModuleList(
            [BatchedRadialFlowLayer(latent_dim, num_classes) for _ in range(flow_length)],
        )

        self.log_base_const = -0.5 * self.latent_dim * math.log(2 * math.pi)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand input x for all classes and apply flow layers."""
        B = x.size(0)  # noqa: N806
        zc = x.unsqueeze(0).expand(self.num_classes, B, self.latent_dim)
        sum_log_jac = torch.zeros(self.num_classes, B, device=x.device)

        for layer in self.layers:
            zc, log_j = layer(zc)
            sum_log_jac = sum_log_jac + log_j

        return zc, sum_log_jac

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Return class-conditional log densities log P(x|c)."""
        zc, sum_log_jac = self.forward(x)  # zc: [C,B,D]

        base_logp = self.log_base_const - 0.5 * (zc**2).sum(dim=-1)
        logp = base_logp + sum_log_jac  # [C,B]

        return logp.transpose(0, 1)  # [B,C]


class RadialFlowLayer(nn.Module):
    """Single radial flow layer for a latent vector z ∈ R^D."""

    def __init__(self, latent_dim: int) -> None:  # noqa: D107
        super().__init__()
        self.latent_dim = latent_dim

        # Learnable parameters:
        # - x0: center of the radial transformation (vector in R^D)
        # - alpha_prime, beta_prime: unconstrained scalars that we transform to valid alpha, beta
        self.x0 = nn.Parameter(torch.zeros(latent_dim))
        self.alpha_prime = nn.Parameter(torch.zeros(1))
        self.beta_prime = nn.Parameter(torch.zeros(1))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the radial flow to latent inputs features.

        Args:
            features: Tensor of shape [B, D].

        Returns:
            features_new: Transformed latent tensor, shape [B, D].
            log_abs_det: Log-absolute determinant of the Jacobian, shape [B].
        """
        # Ensure alpha > 0 and beta > -alpha for invertibility
        alpha = torch.nn.functional.softplus(self.alpha_prime)  # scalar > 0
        beta = -alpha + torch.nn.functional.softplus(self.beta_prime)  # scalar > -alpha

        # z0 is the learnable center (broadcast to [B, D])
        x0 = self.x0  # [D]

        # Difference from the center
        diff = features - x0  # [B, D]
        r = diff.norm(dim=-1)  # Distance to center, shape [B]

        # Radial flow scalar functions h(r) and h'(r)
        h = 1.0 / (alpha + r)  # [B]
        h_prime = -h * h  # [B]
        beta_h = beta * h  # [B]

        # Apply the radial flow transformation:
        features_new = features + beta_h.unsqueeze(-1) * diff  # [B, D]

        # Log determinant of the Jacobian:
        # formula derived in Rezende & Mohamed (2015)
        term1 = (self.latent_dim - 1) * torch.log1p(beta_h)  # [B]
        term2 = torch.log1p(beta_h + beta * h_prime * r)  # [B]
        log_abs_det = term1 + term2  # [B]

        return features_new, log_abs_det


class EDLHead(nn.Module):
    """outputs Dirichlet concentration parameters (alpha)."""

    def __init__(self, latent_dim: int, num_classes: int = 10, hidden_dim: int = 128) -> None:
        """Initialize the EDLHead.

        Args:
            latent_dim: Dimension of the input latent vector.
            num_classes: Number of output classes. Defaults to 10.
            hidden_dim: Dimension of the hidden layer. Defaults to 128.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute Dirichlet concentration parameters (alpha).

        Args:
            features: Input latent tensor.

        Returns:
            torch.Tensor: Dirichlet concentration parameters (alpha).
        """
        alpha = F.softplus(self.net(features)) + 1.0

        return alpha


class RadialFlowDensity(nn.Module):
    """Normalizing flow density p(z) using a stack of radial flows."""

    def __init__(self, latent_dim: int, flow_length: int = 4) -> None:  # noqa: D107
        super().__init__()
        self.latent_dim = latent_dim
        self.layers = nn.ModuleList([RadialFlowLayer(latent_dim=latent_dim) for _ in range(flow_length)])

        # Constant term for log N(z|0, I): -0.5 * D * log(2π)
        self.log_base_const = -0.5 * self.latent_dim * math.log(2 * math.pi)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply all flow layers to x.

        Args:
            x: Tensor of shape [B, D].

        Returns:
            features: Transformed latent tensor after all flows, shape [B, D].
            sum_log_jac: Summed log-det Jacobian across flows, shape [B].
        """
        features = x
        sum_log_jac = torch.zeros(features.size(0), device=features.device)

        for layer in self.layers:
            features, log_j = layer(features)
            sum_log_jac = sum_log_jac + log_j

        return features, sum_log_jac

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log p(x) under the flow-based density.

        Args:
            x: Tensor of shape [B, D].

        Returns:
            logp: Log-density log p(x), shape [B].
        """
        # Apply flow
        features, sum_log_jac = self.forward(x)

        # Base log-prob under N(0, I): -0.5 * (D * log(2π) + ||z||^2)
        base_logp = self.log_base_const - 0.5 * (features**2).sum(dim=-1)  # [B]

        # Add the log-determinant of the Jacobian
        logp = base_logp + sum_log_jac  # [B]
        return logp


class RegressionHead(nn.Module):
    """Head that converts encoded features into evidential Normal-Gamma parameters."""

    def __init__(self, latent_dim: int) -> None:
        """Initialize the head.

        Args:
            latent_dim: Dimension of input features coming from the encoder.
        """
        super().__init__()
        self.linear = nn.Linear(latent_dim, 4)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert features into (mu, kappa, alpha, beta).

        Args:
            features: Feature tensor (N, feature_dim)

        Returns:
            Tuple of four tensors representing Normal-Gamma parameters.
        """
        raw = self.linear(features)

        mu = raw[:, 0:1]
        kappa = F.softplus(raw[:, 1:2])
        alpha = F.softplus(raw[:, 2:3]) + 1.0
        beta = F.softplus(raw[:, 3:4])

        return mu, kappa, alpha, beta


class NatPNClassHead(nn.Module):
    """Dirichlet posterior head for evidential classification.

    Takes latent representations and outputs Dirichlet parameters for uncertainty
    quantification in classification. This head should be used with an encoder to
    create a complete classification model.

    Args:
        latent_dim: Dimension of input latent vectors from an encoder.
        num_classes: Number of classification classes.
        hidden_dim: Dimension of hidden layer. Defaults to 64.
        n_prior: Prior for evidence scaling. If None, defaults to num_classes.
    """

    alpha_prior: torch.Tensor

    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        n_prior: float | None = None,
    ) -> None:
        """Initialize the DirichletHead."""
        super().__init__()

        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

        if n_prior is None:
            n_prior = float(num_classes)

        chi_prior = torch.full((num_classes,), 1.0 / num_classes)
        alpha_prior = n_prior * chi_prior

        self.register_buffer("alpha_prior", alpha_prior)

    def forward(
        self,
        features: torch.Tensor,
        log_pz: torch.Tensor,
        certainty_budget: float,
    ) -> dict[str, torch.Tensor]:
        """Compute Dirichlet parameters for evidential classification.

        Args:
            features: Latent representations of shape [B, latent_dim].
            log_pz: Log probability from density estimator of shape [B].
            certainty_budget: Budget parameter for evidence scaling.

        Returns:
            Dictionary containing:
                - alpha: Dirichlet parameters [B, num_classes]
                - features: Input latent representations
                - log_pz: Log density values
                - evidence: Scaled evidence [B, num_classes]
        """
        logits = self.classifier(features)  # [B, C]
        chi = torch.softmax(logits, dim=-1)  # [B, C]

        # Total evidence n(x)
        n = certainty_budget * log_pz.exp()  # [B]
        n = torch.clamp(n, min=1e-8)

        evidence = n.unsqueeze(-1) * chi  # [B, C]
        alpha = self.alpha_prior.unsqueeze(0) + evidence

        return {
            "alpha": alpha,  # Dirichlet parameters
            "features": features,
            "log_pz": log_pz,
            "evidence": evidence,
        }


class NatPNRegHead(nn.Module):
    """Gaussian posterior head for evidential regression.

    Takes latent representations and outputs mean and variance for Gaussian
    uncertainty quantification in regression. This head should be used with an encoder
    to create a complete regression model.

    Args:
        latent_dim: Dimension of input latent vectors from an encoder.
        out_dim: Dimension of regression output. Defaults to 1 (univariate regression).
    """

    def __init__(
        self,
        latent_dim: int,
        out_dim: int = 1,
    ) -> None:
        """Initialize the GaussianHead."""
        super().__init__()

        self.mean_net = nn.Linear(latent_dim, out_dim)
        self.log_var_net = nn.Linear(latent_dim, out_dim)

    def forward(
        self,
        features: torch.Tensor,
        log_pz: torch.Tensor,
        certainty_budget: float,
    ) -> dict[str, torch.Tensor]:
        """Compute Gaussian parameters for evidential regression.

        Args:
            features: Latent representations of shape [B, latent_dim].
            log_pz: Log probability from density estimator of shape [B].
            certainty_budget: Budget parameter for precision scaling.

        Returns:
            Dictionary containing:
                - mean: Predicted mean [B, out_dim]
                - var: Predicted variance [B, out_dim]
                - features: Input latent representations
                - log_pz: Log density values
                - precision: Scaled precision [B, out_dim]
        """
        mean = self.mean_net(features)  # [B, D]
        log_var = self.log_var_net(features)  # [B, D]

        # Epistemic uncertainty via density scaling
        precision = certainty_budget * log_pz.exp().unsqueeze(-1)
        precision = torch.clamp(precision, min=1e-8)

        var = torch.exp(log_var) / precision

        return {
            "mean": mean,
            "var": var,
            "features": features,
            "log_pz": log_pz,
            "precision": precision,
        }


class IRDHead(nn.Module):
    """Head that converts encoded features into Dirichlet concentration parameters (alpha).

    For multi-class classification, this head outputs K alpha values (one per class),
    where alpha forms a K-dimensional Dirichlet distribution.
    """

    def __init__(self, latent_dim: int, num_classes: int) -> None:
        """Initialize the Dirichlet head.

        Args:
            latent_dim: Dimension of input features from the encoder.
            num_classes: Number of output classes (K in Dirichlet(a)).
        """
        super().__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(latent_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Convert features into Dirichlet concentration parameters (alpha).

        Args:
            features: Feature tensor (batch_size, latent_dim) from encoder.

        Returns:
            Alpha parameters of shape (batch_size, num_classes), all > 0.
        """
        # Linear projection to num_classes dimensions
        logits = self.linear(features)

        # Ensure alpha > 0 by applying softplus and adding small offset
        # alpha = softplus(logits) + 1.0 ensures all values >= 1.0
        alpha = F.softplus(logits) + 1.0

        return alpha


class PrNetHead(nn.Module):
    """Head mapping latent features to Dirichlet concentration parameters."""

    def __init__(self, latent_dim: int, num_classes: int, hidden_dim: int = 256) -> None:
        """Initialize the Dirichlet classification head."""
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Produce positive Dirichlet concentration parameters."""
        return F.softplus(self.net(features)) + 1.0


# ======================================================================================================================


class IntSoftmax(nn.Module):
    """Implementation of the integer softmax layer."""

    def __init__(self) -> None:
        """Initialize the IntSoftmax layer."""
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the IntSoftmax layer."""
        # Extract number of classes
        n_classes = int(x.shape[-1] / 2)

        # Extract center and the radius
        center = x[:, :n_classes]
        radius = x[:, n_classes:]

        # Ensure the nonnegativity of radius
        radius_nonneg = F.softplus(radius)

        # Compute upper and lower probabilities
        exp_center = torch.exp(center)
        exp_center_sum = torch.sum(exp_center, dim=-1, keepdim=True)

        lo = torch.exp(center - radius_nonneg) / (exp_center_sum - exp_center + torch.exp(center - radius_nonneg))
        hi = torch.exp(center + radius_nonneg) / (exp_center_sum - exp_center + torch.exp(center + radius_nonneg))

        # Generate output
        output = torch.cat([lo, hi], dim=-1)

        return output
